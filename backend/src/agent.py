import logging
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


# -----------------------------
# LOAD FRAUD DATABASE
# -----------------------------

fraud_cases: List[Dict[str, Any]] = []
DB_PATH = os.path.join("shared-data", "fraud_cases.json")


def load_cases():
    global fraud_cases
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r", encoding="utf-8") as f:
            fraud_cases = json.load(f)
            logger.info("Loaded fraud cases DB")
    else:
        logger.error("Fraud DB not found. Using empty fallback.")
        fraud_cases = []


load_cases()

active_case: Optional[Dict[str, Any]] = None
call_state = {
    "verified": False,
    "case_loaded": False,
    "decision": None
}


def find_case_by_name(name: str) -> Optional[Dict[str, Any]]:
    for case in fraud_cases:
        if case.get("userName", "").lower() == name.lower():
            return case
    return None


# -----------------------------
# FRAUD AGENT
# -----------------------------

class FraudAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are a Fraud Prevention Officer from Slice Bank (demo environment).
Speak professionally, calmly, and clearly.
NEVER ask for card numbers, PIN, CVV, password, OTP, Aadhaar, PAN or sensitive information.
Only use the verification question stored in the fraud case.

CALL FLOW RULES:
1. Greet the user and introduce yourself as part of Slice Bank Fraud Alert team.
2. Ask for their name to locate their fraud case.
3. If no case found → politely end call.
4. Ask the configured securityIdentifier question (e.g., favorite color).
5. If wrong → call update_case(status="verification_failed") and end call politely.
6. If correct → read out suspicious transaction details:
   - Amount, merchant, time, location, masked card
7. Ask: "Did you authorize this transaction? Yes or No?"
8. If yes → call update_case(status="confirmed_safe", note="Customer approved transaction")
9. If no → call update_case(status="confirmed_fraud", note="Customer denied transaction")
10. Speak short summary and end call.

IMPORTANT:
Use the tool update_case only when decision is final.
""",
        )

    # TOOLS
    @function_tool
    async def load_case(self, context: RunContext, name: str) -> str:
        """Load a fraud case based on customer name."""
        global active_case, call_state
        case = find_case_by_name(name)
        if case:
            active_case = case
            call_state["case_loaded"] = True
            logger.info(f"Loaded case for {name}")
            return f"Case loaded for {name}."
        return "No fraud case found for that name."

    @function_tool
    async def verify_user(self, context: RunContext, answer: str) -> str:
        """Check security identifier answer"""
        global call_state, active_case

        if not active_case:
            return "No case loaded."

        correct = active_case.get("securityIdentifier", "").lower()
        if answer.lower().strip() == correct:
            call_state["verified"] = True
            logger.info("User verified successfully")
            return "verified"
        else:
            call_state["verified"] = False
            logger.info("Verification failed")
            return "failed"

    @function_tool
    async def update_case(self, context: RunContext, status: str, note: str) -> str:
        """Persist case update to fraud_cases.json."""
        global active_case, fraud_cases

        if not active_case:
            return "No case to update."

        active_case["status"] = status
        active_case["note"] = note
        active_case["updated_at"] = datetime.now().isoformat()

        with open(DB_PATH, "w", encoding="utf-8") as f:
            json.dump(fraud_cases, f, indent=2)

        logger.info(f"Case updated: {status} - {note}")
        return "Case updated"


# -----------------------------
# PREWARM & ENTRYPOINT
# -----------------------------

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info(f"Usage summary: {usage_collector.get_summary()}")

    ctx.add_shutdown_callback(log_usage)

    agent = FraudAgent()

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
