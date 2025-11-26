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
# FAQ LOADING & SIMPLE SEARCH
# -----------------------------


FAQ_DATA: Dict[str, Any] = {}
FAQ_ENTRIES: List[Dict[str, str]] = []


def _load_faq() -> None:
    """Load TakeUForward FAQ / company info from JSON."""
    global FAQ_DATA, FAQ_ENTRIES

    possible_paths = [
        os.path.join("..", "shared-data", "tuf_faq.json"),
        os.path.join("shared-data", "tuf_faq.json"),
        "tuf_faq.json",
    ]

    faq_raw: Optional[Dict[str, Any]] = None
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    faq_raw = json.load(f)
                logger.info(f"Loaded TakeUForward FAQ from {path}")
                break
            except Exception as e:
                logger.error(f"Failed to load FAQ from {path}: {e}")

    if faq_raw is None:
        # Fallback minimal data so agent doesn't completely break
        logger.warning("No tuf_faq.json found. Using fallback FAQ data.")
        faq_raw = {
            "company": "TakeUForward",
            "tagline": "Helping learners with DSA and interview preparation.",
            "description": "TakeUForward is focused on teaching data structures, algorithms and interview skills.",
            "pricing": "We have multiple courses and some free resources. Exact pricing depends on the course and time.",
            "faq": [
                {
                    "q": "What does your product do?",
                    "a": "TakeUForward helps people learn DSA, system design and crack coding interviews.",
                },
                {
                    "q": "Who is this for?",
                    "a": "It is for students and professionals preparing for tech interviews.",
                },
                {
                    "q": "Do you have a free tier?",
                    "a": "We provide free YouTube content and some resources; detailed paid courses are available separately.",
                },
            ],
        }

    FAQ_DATA = faq_raw
    FAQ_ENTRIES = faq_raw.get("faq", [])


def faq_search_answer(query: str) -> str:
    """
    Extremely simple FAQ match: keyword overlap count.
    Returns best matching answer or a fallback using description/pricing.
    """
    if not FAQ_ENTRIES:
        desc = FAQ_DATA.get("description", "")
        pricing = FAQ_DATA.get("pricing", "")
        if desc or pricing:
            return desc + ("\n\nPricing: " + pricing if pricing else "")
        return "I don't have enough information to answer that based on my current FAQ data."

    q_tokens = [t for t in query.lower().split() if len(t) > 2]
    best_item = None
    best_score = 0

    for item in FAQ_ENTRIES:
        text = (item.get("q", "") + " " + item.get("a", "")).lower()
        score = sum(1 for tok in q_tokens if tok in text)
        if score > best_score:
            best_score = score
            best_item = item

    if best_item and best_score > 0:
        return best_item.get("a", "").strip() or "I found a related FAQ but it has no answer text."

    # fallback: description / pricing
    desc = FAQ_DATA.get("description", "")
    pricing = FAQ_DATA.get("pricing", "")
    if desc or pricing:
        return desc + ("\n\nPricing: " + pricing if pricing else "")

    return "I don't have enough information in my FAQ data to answer that."


# Load FAQ at import time
_load_faq()


# -----------------------------
# LEAD STATE & PERSISTENCE
# -----------------------------


lead_state: Dict[str, Any] = {
    "name": None,
    "company": None,
    "email": None,
    "role": None,
    "use_case": None,
    "team_size": None,
    "timeline": None,
}

LEADS_DIR = "leads"
LEADS_FILE = os.path.join(LEADS_DIR, "tuf_leads.json")


def reset_lead_state() -> None:
    global lead_state
    lead_state = {
        "name": None,
        "company": None,
        "email": None,
        "role": None,
        "use_case": None,
        "team_size": None,
        "timeline": None,
    }


# -----------------------------
# ASSISTANT (SDR) AGENT
# -----------------------------


class SDRAssistant(Agent):
    def __init__(self) -> None:
        company = FAQ_DATA.get("company", "TakeUForward")
        tagline = FAQ_DATA.get("tagline", "Helping learners with DSA and interviews.")

        super().__init__(
            instructions=f"""
You are a friendly, professional Sales Development Representative (SDR) for the Indian company '{company}'.

High-level:
- You talk to visitors who are interested in {company}.
- You greet them warmly and sound like a human SDR.
- You ask what brought them here and what they're working on.
- You focus the conversation on understanding their needs and whether {company} can help.

You have access to a small FAQ and company info through tools.
VERY IMPORTANT:
- When the user asks about the product, who it's for, what it does, or pricing,
  you MUST call the faq_search tool with their question text.
- Then answer ONLY based on the tool output. Do NOT make up extra factual details.
- If you don't know something, say that you don't have that info and suggest they check the website.

Lead collection:
You need to naturally collect the following fields during the conversation:
- name
- company
- email
- role
- use_case (what they want to use this for)
- team_size
- timeline (now / soon / later or similar)

As the user provides each piece of information:
- Call update_lead(field="<field>", value="<value>").
- Confirm briefly: e.g. "Got it, you're a backend engineer at X", etc.
- Do NOT aggressively interrogate them; keep it conversational and supportive.

End of call:
- If the user says things like "that's all", "I'm done", "thanks", or clearly wants to end:
  1) Give a short verbal summary:
     - who they are (name, role, company if available)
     - what they want to use {company} for (use_case)
     - rough timeline
  2) Then call save_lead(summary="<your short summary>").
  3) After calling save_lead, close politely.

Style:
- Friendly, concise, no slang, no emojis.
- Ask one question at a time.
- If user changes topic, gently bring back to understanding their needs and collecting lead info.
""",
        )

    # ------------- TOOLS -------------

    @function_tool
    async def faq_search(self, context: RunContext, query: str) -> str:
        """
        Search the TakeUForward FAQ / company content for an answer.

        Use this tool whenever the user asks:
        - What does your product do?
        - Who is this for?
        - Do you have free tier / pricing?
        - Any other product / company / pricing related question.

        Args:
            query: The user's question in natural language.
        """
        logger.info(f"FAQ search called with query: {query!r}")
        return faq_search_answer(query)

    @function_tool
    async def update_lead(self, context: RunContext, field: str, value: str) -> str:
        """
        Update a single lead field.

        field: one of name, company, email, role, use_case, team_size, timeline.
        value: the user's provided value for that field.
        """
        global lead_state
        field_norm = field.strip().lower()
        if field_norm not in lead_state:
            return f"Field '{field}' is not a valid lead field."

        lead_state[field_norm] = value.strip()
        logger.info(f"Updated lead field {field_norm} -> {value!r}")
        return f"Updated {field_norm}"

    @function_tool
    async def save_lead(self, context: RunContext, summary: str) -> str:
        """
        Persist the collected lead to a JSON file.

        summary: A short natural language summary of who they are and what they want.
        """
        os.makedirs(LEADS_DIR, exist_ok=True)

        leads: List[Dict[str, Any]] = []
        if os.path.exists(LEADS_FILE):
            try:
                with open(LEADS_FILE, "r", encoding="utf-8") as f:
                    leads = json.load(f)
            except Exception as e:
                logger.error(f"Failed to read existing leads file: {e}")

        entry = {
            "timestamp": datetime.now().isoformat(),
            "lead": lead_state.copy(),
            "summary": summary.strip(),
        }
        leads.append(entry)

        with open(LEADS_FILE, "w", encoding="utf-8") as f:
            json.dump(leads, f, indent=2)

        logger.info(f"Saved lead entry at {entry['timestamp']}")
        # Reset for potential future calls
        reset_lead_state()

        return "Lead saved"


# -----------------------------
# PREWARM & ENTRYPOINT
# -----------------------------


def prewarm(proc: JobProcess):
    # Preload VAD
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging context
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Voice pipeline: Deepgram STT, Gemini LLM, Murf TTS
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

    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    assistant = SDRAssistant()

    # Start the SDR session
    await session.start(
        agent=assistant,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Connect to the room / user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
