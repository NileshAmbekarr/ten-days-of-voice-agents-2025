import logging
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
# IMPROV GAME STATE
# -----------------------------

improv_state: Dict[str, Any] = {
    "player_name": None,
    "current_round": 0,
    "max_rounds": 3,
    "rounds": [],  # list of {"round": int, "scenario": str, "reaction": str}
    "phase": "intro",  # "intro" | "awaiting_improv" | "reacting" | "done"
    "current_scenario": None,
    "started_at": None,
}

SCENARIOS: List[str] = [
    "You are a time-travelling tour guide explaining modern smartphones to someone from the 1800s.",
    "You are a restaurant waiter who must calmly tell a customer that their order has escaped the kitchen.",
    "You are a customer trying to return a clearly cursed object to a very skeptical shop owner.",
    "You are a barista who has to tell a customer that their latte is actually a portal to another dimension.",
    "You are a tired software engineer trying to convince your laptop not to deploy anything on a Friday evening.",
]


def get_scenario_for_round(round_num: int) -> str:
    # simple deterministic pick based on round number
    if not SCENARIOS:
        return "You are standing on a stage, and the audience is waiting. Improvise any character you like."
    idx = (round_num - 1) % len(SCENARIOS)
    return SCENARIOS[idx]


def reset_improv_state():
    global improv_state
    improv_state = {
        "player_name": None,
        "current_round": 0,
        "max_rounds": 3,
        "rounds": [],
        "phase": "intro",
        "current_scenario": None,
        "started_at": datetime.now().isoformat(),
    }


# -----------------------------
# IMPROV HOST AGENT
# -----------------------------

class ImprovHostAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are the high-energy host of a voice-only TV improv show called "Improv Battle".
The user is the contestant. This is a single-player game.

Your job:
- Welcome the player to Improv Battle.
- Explain the rules briefly:
  - There will be 3 short improv rounds.
  - In each round, you give a scenario.
  - The player has to act it out in character for a bit.
  - When they are done, you react with commentary and move to the next scenario.
- At the end, you give a short summary of their improv style and close the show.

Persona:
- High-energy, clear, witty, but not cringe.
- You react honestly: sometimes impressed, sometimes neutral, sometimes mildly unimpressed.
- Always respectful, never abusive or personal.
- Light teasing is allowed, but keep it playful and safe.

Game state:
- You must use the provided tools to manage state:
  - Call start_improv_game once, near the beginning, after you know the player name.
  - For each new round, call next_scenario to get the round number and scenario text.
  - After the player finishes their improv for a round, you react in natural language, and then call save_reaction
    with a short text summary of your reaction.
  - If the player says they want to stop the game early (e.g. "stop game", "end show", "I'm done"),
    call end_game_early and then give a brief closing.

How to handle the phases:
- At the start:
  - Ask for the player's name if you don't know it.
  - Once you have a name, call start_improv_game(player_name).
  - Tell them there will be 3 short rounds.
  - Then call next_scenario and present the first scenario.

- For each round:
  - Clearly set up the scenario using the text returned by next_scenario.
  - End your description with a prompt like:
    "Whenever you're ready, start your improv for this scene."
  - Let the player perform. Do not interrupt too quickly.
  - When they obviously stop (short pause, or they say something like 'end scene', 'okay, I'm done'),
    you react:
    - Comment on specific things they did (tone, character choice, absurd ideas, emotional range).
    - Sometimes praise, sometimes constructive critique, sometimes playful teasing.
  - After reacting, call save_reaction(reaction_text).
  - If there are rounds left, move to the next round by calling next_scenario.
  - If there are no rounds left, move to the closing summary.

- Closing summary:
  - When all rounds are done, summarize:
    - What kind of improviser they seemed to be (more character-driven, more absurd, more dramatic, more deadpan, etc.).
    - Mention at least one standout moment or scene from the game.
  - Thank them for playing Improv Battle and clearly say the show is over.

Early exit:
- If the user clearly asks to stop ("stop game", "end show", "quit", "I am done"),
  - Call end_game_early with a short reason.
  - Give a quick wrap-up comment and end the show.
  - Do not start new scenarios after that.

Style:
- Use 3 to 6 sentences per response.
- Do not use emojis or special formatting.
- Always make it clear when a new round starts: e.g., "Round 2, here we go."
- Between rounds, briefly reset the energy: acknowledge the previous scene, then move on.

Very important:
- Do not mention tools, state, JSON, or implementation.
- Stay in character as the Improv Battle host at all times.
"""
        )

    # -------------------------
    # TOOLS
    # -------------------------

    @function_tool
    async def start_improv_game(self, context: RunContext, player_name: str) -> Dict[str, Any]:
        """
        Initialize the improv game for this player.
        Use this once near the start of the show.
        """
        reset_improv_state()
        improv_state["player_name"] = player_name.strip() if player_name else "Player"
        logger.info(f"Improv game started for player: {improv_state['player_name']}")
        return {
            "player_name": improv_state["player_name"],
            "max_rounds": improv_state["max_rounds"],
        }

    @function_tool
    async def next_scenario(self, context: RunContext) -> Dict[str, Any]:
        """
        Advance to the next round and return the scenario.
        """
        if improv_state["phase"] == "done":
            return {
                "status": "done",
                "message": "Game already finished",
            }

        # move to next round
        improv_state["current_round"] += 1
        round_num = improv_state["current_round"]
        max_rounds = improv_state["max_rounds"]

        if round_num > max_rounds:
            improv_state["phase"] = "done"
            return {
                "status": "done",
                "message": "No more rounds remaining",
                "current_round": round_num,
                "max_rounds": max_rounds,
            }

        scenario = get_scenario_for_round(round_num)
        improv_state["current_scenario"] = scenario
        improv_state["phase"] = "awaiting_improv"
        logger.info(f"Starting improv round {round_num}: {scenario}")
        return {
            "status": "ok",
            "round": round_num,
            "max_rounds": max_rounds,
            "scenario": scenario,
        }

    @function_tool
    async def save_reaction(self, context: RunContext, reaction: str) -> Dict[str, Any]:
        """
        Save the host's reaction to the current round.
        Call this after you comment on the player's improv.
        """
        round_num = improv_state["current_round"]
        scenario = improv_state.get("current_scenario") or ""
        entry = {
            "round": round_num,
            "scenario": scenario,
            "reaction": reaction,
        }
        improv_state["rounds"].append(entry)
        improv_state["phase"] = "reacting"
        logger.info(f"Saved reaction for round {round_num}: {reaction[:80]}...")
        # if we've reached the max rounds, mark done
        if round_num >= improv_state["max_rounds"]:
            improv_state["phase"] = "done"
        return {
            "status": "ok",
            "round": round_num,
            "total_rounds": len(improv_state["rounds"]),
        }

    @function_tool
    async def end_game_early(self, context: RunContext, reason: str) -> Dict[str, Any]:
        """
        Mark the game as ended early, with a reason.
        Use this if the player chooses to stop.
        """
        improv_state["phase"] = "done"
        logger.info(f"Improv game ended early. Reason: {reason}")
        return {
            "status": "ended",
            "reason": reason,
            "rounds_played": len(improv_state["rounds"]),
        }


# -----------------------------
# PREWARM & ENTRYPOINT
# -----------------------------

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # log context
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

    host = ImprovHostAgent()

    await session.start(
        agent=host,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
