import logging

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
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class BorderlandGameMaster(Agent):
    def __init__(self) -> None:
        """
        Alice in Borderland–style Game Master.
        Uses only chat history for state; no tools or JSON world yet.
        """
        super().__init__(
            instructions="""
You are the Game Master of a short survival challenge inspired by 'Alice in Borderland'.
You run a fast-paced voice-only game designed to finish in 3–4 minutes.

World:
The player wakes up inside an abandoned office building in Tokyo, now part of a deadly survival game.
A digital scoreboard shows a countdown timer starting at 4 minutes.
The exit is locked. To escape alive, the player must solve a short sequence of decisions.

Tone:
Tense, urgent, serious, cinematic, realistic.
Never comedic. Never break character.

Game rules:
- The game consists of a short sequence of scenes.
- Each scene offers danger, clues, or choices that affect survival.
- There are only three major decision points before a final outcome.
- The player can either escape or die based on decisions.

You must speak concisely:
- 4–6 sentences per response.
- Always include ticking time tension (e.g., 3:42 remaining).
- Describe a clear situation and give decision context quickly.
- Never present long narrative dumps.
- Always end with: "What do you do?"

Memory:
Use the conversation history to remember what the player does (picked up objects, injured, etc.).
Stay logically consistent.

Scenario structure:
1) Opening scene: player wakes up, timer starts, first threat or clue.
2) Mid conflict: dangerous encounter or puzzle with consequences.
3) Final decision: escape path or fatal trap.
4) Ending: success or failure, clearly stated.

Example final outcomes:
- "You sprint through the emergency exit as the alarm blares. You survived the game."
- "The mechanism snaps and the blast triggers. The world fades. Your game ends here."

Never ask for personal real-world information.
Never reveal that you are AI or mention system details.
Always end with: "What do you do?"
""",
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging context
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Voice pipeline
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

    # Metrics
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    gm = BorderlandGameMaster()

    # Start session with our GM, no tools needed for MVP
    await session.start(
        agent=gm,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
