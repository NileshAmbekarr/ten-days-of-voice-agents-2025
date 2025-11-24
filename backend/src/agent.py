import logging
from datetime import datetime
import json
import os
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
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

checkin_state = {
    "mood": None,
    "energy": None,
    "goals": [],
}


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="""
                    You are a supportive, grounded, and calm Health & Wellness voice companion.
                    You talk to the user once per day for a short check-in.
                    Avoid all medical advice, diagnosis, or therapy language.

                    Your job is to:
                    1. Ask how they are feeling today and their energy level (1-10)
                    2. Ask if anything is stressing them or affecting their mood.
                    3. Ask for 1-3 goals or intentions for the day.
                    4. Ask only one question at a time.
                    5. When all information is collected, summarize concisely.
                    6. Then call the tool `save_checkin` to permanently store the record.

                    You MUST reference previous check-ins when available.
                    For example: “Last time you mentioned low energy. How is today compared to that?”

                    Keep responses encouraging, realistic, grounded, and short.
                    When all fields (mood, energy, goals) are collected, call the tool save_checkin with no additional text.
                    Never end the conversation without calling save_checkin.

                """,
        )

    
    @function_tool
    async def update_checkin(context: RunContext, field: str, value: str):
        """Update part of today's check-in state."""
        global checkin_state
        if field == "goals":
            checkin_state["goals"].append(value)
        else:
            checkin_state[field] = value
        return "updated"


    @function_tool
    async def save_checkin(context: RunContext):
        """Save this check-in entry into wellness_log.json"""

        global checkin_state
        entry = {
            "timestamp": datetime.now().isoformat(),
            "mood": checkin_state["mood"],
            "energy": checkin_state["energy"],
            "goals": checkin_state["goals"],
            "summary": f"Feeling {checkin_state['mood']} with energy {checkin_state['energy']} and goals {checkin_state['goals']}"
        }

        os.makedirs("wellness", exist_ok=True)
        logfile = "wellness/wellness_log.json"
        existing = []

        if os.path.exists(logfile):
            with open(logfile, "r") as f:
                existing = json.load(f)

        existing.append(entry)

        with open(logfile, "w") as f:
            json.dump(existing, f, indent=2)
        
        print("SAVE TOOL EXECUTED")

        return "saved"

    

    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Load previous logs
    previous_entries = []
    logfile = "wellness/wellness_log.json"
    if os.path.exists(logfile):
        with open(logfile, "r") as f:
            previous_entries = json.load(f)

    if len(previous_entries) > 0:
        last = previous_entries[-1]
        hint = f"Last time we talked, you described feeling {last['mood']} with energy level {last['energy']}. How is today compared to that?"
    else:
        hint = "Hey! How are you feeling today?"


    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    
    # Inject this into system message context
    session.llm.system_message = hint


    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
