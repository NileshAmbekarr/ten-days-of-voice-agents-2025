import logging
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    metrics,
    tokenize,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


@dataclass
class TutorSessionInfo:
    current_topic_id: Optional[str] = None   # e.g. "variables"
    current_mode: Optional[str] = None       # "learn" | "quiz" | "teach_back"


def _load_tutor_content() -> Dict[str, Dict]:
    """
    Load tutor concepts from JSON.
    Tries ../shared-data then shared-data relative to backend.
    Falls back to built-in sample if not found.
    """
    paths = [
        os.path.join("..", "shared-data", "day4_tutor_content.json"),
        os.path.join("shared-data", "day4_tutor_content.json"),
        "day4_tutor_content.json",
    ]
    raw: Optional[List[Dict]] = None
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                logger.info(f"Loaded tutor content from {p}")
                break
            except Exception as e:
                logger.error(f"Failed to load tutor content from {p}: {e}")

    if raw is None:
        logger.warning("No day4_tutor_content.json found, using built-in sample content")
        raw = [
            {
                "id": "variables",
                "title": "Variables",
                "summary": "Variables store values so you can reuse them later. They have a name and hold data like numbers or text.",
                "sample_question": "What is a variable and why is it useful?",
            },
            {
                "id": "loops",
                "title": "Loops",
                "summary": "Loops let you repeat an action multiple times without rewriting code. They keep running while a condition is true.",
                "sample_question": "Explain the difference between a for loop and a while loop.",
            },
        ]
    by_id: Dict[str, Dict] = {}
    for item in raw:
        tid = str(item.get("id", "")).strip().lower()
        if not tid:
            continue
        by_id[tid] = item
    return by_id


TUTOR_CONTENT: Dict[str, Dict] = _load_tutor_content()


def _normalize_mode(mode: str) -> Optional[str]:
    m = (mode or "").strip().lower().replace("-", "_").replace(" ", "_")
    if m in ("learn", "quiz", "teach_back"):
        return m
    # accept "teachback", "teach"
    if m in ("teachback", "teach"):
        return "teach_back"
    return None


def _topic_display_list() -> str:
    if not TUTOR_CONTENT:
        return "variables, loops"
    return ", ".join(v.get("title", tid).title() for tid, v in TUTOR_CONTENT.items())


def _topic_id_from_name(name: str) -> Optional[str]:
    if not name:
        return None
    name_norm = name.strip().lower()
    # direct id match
    if name_norm in TUTOR_CONTENT:
        return name_norm
    # title match
    for tid, item in TUTOR_CONTENT.items():
        title = str(item.get("title", "")).strip().lower()
        if title == name_norm:
            return tid
    return None


class LearnAgent(Agent):
    def __init__(self, chat_ctx: Optional[ChatContext] = None):
        topics_list = _topic_display_list()
        super().__init__(
            instructions=f"""
You are an active recall learning tutor in LEARN mode.

Your job:
- Explain one programming concept at a time in simple language.
- Use the provided summary for the selected concept as the base.
- Keep explanations short, structured, and focused on understanding.
- After explaining, ask if the learner wants to continue in LEARN mode,
  switch to QUIZ mode, or TEACH_BACK mode.

Important:
- The current topic id is stored in session.userdata.current_topic_id.
- Use that to decide which concept you're teaching.
- Do NOT invent new topics. Stay within this set: {topics_list}.
- Ask one question at a time.
""",
            chat_ctx=chat_ctx,
            tts=murf.TTS(
                voice="en-US-matthew",  # LEARN voice
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True,
            ),
        )

    async def on_enter(self) -> None:
        userdata: TutorSessionInfo = self.session.userdata
        topic_id = userdata.current_topic_id
        if not topic_id or topic_id not in TUTOR_CONTENT:
            await self.session.generate_reply(
                instructions="We lost track of the topic. Ask the user to choose a valid topic like variables or loops."
            )
            return
        topic = TUTOR_CONTENT[topic_id]
        title = topic.get("title", topic_id.title())
        summary = topic.get("summary", "")
        await self.session.generate_reply(
            instructions=f"""
Explain the concept '{title}' based ONLY on this summary:

{summary}

Keep it structured:
1) What it is
2) Why it matters
3) One simple example

Then ask the user if they want to switch to quiz mode or teach-back mode.
"""
        )

    @function_tool()
    async def switch_mode(self, context: RunContext[TutorSessionInfo], mode: str):
        """Switch learning mode while keeping the current topic. Valid modes: learn, quiz, teach_back."""
        mode_n = _normalize_mode(mode)
        if mode_n is None:
            return "Invalid mode. Valid modes are: learn, quiz, teach_back."
        if not context.userdata.current_topic_id:
            return "No topic is selected yet. Ask the user which topic they want to work on first."
        context.userdata.current_mode = mode_n
        topic_id = context.userdata.current_topic_id
        topic = TUTOR_CONTENT.get(topic_id, {})
        title = topic.get("title", topic_id.title())
        chat_ctx = self.session._chat_ctx
        if mode_n == "learn":
            return None  # already here
        if mode_n == "quiz":
            return QuizAgent(chat_ctx=chat_ctx), f"Switching to quiz mode for {title}."
        if mode_n == "teach_back":
            return TeachBackAgent(chat_ctx=chat_ctx), f"Switching to teach-back mode for {title}."
        return "Unexpected error while switching modes."


class QuizAgent(Agent):
    def __init__(self, chat_ctx: Optional[ChatContext] = None):
        topics_list = _topic_display_list()
        super().__init__(
            instructions=f"""
You are an active recall tutor in QUIZ mode.

Your job:
- Ask short, targeted questions about the current concept.
- Base initial questions on the sample_question from the content file.
- Listen to the user's answer, then:
  - Briefly acknowledge what they got right.
  - Point out 1–2 missing pieces if any.
- Then either:
  - Ask a follow-up quiz question, OR
  - Ask if they want to switch to TEACH_BACK mode.

Rules:
- Ask ONE question at a time.
- Do NOT give full answers before the user tries.
- Be supportive but honest about gaps.
- Do NOT invent new topics. Stay within this set: {topics_list}.
""",
            chat_ctx=chat_ctx,
            tts=murf.TTS(
                voice="en-US-alicia",  # QUIZ voice
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True,
            ),
        )

    async def on_enter(self) -> None:
        userdata: TutorSessionInfo = self.session.userdata
        topic_id = userdata.current_topic_id
        if not topic_id or topic_id not in TUTOR_CONTENT:
            await self.session.generate_reply(
                instructions="We don't have a topic selected. Ask the user which topic they want to be quizzed on."
            )
            return
        topic = TUTOR_CONTENT[topic_id]
        sample_q = topic.get("sample_question") or "Explain this concept in your own words."
        await self.session.generate_reply(
            instructions=f"""
Start quiz mode for the current topic.

First, ask this question (rephrased naturally if you want):

{sample_q}

Wait for the user's answer before continuing.
"""
        )

    @function_tool()
    async def switch_mode(self, context: RunContext[TutorSessionInfo], mode: str):
        """Switch learning mode while keeping the current topic. Valid modes: learn, quiz, teach_back."""
        mode_n = _normalize_mode(mode)
        if mode_n is None:
            return "Invalid mode. Valid modes are: learn, quiz, teach_back."
        if not context.userdata.current_topic_id:
            return "No topic is selected yet. Ask the user which topic they want to work on first."
        context.userdata.current_mode = mode_n
        topic_id = context.userdata.current_topic_id
        topic = TUTOR_CONTENT.get(topic_id, {})
        title = topic.get("title", topic_id.title())
        chat_ctx = self.session._chat_ctx
        if mode_n == "learn":
            return LearnAgent(chat_ctx=chat_ctx), f"Switching to learn mode for {title}."
        if mode_n == "quiz":
            return None  # already here
        if mode_n == "teach_back":
            return TeachBackAgent(chat_ctx=chat_ctx), f"Switching to teach-back mode for {title}."
        return "Unexpected error while switching modes."


class TeachBackAgent(Agent):
    def __init__(self, chat_ctx: Optional[ChatContext] = None):
        topics_list = _topic_display_list()
        super().__init__(
            instructions=f"""
You are an active recall tutor in TEACH-BACK mode.

Your job:
- Ask the learner to explain the current concept back to you in their own words.
- Listen carefully to their explanation.
- Then give brief, honest feedback:
  - What they explained well.
  - What key idea they missed (if any).
- Optionally suggest 1 small improvement or mental model.

Rules:
- Do NOT use clinical or judgmental language.
- Stay focused on learning, not grading.
- Keep feedback to 2–4 short sentences.
- Do NOT invent new topics. Stay within this set: {topics_list}.
""",
            chat_ctx=chat_ctx,
            tts=murf.TTS(
                voice="en-US-ken",  # TEACH-BACK voice
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True,
            ),
        )

    async def on_enter(self) -> None:
        userdata: TutorSessionInfo = self.session.userdata
        topic_id = userdata.current_topic_id
        if not topic_id or topic_id not in TUTOR_CONTENT:
            await self.session.generate_reply(
                instructions="We don't have a topic selected. Ask the user which concept they want to teach back."
            )
            return
        topic = TUTOR_CONTENT[topic_id]
        title = topic.get("title", topic_id.title())
        sample_q = topic.get("sample_question") or f"Explain {title} in your own words."
        await self.session.generate_reply(
            instructions=f"""
Start teach-back mode.

Ask the user to explain the concept '{title}' in their own words.
You can base your request on this sample question:

{sample_q}

After their explanation, give concise feedback:
- 1–2 things they did well
- 1 key detail they could add or clarify
"""
        )

    @function_tool()
    async def switch_mode(self, context: RunContext[TutorSessionInfo], mode: str):
        """Switch learning mode while keeping the current topic. Valid modes: learn, quiz, teach_back."""
        mode_n = _normalize_mode(mode)
        if mode_n is None:
            return "Invalid mode. Valid modes are: learn, quiz, teach_back."
        if not context.userdata.current_topic_id:
            return "No topic is selected yet. Ask the user which topic they want to work on first."
        context.userdata.current_mode = mode_n
        topic_id = context.userdata.current_topic_id
        topic = TUTOR_CONTENT.get(topic_id, {})
        title = topic.get("title", topic_id.title())
        chat_ctx = self.session._chat_ctx
        if mode_n == "learn":
            return LearnAgent(chat_ctx=chat_ctx), f"Switching to learn mode for {title}."
        if mode_n == "quiz":
            return QuizAgent(chat_ctx=chat_ctx), f"Switching to quiz mode for {title}."
        if mode_n == "teach_back":
            return None  # already here
        return "Unexpected error while switching modes."


class RouterAgent(Agent):
    def __init__(self):
        topics_list = _topic_display_list()
        super().__init__(
            instructions=f"""
You are an ACTIVE RECALL COACH that routes the learner between three modes:
- learn: you explain the concept.
- quiz: you ask questions.
- teach_back: you ask the learner to explain and then give feedback.

FIRST STEP:
1) Greet the user.
2) Briefly explain the three modes in simple words.
3) Ask which TOPIC and which MODE they want.

Available topics (from a small internal course file):
{topics_list}

Available modes:
- learn
- quiz
- teach_back

State is stored in:
- session.userdata.current_topic_id
- session.userdata.current_mode

Behavior:
- Use the tool select_topic_and_mode once the user has clearly chosen a topic and mode.
- After handoff, let the specialist agent (learn / quiz / teach_back) control the conversation.
- At any time, if the user says things like "switch to quiz" or "let's do teach back",
  call the switch_mode tool (available on specialist agents) with the requested mode.

Rules:
- Ask one question at a time.
- If the user is vague, ask clarifying questions instead of guessing.
- Do not invent new topics beyond the provided list.
""",
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="""
Greet the user warmly.
Explain that you can help them learn using three modes: learn, quiz, and teach-back.
Then ask:
1) Which concept they want to work on.
2) Which mode they want to start with.
"""
        )

    @function_tool()
    async def select_topic_and_mode(
        self,
        context: RunContext[TutorSessionInfo],
        topic: str,
        mode: str,
    ):
        """
        Select the active topic and initial mode, then hand off to the right tutor agent.

        topic: user-chosen topic name or id (e.g. "variables", "loops")
        mode: one of learn, quiz, teach_back
        """
        tid = _topic_id_from_name(topic)
        if tid is None:
            return (
                f"I couldn't find the topic '{topic}'. Please choose one of: {_topic_display_list()}."
            )
        mode_n = _normalize_mode(mode)
        if mode_n is None:
            return "Invalid mode. Valid modes are: learn, quiz, teach_back."

        context.userdata.current_topic_id = tid
        context.userdata.current_mode = mode_n

        topic_obj = TUTOR_CONTENT.get(tid, {})
        title = topic_obj.get("title", tid.title())
        chat_ctx = self.session._chat_ctx

        if mode_n == "learn":
            return LearnAgent(chat_ctx=chat_ctx), f"Great choice. Let's start by learning {title}."
        if mode_n == "quiz":
            return QuizAgent(chat_ctx=chat_ctx), f"Okay, let's quiz you on {title}."
        if mode_n == "teach_back":
            return TeachBackAgent(chat_ctx=chat_ctx), f"Nice. Let's have you teach back {title}."

        return "Unexpected error while selecting topic and mode."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging context
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Voice pipeline: Deepgram STT, Gemini LLM, Murf TTS
    session: AgentSession[TutorSessionInfo] = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        # Default TTS (per-agent overrides above)
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        userdata=TutorSessionInfo(),
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

    # Start with RouterAgent (mode selector)
    await session.start(
        agent=RouterAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Connect to room / user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
