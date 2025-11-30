import logging
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel
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
# DATA LOADERS
# -----------------------------

CATALOG_PATHS = [
    os.path.join("..", "shared-data", "ecom_catalog.json"),
    os.path.join("shared-data", "ecom_catalog.json"),
    "shared-data/ecom_catalog.json",
]

PRODUCTS: List[Dict[str, Any]] = []
ORDERS: List[Dict[str, Any]] = []

ORDERS_DIR = "orders"
ORDERS_FILE = os.path.join(ORDERS_DIR, "ecom_orders.json")


def load_catalog() -> None:
    global PRODUCTS
    for path in CATALOG_PATHS:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            PRODUCTS = raw.get("products", [])
            logger.info(f"Loaded catalog successfully with {len(PRODUCTS)} products")
            return

    logger.warning("Could not find catalog file. Starting with empty list.")
    PRODUCTS = []


load_catalog()


def filter_products(category=None, max_price=None, color=None, name_query=None):
    results = PRODUCTS
    if category:
        results = [p for p in results if p.get("category", "").lower() == category.lower()]
    if max_price:
        results = [p for p in results if p.get("price", 9999999) <= max_price]
    if color:
        results = [p for p in results if p.get("color", "").lower() == color.lower()]
    if name_query:
        q = name_query.lower()
        results = [p for p in results if q in p["name"].lower() or q in p.get("description", "").lower()]
    return results


def find_product_by_id(pid: str):
    for p in PRODUCTS:
        if p["id"] == pid:
            return p
    return None


def persist_orders() -> None:
    os.makedirs(ORDERS_DIR, exist_ok=True)
    with open(ORDERS_FILE, "w", encoding="utf-8") as f:
        json.dump(ORDERS, f, indent=2)


# -----------------------------
# Pydantic argument models
# -----------------------------

class SearchArgs(BaseModel):
    category: Optional[str] = None
    max_price: Optional[int] = None
    color: Optional[str] = None
    name_query: Optional[str] = None


class OrderArgs(BaseModel):
    product_id: str
    quantity: Optional[int] = 1


# -----------------------------
# ASSISTANT IMPLEMENTATION
# -----------------------------

class EcommerceAssistant(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
You are a friendly voice shopping assistant.
You help users explore items like mugs, t-shirts, hoodies, bottles and accessories.
Your job is to:
- Understand what they want to browse.
- Call the search tool when they ask for products.
- Summarize up to 3 items with name + price.
- Allow references like 'the second hoodie' or 'the black one'.
- Help them place orders using the order tool.
- Ask missing details clearly if required.
- Confirm their order with totals.

When the user asks to view what they previously bought, call the last order tool.

Speak naturally like a helpful store assistant.
Don't mention internal tools, JSON, IDs, or implementation details.
If nothing matches, politely say you don’t have anything like that.

Always end with a question if the conversation should continue.
"""
        )

    # ----------- TOOL: SEARCH PRODUCTS ----------

    @function_tool
    async def search_products(self, context: RunContext, args: SearchArgs):
        results = filter_products(
            category=args.category,
            max_price=args.max_price,
            color=args.color,
            name_query=args.name_query,
        )

        summaries = []
        for idx, p in enumerate(results, start=1):
            summaries.append({
                "index": idx,
                "id": p["id"],
                "name": p["name"],
                "price": p["price"],
                "currency": p["currency"],
                "color": p.get("color"),
                "category": p.get("category"),
            })

        return summaries

    # ----------- TOOL: CREATE ORDER ----------

    @function_tool
    async def create_order(self, context: RunContext, args: OrderArgs):
        global ORDERS
        product = find_product_by_id(args.product_id)
        if not product:
            return {"message": "not_found"}

        quantity = args.quantity or 1
        total = quantity * product["price"]
        order_id = len(ORDERS) + 1

        order = {
            "id": order_id,
            "created_at": datetime.now().isoformat(),
            "items": [
                {
                    "product_id": product["id"],
                    "name": product["name"],
                    "quantity": quantity,
                    "unit_price": product["price"],
                    "total": total
                }
            ],
            "total": total,
            "currency": product["currency"],
        }

        ORDERS.append(order)
        persist_orders()
        return order

    # ----------- TOOL: GET LAST ORDER ----------

    @function_tool
    async def get_last_order(self, context: RunContext):
        if not ORDERS:
            return {"message": "no_orders"}
        return ORDERS[-1]


# -----------------------------
# PREWARM + ENTRYPOINT
# -----------------------------

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
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

    assistant = EcommerceAssistant()

    await session.start(
        agent=assistant,
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
