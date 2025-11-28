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
# CATALOG LOAD + RECIPES
# -----------------------------

CATALOG_PATHS = [
    os.path.join("..", "shared-data", "instamart_catalog.json"),
    os.path.join("shared-data", "instamart_catalog.json"),
    "instamart_catalog.json",
]

CATALOG: List[Dict[str, Any]] = []


def load_catalog():
    global CATALOG
    for p in CATALOG_PATHS:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                CATALOG = data.get("items", [])
                logger.info(f"Loaded Instamart catalog from {p} with {len(CATALOG)} items")
                return
            except Exception as e:
                logger.error(f"Failed to load catalog from {p}: {e}")
    logger.warning("No Instamart catalog found, using empty catalog.")
    CATALOG = []


load_catalog()


def find_item_by_name(name: str) -> Optional[Dict[str, Any]]:
    name = name.lower().strip()
    # simple contains-based match
    for item in CATALOG:
        if name in item["name"].lower():
            return item
    return None


# simple recipes mapping: dish -> list of item names
RECIPES: Dict[str, List[str]] = {
    "peanut butter sandwich": ["Bread", "Peanut Butter"],
    "sandwich": ["Whole Wheat Bread", "Cheese Slices (10 pcs)"],
    "pasta": ["Pasta", "Tomato Pasta Sauce"],
}

# -----------------------------
# CART & ORDER STORAGE
# -----------------------------

cart: List[Dict[str, Any]] = []

ORDERS_DIR = "orders"
ORDERS_FILE = os.path.join(ORDERS_DIR, "instamart_orders.json")


def reset_cart():
    global cart
    cart = []


def cart_total() -> int:
    return sum(item["price"] * item["qty"] for item in cart)


# -----------------------------
# INSTAMART ASSISTANT
# -----------------------------

class InstamartAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are a friendly voice ordering assistant for Swiggy Instamart (demo).
You help users order groceries, snacks, and simple meal ingredients.

Core behavior:
- Greet the user and explain you can help them order groceries and simple meal ingredients.
- Ask what they would like to order.
- For each item request, ask for missing details like quantity or variant if not clear.
- When the user mentions something like:
    "ingredients for a peanut butter sandwich"
    "ingredients for pasta"
  you must call the add_recipe_to_cart tool with the appropriate recipe name.

Cart rules:
- You do NOT invent items. You only use items from the catalog.
- For normal items, call add_item_to_cart(name, quantity).
- To remove items, call remove_item_from_cart(name).
- To show the cart, call list_cart().
- After each add/remove/update, briefly confirm what changed.

Order placement:
- When the user says they are done (e.g. "that's all", "place my order", "I'm done"),
  you MUST:
  1) Call list_cart() to get the final cart summary.
  2) Confirm the total and contents with the user.
  3) Ask for a simple customer name and address in one or two short questions.
  4) Then call place_order(customer_name, address) to persist the order.
  5) After place_order returns, clearly say that the order has been placed.

Constraints:
- Keep responses short and conversational.
- Ask ONE question at a time.
- Do not hallucinate new categories, items, or prices beyond the catalog.
- If an item is not available, say it's not in the Instamart demo catalog.
""",
        )

    # TOOLS

    @function_tool
    async def add_item_to_cart(
        self,
        context: RunContext,
        name: str,
        quantity: int = 1,
    ) -> str:
        """
        Add a specific catalog item to the cart.

        name: partial or full item name (e.g., "bread", "pasta")
        quantity: how many units to add (default 1)
        """
        global cart
        item = find_item_by_name(name)
        if not item:
            return f"I couldn't find '{name}' in the Instamart demo catalog."

        if quantity <= 0:
            quantity = 1

        # Check if already in cart
        for entry in cart:
            if entry["id"] == item["id"]:
                entry["qty"] += quantity
                logger.info(f"Updated cart item {item['name']} to qty {entry['qty']}")
                return f"Updated {item['name']} to quantity {entry['qty']} in your cart."

        cart.append(
            {
                "id": item["id"],
                "name": item["name"],
                "price": item["price"],
                "qty": quantity,
            }
        )
        logger.info(f"Added {quantity} x {item['name']} to cart")
        return f"Added {quantity} x {item['name']} to your cart."

    @function_tool
    async def remove_item_from_cart(
        self,
        context: RunContext,
        name: str,
    ) -> str:
        """
        Remove an item from the cart by name (or reduce quantity if needed).

        name: partial or full item name.
        """
        global cart
        name = name.lower().strip()
        new_cart = []
        removed_any = False
        for entry in cart:
            if name in entry["name"].lower():
                removed_any = True
                logger.info(f"Removed {entry['name']} from cart")
                continue
            new_cart.append(entry)

        cart = new_cart
        if removed_any:
            return f"Removed {name} from your cart."
        return f"I couldn't find {name} in your cart."

    @function_tool
    async def list_cart(self, context: RunContext) -> str:
        """
        List current items in the cart and total price.
        """
        if not cart:
            return "Your cart is currently empty."

        lines = []
        for entry in cart:
            lines.append(
                f"{entry['qty']} x {entry['name']} (₹{entry['price']} each)"
            )
        total = cart_total()
        summary = "; ".join(lines)
        logger.info(f"Cart summary: {summary} | total={total}")
        return f"Your cart has: {summary}. Total so far is ₹{total}."

    @function_tool
    async def add_recipe_to_cart(
        self,
        context: RunContext,
        recipe_name: str,
        servings: int = 1,
    ) -> str:
        """
        Add all items needed for a simple recipe to the cart.

        recipe_name: e.g. "peanut butter sandwich", "pasta"
        servings: multiplier for quantities (basic usage only)
        """
        global cart
        key = recipe_name.lower().strip()
        if key not in RECIPES:
            return f"I don't have a recipe for {recipe_name} in this demo."

        added_items = []
        for item_name in RECIPES[key]:
            item = find_item_by_name(item_name)
            if not item:
                continue
            quantity = max(servings, 1)
            # update or append
            for entry in cart:
                if entry["id"] == item["id"]:
                    entry["qty"] += quantity
                    break
            else:
                cart.append(
                    {
                        "id": item["id"],
                        "name": item["name"],
                        "price": item["price"],
                        "qty": quantity,
                    }
                )
            added_items.append(f"{quantity} x {item['name']}")

        if not added_items:
            return f"I couldn't find any matching items in the catalog for {recipe_name}."
        added_str = ", ".join(added_items)
        logger.info(f"Recipe {recipe_name} added: {added_str}")
        return f"I've added {added_str} to your cart for {recipe_name}."

    @function_tool
    async def place_order(
        self,
        context: RunContext,
        customer_name: str,
        address: str,
    ) -> str:
        """
        Place the current cart as an order and save it to JSON.

        customer_name: simple text name
        address: free-form address string
        """
        global cart

        if not cart:
            return "Your cart is empty; there is nothing to place."

        os.makedirs(ORDERS_DIR, exist_ok=True)

        orders: List[Dict[str, Any]] = []
        if os.path.exists(ORDERS_FILE):
            try:
                with open(ORDERS_FILE, "r", encoding="utf-8") as f:
                    orders = json.load(f)
            except Exception as e:
                logger.error(f"Failed reading existing orders: {e}")

        order_id = len(orders) + 1
        total = cart_total()
        order_items = [
            {
                "name": entry["name"],
                "qty": entry["qty"],
                "price": entry["price"],
                "subtotal": entry["price"] * entry["qty"],
            }
            for entry in cart
        ]

        order = {
            "id": order_id,
            "timestamp": datetime.now().isoformat(),
            "customer_name": customer_name.strip(),
            "address": address.strip(),
            "items": order_items,
            "total": total,
            "status": "placed",
        }

        orders.append(order)

        with open(ORDERS_FILE, "w", encoding="utf-8") as f:
            json.dump(orders, f, indent=2)

        logger.info(f"Placed order #{order_id} for {customer_name}, total ₹{total}")

        # reset cart for next session
        reset_cart()

        return f"Order #{order_id} placed successfully with total ₹{total}."


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

    assistant = InstamartAssistant()

    await session.start(
        agent=assistant,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
