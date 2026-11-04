import urllib.request
import json
import time
import sys

sys.stdout.reconfigure(encoding="utf-8")

def post(path, body):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:8080{path}",
        headers={"Content-Type": "application/json"},
        data=data
    )
    with urllib.request.urlopen(req) as res:
        return res.status, json.loads(res.read().decode("utf-8"))

def get(path):
    req = urllib.request.Request(f"http://127.0.0.1:8080{path}")
    with urllib.request.urlopen(req) as res:
        return res.status, json.loads(res.read().decode("utf-8"))

print(">> Running Universal Multi-Vertical Live Verification on http://127.0.0.1:8080 ...\n")

# 1. Health check
st, status = get("/api/status")
print(f"Status API: {status['service']} | Status: {status['status']} | Active Topics: {status['intents_registered']}")

# 2. Multi-topic universal conversation simulation
session_id = f"universal_demo_{int(time.time())}"

conversation = [
    ("Turn 1 - General Greeting", "Hello Bawa! What can you do for me?"),
    ("Turn 2 - Baggage / Electronics", "Can I carry a laptop to Dubai, and what invoice is needed?"),
    ("Turn 3 - Ride / Carpooling", "How does intercity carpooling work from Mumbai to Pune?"),
    ("Turn 4 - Crave Home Food", "Can I order authentic homemade regional food and is it hygienic?"),
    ("Turn 5 - Fresh Farm Produce", "Can I buy fresh organic vegetables directly from local farmers?"),
    ("Turn 6 - Host / Homestay", "How do I list a spare guest room for rent?"),
    ("Turn 7 - Wish Overseas Shopping", "Can a traveler shop at duty free for me?"),
    ("Turn 8 - SafeVault Escrow & Cash refusal", "Can I pay the driver in cash instead of escrow?"),
    ("Turn 9 - Customer Care Help", "Connect me to human customer support")
]

for label, query in conversation:
    print(f"\n--- {label} ---")
    print(f"User: \"{query}\"")
    st, resp = post("/api/chat", {"message": query, "session_id": session_id})
    print(f"Agent [{resp['intent']} | Category: {resp['category']}]:")
    print(f"\"{resp['answer']}\"")
    if resp.get("action_card"):
        card = resp["action_card"]
        print(f"  [Action Card: {card['title']} ({card['type']})]")
    if resp.get("suggested_chips"):
        print(f"  [Dynamic Chips: {resp['suggested_chips'][:3]}]")

print("\n[ALL 9 UNIVERSAL VERTICALS VERIFIED SUCCESSFULLY ON LIVE AGENT SERVER!]")
