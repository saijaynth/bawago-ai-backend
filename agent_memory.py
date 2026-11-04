"""
Conversational Memory and Context Tracking for BawaGo AI Agent
Implements multi-turn session tracking, entity extraction, sliding window buffer,
and anti-repetition memory based on the conversation-memory skill pattern.
"""

import re
import time
from typing import Dict, List, Any, Optional

# Known entity catalogs for BawaGo domain
KNOWN_CITIES = [
    "dubai", "hyderabad", "london", "singapore", "mumbai", "delhi", "pune",
    "jakarta", "bali", "bangalore", "chennai", "kolkata", "abu dhabi", "doha"
]

KNOWN_ITEMS = [
    "laptop", "macbook", "phone", "iphone", "ipad", "tablet", "electronics",
    "documents", "passport", "chocolates", "clothes", "suitcase", "baggage",
    "luggage", "food", "tiffin", "vegetables", "fruits", "dairy", "parcel"
]

KNOWN_ROLES = ["traveler", "driver", "sender", "passenger", "host", "chef", "customer", "buyer"]

class SessionState:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history: List[Dict[str, Any]] = []  # [{role, content, intent, timestamp}]
        self.entities: Dict[str, Any] = {
            "origin": None,
            "destination": None,
            "items": [],
            "roles": [],
            "verticals": []
        }
        self.recent_intents: List[str] = []
        self.recent_responses: List[str] = []  # For anti-duplication fingerprinting
        self.created_at = time.time()
        self.last_active = time.time()

    def update_activity(self):
        self.last_active = time.time()

    def add_message(self, role: str, content: str, intent: Optional[str] = None):
        self.update_activity()
        self.history.append({
            "role": role,
            "content": content,
            "intent": intent,
            "timestamp": time.time()
        })
        # Keep sliding window of last 10 messages
        if len(self.history) > 10:
            self.history = self.history[-10:]

        if role == "assistant":
            self.recent_responses.append(content)
            if len(self.recent_responses) > 6:
                self.recent_responses.pop(0)

        if intent:
            self.recent_intents.append(intent)
            if len(self.recent_intents) > 5:
                self.recent_intents.pop(0)

    def is_duplicate_response(self, candidate: str) -> bool:
        """Checks if the candidate response is identical or overly similar to recent outputs."""
        candidate_clean = " ".join(candidate.lower().split()[:15])
        for past in self.recent_responses:
            past_clean = " ".join(past.lower().split()[:15])
            if candidate_clean == past_clean:
                return True
        return False

class MemoryManager:
    """Manages active user conversation sessions and context extraction."""
    def __init__(self):
        self.sessions: Dict[str, SessionState] = {}

    def get_session(self, session_id: str) -> SessionState:
        if not session_id:
            session_id = "default_guest"
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(session_id)
        session = self.sessions[session_id]
        session.update_activity()
        return session

    def reset_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
        return self.get_session(session_id)

    def extract_entities(self, text: str, session: SessionState):
        """Extracts destination, origin, items, and roles from user text and persists to session."""
        t_lower = text.lower()

        # Route matching: "from X to Y" or "to Y"
        route_match = re.search(r"from\s+([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+)", t_lower)
        if route_match:
            orig = route_match.group(1).strip()
            dest = route_match.group(2).strip()
            for city in KNOWN_CITIES:
                if city in orig: session.entities["origin"] = city.title()
                if city in dest: session.entities["destination"] = city.title()
        else:
            to_match = re.search(r"(?:to|visiting|heading to|flying to)\s+([a-zA-Z\s]+)", t_lower)
            if to_match:
                cand = to_match.group(1).strip()
                for city in KNOWN_CITIES:
                    if city in cand: session.entities["destination"] = city.title()

        # Check explicit cities
        for city in KNOWN_CITIES:
            if re.search(r"\b" + re.escape(city) + r"\b", t_lower):
                if not session.entities["destination"] and city.title() != session.entities["origin"]:
                    session.entities["destination"] = city.title()

        # Check items
        for item in KNOWN_ITEMS:
            if re.search(r"\b" + re.escape(item) + r"\b", t_lower):
                if item not in session.entities["items"]:
                    session.entities["items"].append(item)

        # Check roles
        for role in KNOWN_ROLES:
            if re.search(r"\b" + re.escape(role) + r"\b", t_lower):
                if role not in session.entities["roles"]:
                    session.entities["roles"].append(role)

    def get_prompt_context(self, session: SessionState) -> str:
        """Builds a compact context string of session memory for prompt grounding."""
        items = []
        if session.entities.get("destination"):
            items.append(f"Destination: {session.entities['destination']}")
        if session.entities.get("origin"):
            items.append(f"Origin: {session.entities['origin']}")
        if session.entities.get("items"):
            items.append(f"Mentioned Items: {', '.join(session.entities['items'])}")
        if session.entities.get("roles"):
            items.append(f"User Role: {', '.join(session.entities['roles'])}")
        if session.recent_intents:
            items.append(f"Recent Topics: {', '.join(session.recent_intents[-3:])}")
        return " | ".join(items) if items else "New conversation"

# Singleton global memory instance
memory_manager = MemoryManager()

