// BawaGo Universal Interactive AI Agent Frontend Controller
const chatStream = document.getElementById("chat-stream");
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const typingIndicator = document.getElementById("typing-indicator");
const statusText = document.getElementById("agent-status-text");
const contextPill = document.getElementById("context-pill");
const dynamicChipsBar = document.getElementById("dynamic-chips-bar");

// Session Management (Preserved in sessionStorage per tab)
let sessionId = sessionStorage.getItem("bawago_agent_session_id");
if (!sessionId) {
  sessionId = "sess_" + Math.random().toString(36).substring(2, 11) + "_" + Date.now();
  sessionStorage.setItem("bawago_agent_session_id", sessionId);
}

// Client-Side Database & Fallback Engine
let clientIntents = null;

async function loadClientIntents() {
  if (clientIntents) return clientIntents;
  const paths = ["intent_answers.json", "/static/intent_answers.json", "./model/intent_answers.json"];
  for (const p of paths) {
    try {
      const res = await fetch(p);
      if (res.ok) {
        clientIntents = await res.json();
        return clientIntents;
      }
    } catch (e) {}
  }
  return null;
}

document.addEventListener("DOMContentLoaded", () => {
  loadClientIntents();
  checkAgentStatus();
  if (userInput) userInput.focus();
});

// Check Server / Model Health
async function checkAgentStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    if (data.status === "Online" && statusText) {
      statusText.innerText = "BawaGo Support · Online";
    }
  } catch (err) {
    if (statusText) statusText.innerText = "BawaGo Support · Online";
  }
}

// User form submit
function handleUserSubmit(event) {
  event.preventDefault();
  const text = userInput.value.trim();
  if (!text) return;
  sendMessage(text);
}

// Starter chips click
function sendStarterPrompt(button) {
  const text = button.innerText.replace(/^[^\w\s]+/, '').trim();
  sendMessage(text);
}

// Main message sender
async function sendMessage(text) {
  userInput.value = "";
  
  // Hide welcome card once conversation starts
  const welcome = document.getElementById("welcome-card");
  if (welcome) welcome.style.display = "none";

  // Append user bubble
  appendUserMessage(text);

  // Show typing indicator
  showTyping(true);

  try {
    const startTime = performance.now();
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        message: text,
        session_id: sessionId 
      })
    });
    if (!res.ok) throw new Error("API not ok");
    const data = await res.json();
    const elapsed = Math.round(performance.now() - startTime);

    // Natural typing delay (minimum 300ms)
    const delay = Math.max(0, 350 - elapsed);
    setTimeout(() => {
      showTyping(false);
      renderBotInteractiveResponse(data);
    }, delay);

  } catch (err) {
    // Client-side fallback engine for Static Spaces & offline environments
    await loadClientIntents();
    const data = predictClientSide(text);
    setTimeout(() => {
      showTyping(false);
      renderBotInteractiveResponse(data);
    }, 320);
  }
}

function showTyping(show) {
  if (!typingIndicator) return;
  if (show) {
    typingIndicator.classList.add("active");
  } else {
    typingIndicator.classList.remove("active");
  }
  scrollToBottom();
}

function appendUserMessage(text) {
  const row = document.createElement("div");
  row.className = "message-row user";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  bubble.innerHTML = `
    <div class="bubble-text">${escapeHtml(text)}</div>
    <div class="bubble-meta">${timeStr}</div>
  `;

  row.appendChild(bubble);
  chatStream.appendChild(row);
  scrollToBottom();
}

function appendSimpleBotMessage(text) {
  const row = document.createElement("div");
  row.className = "message-row bot";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = `<div class="bubble-text">${escapeHtml(text)}</div>`;
  row.appendChild(bubble);
  chatStream.appendChild(row);
  scrollToBottom();
}

// Interactive Typewriter Streaming Renderer
function renderBotInteractiveResponse(data) {
  const row = document.createElement("div");
  row.className = "message-row bot";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  // Clean category tag
  const categoryBadge = data.category ? `<span class="topic-tag">${escapeHtml(data.category)}</span>` : "";

  bubble.innerHTML = `
    <div class="bubble-header">
      <div class="bot-identity">
        <img src="bawa-avatar.png" alt="Bawa" class="bot-badge-img">
        <span class="bot-badge-name">Bawa</span>
      </div>
      ${categoryBadge}
    </div>
    <div class="bubble-text" id="stream-target-${Date.now()}"></div>
    <div class="card-container"></div>
    <div class="feedback-row">
      <button class="feedback-btn" type="button" onclick="sendFeedback(this, '${escapeHtml(data.intent || '')}', 1)" title="Helpful answer">
        👍 Helpful
      </button>
      <button class="feedback-btn" type="button" onclick="sendFeedback(this, '${escapeHtml(data.intent || '')}', 0)" title="Not helpful">
        👎 Needs improvement
      </button>
      <span class="bubble-meta" style="margin-left: auto;">${timeStr}</span>
    </div>
  `;

  row.appendChild(bubble);
  chatStream.appendChild(row);

  const textContainer = bubble.querySelector(".bubble-text");
  const cardContainer = bubble.querySelector(".card-container");

  // Typewriter streaming effect
  streamText(textContainer, data.answer, () => {
    if (data.action_card) {
      renderActionCard(cardContainer, data.action_card);
    }
  });

  // Update context pill in header
  updateContextPill(data.session_context);

  // Update dynamic chips
  renderDynamicChips(data.suggested_chips);

  scrollToBottom();
}

// Typewriter streaming helper
function streamText(container, fullText, onComplete) {
  let index = 0;
  const words = fullText.split(" ");
  
  const cursor = document.createElement("span");
  cursor.className = "typing-cursor";
  container.appendChild(cursor);

  function nextChunk() {
    if (index < words.length) {
      const nextWord = words[index] + " ";
      const textNode = document.createTextNode(nextWord);
      container.insertBefore(textNode, cursor);
      index++;
      scrollToBottom();
      setTimeout(nextChunk, 18);
    } else {
      if (cursor.parentNode) cursor.parentNode.removeChild(cursor);
      if (onComplete) onComplete();
      scrollToBottom();
    }
  }

  nextChunk();
}

// Render Action Card
function renderActionCard(container, card) {
  const cardDiv = document.createElement("div");
  cardDiv.className = `action-card ${escapeHtml(card.type || 'checklist')}`;

  let contentHtml = `<div class="action-card-title">${escapeHtml(card.title)}</div>`;

  if (card.items) {
    contentHtml += `<ul class="action-card-list">`;
    card.items.forEach(it => {
      contentHtml += `<li>${escapeHtml(it)}</li>`;
    });
    contentHtml += `</ul>`;
  } else if (card.steps) {
    contentHtml += `<div class="flow-steps">`;
    card.steps.forEach(st => {
      contentHtml += `<div class="flow-step"><span>🔹</span><span>${escapeHtml(st)}</span></div>`;
    });
    contentHtml += `</div>`;
  }

  cardDiv.innerHTML = contentHtml;
  container.appendChild(cardDiv);
}

// Update header context pill
function updateContextPill(context) {
  if (!contextPill) return;
  if (context && context.category && context.category !== "General") {
    contextPill.innerText = `🌐 ${context.category}`;
    contextPill.style.display = "inline-block";
  } else {
    contextPill.style.display = "none";
  }
}

// Render Dynamic Chips
function renderDynamicChips(chips) {
  if (!dynamicChipsBar) return;
  if (!chips || chips.length === 0) {
    dynamicChipsBar.style.display = "none";
    return;
  }

  dynamicChipsBar.innerHTML = "";
  chips.forEach(chipText => {
    const btn = document.createElement("button");
    btn.className = "dynamic-chip";
    btn.type = "button";
    btn.innerText = chipText;
    btn.onclick = () => {
      const clean = chipText.replace(/^[^\w\s]+/, '').trim();
      sendMessage(clean);
    };
    dynamicChipsBar.appendChild(btn);
  });

  dynamicChipsBar.style.display = "flex";
  scrollToBottom();
}

// Reset conversation
async function resetConversation() {
  sessionId = "sess_" + Math.random().toString(36).substring(2, 11) + "_" + Date.now();
  sessionStorage.setItem("bawago_agent_session_id", sessionId);

  try {
    await fetch("/api/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId })
    });
  } catch (err) {}

  chatStream.innerHTML = "";
  const welcome = document.getElementById("welcome-card");
  if (welcome) welcome.style.display = "flex";
  if (dynamicChipsBar) dynamicChipsBar.style.display = "none";
  if (contextPill) contextPill.style.display = "none";
  if (userInput) userInput.focus();
}

// Send Feedback
async function sendFeedback(btn, intent, score) {
  const row = btn.parentNode;
  row.querySelectorAll(".feedback-btn").forEach(b => {
    b.classList.remove("active-helpful", "active-improve");
  });

  if (score === 1) {
    btn.classList.add("active-helpful");
    btn.innerHTML = "✓ Helpful";
  } else {
    btn.classList.add("active-improve");
    btn.innerHTML = "✓ Feedback noted";
  }

  try {
    await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        intent: intent,
        score: score
      })
    });
  } catch (err) {}
}

function scrollToBottom() {
  setTimeout(() => {
    chatStream.scrollTop = chatStream.scrollHeight;
  }, 50);
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// ==========================================
// CLIENT-SIDE ENGINE (FOR STATIC SPACES / OFFLINE)
// ==========================================
const KEYWORD_INTENTS = {
  "invoice": "baggage-laptop-electronics",
  "macbook": "baggage-laptop-electronics",
  "laptop": "baggage-laptop-electronics",
  "electronics": "baggage-laptop-electronics",
  "carpool": "ride-intercity-carpool",
  "homestay": "host-spare-room-listing",
  "duty free": "wish-personal-shopper-overseas",
  "personal shopper": "wish-personal-shopper-overseas",
  "hygiene": "home-food-hygiene-safety",
  "fssai": "home-food-hygiene-safety",
  "prohibited": "parcel-prohibited-dangerous-goods",
  "contraband": "parcel-prohibited-dangerous-goods",
  "human support": "support-human-agent-escalation",
  "real person": "support-human-agent-escalation",
  "escrow": "safevault-deposit-escrow-hold",
  "otp": "safevault-six-digit-otp-payout",
  "produce": "fresh-farm-produce-ordering",
  "farmer": "fresh-farm-produce-ordering",
  "commission": "pricing-fee-commission-split",
  "percentage": "pricing-fee-commission-split",
  "fee": "pricing-fee-commission-split",
  "suitcase": "baggage-rules-excess",
  "books": "parcel-items-allowed"
};

const clientMemory = {
  destination: null,
  items: [],
  historyCount: 0
};

function predictClientSide(userMessage) {
  const q = userMessage.toLowerCase().trim();
  clientMemory.historyCount++;

  // Entity Extraction
  const cities = ["dubai", "london", "singapore", "mumbai", "delhi", "pune", "hyderabad", "bangalore", "toronto", "bali"];
  cities.forEach(c => {
    if (q.includes(c)) clientMemory.destination = c.charAt(0).toUpperCase() + c.slice(1);
  });
  const items = ["laptop", "macbook", "phone", "food", "tiffin", "spices", "clothes", "books", "medicine"];
  items.forEach(it => {
    if (q.includes(it) && !clientMemory.items.includes(it)) clientMemory.items.push(it);
  });

  let matchedIntent = "general-about-bawago-platform";
  let maxScore = -1;

  // 1. Direct Keyword Boost
  for (const [kw, kwIntent] of Object.entries(KEYWORD_INTENTS)) {
    if (q.includes(kw)) {
      matchedIntent = kwIntent;
      maxScore = 100;
      break;
    }
  }

  // 2. Token Overlap Match
  if (maxScore < 50 && clientIntents) {
    const qTokens = q.replace(/[^\w\s]/g, " ").split(/\s+/).filter(w => w.length > 2);
    for (const [intentId, info] of Object.entries(clientIntents)) {
      let score = 0;
      const targetText = (intentId + " " + (info.canonical_question || "") + " " + (info.answer || "")).toLowerCase();
      qTokens.forEach(tok => {
        if (targetText.includes(tok)) score += 1;
      });
      if (score > maxScore) {
        maxScore = score;
        matchedIntent = intentId;
      }
    }
  }

  const intentInfo = (clientIntents && clientIntents[matchedIntent]) 
    ? clientIntents[matchedIntent] 
    : {
        category: "General",
        answer: "BawaGo connects community members for secure baggage sharing, carpooling, homestays, home-cooked food, personal shopping, and farm produce."
      };

  const category = intentInfo.category || "General";
  let answer = intentInfo.answer || "";

  // Dynamic Action Cards
  let actionCard = null;
  const iLow = matchedIntent.toLowerCase();
  if (iLow.includes("laptop") || iLow.includes("electronics") || iLow.includes("invoice")) {
    actionCard = {
      type: "checklist",
      title: "Aviation Electronics Checklist",
      items: [
        "Cabin Hand-Carry Only (No Checked Cargo)",
        "Sender Purchase Invoice Required",
        "Airport Power-On Inspection Test"
      ]
    };
  } else if (iLow.includes("escrow") || iLow.includes("otp") || iLow.includes("safevault")) {
    actionCard = {
      type: "flow",
      title: "SafeVault 3-Step Escrow Guarantee",
      steps: [
        "1. Deposit Held in Escrow",
        "2. Physical Verification at Handover",
        "3. 6-Digit OTP Unlocks Payout"
      ]
    };
  } else if (iLow.includes("prohibited") || iLow.includes("contraband")) {
    actionCard = {
      type: "checklist",
      title: "Prohibited Goods (Zero-Tolerance)",
      items: [
        "No Liquids, Aerosols, or Unsealed Chemicals",
        "No Loose Lithium Batteries Outside Cabin",
        "No Prescription Drugs Without Valid Medical Doctor Rx"
      ]
    };
  } else if (iLow.includes("hygiene") || iLow.includes("fssai") || iLow.includes("food")) {
    actionCard = {
      type: "checklist",
      title: "Crave Home Kitchen Hygiene Standards",
      items: [
        "FSSAI / Municipal Food Safety Registration",
        "Daily Fresh Cooked Batch Inspection",
        "Tamper-Evident Thermal Sealed Packaging"
      ]
    };
  } else if (iLow.includes("carpool") || iLow.includes("ride")) {
    actionCard = {
      type: "flow",
      title: "Intercity Carpool Booking Process",
      steps: [
        "1. Driver Posts Verified Route & Empty Seats",
        "2. Passenger Reserves & Escrows Fair Share",
        "3. Ride Check-In via Mutual OTP at Departure"
      ]
    };
  } else if (iLow.includes("host") || iLow.includes("room")) {
    actionCard = {
      type: "checklist",
      title: "Homestay Host Verification Steps",
      items: [
        "Government ID & Property Ownership Proof",
        "Safety & Cleanliness Verification",
        "SafeVault Automatic Payout 24h After Guest Check-In"
      ]
    };
  } else if (iLow.includes("wish") || iLow.includes("shopper")) {
    actionCard = {
      type: "checklist",
      title: "Personal Shopper Sourcing Guide",
      items: [
        "Duty-Free & Regional Specialties Allowed",
        "Original Store Purchase Receipt Required",
        "SafeVault Reimburses Cost + Traveler Reward"
      ]
    };
  } else if (iLow.includes("fresh") || iLow.includes("farmer")) {
    actionCard = {
      type: "checklist",
      title: "Farm-to-Doorstep Harvest Standards",
      items: [
        "Direct Harvest From Verified Local Farmers",
        "Pesticide-Free & Organic Traceability",
        "Delivered Fresh Within Hours of Harvest"
      ]
    };
  } else if (iLow.includes("support") || iLow.includes("human")) {
    actionCard = {
      type: "flow",
      title: "Human Support Escalation",
      steps: [
        "1. In-App Ticket or Email: support@bawago.com",
        "2. Dedicated Agent Review Within 2 Hours",
        "3. SafeVault Dispute & Claim Mediation"
      ]
    };
  }

  // Dynamic Category Chips
  const categoryChips = {
    "Baggage": ["✈️ What items are allowed?", "💼 What invoice documents are needed?", "⚖️ What is the maximum luggage weight?"],
    "SafeVault": ["🔒 How does SafeVault protect my funds?", "🔢 Why do I need a 6-digit OTP?", "💵 What is the refund policy?"],
    "Ride": ["🚗 How does carpooling work?", "⛽ How is fuel cost calculated?", "🛡️ How are drivers verified?"],
    "Home Food": ["🍲 How are kitchen hygiene standards checked?", "🌱 Can I order vegan or Jain options?", "📦 How is hot food packaged?"],
    "Host": ["🏡 How do I list my spare room?", "🔑 How does guest check-in work?", "💰 What is the host payout schedule?"],
    "Wish": ["🛍️ Can a traveler shop duty-free for me?", "🧾 Is store purchase invoice required?", "💵 When does the traveler get paid?"],
    "Fresh": ["🌾 Can I order directly from local farmers?", "🚚 How fast is fresh harvest delivered?", "🥦 Are crops organic certified?"],
    "Parcel": ["📦 What items are prohibited?", "⚖️ What is the parcel size limit?", "🔍 How do I track transit in real time?"],
    "Safety": ["🛡️ How is traveler identity verified?", "✈️ Are airline PNRs checked?", "🔒 What happens if a dispute occurs?"],
    "Support": ["📞 Connect me to human agent", "📧 What is the support email?", "⏱️ What is the emergency response time?"]
  };

  const chips = categoryChips[category] || [
    "✈️ Flight baggage pooling",
    "🔒 SafeVault escrow",
    "🚗 Intercity rides",
    "🍲 Crave home meals"
  ];

  return {
    answer: answer,
    intent: matchedIntent,
    category: category,
    suggested_chips: chips,
    action_card: actionCard,
    session_context: {
      destination: clientMemory.destination,
      items: clientMemory.items,
      category: category,
      turns_count: clientMemory.historyCount
    }
  };
}
