// BawaGo Interactive AI Agent Frontend Controller
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

document.addEventListener("DOMContentLoaded", () => {
  checkAgentStatus();
  if (userInput) userInput.focus();
});

// Check Server / Model Health
async function checkAgentStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    if (data.status === "Online" && statusText) {
      statusText.innerText = `BawaGo Universal AI Guide · Online (${data.intents_registered}+ Topics · All 7 Verticals)`;
    }
  } catch (err) {
    if (statusText) statusText.innerText = "BawaGo Universal AI Guide · Connecting...";
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
    const data = await res.json();
    const elapsed = Math.round(performance.now() - startTime);

    // Natural typing delay (minimum 300ms)
    const delay = Math.max(0, 350 - elapsed);
    setTimeout(() => {
      showTyping(false);
      renderBotInteractiveResponse(data);
    }, delay);

  } catch (err) {
    showTyping(false);
    appendSimpleBotMessage("I am temporarily having trouble reaching our AI engine. Please ensure the BawaGo server is running on port 8080.");
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

  // Topic badge (clean, without any robotic match phrase)
  const categoryBadge = data.category ? `<span class="topic-tag">${escapeHtml(data.category)}</span>` : "";

  bubble.innerHTML = `
    <div class="bubble-header">
      <span class="bot-badge-name">🤖 Bawa</span>
      ${categoryBadge}
    </div>
    <div class="bubble-text" id="stream-target-${Date.now()}"></div>
    <div class="card-container"></div>
    <div class="feedback-row">
      <button class="feedback-btn" onclick="sendFeedback(this, '${escapeHtml(data.intent)}', 1)" title="Helpful answer">
        👍 Helpful
      </button>
      <button class="feedback-btn" onclick="sendFeedback(this, '${escapeHtml(data.intent)}', 0)" title="Not helpful">
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
    // Once text typing finishes, render action card if present
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
  
  // Create cursor
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
      setTimeout(nextChunk, 22); // ~45 words/sec natural cadence
    } else {
      if (cursor.parentNode) cursor.parentNode.removeChild(cursor);
      if (onComplete) onComplete();
      scrollToBottom();
    }
  }

  nextChunk();
}

// Render Action Card (Checklist, Flow, Warning)
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
    btn.className = "dynamic-chip-btn";
    btn.innerText = chipText;
    btn.onclick = () => {
      const cleanText = chipText.replace(/^[^\w\s]+/, '').trim();
      sendMessage(cleanText);
    };
    dynamicChipsBar.appendChild(btn);
  });

  dynamicChipsBar.style.display = "flex";
}

// Feedback submission
async function sendFeedback(btn, intent, rating) {
  const parent = btn.parentElement;
  parent.querySelectorAll(".feedback-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");

  try {
    await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        intent: intent,
        rating: rating,
        timestamp: Date.now()
      })
    });
  } catch (err) {
    // Silently continue
  }
}

// Reset Conversation
async function resetConversation() {
  try {
    await fetch("/api/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId })
    });
  } catch (e) {}

  // Generate fresh session
  sessionId = "sess_" + Math.random().toString(36).substring(2, 11) + "_" + Date.now();
  sessionStorage.setItem("bawago_agent_session_id", sessionId);

  // Clear messages
  chatStream.innerHTML = `
    <div class="welcome-card" id="welcome-card">
      <div class="welcome-icon">👋</div>
      <h2 class="welcome-title">Welcome to BawaGo!</h2>
      <p class="welcome-text">
        I am <strong>Bawa</strong>, your interactive AI assistant. I have real-time memory and know all 7 BawaGo verticals (Baggage pooling, SafeVault escrow, Parcels, Rides, Crave Food, Host, Wish, Fresh) and platform safety rules.
      </p>

      <div class="starter-chips-label">Popular questions you can ask:</div>
      <div class="starter-chips">
        <button class="chip-btn" onclick="sendStarterPrompt(this)">✈️ Can I carry a laptop to Dubai?</button>
        <button class="chip-btn" onclick="sendStarterPrompt(this)">🔒 How does SafeVault Escrow protect my money?</button>
        <button class="chip-btn" onclick="sendStarterPrompt(this)">📦 Can I send urgent parcels to another city?</button>
        <button class="chip-btn" onclick="sendStarterPrompt(this)">💰 What are the platform fees?</button>
      </div>
    </div>
  `;

  if (dynamicChipsBar) dynamicChipsBar.style.display = "none";
  if (contextPill) contextPill.style.display = "none";
  if (userInput) userInput.focus();
}

function scrollToBottom() {
  chatStream.scrollTop = chatStream.scrollHeight;
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
