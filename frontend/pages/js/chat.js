/**
 * Chat module for Lexi AI (angepasst für reines Text-Streaming)
 * Security: Uses DOMPurify for XSS protection
 */

// Use DOMPurify sanitization (loaded from CDN in HTML)
function sanitizeHTML(html) {
    // Use window.sanitize from sanitize.js for consistent XSS protection
    // For plain text messages, this escapes HTML entities
    // For future: Can be extended to allow safe HTML/Markdown formatting
    return window.sanitize ? window.sanitize.sanitizeUserContent(html) : html;
}

function addMessage(text, sender, id = null) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    if (id) messageDiv.id = id;

    // XSS-safe: User-generated chat messages must be sanitized
    window.sanitize.setInnerHTML(messageDiv, sanitizeHTML(text));

    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';

    const now = new Date();
    timeDiv.textContent = `${now.getHours()}:${String(now.getMinutes()).padStart(2, '0')}`;

    messageDiv.appendChild(timeDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function updateMessage(id, text) {
    const messageDiv = document.getElementById(id);
    if (messageDiv) {
        const timeDiv = messageDiv.querySelector('.message-time');
        // XSS-safe: Streaming messages must be sanitized
        window.sanitize.setInnerHTML(messageDiv, sanitizeHTML(text));
        if (timeDiv) messageDiv.appendChild(timeDiv);

        const chatMessages = document.getElementById('chat-messages'); // 🔧 DAS ist die fehlende Zeile
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}



function showTypingIndicator() {
    const chatMessages = document.getElementById('chat-messages');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message typing-indicator';
    typingDiv.id = 'typing-indicator';
    typingDiv.innerText = '...';
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTypingIndicator() {
    const typing = document.getElementById('typing-indicator');
    if (typing) typing.remove();
}

function sendMessage() {
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const messageText = chatInput.value.trim();

    if (messageText === '') return;

    addMessage(messageText, 'user');
    chatInput.value = '';

    showTypingIndicator();

    const isStreaming = document.querySelector('input[value="streaming"]').checked;

    fetch('/ui/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            message: messageText,
            user_id: "test_user",
            context: {},
            stream: isStreaming,
            model: document.getElementById('llm-model').value || "gemma3:4b-it-qat"
        })
    })
    .then(response => {
        if (isStreaming) {
            removeTypingIndicator();
            const responseId = 'response-' + Date.now();
            addMessage('', 'bot', responseId);

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let responseText = '';
            let buffer = '';

            function readChunk() {
                return reader.read().then(({ done, value }) => {
                    if (done) return;

                    // Decode the chunk and add to buffer
                    buffer += decoder.decode(value, { stream: true });

                    // Process complete SSE events (split by double newline or single newline)
                    const lines = buffer.split('\n');

                    // Keep the last incomplete line in the buffer
                    buffer = lines.pop() || '';

                    // Process each complete line
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const jsonStr = line.substring(6); // Remove 'data: ' prefix
                                const data = JSON.parse(jsonStr);

                                // Handle chunk data
                                if (data.chunk !== undefined) {
                                    responseText += data.chunk;
                                    updateMessage(responseId, responseText);
                                }
                                // Ignore metadata and completion events
                            } catch (e) {
                                console.warn('Failed to parse SSE data:', line, e);
                            }
                        }
                    }

                    return readChunk();
                }).catch(e => {
                    console.error('Streaming error:', e);
                    updateMessage(responseId, responseText || '⚠️ Fehler bei der Antwort.');
                });
            }

            return readChunk();
        } else {
            return response.json().then(data => {
                removeTypingIndicator();
                if (data.response) {
                    addMessage(data.response, 'bot');
                } else {
                    addMessage('Entschuldigung, ich konnte keine Antwort generieren.', 'bot');
                }
            });
        }
    })
    .catch(error => {
        console.error('Error connecting to the API:', error);
        removeTypingIndicator();
        addMessage('Verbindungsfehler zum API-Server. Bitte überprüfen Sie Ihre Einstellungen.', 'bot');
    });
}

export function initializeChat() {
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-message');

    sendButton.addEventListener('click', sendMessage);

    chatInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
}
