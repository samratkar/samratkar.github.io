// Sleep Assistant Chatbot - Main JavaScript File
// Handles vector database loading, similarity search, and chat functionality

let vectorDB = null;
let chatHistory = [];

// Initialize on page load
window.addEventListener('DOMContentLoaded', async () => {
    await loadVectorDatabase();
});

// Load the vector database
async function loadVectorDatabase() {
    const overlay = document.getElementById('loadingOverlay');
    overlay.classList.remove('hidden');
    
    try {
        const response = await fetch('vectordb.json');
        vectorDB = await response.json();
        console.log('Vector database loaded successfully:', vectorDB.name);
        console.log('Total documents:', vectorDB.documents.length);
    } catch (error) {
        console.error('Error loading vector database:', error);
        addBotMessage('Sorry, I encountered an error loading my knowledge base. Please refresh the page and try again.');
    } finally {
        overlay.classList.add('hidden');
    }
}

// Simple text embedding using character-level TF-IDF (simplified)
function simpleEmbedding(text) {
    const words = text.toLowerCase().split(/\s+/);
    const embedding = new Array(384).fill(0); // Match vectorDB dimension
    
    // Simple hash-based embedding
    words.forEach((word, idx) => {
        for (let i = 0; i < word.length; i++) {
            const charCode = word.charCodeAt(i);
            const index = (charCode * (i + 1) + idx) % 384;
            embedding[index] += 1.0 / (word.length + 1);
        }
    });
    
    // Normalize
    const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    return embedding.map(val => magnitude > 0 ? val / magnitude : 0);
}

// Cosine similarity between two vectors
function cosineSimilarity(vecA, vecB) {
    if (vecA.length !== vecB.length) return 0;
    
    let dotProduct = 0;
    let magA = 0;
    let magB = 0;
    
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        magA += vecA[i] * vecA[i];
        magB += vecB[i] * vecB[i];
    }
    
    magA = Math.sqrt(magA);
    magB = Math.sqrt(magB);
    
    if (magA === 0 || magB === 0) return 0;
    return dotProduct / (magA * magB);
}

// Search for relevant documents using keyword matching and simple semantic similarity
function searchRelevantDocuments(query, topK = 3) {
    if (!vectorDB || !vectorDB.documents) {
        return [];
    }
    
    const queryLower = query.toLowerCase();
    const queryWords = new Set(queryLower.split(/\s+/));
    
    // Calculate relevance scores for each document
    const scores = vectorDB.documents.map((doc, index) => {
        const docLower = doc.toLowerCase();
        
        // Keyword matching score
        let keywordScore = 0;
        queryWords.forEach(word => {
            if (word.length > 3 && docLower.includes(word)) {
                keywordScore += 1;
            }
        });
        
        // Bonus for exact phrase match
        if (docLower.includes(queryLower)) {
            keywordScore += 5;
        }
        
        // Simple position bonus (earlier mentions are more relevant)
        const firstOccurrence = docLower.indexOf(queryLower);
        const positionScore = firstOccurrence >= 0 ? (1000 - firstOccurrence) / 1000 : 0;
        
        return {
            index,
            document: doc,
            score: keywordScore + positionScore
        };
    });
    
    // Sort by score and return top K
    return scores
        .filter(item => item.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, topK);
}

// Extract title and authors from document text
function extractMetadata(document) {
    const lines = document.split('\n').filter(line => line.trim());
    let title = '';
    let authors = '';
    
    // Look for title patterns
    for (let i = 0; i < Math.min(10, lines.length); i++) {
        const line = lines[i].trim();
        if (line.includes('TITLE') || line.includes('Title')) {
            title = lines[i + 1]?.trim() || line.replace(/TITLE|Title/gi, '').trim();
            break;
        }
    }
    
    // Extract first meaningful line as title if not found
    if (!title) {
        title = lines.find(line => line.length > 20 && line.length < 200) || 'Research Paper';
        title = title.substring(0, 150);
    }
    
    // Look for authors
    for (let i = 0; i < Math.min(15, lines.length); i++) {
        const line = lines[i].toLowerCase();
        if (line.includes('author') && !line.includes('corresponding')) {
            const nextLines = lines.slice(i + 1, i + 4).join(' ');
            authors = nextLines.substring(0, 100);
            break;
        }
    }
    
    return { title, authors };
}

// Generate response based on relevant documents
function generateResponse(query, relevantDocs) {
    if (relevantDocs.length === 0) {
        return {
            answer: "I couldn't find specific information about that in my knowledge base. " +
                    "However, I have extensive information about sleep stages, sleep disorders, " +
                    "sleep apnea, vitamin D effects on sleep, and sleep duration recommendations. " +
                    "Could you rephrase your question or try one of the suggested topics?",
            sources: []
        };
    }
    
    // Extract key information from top document
    const topDoc = relevantDocs[0].document;
    const excerptLength = 500;
    
    // Find the most relevant excerpt
    let bestExcerpt = topDoc.substring(0, excerptLength);
    const queryLower = query.toLowerCase();
    const queryWords = queryLower.split(/\s+/).filter(w => w.length > 3);
    
    // Try to find excerpt with most query words
    if (queryWords.length > 0) {
        let bestScore = 0;
        let bestStart = 0;
        
        for (let i = 0; i < topDoc.length - excerptLength; i += 100) {
            const excerpt = topDoc.substring(i, i + excerptLength).toLowerCase();
            const score = queryWords.filter(word => excerpt.includes(word)).length;
            
            if (score > bestScore) {
                bestScore = score;
                bestStart = i;
            }
        }
        
        if (bestScore > 0) {
            // Adjust start to beginning of sentence
            let start = bestStart;
            while (start > 0 && topDoc[start] !== '.' && topDoc[start] !== '\\n') {
                start--;
            }
            if (topDoc[start] === '.' || topDoc[start] === '\\n') start++;
            
            bestExcerpt = topDoc.substring(start, Math.min(start + excerptLength, topDoc.length));
        }
    }
    
    // Clean up the excerpt
    bestExcerpt = bestExcerpt.trim();
    if (bestExcerpt.length === excerptLength && !bestExcerpt.endsWith('.')) {
        bestExcerpt += '...';
    }
    
    // Generate contextual answer
    let answer = `Based on the research literature, here's what I found:\\n\\n${bestExcerpt}\\n\\n`;
    
    if (relevantDocs.length > 1) {
        answer += `I found ${relevantDocs.length} relevant sections in my knowledge base that might help answer your question.`;
    }
    
    // Extract metadata from relevant documents
    const sources = relevantDocs.map((doc, idx) => {
        const metadata = extractMetadata(doc.document);
        return {
            number: idx + 1,
            title: metadata.title,
            authors: metadata.authors,
            score: doc.score.toFixed(2)
        };
    });
    
    return {
        answer: answer,
        sources: sources
    };
}

// Handle sending messages
async function sendMessage() {
    const input = document.getElementById('userInput');
    const message = input.value.trim();
    
    if (!message) return;
    
    // Disable input while processing
    input.disabled = true;
    document.getElementById('sendButton').disabled = true;
    
    // Add user message
    addUserMessage(message);
    input.value = '';
    
    // Show typing indicator
    showTypingIndicator();
    
    // Simulate processing delay for better UX
    await new Promise(resolve => setTimeout(resolve, 800));
    
    // Search and generate response
    const relevantDocs = searchRelevantDocuments(message);
    const response = generateResponse(message, relevantDocs);
    
    // Debug logging
    console.log('Relevant docs found:', relevantDocs.length);
    console.log('Response sources:', response.sources);
    
    // Hide typing indicator and show response
    hideTypingIndicator();
    addBotMessage(response.answer, response.sources);
    
    // Re-enable input
    input.disabled = false;
    document.getElementById('sendButton').disabled = false;
    input.focus();
    
    // Save to chat history
    chatHistory.push({
        user: message,
        bot: response.answer,
        timestamp: new Date().toISOString()
    });
}

// Add user message to chat
function addUserMessage(message) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user';
    messageDiv.innerHTML = `
        <div class="message-content">${escapeHtml(message)}</div>
    `;
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// Add bot message to chat
function addBotMessage(message, sources = []) {
    console.log('addBotMessage called with sources:', sources);
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot';
    
    let sourceHtml = '';
    if (sources && sources.length > 0) {
        sourceHtml = '<div class="source-references">';
        sourceHtml += '<div class="source-title">📚 Sources:</div>';
        sources.forEach(source => {
            sourceHtml += `
                <div class="source-item">
                    <strong>Source ${source.number}</strong> (Relevance: ${source.score})<br>
                    <em>${escapeHtml(source.title)}</em>
                    ${source.authors ? `<br><small>${escapeHtml(source.authors)}</small>` : ''}
                </div>
            `;
        });
        sourceHtml += '</div>';
    }
    
    messageDiv.innerHTML = `
        <div class="message-content">
            ${formatMessage(message)}
            ${sourceHtml}
        </div>
    `;
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// Show typing indicator
function showTypingIndicator() {
    const chatMessages = document.getElementById('chatMessages');
    const indicator = document.createElement('div');
    indicator.className = 'message bot';
    indicator.id = 'typingIndicator';
    indicator.innerHTML = `
        <div class="typing-indicator active">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;
    chatMessages.appendChild(indicator);
    scrollToBottom();
}

// Hide typing indicator
function hideTypingIndicator() {
    const indicator = document.getElementById('typingIndicator');
    if (indicator) {
        indicator.remove();
    }
}

// Handle Enter key press
function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

// Send suggested question
function sendSuggestedQuestion(question) {
    document.getElementById('userInput').value = question;
    sendMessage();
}

// Scroll chat to bottom
function scrollToBottom() {
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Format message with basic markdown-like formatting
function formatMessage(message) {
    return escapeHtml(message)
        .replace(/\\n\\n/g, '</p><p>')
        .replace(/\\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>');
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Export chat history (optional feature)
function exportChatHistory() {
    const dataStr = JSON.stringify(chatHistory, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `sleep_chat_history_${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
}
