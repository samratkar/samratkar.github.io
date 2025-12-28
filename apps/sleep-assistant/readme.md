# 🌙 Sleep Assistant AI Chatbot

An elegant, professional AI-powered chatbot that helps users understand sleep science, disorders, and recommendations based on scientific research papers.

## 🎯 Features

- **Beautiful UI**: Modern, responsive design with gradient themes and smooth animations
- **Smart Search**: Retrieves relevant information from a vector database of sleep research papers
- **Real-time Chat**: Interactive conversation interface with typing indicators
- **Source References**: Shows which research documents were used to generate responses
- **Suggested Questions**: Quick-start chips for common sleep-related queries
- **Browser-Based**: Runs entirely in the browser - no backend required for basic functionality
- **Optional Python Backend**: Advanced features with Flask API for better semantic search

## 🚀 Quick Start (GitHub Pages)

### Access the Chatbot

The chatbot is automatically served via GitHub Pages/Jekyll:

1. Visit: `https://samratkar.github.io/apps/sleep-assistant/`
2. Start chatting about sleep!

### Local Testing

To test locally before publishing:

```bash
# Navigate to your Jekyll site root
cd c:\github\samratkar.github.io

# Start Jekyll server
bundle exec jekyll serve

# Open browser to
http://localhost:4000/_posts/apps/sleep_assistant/index.html
```

## 📂 Files Structure

```
sleep_assistant/
├── index.html          # Main chatbot interface
├── chatbot.js          # JavaScript logic for chat and vector search
├── vectordb.json       # Vector database with sleep research papers
├── app.py             # Optional Python backend (Flask API)
├── requirements.txt    # Python dependencies
└── readme.md          # This file
```

## 💡 How It Works

### Browser-Only Mode (Default)

1. **Load Vector DB**: On page load, fetches `vectordb.json` containing research documents
2. **User Query**: User types a question about sleep
3. **Search**: JavaScript performs keyword matching and simple semantic search
4. **Retrieve**: Finds top 3 most relevant document excerpts
5. **Response**: Displays relevant information with source references

### With Python Backend (Optional, Advanced)

For better semantic search and GPT-powered responses:

1. **Install Dependencies**:
   ```bash
   cd _posts/apps/sleep_assistant
   pip install -r requirements.txt
   ```

2. **Run Backend**:
   ```bash
   python app.py
   ```
   Server starts on `http://localhost:5000`

3. **Configure Frontend**: Update `chatbot.js` to use API endpoint:
   ```javascript
   const API_ENDPOINT = 'http://localhost:5000/chat';
   ```

4. **Optional - OpenAI Integration**:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   python app.py
   ```

## 🎨 Customization

### Modify Appearance

Edit the `<style>` section in `index.html`:

```css
/* Change primary colors */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

/* Adjust chat container size */
.chat-container {
    max-width: 900px;
    height: 85vh;
}
```

### Add More Suggested Questions

In `index.html`, add more chips:

```html
<div class="question-chip" onclick="sendSuggestedQuestion('Your question here')">
    Your question here
</div>
```

### Update Vector Database

Replace `vectordb.json` with your own research documents:

```json
{
  "name": "Your_vectordb",
  "createdAt": "2025-12-28T00:00:00.000Z",
  "dimension": 384,
  "documents": [
    "Your research document text here...",
    "Another document..."
  ]
}
```

## 🔧 Technical Details

### Vector Search Algorithm

The chatbot uses a hybrid approach:
1. **Keyword Matching**: Exact word matches in documents
2. **Phrase Matching**: Bonus for exact phrase occurrences
3. **Position Scoring**: Earlier mentions weighted higher
4. **Simple Embeddings**: Character-level hash-based vectors (fallback)

### Backend Features (Optional)

- **Sentence Transformers**: Better semantic embeddings using `all-MiniLM-L6-v2`
- **Cosine Similarity**: Accurate vector similarity calculation
- **OpenAI GPT**: Natural language response generation
- **CORS Enabled**: Works with GitHub Pages frontend

## 📊 Performance

- **Load Time**: < 2 seconds (vectordb.json is ~1.5MB)
- **Response Time**: < 1 second for queries
- **Accuracy**: Depends on query specificity and document relevance
- **No Rate Limits**: Runs locally in browser

## 🛠️ Troubleshooting

### Chatbot Not Loading

1. Check browser console for errors (F12)
2. Ensure `vectordb.json` is in the same directory
3. Try refreshing the page

### No Relevant Results

1. Try rephrasing your question
2. Use suggested questions to understand scope
3. The chatbot specializes in sleep science topics

### Backend Won't Start

```bash
# Install dependencies
pip install flask flask-cors numpy

# Check if port 5000 is available
netstat -an | findstr :5000

# Run with different port
python app.py --port 8080
```

## 🌟 Example Queries

Try asking:
- "What are the main sleep stages?"
- "How does vitamin D affect sleep?"
- "What is sleep apnea?"
- "How much sleep do children need?"
- "What is REM sleep?"
- "How to improve sleep quality?"

## 📝 Future Enhancements

- [ ] Add user authentication and chat history persistence
- [ ] Implement more advanced RAG techniques
- [ ] Add voice input/output capabilities
- [ ] Multi-language support
- [ ] PDF export of chat conversations
- [ ] Integration with sleep tracking devices

## 🤝 Contributing

To improve the chatbot:

1. Add more research papers to `vectordb.json`
2. Enhance the UI/UX in `index.html`
3. Improve search algorithms in `chatbot.js`
4. Add new features to `app.py`

## 📄 License

This project uses research papers for educational purposes. Please respect copyright and fair use guidelines.

## 🙏 Acknowledgments

- Sleep research papers from the EDEN cohort and other scientific studies
- Built with vanilla JavaScript for maximum compatibility
- Styled with modern CSS3 features
- Optional backend powered by Flask and Sentence Transformers

---

**Built with ❤️ for better sleep education**

*Last updated: December 2025*
