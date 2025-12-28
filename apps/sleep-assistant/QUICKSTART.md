# 🚀 Quick Start Guide

## 🎯 For End Users

### Access the Chatbot Online

**Simply visit:** https://samratkar.github.io/apps/sleep-assistant/

That's it! Start asking questions about sleep.

### How to Use

1. **Type your question** in the input box at the bottom
2. **Press Enter** or click "Send"
3. **Get instant answers** based on research papers
4. **Try suggested questions** for quick topics

### Example Questions

- "What are the main sleep stages?"
- "How does vitamin D affect sleep?"
- "What is sleep apnea?"
- "How much sleep do children need?"

---

## 💻 For Developers

### Test Locally (Browser-Only Mode)

```bash
# Clone or navigate to repository
cd c:\github\samratkar.github.io\_posts\apps\sleep_assistant

# Open index.html in browser
start index.html

# Or use Python simple server
python -m http.server 8000
# Then visit: http://localhost:8000/index.html
```

### Test with Jekyll

```bash
# Navigate to Jekyll site root
cd c:\github\samratkar.github.io

# Start Jekyll server
bundle exec jekyll serve

# Visit: http://localhost:4000/_posts/apps/sleep_assistant/index.html
```

### Run Python Backend (Optional)

```bash
# Install dependencies
pip install -r requirements.txt

# Run backend
python app.py

# Backend runs on: http://localhost:5000
```

### Run Tests

```bash
# Test vector search
python test_chatbot.py
```

---

## 📝 Customize

### Change Colors

Edit `index.html`, find the gradient:

```css
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
```

Change to your colors:

```css
background: linear-gradient(135deg, #your-color-1 0%, #your-color-2 100%);
```

### Add More Questions

Edit `index.html`, add chips:

```html
<div class="question-chip" onclick="sendSuggestedQuestion('Your question')">
    Your question
</div>
```

### Update Research Papers

Replace `vectordb.json` with your documents:

```json
{
  "name": "Your_vectordb",
  "dimension": 384,
  "documents": ["Document 1...", "Document 2..."]
}
```

---

## 🐛 Troubleshooting

### Chatbot Won't Load
- Check browser console (F12)
- Verify `vectordb.json` is in same folder
- Try different browser

### No Search Results
- Use suggested questions first
- Rephrase your question
- Check spelling

### Backend Won't Start
```bash
# Check Python version
python --version  # Should be 3.8+

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

---

## 📚 Documentation

- **Full Documentation**: [readme.md](readme.md)
- **Deployment Guide**: [DEPLOYMENT.md](DEPLOYMENT.md)
- **Source Code**: Check `.js` and `.py` files

---

## 🎓 Learn More

### File Structure
```
sleep_assistant/
├── index.html          ← Main interface
├── chatbot.js          ← Search & chat logic
├── vectordb.json       ← Research database
├── app.py             ← Optional backend
└── readme.md          ← Full documentation
```

### Key Technologies
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Backend**: Python, Flask (optional)
- **Deployment**: GitHub Pages, Jekyll
- **Search**: Keyword matching + semantic similarity

---

## ✅ Next Steps

1. **Try the chatbot** - Click suggested questions
2. **Customize appearance** - Edit colors and text
3. **Add more research** - Update vectordb.json
4. **Share with others** - Send them the URL
5. **Deploy backend** - For advanced features

---

## 💡 Tips

- **Be specific** with your questions
- **Use medical terms** for better results  
- **Try different phrasings** if needed
- **Check source references** for research backing

---

## 🆘 Need Help?

- Check [readme.md](readme.md) for detailed docs
- See [DEPLOYMENT.md](DEPLOYMENT.md) for deployment options
- Review browser console for errors
- Open an issue on GitHub

---

**🌙 Ready to explore sleep science?**

**Start now:** [Launch Sleep Assistant →](index.html)

---

*Made with ❤️ for better understanding of sleep | December 2025*
