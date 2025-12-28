---
layout: page
title: "Sleep Assistant AI Chatbot"
permalink: /apps/sleep-assistant/
---

# 🌙 Sleep Assistant AI Chatbot

<style>
.app-preview {
    border: 2px solid #667eea;
    border-radius: 10px;
    padding: 20px;
    margin: 30px 0;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    text-align: center;
}

.launch-btn {
    display: inline-block;
    padding: 15px 40px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    text-decoration: none;
    border-radius: 30px;
    font-size: 18px;
    font-weight: bold;
    margin: 20px 0;
    transition: transform 0.3s, box-shadow 0.3s;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}

.launch-btn:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    color: white;
}

.features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
    margin: 30px 0;
}

.feature-card {
    padding: 20px;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    background: white;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

.feature-card h3 {
    color: #667eea;
    margin-bottom: 10px;
}
</style>

<div class="app-preview">
    <h2>🤖 Your Personal Sleep Science Assistant</h2>
    <p>Ask questions about sleep disorders, sleep stages, and get answers based on scientific research</p>
    <a href="{{ site.baseurl }}/apps/sleep-assistant/index.html" class="launch-btn" target="_blank">
        🚀 Launch Sleep Assistant
    </a>
</div>

## About This App

The Sleep Assistant AI Chatbot is an intelligent conversational interface that helps you understand sleep science, disorders, and recommendations. It's powered by a vector database containing research papers on sleep studies, including:

- Sleep stage classification
- Obstructive sleep apnea
- Vitamin D effects on sleep
- Sleep duration in children
- Sleep architecture analysis

## ✨ Key Features

<div class="features-grid">
    <div class="feature-card">
        <h3>🎨 Beautiful Interface</h3>
        <p>Modern, responsive design with smooth animations and intuitive controls</p>
    </div>
    
    <div class="feature-card">
        <h3>🔍 Smart Search</h3>
        <p>Advanced retrieval system that finds relevant research based on your questions</p>
    </div>
    
    <div class="feature-card">
        <h3>📚 Research-Backed</h3>
        <p>All answers are derived from scientific sleep research papers</p>
    </div>
    
    <div class="feature-card">
        <h3>⚡ Fast & Local</h3>
        <p>Runs entirely in your browser - no external API calls needed</p>
    </div>
    
    <div class="feature-card">
        <h3>💬 Natural Chat</h3>
        <p>Conversational interface with typing indicators and message history</p>
    </div>
    
    <div class="feature-card">
        <h3>🎯 Source References</h3>
        <p>See which research documents were used for each answer</p>
    </div>
</div>

## 🎓 What You Can Ask

Try questions like:

- **"What are the different stages of sleep?"**
- **"How does vitamin D deficiency affect sleep?"**
- **"What is obstructive sleep apnea?"**
- **"How much sleep do children need?"**
- **"What is REM sleep and why is it important?"**
- **"How can I improve my sleep quality?"**

## 🛠️ Technical Details

This chatbot demonstrates:

- **Frontend**: Pure HTML, CSS, and JavaScript
- **Vector Database**: JSON-based document storage
- **Search Algorithm**: Hybrid keyword + semantic similarity
- **Deployment**: GitHub Pages via Jekyll
- **Optional Backend**: Python Flask API for advanced features

## 📖 Documentation

Full documentation, source code, and setup instructions are available in the [GitHub repository](https://github.com/samratkar/samratkar.github.io/tree/main/apps/sleep-assistant).

## 🚀 Quick Start

1. Click the "Launch Sleep Assistant" button above
2. Type your question in the chat input
3. Press Enter or click Send
4. Get research-backed answers instantly!

## 💡 Tips for Best Results

- **Be specific**: Instead of "sleep problems", try "symptoms of sleep apnea"
- **Use medical terms**: The chatbot understands scientific terminology
- **Try suggested questions**: Click the suggested questions to get started
- **Rephrase if needed**: If you don't get good results, try asking differently

## 🔬 Research Sources

The chatbot's knowledge base includes:

- Deep residual networks for automatic sleep stage classification
- Studies on vitamin D and sleep duration in children
- Research on sleep disorders and treatment
- Sleep architecture and polysomnography studies
- Multi-cohort sleep research data

## 🌟 Future Enhancements

Planned features:
- Voice input/output
- Chat history export
- More research papers
- Advanced RAG techniques
- Multi-language support

---

**Ready to learn about sleep science?** [Launch the Sleep Assistant →]({{ site.baseurl }}/apps/sleep-assistant/index.html)

<div style="text-align: center; margin-top: 40px; padding: 20px; background: #f5f7fa; border-radius: 10px;">
    <p><em>💤 Better understanding leads to better sleep</em></p>
    <p style="font-size: 12px; color: #888;">Built with care for educational purposes | December 2025</p>
</div>
