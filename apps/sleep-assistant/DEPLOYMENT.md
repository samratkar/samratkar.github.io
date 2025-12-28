# 🚀 Deployment Guide - Sleep Assistant Chatbot

## Option 1: GitHub Pages (Recommended - Free & Simple)

Your chatbot is **already deployed** and accessible via GitHub Pages!

### Access URL
```
https://samratkar.github.io/apps/sleep-assistant/
```

### Steps to Publish (if not already live):

1. **Commit and Push Files**
   ```bash
   cd c:\github\samratkar.github.io
   git add _posts/apps/sleep_assistant/
   git commit -m "Add Sleep Assistant AI Chatbot"
   git push origin main
   ```

2. **Enable GitHub Pages** (if not already enabled)
   - Go to your repository on GitHub
   - Settings → Pages
   - Source: Deploy from branch `main`
   - Folder: `/ (root)`
   - Click Save

3. **Wait 1-2 Minutes**
   - GitHub Pages builds automatically
   - Check the Actions tab for build status

4. **Visit Your Chatbot**
   - Navigate to the URL above
   - Start chatting!

### Adding to Your Site Navigation

Edit your site's navigation (usually in `_config.yml` or navigation file):

```yaml
# Add to navigation
navigation:
  - title: "Apps"
    url: /apps/
  - title: "Sleep Assistant"
    url: /_posts/apps/sleep_assistant/index.html
```

---

## Option 2: Local Testing with Jekyll

Test the chatbot locally before publishing:

### Prerequisites
- Ruby and Jekyll installed
- Bundle gem installed

### Steps

1. **Navigate to Repository**
   ```bash
   cd c:\github\samratkar.github.io
   ```

2. **Install Dependencies**
   ```bash
   bundle install
   ```

3. **Start Jekyll Server**
   ```bash
   bundle exec jekyll serve
   ```

4. **Open in Browser**
   ```
   http://localhost:4000/_posts/apps/sleep_assistant/index.html
   ```

5. **Test the Chatbot**
   - Try different queries
   - Check console for errors (F12)
   - Verify vectordb.json loads correctly

---

## Option 3: Python Backend Deployment (Advanced)

Deploy the Python Flask backend for advanced features like better semantic search and GPT integration.

### A. Local Deployment

1. **Install Dependencies**
   ```bash
   cd c:\github\samratkar.github.io\_posts\apps\sleep_assistant
   pip install -r requirements.txt
   ```

2. **Set Environment Variables** (Optional for OpenAI)
   ```bash
   # Windows
   set OPENAI_API_KEY=your-key-here
   
   # Linux/Mac
   export OPENAI_API_KEY=your-key-here
   ```

3. **Run Backend**
   ```bash
   python app.py
   ```

4. **Update Frontend**
   - Edit `chatbot.js`
   - Change API endpoint to `http://localhost:5000`

### B. Heroku Deployment

1. **Create `Procfile`**
   ```
   web: gunicorn app:app
   ```

2. **Add to requirements.txt**
   ```
   gunicorn==21.2.0
   ```

3. **Deploy**
   ```bash
   heroku create sleep-assistant-api
   heroku config:set OPENAI_API_KEY=your-key
   git push heroku main
   ```

4. **Update Frontend**
   - Change API endpoint to your Heroku URL

### C. AWS Lambda Deployment

1. **Install Serverless Framework**
   ```bash
   npm install -g serverless
   ```

2. **Create serverless.yml**
   ```yaml
   service: sleep-assistant
   provider:
     name: aws
     runtime: python3.9
   functions:
     chat:
       handler: app.lambda_handler
       events:
         - http:
             path: chat
             method: post
   ```

3. **Deploy**
   ```bash
   serverless deploy
   ```

### D. Docker Deployment

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["python", "app.py"]
   ```

2. **Build and Run**
   ```bash
   docker build -t sleep-assistant .
   docker run -p 5000:5000 sleep-assistant
   ```

---

## Option 4: Static Site Generators

### Netlify

1. **Connect Repository**
   - Go to Netlify Dashboard
   - New Site from Git
   - Select your repository

2. **Build Settings**
   - Build command: `jekyll build`
   - Publish directory: `_site`

3. **Deploy**
   - Automatic deployment on git push

### Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
cd c:\github\samratkar.github.io
vercel
```

---

## 🔧 Configuration After Deployment

### Update CORS Settings

If using Python backend with GitHub Pages frontend:

```python
# In app.py
CORS(app, origins=[
    'https://samratkar.github.io',
    'http://localhost:4000'
])
```

### Update API Endpoint

In `chatbot.js`:

```javascript
// For production backend
const API_ENDPOINT = 'https://your-backend-url.com/chat';

// Modify sendMessage() function to use API
async function sendMessage() {
    // ... existing code ...
    
    const response = await fetch(API_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: message })
    });
    
    const data = await response.json();
    // ... handle response ...
}
```

---

## 📊 Performance Optimization

### 1. Compress vectordb.json
```bash
# Using gzip
gzip vectordb.json

# Update HTML to load compressed version
fetch('vectordb.json.gz').then(/* decompress */)
```

### 2. Enable Browser Caching

Add to Jekyll `_config.yml`:
```yaml
# Cache static assets
include: ['_posts/apps/sleep_assistant/*.js', '_posts/apps/sleep_assistant/*.json']
```

### 3. Use CDN

Upload `vectordb.json` to CDN (e.g., Cloudflare) and update fetch URL:
```javascript
fetch('https://cdn.example.com/vectordb.json')
```

---

## 🔒 Security Considerations

### For GitHub Pages
- ✅ No server-side code = No security vulnerabilities
- ✅ HTTPS by default
- ✅ No API keys exposed (runs client-side)

### For Python Backend
- 🔐 Use environment variables for API keys
- 🔐 Implement rate limiting
- 🔐 Add authentication if needed
- 🔐 Use HTTPS only in production

---

## 🧪 Testing Checklist

Before going live:

- [ ] Test all suggested questions
- [ ] Try edge cases (empty queries, special characters)
- [ ] Check mobile responsiveness
- [ ] Verify vectordb.json loads correctly
- [ ] Test on different browsers (Chrome, Firefox, Safari)
- [ ] Check console for JavaScript errors
- [ ] Verify links and navigation work
- [ ] Test with slow network connection
- [ ] Check accessibility (keyboard navigation)

---

## 📈 Monitoring

### Google Analytics

Add to `index.html`:
```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
```

### Error Tracking

Add Sentry for error monitoring:
```html
<script src="https://browser.sentry-cdn.com/7.x/bundle.min.js"></script>
<script>
  Sentry.init({ dsn: 'YOUR_DSN' });
</script>
```

---

## 🆘 Troubleshooting

### Chatbot Not Loading
1. Check browser console (F12) for errors
2. Verify `vectordb.json` is accessible
3. Clear browser cache
4. Check network tab for failed requests

### Search Not Working
1. Verify `vectordb.json` structure
2. Check JavaScript console for errors
3. Test with simpler queries

### Backend Connection Failed
1. Verify backend is running
2. Check CORS configuration
3. Verify API endpoint URL
4. Check network firewall

---

## 📞 Support

For issues or questions:
- GitHub Issues: [Create an issue](https://github.com/samratkar/samratkar.github.io/issues)
- Email: [Your email]
- Documentation: See [readme.md](readme.md)

---

**🎉 Your Sleep Assistant Chatbot is Ready!**

Now users can:
- Access it anytime via GitHub Pages
- Get instant answers about sleep science
- Learn from research-backed information
- Enjoy a beautiful, professional interface

**Happy Deploying! 🚀**
