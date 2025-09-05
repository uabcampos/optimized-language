# 🚀 Deployment Guide

## Streamlit Cloud (Recommended)

### 1. Go to [share.streamlit.io](https://share.streamlit.io)
### 2. Sign in with GitHub
### 3. Click "New app"
### 4. Select your repository: `uabcampos/optimized-language`
### 5. Set the main file path: `web_app_advanced.py`
### 6. Add environment variable:
   - Key: `OPENAI_API_KEY`
   - Value: Your OpenAI API key
### 7. Click "Deploy"

## Alternative: Railway

### 1. Go to [railway.app](https://railway.app)
### 2. Sign in with GitHub
### 3. Click "New Project" → "Deploy from GitHub repo"
### 4. Select `uabcampos/optimized-language`
### 5. Add environment variable:
   - Key: `OPENAI_API_KEY`
   - Value: Your OpenAI API key
### 6. Railway will auto-detect it's a Streamlit app

## Alternative: Render

### 1. Go to [render.com](https://render.com)
### 2. Sign in with GitHub
### 3. Click "New" → "Web Service"
### 4. Connect your repository
### 5. Set:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run web_app_advanced.py --server.port $PORT --server.address 0.0.0.0`
### 6. Add environment variable: `OPENAI_API_KEY`

## Environment Variables Required

### Required (at least one):
- `GEMINI_API_KEY`: Your Google Gemini API key (recommended - faster & cheaper)
- `OPENAI_API_KEY`: Your OpenAI API key (backup option)

### Streamlit Cloud Secrets Format:
In your Streamlit Cloud app settings, add these secrets:

```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
OPENAI_API_KEY = "your_openai_api_key_here"
```

### API Selection:
- **Auto mode** (default): Uses Gemini if available, falls back to OpenAI
- **Gemini**: Uses Google Gemini 1.5 Flash (faster, cheaper)
- **OpenAI**: Uses specified OpenAI model

## Notes

- Netlify is for static sites only, not Python applications
- Streamlit Cloud is the easiest option for Streamlit apps
- All platforms support your GitHub repository
- The app will be publicly accessible once deployed
