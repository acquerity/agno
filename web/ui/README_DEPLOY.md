# Deploy Your Agno SuperAgent UI and API

## 1. Deploy Backend (FastAPI) to Render

1. Go to https://render.com/
2. Sign up (free) and click "New Web Service"
3. Connect your GitHub repo, select the root folder (where `web/main.py` and `requirements.txt` are)
4. Set the build command to:
   ```
   pip install -r requirements.txt
   ```
   and the start command to:
   ```
   uvicorn web.main:app --host 0.0.0.0 --port 10000
   ```
5. Set the port to `10000` (or whatever Render suggests)
6. Deploy and wait for your public API URL (e.g., `https://your-backend.onrender.com`)

## 2. Deploy Frontend (React UI) to Vercel

1. Go to https://vercel.com/
2. Sign up (free) and click "New Project"
3. Connect your GitHub repo, select the `web/ui/` folder
4. In the Vercel dashboard, set the environment variable:
   - `VITE_API_BASE_URL=https://your-backend.onrender.com`
5. Deploy and get your public website link (e.g., `https://your-ui.vercel.app`)

## 3. Log In and Use Your Agent
- Open your Vercel site in any browser, anywhere.
- Log in with your credentials (default: `devuser` / `SuperSecureDevPassword123!`).
- Enjoy your fully public, login-protected AI agent UI!

## 4. (Optional) Update API Keys
- Go to the API Keys page in your UI and add your LLM API keys for full functionality.

---

**If you need help, just copy-paste your Render and Vercel URLs here and I'll help you test and debug!** 