---
title: DocAudit
emoji: 🧾
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
hf_oauth: true
pinned: false
short_description: Explainable checks for shopping receipts (OCR + rules)
---

# DocAudit — Explainable Anomaly Detection in Malaysian Shopping Receipts (MYR)

## Quick start — web demo (read this first)

```bash
python3.12 -m venv .venv                      # Windows: py -3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt   # Windows: .venv\Scripts\python ...
.venv/bin/python -m src.ui.app                # or: bash start_ui.sh / start_ui.bat
```

- Open **http://127.0.0.1:7860** (the browser opens by itself).
- Ready when the terminal prints `Running on local URL:  http://127.0.0.1:7860`.
- `python -m src.extract.extract …` is the command-line extractor: it only
  prints JSON in the terminal. The web UI is `python -m src.ui.app`.
- No API key needed for the image flow (local PaddleOCR + rules).
- Upload a receipt → Analyse → review rows and TOTAL (Accept / Reject) →
  Export. Export is allowed only when every row and the TOTAL are accepted.
- 中文说明：`QUICKSTART_zh.md`. Review rules: `src/ui/review.py`.

### Public demo from this computer (accounts + Cloudflare Tunnel)
```bash
brew install cloudflared
PYTHON=.venv/bin/python bash start_public.sh
```
Prints a `https://<random>.trycloudflare.com` link (changes on every start).
`/` landing page · `/signup` create an account · `/app/` the app (login).
Accounts live in `~/.docaudit/users.sqlite` (PBKDF2-hashed passwords).
Behind a VPN/proxy in TUN mode (Shadowrocket, Clash…), add a rule
`DOMAIN-SUFFIX,argotunnel.com,DIRECT` or the tunnel cannot reach Cloudflare.

### Hosted demo (Hugging Face Space — needs a paid plan for Gradio)
- The block at the top of this file is Hugging Face Space configuration
  (`app.py` is the Space entry point; it turns on public mode and
  "Sign in with Hugging Face").
- `site/` is the static landing page deployed on Cloudflare Pages.
- SROIE receipt photos (`data/photos/`) are not in this repository; get the
  dataset from its source (ICDAR 2019 SROIE) if you want to re-run extraction.
  The demo images in `demo/` are enough to try the app.


Read with PaddleOCR · Structure with DeepSeek · Decide with code · Explain with templates

## Setup

1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and set your `DEEPSEEK_API_KEY`
3. Run `python -c "from src.shared import deepseek_client; print(deepseek_client.health_check())"`

## Pipeline

```
Receipt photo → preprocessing → PaddleOCR → DeepSeek mapping → name normalisation
→ arithmetic gate → three checks → explanation → human review → export
```

## Run

```bash
# Extract one receipt
python -m src.extract.extract data/photos/receipt1.jpg

# Launch UI
python -m src.ui.app

# Run tests
pytest tests/
```
