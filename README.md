# DocAudit — Explainable Anomaly Detection in Malaysian Shopping Receipts (MYR)

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
