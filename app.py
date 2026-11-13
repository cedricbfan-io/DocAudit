"""Hugging Face Space entry point (sdk: gradio, app_file: app.py).

Public mode: per-visitor sessions, no LLM button, exports in a temp dir.
On a Space with `hf_oauth: true`, Analyse / Run detection require
"Sign in with Hugging Face" (visitors create a free HF account).
Locally, `python app.py` runs the same public UI without the login gate.
"""

import os

on_space = bool(os.environ.get("SPACE_ID"))
os.environ.setdefault("DOCAUDIT_PUBLIC", "1")
os.environ.setdefault("DOCAUDIT_REQUIRE_LOGIN", "1" if on_space else "0")

from src.ui.app import build_ui, _BRAND_CSS  # noqa: E402  (reads the env above)

demo = build_ui()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0" if on_space else "127.0.0.1",
                css=_BRAND_CSS)
