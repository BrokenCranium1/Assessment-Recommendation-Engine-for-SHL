# SHL Assessment Recommendation System

## Quick Start (what I actually ran)

```bash
# 1. Install deps into a fresh venv
pip install -r requirements.txt

# 2. Precompute embeddings once (optional but faster)
python precompute_embeddings.py

# 3. Start the API
python main.py

# 4. Try it from the browser
# Open index.html or go to http://localhost:8000/docs
```

## Useful scripts

- `scripts/shl_scraper.py` – scrapes and enriches the SHL catalog into `data/shl_catalog_final.csv`.
- `scripts/evaluate.py` – runs the Mean Recall@10 evaluation and writes test predictions.
- `scripts/diagnose_evaluation.py` – quick sanity checks on training data and URL matching.
- `final_validation.py` – final local smoke test against the FastAPI server.

I kept the repo focused on being easy to run locally: one command to start the API, a couple of helper scripts for scraping/evaluation, and plain HTML/JS for the UI. If something feels brittle or overkill, it probably came from iterating quickly for the assignment rather than polishing it as a product.