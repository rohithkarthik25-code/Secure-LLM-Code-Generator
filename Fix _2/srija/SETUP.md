# Setup — required for real spaCy + BART (not the fallback path)

```bash
pip install spacy transformers torch
python -m spacy download en_core_web_sm
```

First run of `run_semantics.py` / `run_prompt.py` will also auto-download
`facebook/bart-large-mnli` (~1.6 GB) from HuggingFace the first time
`prompt_intent.py` is imported — this needs internet access ONCE, then it's
cached locally (`~/.cache/huggingface/hub`) and loads offline after that.

## How to confirm it's really using them (not the fallback)
Run either script and check the console output at startup:

  [Stage 1] spaCy en_core_web_sm loaded with security EntityRuler.
  [Stage 2] BART-large-MNLI zero-shot classifier loaded.

If you instead see a boxed "Falling back..." warning, spaCy/BART did NOT
load — check your internet connection / disk space (BART needs ~1.6GB free)
and re-run.
