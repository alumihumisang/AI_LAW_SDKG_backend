# AI LAW SDKG Backend Handoff

This private repository contains the complete `SDKG_official/` backend package for frontend integration.

Start with:

```bash
cd SDKG_official
cat README.md
```

The recommended frontend demo output is:

```text
SDKG_official/generation_outputs/SDKG_50q_FCH_TOP8_ui_relations_20260809.csv
```

For live API integration, run the backend from the repository root:

```bash
pip install -r SDKG_official/requirements.txt
uvicorn SDKG_official.api_server:app --host 0.0.0.0 --port 8000
```
