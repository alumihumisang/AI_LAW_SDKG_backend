# SDKG Official Backend

This folder is the cleaned SDKG backend package for frontend handoff.

The recommended demo setting is:

- Method: SDKG
- Experiment: `FC-H` / `E14`
- Weights: fact `0.5`, injury `0.2`, compensation `0.3`
- Threshold: `tau = 0.5`
- Default UI retrieval size: `top_k = 8`
- LLM backend: Ollama `gemma3:27b`

For legal-domain users, the frontend should normally keep the SDKG parameters fixed and show the retrieved similar cases, Boolean feature profile, severity relation, and generated complaint. Parameter tuning is mainly for experiment reproduction, not for the main UI.

## What Is Included

- `data/queries_50_0519.xlsx`: the 50 demo queries.
- `data/complaints_6057.xlsx`: full complaint texts used for similar-case display.
- `data/lawyer_inputs_6057.xlsx`: original lawyer-input text used by the Boolean-feature preprocessing script.
- `phase1_boolean_matrix_v1.jsonl`: 4 x 4 Boolean feature matrix for the 6,057 cases.
- `phase1_boolean_severity_v1.jsonl`: Boolean features plus severity scores.
- `experiment_outputs/severity_trees/`: preprocessed LH/HL severity-tree links.
- `generation_outputs/SDKG_50q_FCH_TOP1...TOP10_ui_relations_20260809.*`: official 50-query generation outputs for TOP1 to TOP10.

The `generation_outputs` CSV files already contain the fields most useful for UI inspection:

- `query_text`
- `generated_text`
- `query_feature_profile`
- `query_case_relations`
- `top8_case_ids`
- `top8_distances`
- `top8_retrieval_sources`
- `top8_query_case_relations`
- `top8_case_full_texts`
- `similar_cases`
- `facts_section`
- `laws_section`
- `damages_section`
- `conclusion_section`

For the thesis/demo story, TOP8 is the main display version. TOP9 and TOP10 are included because the experiment was run up to TOP10; they intentionally become slightly more generic because more retrieved cases are mixed into the generation context.

## API Startup

Install dependencies from the repository root:

```bash
pip install -r SDKG_official/requirements.txt
```

Start Ollama separately and make sure `gemma3:27b` is available:

```bash
ollama serve
ollama pull gemma3:27b
```

Start the FastAPI backend:

```bash
uvicorn SDKG_official.api_server:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Generate one complaint:

```bash
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"query_text":"一、事故發生緣由：被告駕車不慎撞擊原告，致原告受傷。","category":"單純原被告各一","experiment":"FC-H","top_k":8}'
```

The API response includes:

- `generated_text`
- `sections.facts`
- `sections.laws`
- `sections.damages`
- `sections.conclusion`
- `parties`
- `retrieval.anchor_case_id`
- `retrieval.anchor_distance`
- `retrieval.similar_cases`

## Batch Generation

Run one query:

```bash
python SDKG_official/SDKG_query_generate.py \
  --query-id 28 \
  --exp-id E14 \
  --top-k 8 \
  --model gemma3:27b \
  --save \
  --output-stem SDKG_demo_query28_top8
```

Run the official TOP1 to TOP8 output:

```bash
bash SDKG_official/run_TOP1_to_TOP8_ui_relations_20260809.sh
```

Run TOP9 and TOP10:

```bash
bash SDKG_official/run_TOP9_to_TOP10_ui_relations_20260809.sh
```

## Main Scripts

- `api_server.py`: FastAPI entry point for frontend integration.
- `web_indictment_generator.py`: API-facing wrapper.
- `SDKG_query_generate.py`: SDKG retrieval and complaint generation.
- `sdkg_distance.py`: activated-feature normalized Boolean distance.
- `sdkg_generation_legal.py`: legal basis and facts-section generation helpers.
- `sdkg_generation_sections.py`: damages, conclusion, and post-processing helpers.
- `SDKG_evaluate_generation.py`: automatic-metric evaluation.
- `SDKG_coverage_audit.py`: coverage checks for generated complaints.

## Notes For Frontend Handoff

1. Use `generation_outputs/SDKG_50q_FCH_TOP8_ui_relations_20260809.csv` first for static UI prototyping.
2. Use `/generate` for live generation after the UI layout is stable.
3. Keep `experiment="FC-H"` and `top_k=8` for the main legal-user demo.
4. Do not expose alpha, beta, lambda, or tau controls in the first UI unless an experiment/debug mode is needed.
5. The current API returns compact retrieval metadata. If the UI needs full similar-case text and full Boolean matrices in live API responses, extend `web_indictment_generator.public_response()` using the fields already produced by `SDKG_query_generate.py`.

## Preprocessing Scope

The included JSONL and severity-tree files are already preprocessed. The frontend handoff does not need to rebuild Phase 1 or Phase 2.

Phase 1/Phase 2 scripts are kept for reproducibility, but rebuilding the full corpus may require original upstream intermediate files outside this frontend package. For frontend work, use the included preprocessed JSONL and severity-tree files.
