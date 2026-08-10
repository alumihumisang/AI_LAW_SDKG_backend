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

The `generation_outputs` CSV files already contain the fields most useful for UI inspection. See "Generation Output Columns" below for the detailed meaning of each column.

For the thesis/demo story, TOP8 is the main display version. TOP9 and TOP10 are included because the experiment was run up to TOP10; they intentionally become slightly more generic because more retrieved cases are mixed into the generation context.

## Generation Output Columns

The recommended frontend prototype file is:

```text
generation_outputs/SDKG_50q_FCH_TOP8_ui_relations_20260809.csv
```

Each row is one query case and one generated complaint. The most important UI fields are `query_text`, `generated_text`, `query_feature_profile`, `query_case_relations`, and `top8_case_full_texts`.

### Basic Query And Experiment Columns

- `query_id`: The ID of the demo query, from 1 to 50.
- `category`: The query type used in the 50-query demo set, such as single plaintiff/single defendant or multiple plaintiffs/defendants.
- `exp_id`: Internal experiment ID. The main demo uses `E14`.
- `exp_name`: Human-readable experiment name. The main demo uses `FC-H`.
- `retrieval_mode`: Retrieval strategy. `dual_tree` means the query uses the SDKG dual severity-tree retrieval process.
- `tree_exp_id`: The severity-tree configuration used for retrieval. It normally matches `exp_id`.
- `weight_code`: Short code for the legal-dimension weight order. In `FC-H`, `F` means Fact has the largest weight and `C` means Compensation is the second-largest weight.
- `tau_code`: Short code for the distance threshold level. `H` is the high-threshold setting.
- `fact_w`: Weight of the Fact dimension.
- `injury_w`: Weight of the Injury dimension.
- `comp_w`: Weight of the Compensation dimension.
- `tau`: Distance threshold used when constructing/retrieving from the SDKG relation structure.
- `top_k`: Number of retrieved cases used as generation references.
- `topk_style_level`: Internal style-control level used by the generation post-processing. Larger TOP values become slightly more generic because more cases are mixed into the context.
- `model`: LLM used for complaint generation.

### Query And Reference Text Columns

- `query_text`: The lawyer-style input for the query case. This is the main input shown to the user.
- `silver_reference_text`: Reference complaint text used for comparison/evaluation.
- `legacy_silver_reference_text`: Earlier preserved reference text. It is kept for traceability and can usually be hidden from the UI.
- `human_reference_text`: Human-written reference complaint. It can be used for internal comparison, not normally shown in the frontend.
- `ground_truth_text`: Ground-truth complaint text used in experiments.
- `gpt_baseline_text`: GPT baseline output used in the thesis experiment. It is useful for research comparison, but not required in the normal SDKG demo UI.

### Anchor And Retrieval Columns

- `anchor_case_id`: The first and closest case selected as the anchor case for the query.
- `anchor_distance`: Distance between the query and the anchor case. Smaller means more similar under the SDKG distance calculation.
- `anchor_score`: Weighted severity score of the anchor case under the selected legal-weight setting.
- `top8_case_ids`: Case IDs of the retrieved cases used for UI display and generation. The name keeps the original TOP8 demo naming, but in TOP1, TOP9, and TOP10 files the number of IDs follows `top_k`.
- `top8_distances`: Distances between the query and the retrieved cases, in the same order as `top8_case_ids`. Smaller values indicate closer retrieved cases.
- `top8_retrieval_sources`: Retrieval source of each retrieved case. `anchor` is the anchor case; `lighter_tree` and `heavier_tree` indicate whether the case came from the lighter or heavier side of the dual-tree relation.
- `top8_query_case_relations`: Query-to-case severity direction for each retrieved case. `LH` means the query is lighter than the retrieved case; `HL` means the query is heavier than the retrieved case.
- `top8_case_full_texts`: Full text of the retrieved similar cases. This is useful for a frontend "similar case details" panel.

### SDKG Explanation Columns

- `query_feature_profile`: JSON string describing the query's extracted legal feature profile. It includes severity levels, severity scores, and Boolean feature information. This is the best source for visualizing the query's Boolean matrix or feature summary.
- `query_case_relations`: JSON string describing each retrieved case's relation to the query, including rank, case ID, retrieval source, anchor information, distance, query score, case score, severity relation, feature profile, and shared feature information. This is the main field for drawing "why this case was retrieved."
- `similar_cases`: JSON string containing compact retrieval metadata for the retrieved cases. It is useful for tables or cards that show rank, case ID, distance, score, and retrieval side.
- `generation_support`: Plain-text support context passed to the generation step. It summarizes similar-case structures, accident facts, injury facts, compensation structures, and parent-case reasoning. This is useful for debugging why the generated complaint used certain wording.

### Generated Complaint Columns

- `facts_section`: Generated facts section of the complaint.
- `laws_section`: Generated legal-basis section.
- `damages_section`: Generated damages/compensation section.
- `conclusion_section`: Generated conclusion and total-claim section.
- `generated_text`: Final assembled complaint text shown to the user. It combines the facts, laws, damages, and conclusion sections.

### Run Status Columns

- `run_status`: Generation status. `ok` means the row was generated successfully.
- `error_message`: Error message if generation failed. It should be empty when `run_status` is `ok`.

### Recommended Frontend Use

For the first frontend version, use these fields:

- Query panel: `query_id`, `category`, `query_text`.
- Generated complaint panel: `generated_text`, or the four section fields if the UI wants tabs.
- Retrieval summary: `top_k`, `anchor_case_id`, `anchor_distance`, `top8_case_ids`, `top8_distances`, `top8_retrieval_sources`.
- SDKG explanation panel: `query_feature_profile` and `query_case_relations`.
- Similar-case detail panel: `top8_case_full_texts`.

The frontend does not need to expose `fact_w`, `injury_w`, `comp_w`, `tau`, or `topk_style_level` in the main legal-user interface. These fields are included for research traceability and debugging.

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
