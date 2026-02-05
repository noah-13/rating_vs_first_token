# rating_vs_first_token

Analyze how Llama-3-as-a-judge produces rating tokens (1-5) and how first-step token probabilities relate to final ratings.

## Contents
- `llama3_as_judge_with_path.py`: Run judge inference and save per-dimension CSV + pure-rating trajectories.
- `analyze_rating_vs_first.py`: Compute metrics comparing gold vs rating-token predictions.
- `analyze_traj.py`: Plot error vs distance to rating emission step.
- `run_all.sh`: End-to-end reproduction script.

## Requirements
- Python >= 3.12
- GPU with enough VRAM for `meta-llama/Meta-Llama-3-8B-Instruct`
- Hugging Face access to the model (set `HF_TOKEN` if needed)

## Setup
Using `uv` (recommended):
```bash
uv sync
```

Or with `pip`:
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r <(python - <<'PY'
import tomllib, sys
from pathlib import Path
py = tomllib.loads(Path('pyproject.toml').read_text())
print('\n'.join(py['project']['dependencies']))
PY
)
```

## Reproduce All Experiments
```bash
./run_all.sh
```

Optional overrides:
```bash
MODEL=meta-llama/Meta-Llama-3-8B-Instruct \
BATCH_SIZE=8 \
MAX_NEW_TOKENS=128 \
LIMIT=100 \
./run_all.sh
```

## Outputs
All outputs are written to `results/`:
- `summeval_results_3tiers_<dim>.csv`: per-sample results for each dimension.
- `pure_rating_traj_<dim>.jsonl`: pure-rating token trajectory per sample.
- `metrics_table.csv`: summary metrics across dimensions and tiers.
- `pure_traj_4dims_fig1_error_firststep.png`: error vs distance plot.
- `pure_traj_4dims_fig2_mass_only.png`: rating-token mass plot.

## Data
- `data/summeval.json`: input dataset used by the judge.
- `data/prompts/*.txt`: prompt templates for each dimension.

## Notes
- The script overwrites files in `results/`.
- If you want a quick sanity check, use `LIMIT=10`.
