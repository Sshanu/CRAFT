# CRAFT: Cost-aware Refinement And Front-aware Tuning of Prompts

CRAFT is a prompt optimizer that searches the **Pareto front** of two
objectives at once: task accuracy and prompt-token cost. Instead of returning a
single prompt, one CRAFT run returns a set of prompts that trade accuracy
against cost, so the operating point can be chosen at deployment time.

This repository contains the reference implementation and the scripts used for
the experiments in the paper.

CRAFT has four components, run as a round-by-round loop:

1. **Refiner** (default: SCULPT) — a structure-aware module that rewrites a
   prompt's tree to improve accuracy.
2. **Condenser** (default: Distill) — a structure-aware module that shortens a
   prompt while preserving task-relevant instructions.
3. **Pareto-gap acquisition** — ranks generated candidates by their distance to
   the current optimistic front and spends a fixed per-round validation budget
   on the closest ones.
4. **NSGA-II selector** — keeps a non-dominated, spread-out population of `k`
   prompts for the next round.

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/Sshanu/CRAFT.git
cd CRAFT
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## API keys

CRAFT calls a hosted LLM as both the *optimizer* (it proposes prompt edits) and
the *target* (it executes prompts to score them). Set the keys for whichever
provider you use:

| Provider | Environment variables | Notes |
|----------|----------------------|-------|
| `openai` (default) | `OPENAI_API_KEY` | OpenAI API. |
| `azure`  | `AZURE_API_KEY`, `AZURE_INFERENCE_ENDPOINT`, `AZURE_OPENAI_ENDPOINT` | Azure OpenAI / AI Foundry. |
| `ollama` | none | Local models via [Ollama](https://ollama.com). |
| `codex`  | (uses the local `codex` CLI) | Optional. |

```bash
export OPENAI_API_KEY="sk-..."
```

## Quick start

Each method is one script under `scripts/`. The first argument is the task:

```bash
bash scripts/craft.sh beaver_tails
```

Tasks: `beaver_tails`, `go_emotions`, `disambiguation_qa`, `causal_judgement`,
`formal_fallacy`, `salient_translation`.

Common overrides are environment variables:

```bash
SEED=91 MODEL=gpt-5 MAX_ROUNDS=8 bash scripts/craft.sh go_emotions
```

Key knobs (see `scripts/_base.sh` for all of them): `PROVIDER`, `MODEL`,
`SEED`, `MAX_ROUNDS` (R), `BEAM_SIZE` (k), `EVAL_ROUNDS` (T), `N_VAL`, `N_TEST`.

## Methods

| Script | Method |
|--------|--------|
| `scripts/craft.sh` | **CRAFT** — full method (NSGA-II selector + refiner + condenser). |
| `scripts/weighted_sum.sh` | Weighted-sum (WPRO) baseline. Set `SCORE_WGT`/`TOKEN_WGT` (paper uses 0.3, 0.5, 0.7). |
| `scripts/sculpt.sh` | SCULPT single-objective baseline (accuracy only). |
| `scripts/distill.sh` | Distill single-objective baseline (compression). |
| `scripts/llmlingua.sh` | LLMLingua-2 single-objective baseline (compression). |
| `scripts/craft_opro.sh` | CRAFT with the refiner swapped to OPRO. |
| `scripts/craft_evoprompt.sh` | CRAFT with the refiner swapped to EvoPrompt. |
| `scripts/craft_r.sh` | CRAFT-R ablation: refiner only, no condenser. |
| `scripts/craft_c.sh` | CRAFT-C ablation: condenser only, no refiner. |

All nine scripts share `scripts/_base.sh`, which holds the common configuration
and the paper's default hyper-parameters.

## Repository layout

```
src/        CRAFT implementation (orchestrator, refiner, condenser,
            acquisition, selector, evaluation, metrics)
scripts/    one runnable script per method (see table above)
scenario/   task definitions, data, and seed prompts for the six benchmarks
```

## Outputs

Each run writes to `outputs/<run_name>/`: per-round prompt candidates, the
retained Pareto front, and the metrics computed at the round snapshot
(best score, mean front score, min feasible tokens, peak efficiency,
hypervolume, IGD, front size).

## Notes

- Weights & Biases logging is **off** by default. Set `WANDB_MODE=online` and
  `wandb login` to enable it.
- LLM responses are cached under `cache/` so re-runs reuse prior calls.
