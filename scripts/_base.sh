#!/usr/bin/env bash
# Shared runner for the CRAFT experiment scripts.
#
# This file is NOT run directly. Each per-method script in this folder
# (craft.sh, weighted_sum.sh, ...) sets the method-specific variables and
# then sources this file, e.g.:
#
#     bash scripts/craft.sh beaver_tails
#
# Every setting below can be overridden from the environment, e.g.:
#
#     SEED=91 MAX_ROUNDS=12 MODEL=gpt-5 bash scripts/craft.sh go_emotions
set -euo pipefail

# ---- task ----
# One of: beaver_tails | go_emotions | disambiguation_qa |
#         causal_judgement | formal_fallacy | salient_translation
TASK="${1:-beaver_tails}"
SEED="${SEED:-50}"
RUN_NAME="${RUN_NAME:-${METHOD:-run}_${TASK}_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"

# ---- optimizer / target LLM (override via env) ----
# PROVIDER is one of: openai | azure | ollama | codex
PROVIDER="${PROVIDER:-openai}"
MODEL="${MODEL:-gpt-5}"
PROVIDER_EVAL="${PROVIDER_EVAL:-$PROVIDER}"
MODEL_EVAL="${MODEL_EVAL:-$MODEL}"

# ---- search budget (paper defaults) ----
MAX_ROUNDS="${MAX_ROUNDS:-8}"     # R: optimization rounds (paper reports R=8)
BEAM_SIZE="${BEAM_SIZE:-4}"       # k: retained population size
EVAL_ROUNDS="${EVAL_ROUNDS:-8}"   # T: number of validation subsets
MAX_THREADS="${MAX_THREADS:-4}"
EVAL_THREADS="${EVAL_THREADS:-12}"

# ---- per-dataset validation/test sizes (paper Table 6) ----
case "$TASK" in
  beaver_tails|go_emotions) N_VAL="${N_VAL:-200}"; N_TEST="${N_TEST:-500}";;
  causal_judgement)         N_VAL="${N_VAL:-37}";  N_TEST="${N_TEST:-130}";;
  *)                        N_VAL="${N_VAL:-50}";  N_TEST="${N_TEST:-175}";;
esac

# ---- fixed hyper-parameters ----
FILTER_FACTOR="${FILTER_FACTOR:-0.7}"   # batch-relative pruning strictness
COMPRESS_MAX="${COMPRESS_MAX:-0.2}"
COMPRESS_MIN="${COMPRESS_MIN:-0.1}"
COMPRESS_N="${COMPRESS_N:-4}"           # n_C: condensations per prompt

# ---- method-specific knobs (set by the calling script) ----
SCORING_METRIC="${SCORING_METRIC:?must be set by the per-method script}"
ENHANCER_TYPE="${ENHANCER_TYPE:-sculpt}"
DISABLE_ENHANCER="${DISABLE_ENHANCER:-False}"
COMPRESS_METHODS="${COMPRESS_METHODS:-distill}"
SCORE_WGT="${SCORE_WGT:-0.5}"           # weighted-sum only
TOKEN_WGT="${TOKEN_WGT:-0.5}"           # weighted-sum only

WEIGHT_ARGS=""
if [ "$SCORING_METRIC" = "weighted" ]; then
  WEIGHT_ARGS="selector.score_weight=$SCORE_WGT selector.token_weight=$TOKEN_WGT"
fi

# Pair UCB acquisition with the selector method. main.py enforces this
# mapping via _validate_selector_ucb_pairing; hardcoding ucb-mo here would
# fail-fast for every score/weighted baseline. Override via UCB_TYPE if you
# really need a non-default pairing.
case "$SCORING_METRIC" in
  score|token)        DEFAULT_UCB_TYPE="ucb-e";;
  weighted)           DEFAULT_UCB_TYPE="ucb-ws";;
  nsga2|nsga2-lcb)    DEFAULT_UCB_TYPE="ucb-mo";;
  *)                  DEFAULT_UCB_TYPE="ucb-mo";;
esac
UCB_TYPE="${UCB_TYPE:-$DEFAULT_UCB_TYPE}"

# Weights & Biases is off by default; set WANDB_MODE=online to enable.
export WANDB_MODE="${WANDB_MODE:-disabled}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "[CRAFT] method=${METHOD:-?} task=$TASK seed=$SEED provider=$PROVIDER model=$MODEL"
PYTHONPATH=src python -m main \
  task_name="$TASK" run_name="$RUN_NAME" output_dir="$OUTPUT_DIR" \
  max_rounds="$MAX_ROUNDS" run_test_evals=True random_seed="$SEED" \
  orchestrator.provider="$PROVIDER" orchestrator.model="$MODEL" \
  orchestrator.n_val_exs="$N_VAL" orchestrator.n_test_exs="$N_TEST" \
  orchestrator.format_prompt=False orchestrator.num_fewshots=20 \
  orchestrator.init_prompt_gen=False orchestrator.max_threads="$MAX_THREADS" \
  selector.method="$SCORING_METRIC" selector.beam_size="$BEAM_SIZE" \
  selector.accuracy_buffer=0.0 selector.token_buffer=0 \
  selector.filter_factor="$FILTER_FACTOR" selector.lambda_factor=1 $WEIGHT_ARGS \
  evaluator.provider="$PROVIDER_EVAL" evaluator.model="$MODEL_EVAL" \
  evaluator.validation_type=ucb evaluator.ucb_type="$UCB_TYPE" \
  evaluator.use_stratified_validation=True evaluator.eval_rounds="$EVAL_ROUNDS" \
  evaluator.max_threads="$EVAL_THREADS" \
  enhancer.provider="$PROVIDER" enhancer.model="$MODEL" \
  enhancer.type="$ENHANCER_TYPE" enhancer.disable="$DISABLE_ENHANCER" \
  enhancer.n_critic=2 enhancer.errors_per_critic=4 \
  enhancer.max_structural_rounds=1 enhancer.aggregate_feedbacks=diversity_agg \
  compressor.provider="$PROVIDER" compressor.model="$MODEL" \
  compressor.methods="$COMPRESS_METHODS" compressor.max_ratio="$COMPRESS_MAX" \
  compressor.min_ratio="$COMPRESS_MIN" compressor.num_compressions="$COMPRESS_N" \
  compressor.max_threads="$MAX_THREADS" \
  mutator.provider="$PROVIDER" mutator.model="$MODEL" mutator.num_mutations=0 \
  cache.enabled=True cache.cache_dir="${CACHE_DIR:-cache}" \
  cache.refresh_cache=False cache.clear_on_start=False cache.print_stats=True
