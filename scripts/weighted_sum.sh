#!/usr/bin/env bash
# Weighted-sum (WPRO) baseline: linear score over normalised accuracy and token reduction.
# Weights default to 0.5/0.5; override with SCORE_WGT / TOKEN_WGT (paper uses 0.3, 0.5, 0.7).
export METHOD="weighted_sum" SCORING_METRIC="weighted" ENHANCER_TYPE="sculpt" DISABLE_ENHANCER="False" COMPRESS_METHODS="distill"
export SCORE_WGT="${SCORE_WGT:-0.5}" TOKEN_WGT="${TOKEN_WGT:-0.5}"
source "$(dirname "$0")/_base.sh" "$@"
