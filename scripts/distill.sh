#!/usr/bin/env bash
# Distill single-objective baseline: score-only selection with Distill compression.
export METHOD="distill" SCORING_METRIC="score" ENHANCER_TYPE="sculpt" DISABLE_ENHANCER="True" COMPRESS_METHODS="distill"
source "$(dirname "$0")/_base.sh" "$@"
