#!/usr/bin/env bash
# LLMLingua single-objective baseline: score-only selection with LLMLingua-2 compression.
export METHOD="llmlingua" SCORING_METRIC="score" ENHANCER_TYPE="sculpt" DISABLE_ENHANCER="True" COMPRESS_METHODS="llmlingua"
source "$(dirname "$0")/_base.sh" "$@"
