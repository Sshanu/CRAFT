#!/usr/bin/env bash
# SCULPT single-objective baseline: score-only selection, SCULPT refiner, no condenser.
export METHOD="sculpt" SCORING_METRIC="score" ENHANCER_TYPE="sculpt" DISABLE_ENHANCER="False" COMPRESS_METHODS="NA"
source "$(dirname "$0")/_base.sh" "$@"
