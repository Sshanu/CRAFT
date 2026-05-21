#!/usr/bin/env bash
# CRAFT-C ablation: NSGA-II selector with the condenser only (no refiner).
export METHOD="craft_c" SCORING_METRIC="nsga2" ENHANCER_TYPE="sculpt" DISABLE_ENHANCER="True" COMPRESS_METHODS="distill"
source "$(dirname "$0")/_base.sh" "$@"
