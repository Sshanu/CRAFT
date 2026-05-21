#!/usr/bin/env bash
# CRAFT-R ablation: NSGA-II selector with the refiner only (no condenser).
export METHOD="craft_r" SCORING_METRIC="nsga2" ENHANCER_TYPE="sculpt" DISABLE_ENHANCER="False" COMPRESS_METHODS="NA"
source "$(dirname "$0")/_base.sh" "$@"
