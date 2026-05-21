#!/usr/bin/env bash
# CRAFT: NSGA-II selector + SCULPT refiner + Distill condenser (full method).
export METHOD="craft" SCORING_METRIC="nsga2" ENHANCER_TYPE="sculpt" DISABLE_ENHANCER="False" COMPRESS_METHODS="distill"
source "$(dirname "$0")/_base.sh" "$@"
