#!/usr/bin/env bash
# CRAFT with the SCULPT refiner replaced by EvoPrompt.
export METHOD="craft_evoprompt" SCORING_METRIC="nsga2" ENHANCER_TYPE="evoprompt" DISABLE_ENHANCER="False" COMPRESS_METHODS="distill"
source "$(dirname "$0")/_base.sh" "$@"
