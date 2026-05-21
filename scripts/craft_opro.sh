#!/usr/bin/env bash
# CRAFT with the SCULPT refiner replaced by OPRO.
export METHOD="craft_opro" SCORING_METRIC="nsga2" ENHANCER_TYPE="opro" DISABLE_ENHANCER="False" COMPRESS_METHODS="distill"
source "$(dirname "$0")/_base.sh" "$@"
