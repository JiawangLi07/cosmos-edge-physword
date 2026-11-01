#!/usr/bin/env bash
# Small, offline badminton pilot only. No network/model training is invoked.
set -euo pipefail
PHYSWORD_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$PHYSWORD_REPO"
PHYSWORD_RUN_PYTHON=${PHYSWORD_PYTHON:-python3}
PHYSWORD_RUN_OUTPUT=${1:?Usage: PHYSWORD_PYTHON=/path/to/python bash tools/physword_data/run_pilot.sh /fresh/output}
PHYSWORD_RUN_MODULE=cosmos_framework.data.generator.physword.cli
PHYSWORD_RUN_CONFIG=tools/physword_data/badminton_pilot.yaml
"$PHYSWORD_RUN_PYTHON" -m "$PHYSWORD_RUN_MODULE" audit --config "$PHYSWORD_RUN_CONFIG" --output "$PHYSWORD_RUN_OUTPUT"
"$PHYSWORD_RUN_PYTHON" -m "$PHYSWORD_RUN_MODULE" prepare --config "$PHYSWORD_RUN_CONFIG" --output "$PHYSWORD_RUN_OUTPUT" \
  --episode badminton__episode_0001 --episode badminton__episode_0017 --episode badminton__episode_0038
"$PHYSWORD_RUN_PYTHON" -m "$PHYSWORD_RUN_MODULE" annotate --config tools/physword_data/badminton_annotation_import.yaml --output "$PHYSWORD_RUN_OUTPUT"
"$PHYSWORD_RUN_PYTHON" -m "$PHYSWORD_RUN_MODULE" build --config "$PHYSWORD_RUN_CONFIG" --output "$PHYSWORD_RUN_OUTPUT"
"$PHYSWORD_RUN_PYTHON" -m "$PHYSWORD_RUN_MODULE" qa --output "$PHYSWORD_RUN_OUTPUT"
"$PHYSWORD_RUN_PYTHON" -m "$PHYSWORD_RUN_MODULE" visualize --config "$PHYSWORD_RUN_CONFIG" --output "$PHYSWORD_RUN_OUTPUT"
