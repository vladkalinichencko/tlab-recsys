#!/bin/bash
# The comparison on KION and both Yambda tasks (or on the run directories given as arguments):
# run_a100.sh where there is an NVIDIA card, run_mac.sh on the Mac.
cd "$(dirname "$0")"
if command -v nvidia-smi > /dev/null; then bash run_a100.sh "$@"; else bash run_mac.sh "$@"; fi
