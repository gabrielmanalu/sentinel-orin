#!/usr/bin/env bash
# Log tegrastats to CSV during a run. Reused from EdgeDrive.
set -euo pipefail
INTERVAL_MS="${1:-1000}"
OUT="${2:-/data/sentinel/benchmarks/tegrastats_$(date +%Y%m%d_%H%M%S).log}"
echo "Logging tegrastats every ${INTERVAL_MS}ms to ${OUT}"
sudo tegrastats --interval "${INTERVAL_MS}" --logfile "${OUT}"
