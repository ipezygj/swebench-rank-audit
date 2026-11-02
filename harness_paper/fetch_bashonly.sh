#!/bin/sh
# per-instance report.json for every bash-only (mini-SWE-agent) submission, public bucket
set -e
for s in $(aws s3 ls --no-sign-request s3://swe-bench-submissions/bash-only/ | awk '/PRE/{print $2}' | tr -d /); do
  aws s3 sync --no-sign-request --only-show-errors --exclude "*" --include "*/report.json" \
    s3://swe-bench-submissions/bash-only/$s/logs/ bashonly/$s/ || echo "FAIL $s"
  echo "$s $(ls bashonly/$s | wc -l)"
done
