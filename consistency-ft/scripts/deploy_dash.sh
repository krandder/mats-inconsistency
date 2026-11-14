#!/bin/sh
set -e

python3 scripts/build_dash.py && rsync -a dash/ deploy-host:fleet/apps/consistency-dash/
