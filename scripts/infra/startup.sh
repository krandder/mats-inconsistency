#!/bin/bash
# GCE startup script for soup-gpu VMs (runs as root at boot).
# Installs clang, bun and the pinned Bend2 checkout, then writes /var/lib/soup/READY
# (or /var/lib/soup/FAILED). gpu-up.sh waits for one of them and logs the result.
# Placeholders __BEND_REF__ __BEND_LOCAL__ __BUN_VERSION__ are filled in by gpu-up.sh.
exec >>/var/log/soup-startup.log 2>&1
set -x
mkdir -p /var/lib/soup /root/runs
[[ -f /var/lib/soup/READY ]] && exit 0
fail() { echo "FAIL $*" >>/var/lib/soup/FAILED; exit 1; }
export DEBIAN_FRONTEND=noninteractive
for i in 1 2 3 4 5; do apt-get update -q && break; sleep 10; done
apt-get install -y -q clang clang-19 rsync git unzip curl build-essential || fail apt
# GCE startup services may have no HOME; the interactive Bun installer assumes
# it exists. Install the same pinned x86_64 release without shell-profile edits.
curl -fsSL "https://github.com/oven-sh/bun/releases/download/bun-v__BUN_VERSION__/bun-linux-x64.zip" -o /var/lib/soup/bun.zip || fail bun-download
unzip -oq /var/lib/soup/bun.zip -d /opt/bun || fail bun-unzip
install -Dm755 /opt/bun/bun-linux-x64/bun /opt/bun/bin/bun || fail bun-install
[[ "$(/opt/bun/bin/bun --version)" == "__BUN_VERSION__" ]] || fail bun-version
ln -sf /opt/bun/bin/bun /usr/local/bin/bun
git clone --depth 1 --branch __BEND_REF__ https://github.com/bendlang/bend.git /opt/bend2 || fail bend-clone
# Same absolute path as on General, so ../selfmod/bend2-lab/run.sh style scripts work unchanged.
mkdir -p "$(dirname __BEND_LOCAL__)" && ln -sfn /opt/bend2 __BEND_LOCAL__
echo 'export PATH=/usr/local/cuda/bin:/opt/bun/bin:$PATH' >/etc/profile.d/soup.sh
export PATH=/usr/local/cuda/bin:$PATH
{
  echo "gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | head -1)"
  echo "nvcc=$(nvcc --version 2>&1 | tail -1)"
  echo "clang=$(clang --version | head -1)"
  echo "bun=$(bun --version)"
  echo "bend=__BEND_REF__@$(git -C /opt/bend2 rev-parse --short HEAD)"
} >/var/lib/soup/READY.tmp && mv /var/lib/soup/READY.tmp /var/lib/soup/READY
