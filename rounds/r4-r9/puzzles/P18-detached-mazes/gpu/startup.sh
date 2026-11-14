#!/bin/bash
# P18 VM startup: stop apt from grabbing the dpkg lock or the network during runs
for u in apt-daily.timer apt-daily-upgrade.timer apt-daily.service apt-daily-upgrade.service unattended-upgrades.service; do
  systemctl stop $u 2>/dev/null; systemctl mask $u 2>/dev/null; done
echo "startup done $(date -u +%FT%TZ)" > /var/log/p18-startup.txt
