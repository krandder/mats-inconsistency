#!/bin/bash
# run a command as root on the P1 VM: vm.sh 'cmd'
ssh -i [ssh-key-path-removed] -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=15 -o ServerAliveInterval=30 p1@[ip-removed] "sudo bash -c $(printf '%q' "$1")"
