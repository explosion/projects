#!/bin/bash
# Utility for robustly downloading large files (i.e. retrying on dropped connections).
# Source: https://superuser.com/a/689340

while [ 1 ]; do
wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 0 --continue $1
if [ $? = 0 ]; then break; fi; # check return value, break if successful (0)
sleep 1s;
done