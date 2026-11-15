#!/usr/bin/env bash
# Local watcher: exits (printing the line) when <logfile> on the AWS host contains <regex>,
# or after 3 consecutive SSH failures. Usage: watch_aws.sh <logfile> <regex>
source "$HOME/Downloads/ocr-rnd/ocr-rnd.env"
fails=0
while true; do
  if out=$(ssh -i "$HOME/Downloads/ocr-rnd/ocr-rnd.pem" -o BatchMode=yes -o ConnectTimeout=20 "ubuntu@$OCR_RND_EIP" \
           "grep -E '$2' ~/wm_research/$1 | tail -3" 2>&1); then
    fails=0
    [ -n "$out" ] && { echo "$1: $out"; exit 0; }
  else
    fails=$((fails+1)); [ $fails -ge 3 ] && { echo "SSH FAILING: $out"; exit 1; }
  fi
  sleep 300
done
