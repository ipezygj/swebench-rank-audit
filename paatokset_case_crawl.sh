#!/bin/bash
# Fetch paatokset.hel.fi case pages and list every ahjojulkaisu.hel.fi PDF
# attachment linked from each - one case page yields all its attachments,
# not just the one a WebSearch snippet happened to match.
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/131.0 Safari/537.36"
while read -r case; do
  [ -z "$case" ] && continue
  url="https://paatokset.hel.fi/fi/asia/${case}"
  html=$(curl -sLk --max-time 20 -A "$UA" "$url")
  links=$(echo "$html" | grep -oE 'href="https://ahjojulkaisu\.hel\.fi/[^"]*\.pdf"' | sed -E 's/href="(.*)"/\1/' | sort -u)
  if [ -n "$links" ]; then
    while read -r l; do
      echo "$case	$l"
    done <<< "$links"
  fi
  sleep 0.2
done < "$1"
