#!/usr/bin/env bash
set -ex

curl "https://api.warpcast.com/v2/recent-casts?limit=1000" \
  -H "accept: application/json"	                	\
  -H "authorization: Bearer $MERKLE_SECRET"		\
  -o "$PWD/data/recent_casts.json"
