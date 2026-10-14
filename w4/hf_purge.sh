#!/bin/sh
# Delete HF blobs (and their .lock/.refs) in W4's cache that no snapshot links to. This hub version shares
# blobs/ across repos, so `rm -rf models--X` alone frees nothing.  Usage: sh w4/hf_purge.sh [models--repo ...]
cd "$(dirname "$0")/../data/w4/hf" || exit 1
for m in "$@"; do rm -rf "$m"; done
find models--* -type l -exec readlink -f {} \; 2>/dev/null | sort -u > .live
find "$PWD/blobs" -type f ! -name ".*" ! -name "*.lock" ! -name "*.refs" | sort | comm -23 - .live |
    while read -r b; do rm -fv "$b" "$b.lock" "$b.refs"; done
rm -f .live
