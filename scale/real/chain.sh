#!/bin/bash
# S1: one cut shard through the stages.   bash scale/real/chain.sh <tag> stages/cut/<run>
#   audio_qc, dedupe, transcribe: local, light CPU. One shard's audio on disk at a time (LOCK; 1 GB laptop budget).
#   align, asr_check: SageMaker ml.c5.2xlarge spot, at most SLOTS of S1's jobs at once (spot is capped at 10 instances account-wide).
#   translate: local (the SageMaker role has no Bedrock access), last, so asr_check's drops cost no tokens.
# Run ids are <time>-<sha>-<tag>, logged to data/scale-real/chains/<tag>.tsv as "<stage> <run> started|ok".
# A rerun skips steps logged ok; a local step resumes into its logged run, a SageMaker step gets a new run.
set -uo pipefail
cd "$(dirname "$0")/../.."
tag=$1 prev=$2
D=/home/harsh1/github-repos/sarvam-high-prep/data/scale-real
CACHE=/home/harsh1/github-repos/sarvam-high-prep/data/cache
LOG=$D/chains/$tag.tsv SLOTS=${SLOTS:-4}  # the account allows 10 spot instances across ALL training jobs, shared
mkdir -p "$D/chains"
touch "$LOG"
echo "$prev" > "$D/chains/$tag.cut"
set -a; . /home/harsh1/github-repos/sarvam-high-prep/.env; set +a
cut_audio="$CACHE/audio/${prev//\//_}_audio"  # run.run_local's shared audio cache for this cut run

declare -A EXTRA=([translate]="-- --workers 16")
say() { echo "$(date -u +%FT%TZ) [$tag] $*"; }
run_step() {  # run_step <stage> local|sm <instance> [-- params]; chains $prev -> stages/<stage>/<run>
  local stage=$1 where=$2 inst=$3; shift 3
  local ok rid rc out
  ok=$(awk -v s="$stage" '$1==s && $3=="ok"{print $2}' "$LOG" | tail -1)
  if [ -n "$ok" ]; then prev="stages/$stage/$ok"; return 0; fi
  rid=$(awk -v s="$stage" '$1==s && $3=="started"{print $2}' "$LOG" | tail -1)
  while true; do
    if [ "$where" = sm ] || [ -z "$rid" ]; then rid="$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short HEAD)-$tag"; fi
    printf '%s\t%s\tstarted\n' "$stage" "$rid" >> "$LOG"
    say "$stage $rid ($where) <- $prev"
    if [ "$where" = sm ]; then
      exec {sl}>"$D/.sm.submit2"; flock "$sl"  # one S1 submit at a time, so the account-wide count can't race
      while [ "$(spot_used)" -ge 8 ]; do sleep 60; done  # leave 2 of the account's 10 spot slots to other sessions
      # {sl}>&-: the job must not inherit the submit lock, or it holds it until the job ends
      uv run pipeline/run.py "$stage" --in "$prev" --run "$rid" --sm "$inst" --spot "$@" > "$D/chains/$tag.$stage.out" 2>&1 {sl}>&- &
      local pid=$!
      until aws sagemaker describe-training-job --training-job-name "sarvam-${stage//_/-}-$rid" >/dev/null 2>&1 || ! kill -0 $pid 2>/dev/null; do sleep 5; done
      exec {sl}>&-
      wait $pid; rc=$?; out=$(cat "$D/chains/$tag.$stage.out")
    else
      out=$(uv run pipeline/run.py "$stage" --in "$prev" --run "$rid" --local "$@" 2>&1); rc=$?
    fi
    echo "$out" | tail -5 > "$D/chains/$tag.$stage.last"
    if [ $rc -eq 0 ]; then break; fi
    say "$stage FAILED (rc $rc): $(echo "$out" | tail -3 | tr '\n' ' ')"; return 1
  done
  printf '%s\t%s\tok\n' "$stage" "$rid" >> "$LOG"
  prev="stages/$stage/$rid"
}
spot_used() {  # spot training instances in use account-wide, every session's (each job here is 1 instance)
  local n=0 j
  for j in $(aws sagemaker list-training-jobs --status-equals InProgress --max-results 100 --query "TrainingJobSummaries[].TrainingJobName" --output text); do
    [ "$(aws sagemaker describe-training-job --training-job-name "$j" --query EnableManagedSpotTraining --output text)" = True ] && n=$((n + 1))
  done
  echo $n
}
with_slot() {  # run "$@" holding one of SLOTS SageMaker slots
  while true; do
    for i in $(seq 0 $((SLOTS - 1))); do
      exec {fd}>"$D/.sm.slot$i"
      if flock -n "$fd"; then "$@"; local rc=$?; exec {fd}>&-; return $rc; fi
      exec {fd}>&-
    done
    sleep 60
  done
}
local_steps() {  # under LOCK: download the shard's audio once, run the light stages, free the disk
  exec {lk}>"$D/.local.lock"; flock "$lk"
  for s in "$@"; do run_step "$s" local - ${EXTRA[$s]:-} || { rm -rf "$cut_audio"; exit 1; }; done
  rm -rf "$cut_audio"
  exec {lk}>&-
}

uv run scale/real/report.py heldout "$prev" || { say "held-out check failed"; exit 1; }
local_steps audio_qc dedupe transcribe
with_slot run_step align sm ml.c5.2xlarge || exit 1
with_slot run_step asr_check sm ml.c5.2xlarge || exit 1
local_steps translate
say "done: $prev"
