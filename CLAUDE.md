# sarvam-high-prep

IIT Kharagpur team repo for Inter IIT Tech Meet 15.0, Sarvam high-prep PS: streaming Hindi (code-mixed) → English speech-to-speech translation.
- Part A: build a ≤200 h dataset by the mid-term.
- Part B: fine-tune Sarvam's base checkpoint and serve it over WebSocket.

## Stack

Not chosen yet. Decisions are tracked in `tasks/`.

## Layout

```
ps.md    the problem statement, verbatim, with PDF page markers. Cite as §3A, §5.1 R4, §10 R6, FAQ
tasks/   workstreams W0–W8: goal, PS rules, research, done-when. Owners append findings to their own file
```

## Gotchas

- **Keep this repo and every HF repo private.** They hold Sarvam API outputs (Bulbul audio, transcripts, translations), which must never be published (PS §10 R3).
- Bulbul's default sample rate is 22,050 Hz. Always pass `speech_sample_rate=24000`.
- Always pass `model="bulbul:v3"`. Sarvam's docs push `bulbul:v4-flash`, where `shubh` is not a valid speaker (§3B needs v3 `shubh`).
- Record IDs and `source_file` names are frozen at the mid-term. Never renumber or rename (G3).
- Augmented audio is never a dataset record; augment only at training time.
- Hindi transcripts contain no Latin letters and no digits: English words go in Devanagari, numbers as words.

## Don't

- Re-synthesise TTS in `rebuild`. Bulbul has no seed; use the archived outputs.
- Add files outside the PS deliverable list to submissions (§9.2 penalty).
