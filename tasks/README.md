# Workstreams

The problem statement ([`../ps.md`](../ps.md), verbatim) is split into parallel workstreams; every § reference in these files points into it. Each `W*.md` file holds one stream's goal, the PS rules that bind it, its research questions, its experiments and its done-when. Owners append their findings to their own file.

**How we stay parallel:**

1. **Day 1–3:** W0 freezes the contracts: `SPEC.md`, `validate.py` and the intermediate record format.
2. **Week 1:** build a **pilot set** of about 50 real clips + 50 synthetic clips. Every stream develops on the pilot; nobody waits for upstream to scale.

## Dependency map

```
            ┌──────────────── W0  Contracts + validator (days 1–3) ────────────────┐
            ▼                        ▼                        ▼                     ▼
REAL:   W1 Real-audio sourcing → W4 Transcription ─┐
                                                   ├→ W5 Hindi alignment → W6 Translation → W7 Assembly + release
SYNTH:  W2 Script generation  → W3 Hindi TTS + QA ─┘                                          (mid-term)

PART B prep (independent of Part A, start now):
        W8a Streaming server + harness   W8b English target pipeline
        W8c Eval harness + dev set       W8d Model literature + playbooks
```

## Streams

| Stream | File | Goal | Input from | People | Owner |
|---|---|---|---|---|---|
| W0 | [W0-contracts.md](W0-contracts.md) | Rules + validator everyone codes against | — | 1 | |
| W1 | [W1-real-audio.md](W1-real-audio.md) | Find and cut usable real Hindi audio | W0 | 1–2 | |
| W2 | [W2-scripts.md](W2-scripts.md) | Entity/intent-dense Hinglish monologue scripts | W0 | 1–2 | |
| W3 | [W3-hindi-tts.md](W3-hindi-tts.md) | Scripts → clean synthetic Hindi audio | W2 | 1–2 | |
| W4 | [W4-transcription.md](W4-transcription.md) | L1-compliant transcripts for real audio | W1 | 1–2 | |
| W5 | [W5-alignment.md](W5-alignment.md) | Hindi word timings | W3, W4 | 1 | |
| W6 | [W6-translation.md](W6-translation.md) | Streaming-friendly English targets | W3, W4 | 1–2 | |
| W7 | [W7-assembly.md](W7-assembly.md) | Generate/rebuild pipeline, HF release, stats | all | 1 | |
| W8 | [W8-partb-prep.md](W8-partb-prep.md) | Server, English targets, eval, model playbooks | — | 2–3 | |

With fewer than about 8 people, merge W0+W7, W1+W4, W2+W3 and W5+W6, and run W8 as two people (8a+8c, 8b+8d).

## Week-1 deliverables

| Stream | Deliverable |
|---|---|
| W0 | `SPEC.md` + validator v0 (day 3) |
| W1 | Access to IndicVoices, SPRING-INX, Vaani, Kathbath; 50 real pilot clips cut |
| W2 | Coverage matrix v0; 50 pilot scripts |
| W3 | 50 pilot scripts synthesised with 2 engines |
| W4 | 1 h ASR benchmark set, hand-checked |
| W5 | Gold alignment set started; MMS-FA baseline running |
| W6 | Style guide v0; 3 translation engines run on the pilot |
| W7 | `rebuild` stub; private HF dataset repo created |
| W8a | Harness that streams real-time audio and flags every §5.1 violation |

## Rules that bind everyone

- **The audio is frozen at the mid-term** (§3B, §8 G3). After that we can improve text but never add records or hours. Record IDs and `source_file` names are never renumbered or renamed.
- **Sarvam API outputs are never published** (§10 R3). Keep this repo and all HF repos **private**.
- **Sarvam owns the IP of submitted code** (§10 R5). Don't paste in code we can't assign.
- **Credits** (§11 FAQ): Sarvam's $1,000 must also pay for the Part B English targets, so Part A spends ≤30–40% of it. Modal tranche 1 is all we have before the mid-term.
- Some questions can't be answered by reading. Decide on **measured** pilot results, not paper claims.
