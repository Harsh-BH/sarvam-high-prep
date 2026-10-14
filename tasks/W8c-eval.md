# W8c: held-out split, eval harness, SeamlessStreaming baseline

**Goal (learning version of W8c in W8-partb-prep.md):** measure a streaming Hi→En S2S system the same way before and after fine-tuning (§8 G2), on held-out data nobody trains on. Sarvam gives no KPI feedback (§6), so the KPIs of §7 get proxies.

Owner files: `w8c/`, `data/w8c/`, this file. Base model (Harsh's decision): SeamlessStreaming (Meta `seamless_communication`, **CC BY-NC 4.0 weights: NC, private eval only, never publish outputs**).

## Split rule

Code: `w8c/split.py` (`uv run w8c/split.py`). Files: `w8c/splits/{train,dev,test}.txt` + `heldout.json` (git), copied to `data/w8c/splits/`.

- **Train** = release/pilot-v2 minus every record `is_heldout()` flags.
- **Held-out synthetic:** whole voices, per engine, in crc32(model/voice) order, until ≥ 10% of that engine's pilot-v2 hours. Each pilot-v2 script has exactly one take, so whole scripts go with their voice. A cloned OmniVoice voice is eligible only if the IndicVoices person it imitates is not a pilot-v2 real speaker. Voices go to dev until dev holds ⅓ of the held-out synthetic hours, then to test.
- **Held-out real:** IndicVoices Hindi *valid* (data/w1/iv_full), cut with W1's cutter and drop rule (`w1/cut_pilot.file_candidates`, pinned ffmpeg).
  - Eligible speakers: not a pilot-v2 real speaker, not a w3/voices.tsv reference speaker.
  - Speaker → dev if crc32(speaker) % 3 == 0, else test. Dev and test are speaker-disjoint.
- **Future records** (scale sessions) are train unless `is_heldout(record)` says otherwise. It flags:
  - a dev/test id;
  - a held-out person: a real `speaker_id`, or an `iv-<speaker>` voice that imitates one;
  - a held-out (model, voice);
  - a held-out script pid (`_w2.pid` / `_w3.pid`).
- **IDs:** `rec-800000`+ from `w8c/ids.tsv`. This is the SPEC §4 reserved range, and that file is its only allocator. It never collides with W1/S1 (`w1/ids.tsv`) or W3/S2 (`w3/ids.tsv`). These clips are eval data, not Part A records.
- **FLEURS hi_in test** (CC BY 4.0, read speech): a third, eval-only set; English references joined from en_us on the FLoRes sentence id.

**Finding: IndicVoices valid is NOT speaker-disjoint from train.**
- 255 of the 483 valid speakers occur in our 2,273-file IV-train sample, all 255 with identical gender/age/district.
- 96 of the 270 pilot-v2 real speakers occur in valid, as do 26 of the 31 OmniVoice reference speakers.
- So "valid = unseen speakers" is false. Disjointness comes from the speaker filter above, not from IV's split.

## Numbers (2026-10-15)

| | synthetic (pilot-v2) | real (IV valid, before stages) |
|---|---|---|
| dev | 33 rec, 0.193 h: bulbul kavya, OmniVoice iv-S4257045200388748, iv-S4258798800383835 | 101 clips, 0.626 h, 49 speakers |
| test | 45 rec, 0.298 h: OmniVoice iv-S4258214800384862, iv-S4259922900308772, magpie Aria | 187 clips, 1.124 h, 76 speakers |
| train | 1,010 of 1,088 pilot-v2 records, 6.090 h | |

- **Supply limit:** every eligible valid clip is used.
  - The cutter finds 1.75 h of 10–60 s single-speaker windows in 9.3 h of eligible raw valid audio. Breaks: 609 loud gaps, 478 second-voice/partner segments, 84 disfluency segments, 79 conversations without a partner JSON.
  - So real dev/test land below the 1 h / 2 h target. FLEURS test adds read speech on top.
- **First try:** also excluding the 1,162 IV-train-sample speakers (so S1 loses no train speakers) left only 0.95 h. Dropped: S1 just skips the held-out speakers' train files via is_heldout.

## Spend

| Date | What | Cost |
|---|---|---|
