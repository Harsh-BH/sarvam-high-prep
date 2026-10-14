# W6 translation style guide v1

Hindi (code-mixed, spoken) → English sentence-level targets. Binding for `sentences[].en_text` / `en_alt`. The
section "Rules" is pasted verbatim into the translator and judge prompts (`w6/translate.py`), so editing it changes
every request key and re-translates: bump the version line when you do. Rationale and PS citations: `tasks/W6-translation.md`.

Version: w6-style-v1

## Rules

You translate one recording of spontaneous or read Hindi speech (often mixed with English words) into English. The
Hindi is a verbatim transcript: fillers, repetitions, false starts and self-corrections are written as spoken, English
words are written in Devanagari, numbers are written as words. Your English becomes the target that a streaming
speech-to-speech model learns to speak, and is read aloud by an English TTS.

1. **Segments.** The input is a list of Hindi segments. Output exactly one English segment per Hindi segment, in order.
   Each English segment translates only its own Hindi segment: never move words or facts to another segment, never
   merge or split segments. A segment may be a fragment (the speaker trailed off, or the split is mid-phrase):
   translate it as a fragment ("At that rate.", "If their."), using the other segments only to understand it.
2. **Faithful and natural.** Translate the meaning completely: no summarising, no omissions, no additions, no
   explanations. Spoken register, first person kept, contractions allowed. Where English allows it, keep the speaker's
   order of information (who, what, numbers) rather than reordering for elegance.
3. **Disfluencies.** Drop fillers (अ, अं, उम्म, हम्म, मतलब/यानी used as fillers), stutters, repeated words and abandoned
   false starts; never start a segment with "Uh," or "Um,". A segment that is only a filler becomes "Uh." (never empty).
4. **Self-corrections.** When the speaker abandons a value or word and replaces it inside the same segment
   ("साढ़े तीन लाख … नहीं नहीं पौने चार लाख"), `en` states only the corrected meaning: "three and three-quarter lakh
   rupees were debited". Never state the abandoned value as fact. When the correction is in a later segment, translate
   each segment as said: the later one is an explicit correction ("Sorry, the last three digits are two one nine, not
   zero."). A contrast is not a correction: "तीस किलो है पच्चीस नहीं" = "it is thirty kilos, not twenty-five".
5. **`alt` (only for rule-4 segments).** When both the abandoned value and its correction are inside this one segment,
   also give `alt`: the same sentence rendered as spoken, with the abandoned value and an explicit repair ("Three and a
   half lakh rupees were debited, no, no, three and three-quarter lakh."). `alt` uses only this segment's words, never
   words from another segment. For every other segment, including one that corrects an earlier segment, give no `alt`.
6. **Numbers in words, always.** Never digits, never symbols (₹, %, /, :). Indian units: thousand, lakh, crore; never
   million or billion. Keep the speaker's amount: "साढ़े तीन लाख" = "three and a half lakh" or "three lakh fifty
   thousand"; "सवा करोड़" = "one and a quarter crore"; "पौने दो करोड़" = "one and three-quarter crore"; "डेढ़ हज़ार" =
   "one and a half thousand" or "fifteen hundred". "रुपये" = "rupees". Hyphenate twenty-one to ninety-nine. Say every
   number in full, prices too: "three hundred and ninety-nine", not "three ninety-nine".
   - Times: "साढ़े चार बजे" = "half past four" or "four thirty"; "पौने नौ" = "a quarter to nine". Dates:
     "सत्ताईस सितंबर" = "the twenty-seventh of September".
   - Digit strings (phone, account, Aadhaar, PNR, OTP, IDs) are spoken digit by digit, in the speaker's grouping:
     "nine eight seven six…", keeping "double seven" / "triple five" when said. शून्य and ज़ीरो are "zero".
   - Letters in codes are capital letters separated by spaces: "K X seven Y four M". Acronyms are written as
     acronyms, with exactly the letters the speaker said (पी आर ए एन = PRAN, not PAN): UPI, OTP, EMI, GST, IRCTC.
7. **English words the speaker used stay the same English words**, spelled correctly: "क्रेडिट कार्ड" = "credit card",
   "सेविंग्स अकाउंट" = "savings account", "रिफ़ंड" = "refund" (not "reimbursement"). Mispronounced English is restored
   to the intended word ("इस्क्रीन" = "screen"). Brands and products in their official spelling: PhonePe, Paytm,
   IndiGo, MakeMyTrip, Flipkart, BigBasket, MacBook.
8. **Names and places** are romanised in their usual spelling, never translated: "सुनीता राघवन" = "Sunita Raghavan",
   "मुज़फ़्फ़रपुर" = "Muzaffarpur", "केम्पेगौड़ा" = "Kempegowda". Keep the form the speaker said (हुबली = "Hubli").
   Titles: "डॉक्टर" = "Doctor"; "सर", "मैम" = "sir", "ma'am"; "जी" after a name may be dropped.
9. **Intent explicit.** Negations are never dropped or softened. Questions stay questions and end with "?", including
   intonation-only ones (the Hindi ends with "?"). Conditions stay conditions ("अगर … तो" = "If …, …"). Requests,
   refusals and confirmations keep their force. Who does what to whom stays the same.
10. **Output text only.** No Devanagari, no notes, no brackets, no quotation marks around the segment, no
    "translates to". Plain ASCII punctuation; each segment ends with ".", "?" or "!".
