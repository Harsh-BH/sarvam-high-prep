"""W6 automatic translation QA (tasks/W6-translation.md "Automatic QA"). Pure functions, no I/O.

    check(hi, en, glosses=()) -> {flag: detail}   one sentence; flags in HARD / CRITICAL / SOFT
    hi_numbers(hi), en_numbers(en) -> list of numbers (float) and digit strings (str)

Numbers: every Hindi number/amount must appear in the English with the same value, and lakh/crore stay lakh/crore
(PS §3A). A Hindi number missing from the English is only soft when the sentence has a self-correction marker (FAQ:
the abandoned value is dropped). Digit runs (phone, account, PNR) compare as digit strings, `double`/`triple` expanded.
"""
import re, unicodedata
from collections import Counter

nfc = lambda s: unicodedata.normalize("NFC", s)
DEVA = re.compile(r"[ऀ-ॿ]")

_HI = """शून्य ज़ीरो जीरो|एक|दो दौ|तीन|चार|पाँच पांच|छह छः छै छे छेह|सात|आठ|नौ|दस|ग्यारह|बारह|तेरह|चौदह|पंद्रह पन्द्रह|सोलह|सत्रह|अठारह|उन्नीस
|बीस|इक्कीस|बाईस|तेईस|चौबीस|पच्चीस|छब्बीस|सत्ताईस|अट्ठाईस अठ्ठाईस अठाईस|उनतीस उन्तीस|तीस|इकतीस इकत्तीस|बत्तीस|तैंतीस|चौंतीस
|पैंतीस|छत्तीस|सैंतीस|अड़तीस|उनतालीस उन्तालीस|चालीस|इकतालीस|बयालीस बियालीस|तैंतालीस|चवालीस चौवालीस|पैंतालीस|छियालीस|सैंतालीस|अड़तालीस
|उनचास|पचास|इक्यावन|बावन|तिरपन तिरेपन|चौवन|पचपन|छप्पन|सत्तावन|अट्ठावन अठावन|उनसठ|साठ|इकसठ|बासठ|तिरसठ|चौंसठ|पैंसठ|छियासठ
|सड़सठ सरसठ सढ़सठ|अड़सठ अढ़सठ|उनहत्तर|सत्तर|इकहत्तर|बहत्तर|तिहत्तर|चौहत्तर|पचहत्तर|छिहत्तर|सतहत्तर|अठहत्तर|उन्यासी उनासी|अस्सी|इक्यासी
|बयासी बैयासी बियासी|तिरासी|चौरासी|पचासी|छियासी|सत्तासी|अट्ठासी अठासी|नवासी|नब्बे|इक्यानवे|बानवे|तिरानवे|चौरानवे|पचानवे|छियानवे|सत्तानवे
|अट्ठानवे|निन्यानवे"""
# English number words as spoken inside Hindi, written in Devanagari (SPEC R7)
_HI_EN = {"वन": 1, "टू": 2, "थ्री": 3, "फ़ोर": 4, "फोर": 4, "फ़ाइव": 5, "फाइव": 5, "सिक्स": 6, "सेवन": 7, "एट": 8, "ऐट": 8,
          "नाइन": 9, "टेन": 10, "इलेवन": 11, "ट्वेल्व": 12, "थर्टीन": 13, "फ़ोर्टीन": 14, "फोर्टीन": 14, "फ़िफ़्टीन": 15,
          "फिफ्टीन": 15, "सिक्सटीन": 16, "सेवनटीन": 17, "सेवेनटीन": 17, "एटीन": 18, "नाइनटीन": 19, "ट्वेंटी": 20,
          "ट्वेन्टी": 20, "थर्टी": 30, "फ़ोर्टी": 40, "फोर्टी": 40, "फ़िफ़्टी": 50, "फिफ्टी": 50, "सिक्सटी": 60,
          "सेवेंटी": 70, "सेवनटी": 70, "एटी": 80, "नाइनटी": 90}
HI_SMALL = {nfc(w): i for i, ws in enumerate(_HI.replace("\n", "").split("|")) for w in ws.split()}
HI_SMALL.update({nfc(w): v for w, v in _HI_EN.items()})
HI_SMALL.update({k[:-2] + "बे": v for k, v in list(HI_SMALL.items()) if k.endswith("नवे")})  # इक्यानबे …
HI_MULT = {nfc(w): v for w, v in {"सौ": 100, "हंड्रेड": 100, "हज़ार": 1e3, "हजार": 1e3, "थाउज़ेंड": 1e3, "थाउजेंड": 1e3,
                                  "लाख": 1e5, "करोड़": 1e7, "करोड": 1e7, "मिलियन": 1e6, "बिलियन": 1e9}.items()}
HI_FRAC = {nfc(w): v for w, v in {"साढ़े": .5, "साढे": .5, "सवा": .25, "पौने": -.25}.items()}  # prefix: X+v
HI_HALF = {nfc(w): v for w, v in {"डेढ़": 1.5, "डेढ": 1.5, "ढाई": 2.5}.items()}
HI_REP = {nfc("डबल"): 2, nfc("ट्रिपल"): 3}
# दो is also the imperative "give/do": कर दो, भेज दो. Treat it as a verb after these stems or at the sentence end.
_HOW_MANY = set(map(nfc, "कितने कितना कितनी कई".split()))
_DO_VERB = set(map(nfc, "कर दे ले बता भेज रख चला बोल मंगा मँगा दिखा लगा निकाल डाल छोड़ उठा सुना समझा लिख करवा बदल हटा जोड़ खोल बंद".split()))

EN_SMALL = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
                                       "fifteen sixteen seventeen eighteen nineteen".split())}
EN_SMALL.update({w: 10 * i for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), 2)})
EN_SMALL.update({w: i for i, w in enumerate("_ first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth "
                                            "thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth".split()) if i})
EN_SMALL.update({w: 10 * i for i, w in enumerate("twentieth thirtieth".split(), 2)})
EN_MULT = {"hundred": 100, "thousand": 1e3, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "crore": 1e7, "crores": 1e7,
           "million": 1e6, "billion": 1e9}
EN_REP = {"double": 2, "triple": 3}

HI_NEG = set(map(nfc, "नहीं नही नहि नी मत बिना नो".split()))  # not नई: it is also "new" (नई दिल्ली)
_COPULA = set(map(nfc, "है हैं था थी थे हो".split()))  # न after these is the tag "isn't it", not a negation
EN_NEG = re.compile(r"\b(?:not|no|never|nothing|nobody|none|neither|nor|without|cannot|nowhere)\b|n['’]t\b", re.I)
CORRECTION = set(map(nfc, "सॉरी मतलब रुको गलत माफ़ी मिनट".split()))  # + any नहीं: see _corrected()

HARD = {"devanagari", "empty", "meta"}                     # never acceptable: retry
CRITICAL = {"num_extra", "num_missing", "unit"}            # value or unit changed: repair
SOFT = {"num_corr", "digits", "neg", "length", "loanword"}  # route to the judge / review


class _Run:
    """Accumulates one run of number words into groups (arithmetic numbers)."""
    def __init__(self):
        self.groups, self.total, self.cur, self.frac, self.rep, self.last = [], 0.0, None, 0.0, 1, None

    def _flush_num(self):
        if self.cur is not None or self.total:
            self.groups.append(self.total + (self.cur or 0))
        self.total, self.cur, self.frac = 0.0, None, 0.0

    def small(self, n):
        if self.rep > 1:  # double seven -> 7, 7 (digit string)
            self._flush_num()
            self.groups += [n] * self.rep
            self.rep, self.last = 1, "digit"
            return
        c = self.cur
        joins = c is None and not self.total or (c is not None and c >= 100 and c % 100 == 0 and n < 100) or \
            (c is not None and 20 <= c % 100 and c % 10 == 0 and n < 10 and self.last == "small") or \
            (c is None and self.total and n < 1000)
        if not joins:
            self._flush_num()
        self.cur = (self.cur or 0) + n + self.frac
        self.frac, self.last = 0.0, "small"

    def mult(self, m):
        c = 1 if self.cur is None else self.cur
        if m == 100 and c < 100:
            self.cur = c * 100
        elif self.total and m >= 1e3 and self.total < m:  # ascending (हज़ार … लाख): a new number
            self._flush_num()
            self.total = m
        else:
            self.total, self.cur = self.total + c * m, None
        self.last = "mult"

    def add_frac(self, v):  # "and a half" / साढ़े: applied to the number being built
        if self.cur is not None:
            self.cur += v
        else:
            self.frac = v

    def done(self):
        self._flush_num()
        g = self.groups
        self.groups = []
        if len(g) == 2 and g[0] == g[1]:  # सात सात खिलाड़ी: "seven each"
            g = g[:1]
        if len(g) >= 3 and all(float(x).is_integer() for x in g):  # a spoken digit string: compare as a string
            return ["".join(str(int(x)) for x in g)]
        return g


def _finish(run, out):
    out += [x for x in run.done() if x != 1]  # bare "one"/एक is mostly an article: ignored on both sides


def hi_numbers(hi):
    toks = nfc(hi).rstrip("।?! ").split()
    out, run = [], _Run()
    for i, t in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        if t == nfc("दो") and (nxt is None or (i and toks[i - 1] in _DO_VERB)):
            _finish(run, out); run = _Run()
        elif t in HI_SMALL:
            run.small(HI_SMALL[t])
        elif t in HI_HALF:
            run.small(0); run.cur, run.last = HI_HALF[t], "half"
        elif t in HI_MULT:
            if run.cur is None and not run.total and i and toks[i - 1] in _HOW_MANY:
                continue  # कितने लाख: "how many lakh", no value
            run.mult(HI_MULT[t])
        elif t in HI_FRAC:
            if t == nfc("सवा") and nxt not in HI_SMALL:  # सवा करोड़ = 1.25 crore, सवा किलो = 1.25 kilo
                run.small(0); run.cur = 1.25
            else:
                run.frac = HI_FRAC[t]
        elif t in HI_REP:
            run.rep = HI_REP[t]
        else:
            _finish(run, out); run = _Run()
    _finish(run, out)
    return out


_EN_TOK = re.compile(r"\d[\d,]*(?:\.\d+)?k?|[a-z]+(?:'[a-z]+)?")


def en_numbers(en):
    toks = _EN_TOK.findall(en.lower().replace("-", " "))
    out, run, i = [], _Run(), 0
    while i < len(toks):
        t, nxt = toks[i], toks[i + 1:i + 4]
        if t[0].isdigit():
            v = float(t.rstrip("k").replace(",", "")) * (1e3 if t.endswith("k") else 1)
            run.small(0); run.cur = v
        elif t == "one" and run.last == "mult" and not (nxt and (nxt[0] in EN_SMALL or nxt[0] in EN_MULT)):
            _finish(run, out); run = _Run()  # "the eighty-five lakh one": a pronoun, not 8500001
        elif t in EN_SMALL:
            run.small(EN_SMALL[t])
        elif t in EN_MULT and (run.cur is not None or run.total):
            run.mult(EN_MULT[t])
        elif t == "a" and nxt[:1] and nxt[0] in EN_MULT and nxt[0] != "k":
            run.small(0); run.cur = 1  # a lakh, a hundred
        elif t == "and" and nxt[:2] == ["a", "half"]:
            run.add_frac(.5); i += 2
        elif t == "and" and nxt[:2] == ["a", "quarter"]:
            run.add_frac(.25); i += 2
        elif t == "and" and nxt[:1] == ["three"] and nxt[1:2] and nxt[1].startswith("quarter"):
            run.add_frac(.75); i += 2
        elif t == "and" and run.last == "mult" and nxt[:1] and nxt[0] in EN_SMALL:
            pass  # one hundred and two
        elif t in ("half", "quarter") and nxt[:1] and nxt[0] in ("past", "to") and nxt[1:2] and nxt[1] in EN_SMALL:
            _finish(run, out); run = _Run()
            h = EN_SMALL[nxt[1]]
            out.append(h + (.5 if t == "half" else .25 if nxt[0] == "past" else -.25)); i += 2
        elif t == "point" and run.cur is not None and nxt[:1] and nxt[0] in EN_SMALL and EN_SMALL[nxt[0]] < 10:
            run.cur += EN_SMALL[nxt[0]] / 10; i += 1
        elif t in EN_REP:
            run.rep = EN_REP[t]
        else:
            _finish(run, out); run = _Run()
        i += 1
    _finish(run, out)
    # spoken clock time: "seven thirty" -> 7.5 (Hindi says साढ़े सात)
    j, res = 0, []
    while j < len(out):
        if j + 1 < len(out) and isinstance(out[j], float) and isinstance(out[j + 1], float) and 1 <= out[j] <= 12 \
                and out[j + 1] in (15, 30, 45) and out[j].is_integer():
            res.append(out[j] + out[j + 1] / 60); j += 2
        else:
            res.append(out[j]); j += 1
    return res


def _key(x):
    return x if isinstance(x, str) else round(float(x), 4)


_NUMWORD = HI_SMALL.keys() | HI_MULT.keys() | HI_FRAC.keys() | HI_HALF.keys() | HI_NEG


def _corrected(hi):
    """Self-correction marker: सॉरी/मतलब/रुको…, or a नहीं next to a number word or another नहीं (`ढाई नहीं साढ़े तीन`)."""
    t = nfc(hi).rstrip("।?! ").split()
    return bool(CORRECTION & set(t)) or any(w in HI_NEG and ((k and t[k - 1] in _NUMWORD) or (k + 1 < len(t) and t[k + 1] in _NUMWORD))
                                            for k, w in enumerate(t))


def neg_count(hi=None, en=None):
    if en is not None:
        return len(EN_NEG.findall(en))
    t = nfc(hi).rstrip("।?! ").split()
    neg = lambda k: t[k] in HI_NEG or (t[k] == "न" and k + 1 < len(t) and not (k and t[k - 1] in _COPULA))
    return sum(1 for k in range(len(t)) if neg(k) and not (k and t[k - 1] == t[k]))  # नहीं नहीं counts once


def _word_in(word, en):
    w = re.escape(word.lower().strip(" .'"))
    return re.search(rf"\b{w}(?:s|es|ed|d|ing)?\b", en.lower()) is not None


LOAN_EQUIV = {"ok": "okay", "okay": "ok", "madam": "ma'am", "mam": "ma'am", "mummy": "mom", "papa": "dad"}


def check(hi, en, glosses=(), spoken=False):
    """-> {flag: detail} for one sentence (empty = clean). `glosses`: Latin forms of English words in this Hindi.
    `spoken`: `en` is the as-spoken rendering (en_alt): every Hindi number must be there, abandoned ones included."""
    f = {}
    if not en or not en.strip():
        return {"empty": True}
    if DEVA.search(en):
        f["devanagari"] = DEVA.findall(en)[:5]
    if re.search(r"\btranslat|\bin english\b|^\s*english\s*:", en, re.I):
        f["meta"] = en[:60]
    h, e = Counter(map(_key, hi_numbers(hi))), Counter(map(_key, en_numbers(en)))
    missing, extra = sorted((h - e).elements(), key=str), sorted((e - h).elements(), key=str)
    if extra:
        f["num_extra"] = extra
    if missing:  # in a self-correction the abandoned value is meant to go (FAQ): soft, the judge decides
        f["num_corr" if _corrected(hi) and not spoken else "num_missing"] = missing
    hn, el = nfc(hi), en.lower()
    unit = [u for u, hw, ew in (("lakh", "लाख", r"\bla(?:kh|c)s?\b"), ("crore", "करोड", r"\bcrores?\b"))
            if hw in hn and not re.search(ew, el)]
    if re.search(r"\b(?:million|billion)s?\b", el) and not re.search("मिलियन|बिलियन", hn):
        unit.append("million")
    if unit:
        f["unit"] = unit
    if re.search(r"\d", en):
        f["digits"] = re.findall(r"\S*\d\S*", en)[:5]
    hneg, eneg = neg_count(hi=hi), neg_count(en=en)
    if (hneg > 0) != (eneg > 0) and not (hneg and _corrected(hi)) and not spoken:
        f["neg"] = {"hi": hneg, "en": eneg}
    hw, ew = len(hi.split()), len(en.split())
    if hw >= 4 and not 0.4 <= ew / hw <= 2.2:  # ponytail: fixed bounds from the pilot, refit on the full run if noisy
        f["length"] = round(ew / hw, 2)
    miss = [g for g in glosses if not _word_in(g, en) and not (g.lower() in LOAN_EQUIV and _word_in(LOAN_EQUIV[g.lower()], en))]
    if miss:
        f["loanword"] = miss
    return f


def glosses(segments):
    """IndicVoices L2 `normalized` marks English words as `देवनागरी [Latin]`; tags appear in both L1 and L2, glosses only
    in L2. -> {Devanagari word: Latin}, single letters and numbers excluded (acronyms and numbers are checked elsewhere)."""
    pat = re.compile(r"(\S+)\s*\[([^\]]+)\]")
    out = {}
    for s in segments:
        tags = {b for _, b in pat.findall(s.get("verbatim", ""))}
        for d, latin in pat.findall(s.get("normalized", "")):
            if latin not in tags and DEVA.search(d) and len(latin) > 1 and latin.lower() not in EN_SMALL and "_" not in latin:
                out[nfc(d)] = latin
    return out


def sentence_glosses(hi, gl):
    words = set(nfc(hi).rstrip("।?! ").split())
    return [latin for d, latin in gl.items() if d in words]


if __name__ == "__main__":
    n = lambda s: [x if isinstance(x, str) else float(x) for x in hi_numbers(s)]
    assert n("कल साढ़े तीन लाख रुपये कट गए") == [350000.0]
    assert n("एक लाख सत्तानवे हज़ार का") == [197000.0]
    assert n("बारह सौ पचास रुपये") == [1250.0]
    assert n("सवा करोड़ का") == [12500000.0] and n("पौने दो करोड़ रहा") == [17500000.0]
    assert n("पौने नौ बजे") == [8.75] and n("डेढ़ लाख है") == [150000.0] and n("ढाई हज़ार भेजे") == [2500.0]
    assert n("नंबर है नाइन एट सेवन सिक्स फ़ाइव") == ["98765"]
    assert n("एट डबल वन थ्री ट्रिपल फ़ाइव") == ["8113555"]
    assert n("पूरा कैंसिल कर दो") == [] and n("दो कोल्ड ड्रिंक") == [2.0] and n("एक फ़ोन") == []
    assert n("और सवा किलो प्याज़") == [1.25] and n("दोनों तरफ सात सात खिलाड़ी") == [7.0]
    assert n("कितने लाख तक का लोन") == [] and n("छेह रूट") == [6.0]
    assert n("टी एन सैंतीस दो शून्य एक नौ") == ["372019"]
    assert n("चार हज़ार नौ सौ निन्यानवे") == [4999.0] and n("चार सौ बहत्तर रुपये") == [472.0]
    e = lambda s: [x if isinstance(x, str) else float(x) for x in en_numbers(s)]
    assert e("three and a half lakh rupees") == [350000.0] and e("three lakh fifty thousand") == [350000.0]
    assert e("one and a quarter crore") == [12500000.0] and e("one and three-quarter crore") == [17500000.0]
    assert e("twelve hundred and fifty rupees") == [1250.0] and e("one thousand two hundred fifty") == [1250.0]
    assert e("at a quarter to nine") == [8.75] and e("at eight forty-five") == [8.75] and e("half past four") == [4.5]
    assert e("nine eight double seven") == ["9877"] and e("a lakh") == [100000.0] and e("4,999 rupees") == [4999.0]
    assert e("the twenty-seventh of September") == [27.0] and e("two point five GB") == [2.5]
    assert e("four hundred seventy-two rupees") == [472.0] and e("five hundred forty-eight") == [548.0]
    assert e("one lakh ninety-seven thousand") == [197000.0] and e("thirty-seven two zero one nine") == ["372019"]
    assert check("मेरे खाते से साढ़े तीन लाख रुपये कट गए।", "Three and a half lakh rupees were debited from my account.") == {}
    assert set(check("साढ़े तीन लाख रुपये कट गए।", "350,000 rupees were debited.")) == {"unit", "digits"}
    assert "unit" in check("सवा करोड़ का लोन।", "A loan of 12.5 million.")
    assert "num_extra" in check("चार लाख कटे।", "Five lakh was debited.")
    f = check("ढाई हज़ार भेजे हैं नहीं यार ढाई नहीं साढ़े तीन हज़ार भेजे हैं।", "I've sent three and a half thousand.")
    assert set(f) == {"num_corr"}, f
    assert check("साढ़े तीन लाख नहीं नहीं पौने चार लाख।", "Three and a half lakh, no, no, three and three-quarter lakh.", spoken=True) == {}
    assert "num_missing" in check("साढ़े तीन लाख नहीं नहीं पौने चार लाख।", "Three and three-quarter lakh.", spoken=True)
    assert check("मैंने नहीं बताया।", "I told them.") == {"neg": {"hi": 1, "en": 0}}
    assert check("मैंने नहीं बताया।", "I didn't tell them.") == {}
    assert "devanagari" in check("ठीक है।", "Okay है.")
    assert check("फ़ेमस है।", "It is famous.", ["Famous"]) == {} and "loanword" in check("फ़ेमस है।", "It is well known.", ["Famous"])
    g = glosses([{"verbatim": "और [uhh] फ़ेमस है", "normalized": "और [uhh] फ़ेमस [Famous] है पी [P]"}])
    assert g == {nfc("फ़ेमस"): "Famous"}, g
    assert e("the eighty-five lakh one is fine") == [8500000.0] and e("A B C four four seven three K") == ["4473"]
    assert e("one lakh one thousand") == [101000.0]
    assert neg_count(hi="पॉलिसी लैप्स न हो") == 1 and neg_count(hi="ठीक है न आप आइए") == 0 and neg_count(en="I don’t") == 1
    print("ok: qa")
