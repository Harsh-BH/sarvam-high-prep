# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""W2 script generator: coverage matrix -> per-script spec (entities pre-verbalised by banks.py) -> LLM on Bedrock
(every call through pipeline/archive.archive_call) -> checks -> scripts in W3's probe format.

    uv run w2/gen.py bake [--n 30] [--models a,b]     # generator bake-off: same 30 specs per model, first take only
    uv run w2/gen.py pilot --model NAME --n 500        # data/w2/scripts.jsonl + rejects.jsonl: --takes fresh tries,
                                                       # then --repairs turns that show the model its answer and what broke

The LLM writes markup: {देवनागरी|latin} = English words, [text|type] = entity. W3's w3/probe.py:parse turns it into
tts_dev / tts_lat / hi_text / sentences / entities / en_words and runs validate.py's SPEC §2 checks on the text.
A rerun never pays twice: the archive returns every request it has seen (request = model, prompt, config, take).
"""
import argparse
import json
import random
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "w2"), str(ROOT / "w3"), str(ROOT / "pipeline")]
import banks  # noqa: E402
from banks import DOMAINS, NUMFORMS, Ent, nfc  # noqa: E402
import probe  # noqa: E402  W3's markup parser; it imports validate.check_record

_common = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                         capture_output=True, text=True).stdout.strip()
DATA = (Path(_common).parent if _common else ROOT) / "data/w2"  # data lives in the main checkout, not the worktree

PROMPT_V = "w2-v4"
MODELS = {  # name -> (Bedrock model id, USD per 1K input tokens, per 1K output; ap-south-1 on-demand, Pricing API 2026-10-14)
    "qwen3-235b": ("qwen.qwen3-235b-a22b-2507-v1:0", 0.00026, 0.00104),
    "gpt-oss-120b": ("openai.gpt-oss-120b-1:0", 0.00018, 0.00071),
    "mistral-large-3": ("mistral.mistral-large-3-675b-instruct", 0.00059, 0.00176),
    "deepseek-v3.2": ("deepseek.v3.2", 0.00074, 0.00222),
}
USD_INR = 88.0
# ponytail: the first 1,434 archived w2 calls (to 2026-10-14 16:40) recorded an upper bound as `inr` (input + all 4,096
# output tokens), ≈6x their real cost, and archive.spent() sums `inr`. Measured from the archived usage: recorded ₹1,326.70,
# actual $2.5254 = ₹222.24. The cap is raised by that overcount so it still means $15 real; calls now record a realistic
# estimate (p50 output was 255 tokens). Upgrade: archive_call recording the response's actual cost.
OVERCOUNT_INR = 1326.70 - 222.24
CAP_INR = 15 * USD_INR + OVERCOUNT_INR  # W2's AWS cap ($15, SageMaker + Bedrock together); W2 uses no SageMaker
EST_OUT = 600  # output tokens charged per call before it runs
CONFIG = {"maxTokens": 4096, "temperature": 0.8, "topP": 0.95}
S_PER_CHAR = 0.099  # W3: median TTS seconds per non-space char (data/w3/renders.jsonl)
STREAM = "w2"  # archive spend ledger; bulk() switches it
# Off for the pilot (so pilot-v1 replays from the archive unchanged); bulk() sets them:
AVOID = set()  # bank values never drawn
AMOUNT_CAP = {}  # domain -> largest amount drawn
MAX_CHARS = None  # longest hi_text (non-space chars) kept
RESPELL = {}  # whole word -> spelling W3's ASRs agree on

# --- coverage matrix v0 (targets; tasks/W2-scripts.md explains them) ---------------------------------------------
DOMAIN_W = {"banking": 16, "upi": 14, "telecom": 11, "ecommerce": 13, "travel": 12, "health": 12, "gov": 10, "everyday": 12}
FORMS = ["lakh_crore", "frac", "num2199", "digits", "idcode", "endigits", "date", "time", "small"]
NUMFORM_W = {  # P(primary number form | domain), %
    "banking":   [25, 15, 10, 15, 15, 5, 5, 0, 10],
    "upi":       [5, 15, 15, 15, 15, 10, 5, 5, 15],
    "telecom":   [0, 10, 20, 20, 10, 15, 10, 5, 10],
    "ecommerce": [5, 10, 20, 10, 10, 10, 15, 5, 15],
    "travel":    [0, 15, 10, 10, 15, 10, 15, 25, 0],
    "health":    [20, 15, 15, 5, 10, 5, 10, 15, 5],
    "gov":       [15, 10, 10, 10, 25, 5, 15, 5, 5],
    "everyday":  [10, 25, 15, 10, 5, 10, 5, 15, 5],
}
INTENTS = {  # key: (target %, instruction). Detectable ones are checked in the output (DETECT)
    "neg_place": (10, "Use negation (नहीं, न) at least twice, in different positions, e.g. before the verb in one clause and after it with ही in another."),
    "tag_na": (10, "End a sentence with the tag 'ना' that is NOT a negation, e.g. 'आप आज कर देंगे ना?'."),
    "nahi_to": (7, "Use 'नहीं तो' meaning 'otherwise' (do X, नहीं तो Y will happen)."),
    "mat": (8, "Use the negative imperative 'मत' (मत करना, मत भेजिए)."),
    "cond": (10, "Use a condition with 'अगर ... तो' and/or 'वरना'."),
    "intonation_q": (9, "Ask at least one yes/no question with NO question word at all (not even क्या), marked only by intonation and ending with '?'."),
    "refusal": (8, "The speaker clearly refuses something that fits the situation: an offer, a request or a charge."),
    "confirmation": (8, "The speaker explicitly confirms something that fits the situation (a detail, a booking, a request)."),
    "causative": (9, "Contrast doing something yourself with getting it done by someone else, with a causative verb (करवाया, भिजवाया, बनवाया, लगवाया)."),
    "kal_parson": (8, "Use 'कल' or 'परसों' where only the verb tense tells whether it means the past or the future."),
    "prodrop": (6, "Drop the subject pronoun in several clauses where context makes it clear."),
    "ne_ko": (7, "Make clear who did what to whom with ने and को, including one role that a listener could easily get backwards."),
}
CORR_W = {"none": 40, "entity": 45, "other": 15}
MARKERS = ["नहीं नहीं", "मतलब", "{सॉरी|sorry}", "{आई मीन|I mean}", "एक मिनट"]
MIX = {"light": (30, "light: 1 to 3 English words"),
       "medium": (50, "medium: 4 to 8 English words or short phrases"),
       "heavy": (20, "heavy: 9 or more English words, including at least one whole English phrase or clause of your own that fits the situation")}
LENGTH = {"short": (30, (30, 50)), "medium": (50, (55, 95)), "long": (20, (105, 140))}  # words
GENDER_W = {"f": 50, "m": 50}
DISFL_W = {"filler": {"yes": 50, "no": 50}, "repeat": {"yes": 30, "no": 70}}  # IV: 1.34 fillers, 0.53 repeats per 100 words
N_NAMED = {"short": 1, "medium": 2, "long": 3}
SCENARIOS = {
    "banking": ["money debited without the speaker doing any transaction", "debit card lost, wants it blocked", "loan EMI amount or date query", "wants to break or renew a fixed deposit", "a cheque bounced", "KYC update pending and the account is frozen", "transferred money to a wrong account", "ATM did not give cash but the account was debited", "credit card bill has a charge they never made", "closing an old account", "adding a nominee", "home loan disbursement delayed"],
    "upi": ["payment failed but money was debited", "sent money to the wrong number", "refund from a merchant not received", "splitting a dinner bill with friends", "shopkeeper says the payment did not arrive", "forgot the UPI PIN and needs to reset it", "got a call asking for an OTP", "promised cashback not credited", "paying the monthly rent", "wants to cancel an autopay mandate"],
    "telecom": ["no network since yesterday", "recharge done but the plan is not active", "postpaid bill much higher than usual", "wants to port the number to another operator", "home broadband down and the technician did not come", "SIM card damaged, needs a replacement", "daily data finishing too early", "international roaming pack before a trip", "DTH channels stopped after a recharge"],
    "ecommerce": ["wrong item delivered", "refund status for a returned item", "wants a replacement, not a refund", "delivery delayed again and again", "package marked delivered but not received", "cash on delivery problem", "wants to cancel an order", "product arrived damaged", "size exchange for clothes or shoes", "warranty claim for an appliance", "food order arrived late and cold"],
    "travel": ["flight cancelled, wants to rebook", "train ticket on the waiting list", "cab did not arrive or overcharged", "hotel booking not found at check-in", "bus ticket refund", "baggage lost at the airport", "planning a family trip", "tatkal ticket for an emergency"],
    "health": ["booking or moving a doctor's appointment", "asking for a lab or scan report", "hospital bill and the insurance claim amount", "medicine order or dosage confusion", "insurance premium due, the policy may lapse", "insurance claim rejected", "cashless approval pending at the hospital", "a parent's health checkup", "a child's vaccination"],
    "gov": ["Aadhaar address update", "linking PAN with Aadhaar", "PF withdrawal stuck", "passport appointment", "ration card name correction", "electricity bill too high or a meter problem", "property tax payment", "driving licence renewal", "PM Kisan instalment not received", "birth certificate for a child", "income tax refund pending"],
    "everyday": ["planning a movie or an outing with friends", "arranging a family function", "lending or borrowing money between friends", "grocery shopping for the week", "school fees and admission", "house rent and the landlord", "giving directions to a guest", "wedding preparations", "telling family about travel plans", "booking a gas cylinder"],
}
SETTINGS = ["a phone call to customer care", "a phone call to customer care", "talking to an agent at the counter",
            "a voice note to a friend about it", "telling a family member what happened"]
SETTINGS_EVERYDAY = ["a voice note to a friend", "a phone call to a relative", "talking to a family member at home", "a phone call to a friend"]
REGIONS = ["Uttar Pradesh", "Bihar", "Delhi", "Rajasthan", "Madhya Pradesh", "Punjab", "Haryana", "West Bengal", "Odisha",
           "Jharkhand", "Maharashtra", "Gujarat", "Karnataka", "Tamil Nadu", "Telangana", "Kerala", "Assam"]
AGES = ["about 22", "about 35", "about 50", "about 65"]

SYSTEM = nfc("""You write short monologues in spoken Hindi with natural English code-mixing, as one real Indian speaker says them out loud: on a phone call, in a voice note, or talking to someone. A Hindi TTS will read them and they become speech-translation training data, so every word must be speakable and written exactly as it is spoken.

How it must sound:
- Colloquial spoken Hindi, not written or formal Hindi, in short clauses. Use discourse words (मतलब, यानी, अच्छा, देखिए, यार) sparingly, one or two at most.
- Fillers and repetitions only where the request asks for them. The only fillers are उम्म and हम्म; never अ, अं or ए.
- Make the story fit the given entities: if an amount, date or name looks unusual for the situation, adjust the situation so it is plausible.
- No bookish words such as एवं, तथा, हेतु, अतः, उक्त, किंतु, परंतु, यदि.
- Only this one speaker talks: no other voice, no dialogue labels, no stage directions, no narration.
- Vary the opening; do not start with नमस्ते मेरा नाम.

Writing rules (strict):
- Devanagari only. No digits. No Latin letters except inside the markup below. No symbols at all (no ₹ % / - : . " ' … or brackets other than the markup).
- Numbers are written as the spoken words.
- Every English word or phrase, including everyday loanwords and acronyms, is written in Devanagari as an Indian speaker pronounces it and wrapped as {देवनागरी|latin}, e.g. {कस्टमर केयर|customer care}, {रिफ़ंड|refund}, {फ़ोन|phone}, {बैंक|bank}, {अकाउंट|account}, {टाइम|time}, {ओ टी पी|OTP}. Never leave an English word unwrapped, and never write Latin outside the braces.
- Wrap every entity as [text|type], type one of: name place org product amount number date time phone account id. Copy the entities you are given exactly, character for character, including the {..|..} inside them.
- Punctuation: commas for short pauses; every sentence ends with । or ? or !. Nothing else.
- Spelling: anusvara before a consonant (हिंदी, बंद), chandrabindu for a nasal vowel when no matra rises above the line (हाँ, पाँच), nukta only for z and f sounds (ज़रूरत, फ़ोन).

Output only the monologue, on one line, between <script> and </script>.""")

SHOTS = [(nfc("""Domain: banking. Situation: money debited without the speaker doing any transaction. Setting: a phone call to customer care.
Speaker: a woman, about 50, from Tamil Nadu. Every first-person verb is feminine (बोल रही हूँ, गई थी, करूँगी).
Length: 55 to 95 words.
Code-mixing: medium: 4 to 8 English words or short phrases.
Entities to include, copied exactly:
- [सुनीता राघवन|name] (the speaker's own name)
- [{एच डी एफ सी बैंक|HDFC Bank}|org]
Self-correction: the speaker first says [साढ़े तीन लाख रुपये|amount] by mistake, then corrects it with "नहीं नहीं" to [पौने चार लाख रुपये|amount]. Only the corrected value is true.
Also: End a sentence with the tag 'ना' that is NOT a negation."""),
          nfc("""<script>हाँ जी, मैं [सुनीता राघवन|name] बोल रही हूँ, [{एच डी एफ सी बैंक|HDFC Bank}|org] से। उम्म, मेरे {सेविंग्स अकाउंट|savings account} से कल रात [साढ़े तीन लाख रुपये|amount], नहीं नहीं, [पौने चार लाख रुपये|amount] कट गए हैं। मैंने कोई {ट्रांज़ैक्शन|transaction} किया ही नहीं, मतलब {कार्ड|card} तो मेरे पास ही है, घर पे। आप इसको अभी के अभी {ब्लॉक|block} कर दीजिए ना? और पैसे वापस कब तक आएँगे, वो भी बता दीजिए।</script>""")),
         (nfc("""Domain: everyday. Situation: got a call asking for an OTP. Setting: a voice note to his mother.
Speaker: a man, about 22, from Bihar. Every first-person verb is masculine (बोल रहा हूँ, गया था, करूँगा).
Length: 30 to 50 words.
Code-mixing: light: 1 to 3 English words.
Entities to include, copied exactly:
- [{सिक्स वन|six one}, {डबल फ़ोर|double four}, {नाइन टू|nine two}|id]
Self-correction: none.
Also: Use the negative imperative 'मत'."""),
          nfc("""<script>मम्मी, सुनो, अभी किसी का {फ़ोन|phone} आएगा, बोलेगा {बैंक|bank} से बोल रहे हैं। उसको कुछ मत बताना, हाँ? मेरे {फ़ोन|phone} पे जो कोड आया है, [{सिक्स वन|six one}, {डबल फ़ोर|double four}, {नाइन टू|nine two}|id], वो वो किसी को भी मत देना। मैं शाम को आके देख लूँगा।</script>"""))]

# --- design: quotas per dimension, crossed by independent shuffles ------------------------------------------------


def quota(n, weights, r):
    """n items with counts proportional to weights (largest remainder), shuffled."""
    keys, w = list(weights), list(weights.values())
    exact = [n * x / sum(w) for x in w]
    cnt = [int(e) for e in exact]
    for i in sorted(range(len(w)), key=lambda i: exact[i] - cnt[i], reverse=True)[:n - sum(cnt)]:
        cnt[i] += 1
    out = [k for k, c in zip(keys, cnt) for _ in range(c)]
    r.shuffle(out)
    return out


def design(n, seed):
    r = random.Random(seed)
    doms = quota(n, DOMAIN_W, r)
    forms = [None] * n
    for d in DOMAINS:
        idx = [i for i, x in enumerate(doms) if x == d]
        for i, f in zip(idx, quota(len(idx), dict(zip(FORMS, NUMFORM_W[d])), r)):
            forms[i] = f
    corr = quota(n, CORR_W, r)
    marks = iter(quota(sum(c != "none" for c in corr), dict.fromkeys(MARKERS, 1), r))
    cols = {"intent": quota(n, {k: v[0] for k, v in INTENTS.items()}, r), "mix": quota(n, {k: v[0] for k, v in MIX.items()}, r),
            "length": quota(n, {k: v[0] for k, v in LENGTH.items()}, r), "gender": quota(n, GENDER_W, r)}
    cols.update({k: quota(n, w, r) for k, w in DISFL_W.items()})  # drawn last, so earlier columns keep their values
    return [{"domain": doms[i], "numform": forms[i], "corr": corr[i], "marker": next(marks) if corr[i] != "none" else None,
             **{k: v[i] for k, v in cols.items()}} for i in range(n)]


def dev(marked):
    """Marked text -> what is spoken (Devanagari only, no commas)."""
    return probe.transcript(probe.EN.sub(r"\1", marked))


def ok(e, d):
    """Bulk filters: no bank value W3's TTS kept failing on, no lakh-scale amount on a small purchase."""
    return not (set(e.value.split()) & AVOID or e.type == "amount" and int(e.value) > AMOUNT_CAP.get(d, float("inf")))


def draw(g, r, d, *a):
    """g(r, *a) redrawn until ok(); with the filters off it is one call, so pilot-v1 cells keep their entities."""
    for _ in range(10000):
        if ok(e := g(r, *a), d):
            return e
    raise ValueError(f"{g.__name__}: nothing passes the bulk filters in {d}")


def spec(cell, key):
    """Cell -> everything the prompt needs, with entities made by banks.py. Deterministic per key."""
    r = random.Random(key)
    d = cell["domain"]
    prim = draw(NUMFORMS[cell["numform"]], r, d, d)
    ents = [(prim, None)]
    if cell["length"] != "short":
        others = {f: w for f, w in zip(FORMS, NUMFORM_W[d]) if f != cell["numform"] and w}
        ents.append((draw(NUMFORMS[r.choices(list(others), list(others.values()))[0]], r, d, d), None))
    w = {"name": 3, "org": 3, "place": 1, "product": 2}
    for kind in sorted(w, key=lambda k: -r.random() ** (1 / w[k]))[:N_NAMED[cell["length"]]]:  # weighted, no repeats
        if kind == "name":
            own = r.random() < 0.5
            ents.append((draw(banks.g_name, r, d, cell["gender"] if own else None),
                         "the speaker's own name" if own else "a person the speaker mentions"))
        else:
            ents.append((draw({"org": banks.g_org, "product": banks.g_product}[kind], r, d, d) if kind != "place"
                         else draw(banks.g_place, r, d), None))
    corr = None
    if cell["corr"] == "entity":
        i = 0 if r.random() < 0.7 or len(ents) == 1 else r.randrange(1, len(ents))
        right = ents[i][0]
        for _ in range(50):
            wrong = (draw(NUMFORMS[cell["numform"]], r, d, d) if i == 0 else draw(banks.g_name, r, d) if right.type == "name"
                     else draw(banks.g_place, r, d) if right.type == "place" else draw(banks.g_org, r, d, d) if right.type == "org"
                     else draw(banks.g_product, r, d, d))
            if dev(wrong.text) != dev(right.text) and wrong.type == right.type and near(wrong, right):
                break
        corr = {"wrong": wrong, "right": right, "marker": cell["marker"]}
    elif cell["corr"] == "other":
        corr = {"marker": cell["marker"]}
    return {"cell": cell, "ents": ents, "corr": corr, "scenario": r.choice(SCENARIOS[d]),
            "setting": r.choice(SETTINGS_EVERYDAY if d == "everyday" else SETTINGS), "region": r.choice(REGIONS), "age": r.choice(AGES)}


def near(a, b):
    """A plausible slip: a count keeps its unit (साल stays साल), an amount stays within 3x."""
    if a.type == "number":
        return a.value.split(" ", 1)[1:] == b.value.split(" ", 1)[1:]
    if a.type == "amount":
        return 1 / 3 <= int(a.value) / int(b.value) <= 3
    return True


def wrap(e):
    return f"[{e.text}|{e.type}]"


def user_prompt(s):
    c = s["cell"]
    lo, hi = LENGTH[c["length"]][1]
    sp = ("a woman", "feminine (बोल रही हूँ, गई थी, करूँगी)") if c["gender"] == "f" else ("a man", "masculine (बोल रहा हूँ, गया था, करूँगा)")
    lines = [f"Domain: {c['domain']}. Situation: {s['scenario']}. Setting: {s['setting']}.",
             f"Speaker: {sp[0]}, {s['age']}, from {s['region']}. Every first-person verb is {sp[1]}.",
             f"Length: {lo} to {hi} words.", f"Code-mixing: {MIX[c['mix']][1]}.", "Entities to include, copied exactly:"]
    corr = s["corr"]
    for e, role in s["ents"]:
        if not (corr and "right" in corr and e is corr["right"]):
            lines.append(f"- {wrap(e)}" + (f" ({role})" if role else ""))
    if corr and "right" in corr:
        lines.append(f"Self-correction: the speaker first says {wrap(corr['wrong'])} by mistake, then corrects it with "
                     f"\"{corr['marker']}\" to {wrap(corr['right'])}. Only the corrected value is true.")
    elif corr:
        lines.append(f"Self-correction: the speaker corrects one ordinary word (not an entity), such as a day, a person or an "
                     f"action, using \"{corr['marker']}\", e.g. 'मेरे भाई ने, {corr['marker']}, मेरी बहन ने भेजा'.")
    else:
        lines.append("Self-correction: none.")
    lines.append(f"Also: {INTENTS[c['intent']][1]}")
    d = [x for k, x in (("filler", "the filler उम्म or हम्म once, where the speaker hesitates"),
                        ("repeat", "one word said twice in a row, as people do when they restart (मैं मैं, वो वो, मेरा मेरा)")) if c.get(k) == "yes"]
    lines.append("Disfluency (required): " + " AND ".join(d) + "." if d else "Disfluency: none, no fillers.")
    return nfc("\n".join(lines))


# --- the LLM call (archived) ---------------------------------------------------------------------------------------

_bedrock = None


def converse(req):
    global _bedrock
    if _bedrock is None:
        import s3
        from botocore.config import Config
        _bedrock = s3.session().client("bedrock-runtime", config=Config(read_timeout=300, retries={"mode": "adaptive", "max_attempts": 8}))
    msgs = [{"role": role, "content": [{"text": t}]} for role, t in req["messages"]]
    r = _bedrock.converse(modelId=req["model"], system=[{"text": req["system"]}], messages=msgs, inferenceConfig=req["config"])
    blocks = r["output"]["message"]["content"]
    return {"text": "".join(b.get("text", "") for b in blocks), "usage": r["usage"], "stop": r["stopReason"],
            "ms": r["metrics"]["latencyMs"],
            "reasoning_chars": sum(len(b["reasoningContent"].get("reasoningText", {}).get("text", "")) for b in blocks if "reasoningContent" in b)}, None


def llm(name, user, take, hist=()):
    """hist: a repair turn's [role, text] messages after the user prompt (empty = a fresh take, same request as before)."""
    import archive
    model, pin, pout = MODELS[name]
    msgs = [x for q, a in SHOTS for x in (["user", q], ["assistant", a])] + [["user", user]] + [list(m) for m in hist]
    req = {"model": model, "system": SYSTEM, "messages": msgs, "config": CONFIG, "take": take, "v": PROMPT_V}
    est_in = (len(SYSTEM) + sum(len(t) for _, t in msgs)) / 2  # Devanagari tokenises at ~2 chars/token or worse
    inr = (est_in * pin + EST_OUT * pout) / 1000 * USD_INR  # estimate charged against the cap before the call
    resp, _ = archive.archive_call("bedrock", req, converse, stream=STREAM, inr=inr, cap_inr=CAP_INR)
    u = resp["usage"]
    return resp, archive.request_key(req), (u["inputTokens"] * pin + u["outputTokens"] * pout) / 1000


# --- checks ---------------------------------------------------------------------------------------------------------

TYPED = re.compile(r"\[([^\[\]]+?)(?:\|([a-z]+))?\]")
TYPES = {"name", "place", "org", "product", "amount", "number", "date", "time", "phone", "account", "id"}
BANNED = {"अ", "अं", "ए"}  # W3: single-letter fillers can't be verified after TTS
FILLERS = {"उम्म", "हम्म"}
QW = set(nfc("क्या कब कहाँ कहां कैसे कैसी कैसा कौन कौनसा कौनसी क्यों कितना कितने कितनी किस किसे किसको किसका किसकी किसके किधर").split())
CAUS = re.compile(nfc(r"^(करवा|कराय|कराई|कराए|कराना|करानी|बनवा|भिजवा|दिलवा|लगवा|खुलवा|बदलवा|भरवा|निकलवा|रुकवा|मँगवा|मंगवा|"
                      r"कटवा|लिखवा|छुड़वा|उठवा|पहुँचवा|दिखवा|सुधरवा|रखवा|हटवा|बुलवा|जमा करवा|चलवा|बँधवा|धुलवा|बिकवा|सिलवा)"))
NEG = {"नहीं", "न", "मत"}
DETECT = {"tag_na", "mat", "nahi_to", "cond", "intonation_q", "causative", "kal_parson", "neg_place"}
HUN = {"हूँ", "हूं"}
NOT_GENDER = {"भी", "ही", "जी", "वही", "नहीं", "यहीं", "वहीं"}


def detect(sentences):
    """Intent phenomena visible in the text. refusal/confirmation/prodrop/ne_ko need a reader: not detected."""
    out, toks = set(), []
    for s in sentences:
        t = s[:-1].split()
        toks += t
        if t and t[-1] == "ना":
            out.add("tag_na")
        if s[-1] == "?" and not set(t) & QW and t and t[-1] != "ना":
            out.add("intonation_q")
    if "मत" in toks:
        out.add("mat")
    if any(a == "नहीं" and b == "तो" for a, b in zip(toks, toks[1:])):
        out.add("nahi_to")
    if {"अगर", "वरना"} & set(toks):
        out.add("cond")
    if any(CAUS.match(t) for t in toks):
        out.add("causative")
    if {"कल", nfc("परसों")} & set(toks):
        out.add("kal_parson")
    if sum(t in NEG for t in toks) >= 2:
        out.add("neg_place")
    return out


def wrap_missing(marked, e):
    """The LLM spoke a given entity verbatim but forgot its [..|type] wrapper: wrap the first occurrence that sits
    outside every entity span. Exact text only, so the value stays known."""
    i = marked.find(e.text)
    while i >= 0:
        if marked.count("[", 0, i) == marked.count("]", 0, i):
            return f"{marked[:i]}[{e.text}|{e.type}]{marked[i + len(e.text):]}"
        i = marked.find(e.text, i + 1)
    return marked


def repeats(toks):
    return any(a == b and a not in banks.NUMWORDS for a, b in zip(toks, toks[1:]))


def gender_marks(toks):
    m = f = 0
    for a, b in zip(toks, toks[1:]):
        if b in HUN and a not in NOT_GENDER:
            m += a.endswith(("ा", "आ"))
            f += a.endswith(("ी", "ई"))
    for t in toks:
        m += t.endswith(("ूँगा", "ऊँगा", "ूंगा", "ऊंगा"))
        f += t.endswith(("ूँगी", "ऊँगी", "ूंगी", "ऊंगी"))
    return m, f


def realised_tags(item, spans):
    toks = item["hi_text"][:-1].split()
    etoks = [w for e in item["entities"] for w in e.split()]
    tags = set()
    tags |= {"lakh"} if "लाख" in etoks else set()
    tags |= {"crore"} if nfc("करोड़") in etoks else set()
    tags |= {"frac"} if set(banks.FRAC) & set(etoks) else set()
    tags |= {"num2199"} if banks.N2199 & set(etoks) else set()
    tags |= {"enum"} if set(banks.EN_D) & set(etoks) else set()
    tags |= {"acronym"} if any(a in banks.LETTERS and b in banks.LETTERS for a, b in zip(toks, toks[1:])) else set()
    for sp in spans:
        t = sp["type"]
        if t in ("phone", "account", "id") and sum(w in banks.H[:10] or w in banks.EN_D for w in sp["text"].split()) >= 4:
            tags.add("digits")
        tags |= {{"org": "brand", "product": "brand"}.get(t, t)} if t in ("date", "time", "name", "place", "org", "product") else set()
    tags |= {"filler"} if FILLERS & set(toks) else set()
    tags |= {"en"} if item["en_words"] else set()
    tags |= {"selfcorr"} if item["corrections"] else set()
    return sorted(tags | {f"i:{x}" for x in item["detected"]})


def check(sp, raw, pid):
    """-> (item, fails). item carries W3's probe fields + W2's metadata; fails lists the rules it broke."""
    fails = []
    m = re.search(r"<script>(.*?)</script>", raw, re.S)
    if not m:
        return None, ["format"]
    marked = nfc(" ".join(m.group(1).split()))
    for a, b in RESPELL.items():
        marked = re.sub(f"(?<![\u0900-\u097f]){a}(?![\u0900-\u097f])", b, marked)
    wrapped = []
    for e in [e for e, _ in sp["ents"]] + ([sp["corr"]["wrong"]] if sp["corr"] and "wrong" in sp["corr"] else []):
        if dev(e.text) not in [dev(x) for x, _ in TYPED.findall(marked)]:
            marked2 = wrap_missing(marked, e)
            wrapped += [dev(e.text)] if marked2 != marked else []
            marked = marked2
    typed = TYPED.findall(marked)
    plain = TYPED.sub(r"[\1]", marked)
    try:
        item = probe.parse(f"{pid}\t{sp['cell']['domain']}\t-\t{plain}")
    except (AssertionError, ValueError) as e:
        return {"marked": plain, "error": str(e)[:300]}, ["spec_text"]
    c, corr = sp["cell"], sp["corr"]
    given = {dev(e.text): e for e, _ in sp["ents"]}
    if corr and "wrong" in corr:
        given[dev(corr["wrong"].text)] = corr["wrong"]
    spans = []
    for (inner, typ), d in zip(typed, item["entities"]):
        g = given.get(d)
        spans.append({"text": d, "type": typ or None, "value": g.value if g else None, "kind": g.kind if g else None,
                      "given": bool(g), "retracted": bool(corr and "wrong" in corr and d == dev(corr["wrong"].text))})
    missing = [d for d in given if d not in item["entities"]]
    if missing:
        fails.append("entity_missing")
    if any(s["type"] not in TYPES for s in spans):
        fails.append("entity_type")
    hi = item["hi_text"]
    item["corrections"] = []
    if corr:
        mk = dev(corr["marker"])
        if "wrong" in corr:
            w, r_ = dev(corr["wrong"].text), dev(corr["right"].text)
            a = hi.find(w)
            b = hi.find(mk, a + len(w)) if a >= 0 else -1
            ok = a >= 0 and b >= 0 and hi.find(r_, b) >= 0
            item["corrections"] = [{"wrong": w, "right": r_, "marker": mk, "type": corr["right"].type, "value": corr["right"].value}]
        else:
            ok = f" {mk} " in f" {hi[:-1]} "
            item["corrections"] = [{"marker": mk}]
        if not ok:
            fails.append("correction")
    toks = [t for s in item["sentences"] for t in s[:-1].split()]
    mm, ff = gender_marks(re.sub(r"[।?!]", " ", dev(probe.ENT.sub(" ENT ", plain))).split())  # a name before हूँ is no verb
    if (c["gender"] == "f" and mm) or (c["gender"] == "m" and ff):
        fails.append("gender")
    bare = probe.transcript(probe.ENT.sub(r"\1", probe.EN.sub(" ", plain))).replace("।", " ").replace("?", " ").replace("!", " ")
    if BANNED & set(bare.split()):  # outside English spans: ए is also the letter A ({ए टी एम|ATM})
        fails.append("filler")
    est = item["chars"] * S_PER_CHAR
    if not 9 <= est <= 57 or MAX_CHARS and item["chars"] > MAX_CHARS:
        fails.append("length")
    item["detected"] = sorted(detect(item["sentences"]))
    if c["intent"] in DETECT and c["intent"] not in item["detected"]:
        fails.append("intent")
    if (c.get("filler") == "yes" and not FILLERS & set(toks)) or (c.get("repeat") == "yes" and not repeats(toks)):
        fails.append("disfluency")
    item.update(gender=c["gender"], marked=plain, autowrapped=wrapped, est_sec=round(est, 1), words=len(toks), entity_spans=spans, cell=c,
                gender_marks=[mm, ff], scenario=sp["scenario"], setting=sp["setting"])
    item["tags"] = realised_tags(item, spans)
    return item, fails


LATIN = re.compile(r"[A-Za-z][A-Za-z'’ ]*")
RULE = {"entity_type": "Every entity is [text|type] with type one of: " + " ".join(sorted(TYPES)) + ".",
        "gender": "Every first-person verb must agree with the speaker's gender given in the request.",
        "filler": "Never use the fillers अ, अं or ए; only उम्म or हम्म.",
        "length": "Keep to the requested length.",
        "format": "Output only the monologue, on one line, between <script> and </script>."}


def feedback(sp, item, fails):
    """A repair turn's message: each rule the answer broke, quoting the offending text where code can find it."""
    out = []
    if "spec_text" in fails:
        bare = probe.ENT.sub(r"\1", probe.EN.sub(r"\1", item["marked"]))  # Latin on the Devanagari side counts too
        lat = [m.strip() for m in LATIN.findall(bare)]
        sym = sorted({ch for ch in bare if not ("\u0900" <= ch <= "\u097f" or ch in " ,।?!" or ch.isascii() and ch.isalpha())})
        if lat:
            out.append("Latin letters outside the {देवनागरी|latin} markup: " + "; ".join(f'"{x}"' for x in lat) + ". Write that "
                       "English in Devanagari as an Indian speaker pronounces it, wrapped as {देवनागरी|latin} with the English on the right.")
        if sym:
            out.append("Symbols that are not allowed: " + " ".join(sym) + ". Use only commas and । ? ! as punctuation.")
        if not (lat or sym) or set(sym) & set("[]{}|"):
            out.append("The markup is broken: every { needs its |latin}, every [ its |type], and nothing else uses brackets.")
    if "entity_missing" in fails:
        have = item.get("entities", [])
        ents = [e for e, _ in sp["ents"]] + ([sp["corr"]["wrong"]] if sp["corr"] and "wrong" in sp["corr"] else [])
        out.append("These entities are missing or altered; copy each exactly, character for character, wrapped: "
                   + ", ".join(wrap(e) for e in ents if dev(e.text) not in have) + ".")
    if "correction" in fails:
        c = sp["corr"]
        out.append(f"The self-correction must say {wrap(c['wrong'])}, then \"{c['marker']}\", then {wrap(c['right'])}, in that order."
                   if "wrong" in c else f"The self-correction must use \"{c['marker']}\" between the wrong word and the right one.")
    if "intent" in fails:
        out.append(INTENTS[sp["cell"]["intent"]][1])
    if "disfluency" in fails:
        out.append("Include the required disfluency: " + " and ".join(x for k, x in (("filler", "the filler उम्म or हम्म once"),
                   ("repeat", "one word said twice in a row")) if sp["cell"].get(k) == "yes") + ".")
    out += [RULE[f] for f in fails if f in RULE]
    return nfc("Your monologue breaks these rules:\n- " + "\n- ".join(out) + "\nFix only these, keep everything else, and "
               "output the whole corrected monologue between <script> and </script>.")


def generate(name, cell, pid, seed, takes, repairs=0):
    """One script: `takes` fresh tries, then `repairs` turns that show the model its last answer and what broke.
    -> record with model/take/key/cost and fails ([] = kept)."""
    sp = spec(cell, f"{seed}:{pid}")
    user = user_prompt(sp)
    cost, tries, hist = 0.0, [], []
    for take in range(takes + repairs):
        if take >= takes:
            hist += [["assistant", resp["text"]], ["user", feedback(sp, item, fails)]]
        resp, key, usd = llm(name, user, take, hist)
        cost += usd
        item, fails = check(sp, resp["text"], pid)
        tries.append({"take": take, "repair": take >= takes, "key": key, "fails": fails,
                      "out_tokens": resp["usage"]["outputTokens"], "ms": resp["ms"]})
        if not fails:
            break
    rec = {"pid": pid, **(item or {}), "fails": fails, "model": name, "model_id": MODELS[name][0], "take": take,
           "_gen": {"prompt_v": PROMPT_V, "user": user, "tries": tries, "usd": round(cost, 6), "raw": resp["text"]}}
    rec.setdefault("cell", cell)
    return rec


def run(name, cells, prefix, seed, takes, repairs=0, workers=8):
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(generate, name, c, f"{prefix}{i + 1:05d}", seed, takes, repairs) for i, c in enumerate(cells)]
        out = []
        for f in futs:
            out.append(f.result())
            r = out[-1]
            print(r["pid"], name, r["take"], ",".join(r["fails"]) or "ok", file=sys.stderr, flush=True)
    return out


def write(path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def bulk():
    """Prompt v5 for the bulk run (tasks/S2-scale-synth.md), on top of v4: the pilot's three fixes and W3's handoff.
    Rebinds module settings, so call it before generating; pilot-v1 never sees them."""
    global PROMPT_V, STREAM, CAP_INR, MIX, LENGTH, AVOID, AMOUNT_CAP, MAX_CHARS, RESPELL, RULE
    PROMPT_V, STREAM = "w2-v5", "scale-synth"
    CAP_INR = 15 * USD_INR  # S2 Bedrock share of its $30 AWS cap, as archive estimates (~1.7x the real cost)
    # fix 1: heavy first takes passed 12% because the English clause came out in Latin, outside the markup
    MIX = {**MIX, "heavy": (20, "heavy: 9 or more English words, including at least one whole English phrase or clause of "
                                "your own that fits the situation, written in Devanagari inside one {देवनागरी|latin} span "
                                "like every other English word, never in Latin outside the braces")}
    # fix 2: long scripts came out at ~93 words when 105-140 were asked; W3 lost scripts over ~400 chars past 60 s
    LENGTH = {**LENGTH, "long": (20, (90, 110))}
    MAX_CHARS = 400
    RULE = {**RULE, "length": f"Too long: keep the monologue under {MAX_CHARS} Devanagari letters (about 100 words)."}
    # fix 3: no lakh-scale amount on a small purchase (pilot: 26/500, e.g. ₹19.5 lakh for DTH)
    AMOUNT_CAP = {"telecom": 99_999, "ecommerce": 300_000, "upi": 200_000, "travel": 500_000, "everyday": 1_000_000}
    # W3 pilot takes: >= 2 entity misses and >= 25% of the takes that had them (Medanta 13/17, Lhakpa 12/16 ...)
    AVOID = {"Medanta", "Lhakpa", "Fortis", "Uber", "Thyronorm", "David"}
    RESPELL = {"आइ": "आई", "भई": "भाई"}  # both ASRs rewrote these on 38 and 8 pilot takes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["bake", "pilot", "bulk"])
    ap.add_argument("--n", type=int)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--model")
    ap.add_argument("--takes", type=int, default=5)
    ap.add_argument("--repairs", type=int, default=2)
    ap.add_argument("--seed", default="bulk-v1", help="bulk: design seed")
    a = ap.parse_args()
    if a.mode == "bake":
        cells = design(a.n or 30, "bake-v1")
        for name in a.models.split(","):
            recs = run(name, cells, "b-", "bake-v1", takes=1)
            write(DATA / f"bake/{name}-{PROMPT_V}.jsonl", recs)
            print(name, Counter(f for r in recs for f in r["fails"] or ["ok"]), file=sys.stderr)
    elif a.mode == "bulk":  # never the pilot-v1 cells: own seed, own pids, own folder
        bulk()
        recs = run(a.model, design(a.n, a.seed), "w2b-", a.seed, takes=a.takes, repairs=a.repairs)
        out = DATA.parent / "scale-synth" / a.seed
        write(out / "scripts.jsonl", [r for r in recs if not r["fails"]])
        write(out / "rejects.jsonl", [r for r in recs if r["fails"]])
        print(f"{sum(not r['fails'] for r in recs)} kept, {sum(bool(r['fails']) for r in recs)} rejected; "
              f"${sum(r['_gen']['usd'] for r in recs):.3f} -> {out}", file=sys.stderr)
    else:
        recs = run(a.model, design(a.n or 500, "pilot-v1"), "w2-", "pilot-v1", takes=a.takes, repairs=a.repairs)
        write(DATA / "scripts.jsonl", [r for r in recs if not r["fails"]])
        write(DATA / "rejects.jsonl", [r for r in recs if r["fails"]])
        print(f"{sum(not r['fails'] for r in recs)} kept, {sum(bool(r['fails']) for r in recs)} rejected; "
              f"${sum(r['_gen']['usd'] for r in recs):.3f}", file=sys.stderr)


if __name__ == "__main__":
    cells = design(500, "t")
    assert len(cells) == 500 and Counter(c["gender"] for c in cells) == {"f": 250, "m": 250}
    assert Counter(c["domain"] for c in cells)["banking"] == 80 and sum(c["corr"] != "none" for c in cells) == 300
    s = spec({"domain": "banking", "numform": "frac", "corr": "entity", "marker": "नहीं नहीं", "intent": "tag_na",
              "mix": "medium", "length": "medium", "gender": "f"}, "x")
    shot_item, shot_fails = check({**s, "ents": [(Ent("सुनीता राघवन", "name", "Sunita Raghavan", "name"), None)],
                                   "corr": {"wrong": Ent("साढ़े तीन लाख रुपये", "amount", "350000", "frac"),
                                            "right": Ent("पौने चार लाख रुपये", "amount", "375000", "frac"), "marker": "नहीं नहीं"}},
                                  SHOTS[0][1], "t1")
    assert shot_fails == [], shot_fails  # the few-shot answer passes its own checks
    assert shot_item["corrections"][0]["value"] == "375000" and "i:tag_na" in shot_item["tags"], shot_item["tags"]
    assert [x["retracted"] for x in shot_item["entity_spans"]] == [False, False, True, False]
    assert detect(["आप आएँगे ना?", "पैसा आ जाएगा?", "मैंने भाई से करवाया।"]) >= {"tag_na", "intonation_q", "causative"}
    assert gender_marks("मैं बोल रही हूँ और कल आऊँगी".split()) == (0, 2)
    assert wrap_missing("मैं {क्रोमा|Croma} से [{क्रोमा|Croma} स्टोर|org]", Ent("{क्रोमा|Croma}", "org", "Croma", "org")) == \
        "मैं [{क्रोमा|Croma}|org] से [{क्रोमा|Croma} स्टोर|org]"
    assert wrap_missing("[{क्रोमा|Croma} स्टोर|org]", Ent("{क्रोमा|Croma}", "org", "Croma", "org")) == "[{क्रोमा|Croma} स्टोर|org]"
    assert repeats("मैं मैं गया".split()) and not repeats("नौ नौ".split())
    assert near(Ent("x", "number", "66 years", ""), Ent("y", "number", "21 years", "")) and not near(Ent("x", "number", "66 years", ""), Ent("y", "number", "98 minutes", ""))
    assert near(Ent("x", "amount", "4250", ""), Ent("y", "amount", "1500", "")) and not near(Ent("x", "amount", "4250", ""), Ent("y", "amount", "1350", ""))
    atm = {**s, "ents": [], "corr": None, "cell": {**s["cell"], "intent": "refusal", "filler": "no", "repeat": "no"}}
    assert "filler" not in check(atm, "<script>नहीं, मुझे नया {ए टी एम|ATM} कार्ड नहीं चाहिए, बिल्कुल नहीं।</script>", "t2")[1]
    assert "filler" in check(atm, "<script>ए, मुझे नया कार्ड नहीं चाहिए, बिल्कुल नहीं।</script>", "t3")[1]
    bad, f = check(atm, "<script>मुझे I will get it, but नहीं चाहिए - बिल्कुल नहीं।</script>", "t4")
    fb = feedback(atm, bad, f)
    assert f == ["spec_text"] and '"I will get it"' in fb and '"but"' in fb and ": -." in fb, fb
    bad, f = check(atm, "<script>{डबल|double} हाँ - नहीं।</script>", "t5")
    assert f == ["spec_text"] and "double" not in feedback(atm, bad, f), "Latin inside markup is fine"
    bad, f = check(atm, "<script>मेरा {KYC|KYC} नहीं हुआ।</script>", "t6")
    assert '"KYC"' in feedback(atm, bad, f)
    fem = {**atm, "cell": {**atm["cell"], "gender": "f"}}
    assert "gender" not in check(fem, "<script>मैं [ग्रेस डिसूज़ा|name] हूँ, नहीं।</script>", "t7")[1]
    assert "gender" in check(fem, "<script>मैं आ रहा हूँ, नहीं।</script>", "t8")[1]
    if (DATA / "scripts.jsonl").exists():  # pilot-v1 cells and prompts are frozen: they must replay from the archive
        pilot = [json.loads(l) for l in open(DATA / "scripts.jsonl", encoding="utf-8")]
        cells = {f"w2-{i + 1:05d}": c for i, c in enumerate(design(500, "pilot-v1"))}
        assert all(r["cell"] == cells[r["pid"]] and user_prompt(spec(r["cell"], f"pilot-v1:{r['pid']}")) == r["_gen"]["user"]
                   for r in pilot), "pilot-v1 prompts changed"
    if sys.argv[1:2] == ["bulk"]:
        bulk()
        for i, c in enumerate(design(1600, "bulk-v1")):
            sp = spec(c, f"bulk-v1:w2b-{i + 1:05d}")
            es = [e for e, _ in sp["ents"]] + ([sp["corr"]["wrong"]] if sp["corr"] and "wrong" in sp["corr"] else [])
            assert all(ok(e, c["domain"]) for e in es), (c, es)
        assert "inside one" in user_prompt(spec({**s["cell"], "mix": "heavy"}, "x"))
        it, f = check(atm, "<script>{आइ मीन|I mean} भई, नहीं।</script>", "t9")
        assert it["hi_text"] == "आई मीन भाई नहीं।" and "आइए" in check(atm, "<script>आइए, नहीं।</script>", "t10")[0]["hi_text"]
        assert "length" in check(atm, "<script>" + "बहुत " * 110 + "नहीं।</script>", "t11")[1]
    main()
