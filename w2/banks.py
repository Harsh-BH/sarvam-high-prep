"""W2 entity banks and the Hindi verbaliser. Every entity a script must contain is made here, so its spoken
form and its canonical value are known by construction (the LLM only writes the sentences around it).

Markup (same as w3/probe.tsv): {देवनागरी|latin} = English word(s). gen.py wraps entities as [text|type].
Each generator returns Ent(text, type, value, kind): text = marked spoken form, value = canonical value for
W6/W8c entity checks, kind = the coverage-matrix number form it exercises.

    uv run --no-project python w2/banks.py     # self-check
"""
import random
import re
import unicodedata
from collections import namedtuple

Ent = namedtuple("Ent", "text type value kind")


def nfc(s):
    return unicodedata.normalize("NFC", s)


# --- numbers -------------------------------------------------------------------------------------------------

H = nfc("""शून्य एक दो तीन चार पाँच छह सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस
बीस इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस तीस इकतीस बत्तीस तैंतीस चौंतीस पैंतीस छत्तीस
सैंतीस अड़तीस उनतालीस चालीस इकतालीस बयालीस तैंतालीस चवालीस पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास पचास
इक्यावन बावन तिरपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ साठ इकसठ बासठ तिरसठ चौंसठ पैंसठ छियासठ सड़सठ अड़सठ
उनहत्तर सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर उन्यासी अस्सी इक्यासी बयासी तिरासी
चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी नब्बे इक्यानवे बानवे तिरानवे चौरानवे पचानवे छियानवे सत्तानवे अट्ठानवे
निन्यानवे""").split()
N2199 = set(H[21:])
EN_D = nfc("ज़ीरो वन टू थ्री फ़ोर फ़ाइव सिक्स सेवन एट नाइन").split()
EN_DL = "zero one two three four five six seven eight nine".split()
EN_TEEN = nfc("टेन इलेवन ट्वेल्व थर्टीन फ़ोर्टीन फ़िफ़्टीन सिक्सटीन सेवेंटीन एटीन नाइनटीन").split()
EN_TEENL = "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
EN_TENS = nfc("ट्वेंटी थर्टी फ़ोर्टी फ़िफ़्टी सिक्सटी सेवेंटी एटी नाइंटी").split()
EN_TENSL = "twenty thirty forty fifty sixty seventy eighty ninety".split()
LETTERS = nfc("ए बी सी डी ई एफ जी एच आई जे के एल एम एन ओ पी क्यू आर एस टी यू वी डब्ल्यू एक्स वाई ज़ेड").split()  # SPEC R8
FRAC = nfc("डेढ़ ढाई सवा साढ़े पौने").split()
UNITS = [(10**7, nfc("करोड़")), (10**5, "लाख"), (1000, nfc("हज़ार")), (100, "सौ")]
RUPEES = "रुपये"
MONTHS = list(zip(nfc("जनवरी फ़रवरी मार्च अप्रैल मई जून जुलाई अगस्त सितंबर अक्टूबर नवंबर दिसंबर").split(),
                  "January February March April May June July August September October November December".split()))
DOUBLE, TRIPLE = nfc("{डबल|double}"), nfc("{ट्रिपल|triple}")


def hi(n):
    """Integer -> Hindi words, Indian grouping (करोड़/लाख/हज़ार/सौ)."""
    if n < 100:
        return H[n]
    for v, w in UNITS:
        if n >= v:
            q, r = divmod(n, v)
            return f"{hi(q)} {w}" + (f" {hi(r)}" if r else "")


def hi_sau(n):
    """1100-9999 the colloquial way: बारह सौ पचास (1250)."""
    q, r = divmod(n, 100)
    return f"{H[q]} सौ" + (f" {H[r]}" if r else "")


def en(n):
    """0-99 -> (Devanagari, Latin) English number words."""
    if n < 10:
        return EN_D[n], EN_DL[n]
    if n < 20:
        return EN_TEEN[n - 10], EN_TEENL[n - 10]
    t, o = divmod(n, 10)
    return (EN_TENS[t - 2] + (f" {EN_D[o]}" if o else ""), EN_TENSL[t - 2] + (f" {EN_DL[o]}" if o else ""))


def frac_words(r, unit_word):
    """A fractional spoken quantity: (words, multiplier). डेढ़/ढाई alone; सवा/साढ़े/पौने + a count."""
    f = r.choice(FRAC)
    if f in FRAC[:2]:
        return f"{f} {unit_word}", 1.5 if f == FRAC[0] else 2.5
    if f == FRAC[2]:  # सवा
        n = r.choice([1, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12])
        return (f"{f} {unit_word}" if n == 1 else f"{f} {H[n]} {unit_word}"), n + 0.25
    n = r.randint(3, 20) if f == FRAC[3] else r.randint(2, 20)
    return f"{f} {H[n]} {unit_word}", n + (0.5 if f == FRAC[3] else -0.25)


# --- digit strings --------------------------------------------------------------------------------------------

def say_digits(s, lang, r):
    """'98877' -> spoken digits; runs of 2/3 become डबल/ट्रिपल (70%). lang: hi | en. English groups are one span."""
    out, i = [], 0
    zero = r.choice(["hi", "en"])  # Hindi speakers often say ज़ीरो
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i] and j - i < 3:
            j += 1
        run, d = j - i, int(s[i])
        if run > 1 and r.random() < 0.7:
            out.append(("double" if run == 2 else "triple", d))
            i = j
        else:
            out.append(("one", d))
            i += 1
    if lang == "en":
        dev, lat = [], []
        for k, d in out:
            if k != "one":
                dev.append(nfc("डबल" if k == "double" else "ट्रिपल")), lat.append(k)
            dev.append(EN_D[d]), lat.append(EN_DL[d])
        return "{" + " ".join(dev) + "|" + " ".join(lat) + "}"
    words = []
    for k, d in out:
        if k != "one":
            words.append(DOUBLE if k == "double" else TRIPLE)
        words.append(nfc("{ज़ीरो|zero}") if d == 0 and zero == "en" else H[d])
    return " ".join(words)


def digit_string(r, n, first="0123456789", rep=0.2):
    s = r.choice(first)
    while len(s) < n:
        s += s[-1] if r.random() < rep else r.choice("0123456789")
    return s


def grouped(s, sizes, lang, r):
    parts, i = [], 0
    for g in sizes:
        parts.append(s[i:i + g])
        i += g
    return ", ".join(say_digits(p, lang, r) for p in parts), " ".join(parts)


def letters(s):
    """'ABC' -> {ए बी सी|A B C}."""
    return "{" + " ".join(LETTERS[ord(c) - 65] for c in s) + "|" + " ".join(s) + "}"


def alnum(s, lang, r):
    """Mixed letters/digits ('KX7Y4M') -> letter spans + spoken digits."""
    return " ".join(letters(g) if g.isalpha() else say_digits(g, lang, r) for g in re.findall(r"[A-Z]+|\d+", s))


# --- entity generators: (r, domain, lang) -> Ent ----------------------------------------------------------------

def g_lakh_crore(r, dom, lang="hi"):
    c = r.random()
    if c < 0.4:
        v = r.randint(1, 99) * 10**5
    elif c < 0.65:
        v = r.randint(1, 99) * 10**5 + r.randint(1, 99) * 1000
    elif c < 0.85:
        v = r.randint(1, 25) * 10**7
    else:
        v = r.randint(1, 15) * 10**7 + r.randint(1, 99) * 10**5
    return Ent(hi(v) + (f" {RUPEES}" if r.random() < 0.6 else ""), "amount", str(v), "lakh_crore")


def g_frac(r, dom, lang="hi"):
    c = r.random()
    if c < 0.55:  # amount
        unit, uw = r.choice(UNITS)
        w, m = frac_words(r, uw)
        return Ent(w + (f" {RUPEES}" if r.random() < 0.5 else ""), "amount", str(round(m * unit)), "frac")
    if c < 0.8:
        return g_time(r, dom, lang, frac=True)
    uw, ul = r.choice([("किलो", "kg"), ("घंटे", "hours"), ("साल", "years"), ("लीटर", "litres"), ("महीने", "months"),
                       (nfc("{जी बी|G B}"), "GB")])
    w, m = frac_words(r, uw)
    return Ent(w, "number", f"{m:g} {ul}", "frac")


def g_num2199(r, dom, lang="hi"):
    n, c = r.randint(21, 99), r.random()
    if c < 0.35:
        u = r.choice([1000, 1000, 10**5])
        return Ent(f"{H[n]} {dict(UNITS[1:3])[u] if u == 10**5 else UNITS[2][1]}" + (f" {RUPEES}" if r.random() < 0.5 else ""),
                   "amount", str(n * u), "num2199")
    if c < 0.55:
        return Ent(f"{H[n]} साल", "number", f"{n} years", "num2199")
    if c < 0.8:
        noun, nl = r.choice([("लोग", "people"), ("दिन", "days"), ("प्लेट", "plates"), (nfc("किलोमीटर"), "km"),
                             ("मिनट", "minutes"), (nfc("{पीस|pieces}"), "pieces"), (nfc("{ई एम आई|E M I}"), "EMIs")])
        return Ent(f"{H[n]} {noun}", "number", f"{n} {nl}", "num2199")
    d = r.randint(21, 31)
    return Ent(f"{H[d]} तारीख", "date", f"day {d}", "num2199")


def g_small(r, dom, lang="hi"):
    c = r.random()
    if c < 0.25:
        v = r.randint(2, 9) * 100 + r.choice([0, 0, 50, r.randint(1, 99)])
    elif c < 0.5:
        v = r.randint(11, 99) * 100 + r.choice([0, 50, r.randint(1, 99)])
        if v % 1000 // 100:  # not a round thousand: say it the colloquial way
            return Ent(hi_sau(v) + f" {RUPEES}", "amount", str(v), "small")
    elif c < 0.85:
        v = r.randint(1, 99) * 1000 + r.choice([0, 0, r.randint(1, 9) * 100, r.randint(1, 999)])
    else:  # in English
        n = r.randint(2, 99)
        d, l = en(n)
        return Ent("{" + f"{d} {nfc('थाउज़ेंड')}|{l} thousand" + "}" + f" {RUPEES}", "amount", str(n * 1000), "small")
    return Ent(hi(v) + f" {RUPEES}", "amount", str(v), "small")


def g_date(r, dom, lang="hi"):
    m = r.randrange(12)
    d = r.randint(1, [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m])
    month = "{" + f"{MONTHS[m][0]}|{MONTHS[m][1]}" + "}"
    if r.random() < 0.25:
        y = r.choice([2025, 2026, 2027])
        if r.random() < 0.5:
            return Ent(f"{H[d]} {month} {hi(y)}", "date", f"{d} {MONTHS[m][1]} {y}", "date")
        a, b = en(y % 100)
        return Ent(f"{H[d]} {month} " + "{" + f"{nfc('ट्वेंटी')} {a}|twenty {b}" + "}", "date", f"{d} {MONTHS[m][1]} {y}", "date")
    return Ent(f"{H[d]} {month}", "date", f"{d} {MONTHS[m][1]}", "date")


DAYPART = [("सुबह", range(5, 12)), ("दोपहर", range(12, 16)), ("शाम", range(4, 8)), ("रात", range(8, 12))]


def g_time(r, dom, lang="hi", frac=False):
    part, hours = r.choice(DAYPART)
    h = r.choice(list(hours))
    h12 = h if h <= 12 else h - 12
    pre = f"{nfc(part)} " if r.random() < 0.6 else ""
    if frac:
        f = r.choice(FRAC)
        if f == FRAC[0] or f == FRAC[1]:
            h12, mm, words = (1, 30, f) if f == FRAC[0] else (2, 30, f)
            pre = "दोपहर " if r.random() < 0.5 else ""
        elif f == FRAC[2]:
            mm, words = 15, f"{f} {H[h12]}"
        elif f == FRAC[3]:
            h12 = max(h12, 3)
            mm, words = 30, f"{f} {H[h12]}"
        else:
            h12 = max(h12, 2)
            mm, words = 45, f"{f} {H[h12]}"
            h12 -= 1
        text = f"{pre}{words} बजे"
    else:
        c = r.random()
        mm = 0 if c < 0.4 else r.choice(range(5, 60, 5)) if c < 0.85 else r.randint(1, 59)
        if c >= 0.85 or (lang == "en" and mm):  # English clock time: {टेन थर्टी|ten thirty}
            mm = mm or 30
            a, b = en(h12)
            m1, m2 = en(mm)
            text = pre + "{" + f"{a} {m1}|{b} {m2}" + "}"
        else:
            text = pre + (f"{H[h12]} बजे" if mm == 0 else f"{H[h12]} बजकर {H[mm]} मिनट")
    hh = h12 + (12 if pre and not pre.startswith("सुबह") and h12 != 12 else 0)
    return Ent(nfc(text), "time", f"{hh}:{mm:02d}" if pre else f"{h12}:{mm:02d}", "frac" if frac else "time")


def g_digits(r, dom, lang="hi"):
    c = r.random()
    if c < 0.5:
        s = digit_string(r, 10, "6789")
        sizes = r.choice([(5, 5), (3, 3, 4), (2, 2, 2, 2, 2), (4, 3, 3)])
        typ = "phone"
    elif c < 0.8:
        s = digit_string(r, r.randint(11, 14))
        sizes = r.choice([(4,) * (len(s) // 4) + ((len(s) % 4,) if len(s) % 4 else ()), (3,) * (len(s) // 3) + ((len(s) % 3,) if len(s) % 3 else ())])
        typ = "account"
    else:
        s = digit_string(r, r.choice([8, 10, 12]))
        sizes = (4,) * (len(s) // 4) if len(s) % 4 == 0 else (5, 5)
        typ = "id"
    text, val = grouped(s, sizes, lang, r)
    return Ent(text, typ, val, "endigits" if lang == "en" else "digits")


def g_endigits(r, dom, lang="en"):
    return g_digits(r, dom, "en") if r.random() < 0.6 else g_idcode(r, dom, "en")


IFSC = {"एस बी आई": "SBIN", "एच डी एफ सी": "HDFC", "आई सी आई सी आई": "ICIC", "एक्सिस": "UTIB", "पंजाब नेशनल": "PUNB",
        "बड़ौदा": "BARB", "केनरा": "CNRB", "कोटक": "KKBK", "यस": "YESB", "यूनियन": "UBIN", "इंडियन बैंक": "IDIB", "फ़ेडरल": "FDRL"}
ID_MENU = {"banking": ["ifsc", "pan", "otp", "card4"], "upi": ["otp", "upiref"], "telecom": ["otp", "pin", "sim"],
           "ecommerce": ["otp", "pin", "order"], "travel": ["flightpnr", "trainpnr"], "health": ["patient", "policy", "otp"],
           "gov": ["aadhaar", "pan", "pin"], "everyday": ["otp", "pin"]}


def g_idcode(r, dom, lang="hi"):
    k = r.choice(ID_MENU[dom])
    L = lambda n: "".join(r.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(n))
    if k == "aadhaar":
        text, val = grouped(digit_string(r, 12, "23456789"), (4, 4, 4), lang, r)
    elif k == "pan":
        s = L(3) + "P" + L(1) + digit_string(r, 4) + L(1)
        text, val = f"{letters(s[:5])} {say_digits(s[5:9], lang, r)} {letters(s[9])}", s
    elif k == "ifsc":
        code = r.choice(list(IFSC.values()))
        s = code + "0" + digit_string(r, 6)
        text, val = f"{letters(code)} {say_digits(s[4:], lang, r)}", s
    elif k == "otp":
        n = r.choice([4, 6, 6])
        text, val = grouped(digit_string(r, n), (n,) if n == 4 else r.choice([(3, 3), (2, 2, 2), (6,)]), lang, r)
    elif k == "pin":
        text, val = grouped(digit_string(r, 6, "12345678"), r.choice([(3, 3), (6,)]), lang, r)
    elif k == "card4":
        text, val = grouped(digit_string(r, 4), (4,), lang, r)
    elif k in ("upiref", "order", "sim", "policy", "trainpnr"):
        n = {"upiref": 12, "order": 10, "sim": 10, "policy": 9, "trainpnr": 10}[k]
        text, val = grouped(digit_string(r, n), {12: (4, 4, 4), 10: (3, 3, 4), 9: (3, 3, 3)}[n], lang, r)
    elif k == "flightpnr":
        s = "".join(r.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") if r.random() < 0.7 else r.choice("23456789") for _ in range(6))
        text, val = alnum(s, lang, r), s
    else:  # patient id: letters + digits
        s = L(r.choice([2, 3])) + digit_string(r, 4)
        text, val = alnum(s, lang, r), s
    return Ent(text, "id", val, "endigits" if lang == "en" else "idcode")


NUMFORMS = {"lakh_crore": g_lakh_crore, "frac": g_frac, "num2199": g_num2199, "digits": g_digits, "idcode": g_idcode,
            "endigits": g_endigits, "date": g_date, "time": g_time, "small": g_small}


# --- named entities: (Devanagari, Latin) ------------------------------------------------------------------------

def pairs(s):
    return [tuple(nfc(x).split("|")) for x in s.split(";") if x.strip()]


NAMES = {  # region -> male first, female first, surnames
    "north": (pairs("राजेश|Rajesh;अमित|Amit;विकास|Vikas;सुनील|Sunil;मनोज|Manoj;अंकित|Ankit;रोहित|Rohit;दीपक|Deepak;संजय|Sanjay;आशीष|Ashish;गौरव|Gaurav;रामकिशन|Ramkishan"),
              pairs("पूजा|Pooja;नेहा|Neha;सुनीता|Sunita;रेखा|Rekha;प्रियंका|Priyanka;अंजलि|Anjali;कविता|Kavita;ममता|Mamta;सीमा|Seema;आरती|Aarti;स्वाति|Swati;रश्मि|Rashmi"),
              pairs("शर्मा|Sharma;वर्मा|Verma;यादव|Yadav;गुप्ता|Gupta;मिश्रा|Mishra;तिवारी|Tiwari;चौहान|Chauhan;पांडेय|Pandey;श्रीवास्तव|Srivastava;अग्रवाल|Agarwal;मीणा|Meena;कुशवाहा|Kushwaha;राठौर|Rathore")),
    "punjab": (pairs("गुरप्रीत|Gurpreet;हरप्रीत|Harpreet;जसविंदर|Jaswinder;मनप्रीत|Manpreet;कुलदीप|Kuldeep;बलजीत|Baljeet"),
               pairs("सिमरन|Simran;हरलीन|Harleen;जसलीन|Jasleen;नवनीत|Navneet;गुरलीन|Gurleen"),
               pairs("सिंह|Singh;ढिल्लों|Dhillon;गिल|Gill;संधू|Sandhu;बाजवा|Bajwa;ग्रेवाल|Grewal")),
    "east": (pairs("सौरव|Sourav;अनिर्बान|Anirban;देबाशीष|Debashish;सुब्रत|Subrata;अरिजीत|Arijit;बिस्वजीत|Biswajit;प्रणब|Pranab"),
             pairs("शर्मिष्ठा|Sharmistha;मौसमी|Mousumi;पौलमी|Poulami;इंद्राणी|Indrani;सुतापा|Sutapa;रिया|Riya;अर्पिता|Arpita"),
             pairs("चटर्जी|Chatterjee;बनर्जी|Banerjee;मुखर्जी|Mukherjee;दास|Das;घोष|Ghosh;बोस|Bose;मोहंती|Mohanty;पटनायक|Patnaik;बरुआ|Barua;गोगोई|Gogoi;बोरा|Bora")),
    "west": (pairs("प्रसाद|Prasad;अमोल|Amol;सचिन|Sachin;निखिल|Nikhil;हार्दिक|Hardik;जिग्नेश|Jignesh;केतन|Ketan"),
             pairs("स्नेहा|Sneha;मृणाल|Mrinal;अश्विनी|Ashwini;प्राजक्ता|Prajakta;हेतल|Hetal;धारा|Dhara;फाल्गुनी|Falguni"),
             pairs("पाटिल|Patil;देशमुख|Deshmukh;कुलकर्णी|Kulkarni;जोशी|Joshi;गायकवाड़|Gaikwad;शिंदे|Shinde;पटेल|Patel;शाह|Shah;मेहता|Mehta;देसाई|Desai;परमार|Parmar")),
    "south": (pairs("कार्तिक|Karthik;सुरेश|Suresh;वेंकटेश|Venkatesh;श्रीनिवास|Srinivas;अरविंद|Arvind;प्रवीण|Praveen;बालाजी|Balaji;मुरली|Murali"),
              pairs("मीनाक्षी|Meenakshi;लक्ष्मी|Lakshmi;दिव्या|Divya;कीर्तना|Keerthana;अनुषा|Anusha;सौम्या|Soumya;पद्मा|Padma"),
              pairs("अय्यर|Iyer;नायर|Nair;रेड्डी|Reddy;राव|Rao;पिल्लई|Pillai;मेनन|Menon;नायडू|Naidu;गौड़ा|Gowda;हेगड़े|Hegde;शेट्टी|Shetty;कृष्णन|Krishnan")),
    "muslim": (pairs("इरफ़ान|Irfan;आसिफ़|Asif;इमरान|Imran;सलमान|Salman;फ़ैज़ान|Faizan;शाहिद|Shahid;आरिफ़|Arif"),
               pairs("सना|Sana;फ़ातिमा|Fatima;आयशा|Ayesha;नाज़िया|Nazia;शबनम|Shabnam;ज़ैनब|Zainab;रुबीना|Rubina"),
               pairs("अंसारी|Ansari;ख़ान|Khan;क़ुरैशी|Qureshi;सिद्दीक़ी|Siddiqui;शेख़|Shaikh;मंसूरी|Mansuri;हुसैन|Hussain")),
    "other": (pairs("जोसेफ़|Joseph;थॉमस|Thomas;डेविड|David;ल्हाकपा|Lhakpa;ज़ुबिन|Zubin;रॉबिन|Robin"),
              pairs("मैरी|Mary;ग्रेस|Grace;एंजेला|Angela;डॉली|Dolly;रोज़मेरी|Rosemary"),
              pairs("डिसूज़ा|D'Souza;फ़र्नांडिस|Fernandes;मैथ्यू|Mathew;लेप्चा|Lepcha;संगमा|Sangma;लिंगदोह|Lyngdoh;ईरानी|Irani")),
}
PLACES = pairs("""दिल्ली|Delhi;मुंबई|Mumbai;कोलकाता|Kolkata;चेन्नई|Chennai;बेंगलुरु|Bengaluru;हैदराबाद|Hyderabad;पुणे|Pune;
अहमदाबाद|Ahmedabad;जयपुर|Jaipur;लखनऊ|Lucknow;कानपुर|Kanpur;पटना|Patna;भोपाल|Bhopal;इंदौर|Indore;नागपुर|Nagpur;सूरत|Surat;
वडोदरा|Vadodara;लुधियाना|Ludhiana;अमृतसर|Amritsar;चंडीगढ़|Chandigarh;देहरादून|Dehradun;शिमला|Shimla;जम्मू|Jammu;गुवाहाटी|Guwahati;
शिलांग|Shillong;भुवनेश्वर|Bhubaneswar;कटक|Cuttack;रांची|Ranchi;जमशेदपुर|Jamshedpur;रायपुर|Raipur;विशाखापट्टनम|Visakhapatnam;
विजयवाड़ा|Vijayawada;कोच्चि|Kochi;तिरुवनंतपुरम|Thiruvananthapuram;कोयंबटूर|Coimbatore;मदुरै|Madurai;मैसूरु|Mysuru;मंगलुरु|Mangaluru;
हुबली|Hubballi;नासिक|Nashik;औरंगाबाद|Aurangabad;वाराणसी|Varanasi;प्रयागराज|Prayagraj;आगरा|Agra;मेरठ|Meerut;गोरखपुर|Gorakhpur;
गया|Gaya;मुज़फ़्फ़रपुर|Muzaffarpur;दरभंगा|Darbhanga;सीतापुर|Sitapur;बरेली|Bareilly;अजमेर|Ajmer;उदयपुर|Udaipur;जोधपुर|Jodhpur;
बीकानेर|Bikaner;कोटा|Kota;ग्वालियर|Gwalior;जबलपुर|Jabalpur;बिलासपुर|Bilaspur;सिलीगुड़ी|Siliguri;आसनसोल|Asansol;धनबाद|Dhanbad;
हल्द्वानी|Haldwani;पानीपत|Panipat;रोहतक|Rohtak;गुरुग्राम|Gurugram;नोएडा|Noida;गाज़ियाबाद|Ghaziabad;फ़रीदाबाद|Faridabad;
लक्ष्मी नगर|Laxmi Nagar;करोल बाग|Karol Bagh;अंधेरी|Andheri;बोरीवली|Borivali;ठाणे|Thane;कोरमंगला|Koramangala;जयनगर|Jayanagar;
गोमती नगर|Gomti Nagar;हज़रतगंज|Hazratganj;कंकड़बाग|Kankarbagh;अरेरा कॉलोनी|Arera Colony;विजय नगर|Vijay Nagar;सराफ़ा|Sarafa;
चांदनी चौक|Chandni Chowk;बंजारा हिल्स|Banjara Hills;सॉल्ट लेक|Salt Lake;मालवीय नगर|Malviya Nagar""")


def en_list(s):
    """'देव|Lat;...' -> ['{देव|Lat}', ...]: org and product names are English spans for the TTS Latin variant."""
    return ["{" + f"{d}|{l}" + "}" for d, l in pairs(s)]


ORGS = {
    "banking": en_list("एस बी आई|S B I;एच डी एफ सी बैंक|HDFC Bank;आई सी आई सी आई बैंक|ICICI Bank;एक्सिस बैंक|Axis Bank;पंजाब नेशनल बैंक|Punjab National Bank;बैंक ऑफ़ बड़ौदा|Bank of Baroda;केनरा बैंक|Canara Bank;कोटक बैंक|Kotak Bank;यस बैंक|Yes Bank;यूनियन बैंक|Union Bank;इंडियन बैंक|Indian Bank;फ़ेडरल बैंक|Federal Bank;बजाज फ़ाइनेंस|Bajaj Finance;मुथूट फ़ाइनेंस|Muthoot Finance"),
    "upi": en_list("गूगल पे|Google Pay;फ़ोनपे|PhonePe;पेटीएम|Paytm;भीम|BHIM;क्रेड|CRED;अमेज़न पे|Amazon Pay;एस बी आई|S B I;एच डी एफ सी बैंक|HDFC Bank;एन पी सी आई|N P C I"),
    "telecom": en_list("जियो|Jio;एयरटेल|Airtel;वोडाफ़ोन आइडिया|Vodafone Idea;बी एस एन एल|B S N L;एक्ट फ़ाइबरनेट|ACT Fibernet;टाटा प्ले|Tata Play;हैथवे|Hathway"),
    "ecommerce": en_list("अमेज़न|Amazon;फ़्लिपकार्ट|Flipkart;मिंत्रा|Myntra;मीशो|Meesho;नायका|Nykaa;स्विगी|Swiggy;ज़ोमैटो|Zomato;ब्लिंकिट|Blinkit;बिगबास्केट|BigBasket;अजियो|AJIO;क्रोमा|Croma;रिलायंस डिजिटल|Reliance Digital"),
    "travel": en_list("इंडिगो|IndiGo;एयर इंडिया|Air India;स्पाइसजेट|SpiceJet;आकासा एयर|Akasa Air;आई आर सी टी सी|I R C T C;मेकमायट्रिप|MakeMyTrip;ओला|Ola;उबर|Uber;रैपिडो|Rapido;रेडबस|redBus;ओयो|OYO;इंडियन रेलवे|Indian Railways"),
    "health": en_list("अपोलो हॉस्पिटल|Apollo Hospital;फ़ोर्टिस|Fortis;मैक्स हॉस्पिटल|Max Hospital;एम्स|AIIMS;मेदांता|Medanta;स्टार हेल्थ|Star Health;एल आई सी|L I C;निवा बूपा|Niva Bupa;केयर हेल्थ|Care Health;प्रैक्टो|Practo;फ़ार्मईज़ी|PharmEasy;टाटा वन एम जी|Tata 1mg"),
    "gov": en_list("यू आई डी ए आई|U I D A I;ई पी एफ ओ|E P F O;पासपोर्ट सेवा केंद्र|Passport Seva Kendra;आर टी ओ|R T O;नगर निगम|Nagar Nigam;बिजली विभाग|electricity department;इनकम टैक्स डिपार्टमेंट|Income Tax Department;तहसील ऑफ़िस|tehsil office;पोस्ट ऑफ़िस|post office"),
    "everyday": en_list("डी मार्ट|DMart;हल्दीराम|Haldiram's;पी वी आर|P V R;डोमिनोज़|Domino's;रिलायंस फ़्रेश|Reliance Fresh;बिग बाज़ार|Big Bazaar;कैफ़े कॉफ़ी डे|Cafe Coffee Day;केंद्रीय विद्यालय|Kendriya Vidyalaya"),
}
PRODUCTS = {
    "banking": en_list("सिंपलीक्लिक क्रेडिट कार्ड|SimplyCLICK credit card;रुपे डेबिट कार्ड|RuPay debit card;होम लोन|home loan;पर्सनल लोन|personal loan;गोल्ड लोन|gold loan;फ़िक्स्ड डिपॉज़िट|fixed deposit;सेविंग्स अकाउंट|savings account;रेकरिंग डिपॉज़िट|recurring deposit"),
    "upi": en_list("यू पी आई लाइट|UPI Lite;रुपे क्रेडिट कार्ड|RuPay credit card;ऑटोपे|AutoPay;यू पी आई आई डी|UPI ID;वॉलेट|wallet"),
    "telecom": en_list("जियो फ़ाइबर|JioFiber;एयरटेल एक्सस्ट्रीम|Airtel Xstream;पोस्टपेड प्लान|postpaid plan;प्रीपेड प्लान|prepaid plan;ई सिम|eSIM;फ़ाइव जी|5G;डी टी एच कनेक्शन|DTH connection"),
    "ecommerce": en_list("आईफ़ोन फ़िफ़्टीन|iPhone fifteen;रेडमी नोट थर्टीन|Redmi Note thirteen;वनप्लस नॉर्ड|OnePlus Nord;बोट ईयरबड्स|boAt earbuds;प्रेस्टीज कुकर|Prestige cooker;लेनोवो लैपटॉप|Lenovo laptop;सैमसंग फ़्रिज|Samsung fridge;नाइकी के जूते|Nike shoes;वर्लपूल वॉशिंग मशीन|Whirlpool washing machine;मिक्सर ग्राइंडर|mixer grinder"),
    "travel": en_list("वंदे भारत एक्सप्रेस|Vande Bharat Express;राजधानी एक्सप्रेस|Rajdhani Express;शताब्दी|Shatabdi;विंडो सीट|window seat;स्लीपर क्लास|sleeper class;थर्ड एसी|third AC;डीलक्स रूम|deluxe room"),
    "health": en_list("मेडिक्लेम पॉलिसी|mediclaim policy;टर्म प्लान|term plan;डोलो सिक्स फ़िफ़्टी|Dolo six fifty;मेटफ़ॉर्मिन|metformin;थायरोनॉर्म|Thyronorm;आयुष्मान कार्ड|Ayushman card;हेल्थ चेकअप पैकेज|health checkup package;एम आर आई|MRI;सी टी स्कैन|CT scan"),
    "gov": en_list("पी एम किसान|P M Kisan;आयुष्मान भारत|Ayushman Bharat;राशन कार्ड|ration card;ड्राइविंग लाइसेंस|driving licence;पासपोर्ट|passport;वोटर आई डी|voter ID;पैन कार्ड|PAN card;सुकन्या समृद्धि|Sukanya Samriddhi;अटल पेंशन योजना|Atal Pension Yojana"),
    "everyday": en_list("नेटफ़्लिक्स|Netflix;हॉटस्टार|Hotstar;एक्टिवा|Activa;मारुति स्विफ़्ट|Maruti Swift;स्प्लेंडर|Splendor;इंडेन गैस|Indane gas;प्रेशर कुकर|pressure cooker;स्मार्ट टीवी|smart TV"),
}


def g_name(r, gender=None):
    reg = r.choice(sorted(NAMES))
    m, f, last = NAMES[reg]
    g = gender or r.choice("mf")
    first = r.choice(m if g == "m" else f)
    if r.random() < 0.25:
        return Ent(first[0], "name", first[1], "name")
    sur = r.choice(last)
    if reg == "punjab" and g == "f" and r.random() < 0.5:
        sur = (nfc("कौर"), "Kaur")
    return Ent(f"{first[0]} {sur[0]}", "name", f"{first[1]} {sur[1]}", "name")


def g_place(r):
    d, l = r.choice(PLACES)
    return Ent(d, "place", l, "place")


def g_org(r, dom):
    t = r.choice(ORGS[dom])
    return Ent(t, "org", t[t.index("|") + 1:-1], "org")


def g_product(r, dom):
    t = r.choice(PRODUCTS[dom])
    return Ent(t, "product", t[t.index("|") + 1:-1], "product")


DOMAINS = ["banking", "upi", "telecom", "ecommerce", "travel", "health", "gov", "everyday"]


if __name__ == "__main__":
    assert len(H) == 100 and len(LETTERS) == 26 and H[67] == nfc("सड़सठ") and H[99] == "निन्यानवे"
    assert hi(375000) == nfc("तीन लाख पचहत्तर हज़ार") and hi(12500000) == nfc("एक करोड़ पच्चीस लाख"), hi(12500000)
    assert hi(1250) == nfc("एक हज़ार दो सौ पचास") and hi_sau(1250) == "बारह सौ पचास"
    assert en(45) == (nfc("फ़ोर्टी फ़ाइव"), "forty five")
    r = random.Random(1)
    assert say_digits("9877", "hi", random.Random(3)).count("डबल") <= 1
    assert say_digits("98877", "en", r) == "{नाइन डबल एट सेवन सेवन|nine double eight seven seven}"
    assert letters("SBI") == "{एस बी आई|S B I}"
    for k, g in NUMFORMS.items():  # every generator, every domain: no digits/Latin outside markup, value set
        for dom in DOMAINS:
            for i in range(200):
                e = g(random.Random(f"{k}{dom}{i}"), dom)
                bare = re.sub(r"\{[^{}|]+\|[^{}]+\}", "", e.text)
                assert e.value and not re.search(r"[0-9A-Za-z]", bare), (k, e)
                assert e.text == nfc(e.text), e
    t = g_time(random.Random(5), "travel", frac=True)
    assert t.kind == "frac" and t.type == "time", t
    print("banks ok:", {k: g(random.Random(7), "banking") for k, g in NUMFORMS.items()})
