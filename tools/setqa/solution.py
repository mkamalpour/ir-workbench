"""Set QA System — Optimised inventory tracker. NUS CS3244 Kaggle."""

import re, csv, os, argparse
from collections import defaultdict

# ── Verb Lists ────────────────────────────────────────────────────────────────

PICKUP_VERBS = [
    "picked up", "picks up", "took hold of", "took possession of", "got hold of",
    "came across", "took out", "pulled out", "took from", "collected from",
    "retrieved from", "took home", "brought home", "picked off", "picked out",
    "journeyed with", "travelled with", "moved with", "carried with", "kept with",
    "brought along", "took away", "brought back",
    "grabbed", "snagged", "snatched", "lifted", "obtained", "fetched", "retrieved",
    "collected", "acquired", "received", "accepted", "found", "discovered", "picked",
    "pulled", "extracted", "carried", "carrying", "took", "got", "held", "had",
    "possessed", "kept",
]
DROP_VERBS = [
    "put down", "sets down", "set down", "left behind", "let go of", "set aside",
    "put aside", "pushed away", "moved away from", "dropped off", "laid down",
    "placed down", "put back", "walked away from", "stepped away from",
    "stepped back from", "moved from", "went away from", "turned away from",
    "took away from", "removed from", "left it", "left there", "put away",
    "took off", "pulled away", "sent away", "placed on", "set on", "put on",
    "gave back", "gave away", "gave up",
    "dropped", "discarded", "threw", "thrown", "placed", "abandoned", "released",
    "surrendered", "dismissed", "lost", "forgot", "left",
]
TRANSFER_VERBS = [
    "gave", "handed", "passed", "tossed", "returned", "brought", "sent",
    "offered", "showed", "fed", "delivered", "lent", "loaned", "entrusted",
]

LOCATION_WORDS = {
    "garden", "hallway", "bedroom", "bathroom", "office", "kitchen", "park",
    "school", "house", "room", "door", "window", "floor", "yard", "corridor",
    "cellar", "attic", "basement", "pantry", "cupboard", "closet", "shelf",
    "table", "chair", "sofa",
}
NOT_A_CHARACTER = LOCATION_WORDS | {
    "the", "a", "an", "what", "where", "there", "then", "here", "to", "from",
    "into", "onto", "inside", "outside", "back", "is", "was", "has", "have",
    "had", "it", "its", "this", "that", "these", "those", "in", "on", "at",
    "by", "yes", "no", "not", "with", "for", "and", "or", "but",
    "went", "moved", "journeyed", "travelled", "walked",
}
SKIP_WORDS = {
    "the", "a", "an", "to", "for", "with", "from", "on", "in", "at", "by",
    "into", "onto", "up", "it", "its", "this", "that", "there", "here",
    "some", "his", "her", "their", "my", "your", "our",
}
ANAPHORS = {"it", "them", "those", "these", "there"}

# ── Compiled Patterns ─────────────────────────────────────────────────────────

def _re(verbs): return re.compile(
    r'\b(' + '|'.join(re.escape(v) for v in sorted(verbs, key=len, reverse=True)) + r')\b', re.I)

_pickup_re  = _re(PICKUP_VERBS)
_drop_re    = _re(DROP_VERBS)
_tv         = '|'.join(re.escape(v) for v in sorted(TRANSFER_VERBS, key=len, reverse=True))
_transfer_B = re.compile(rf'([A-Z][a-z]+)\s+(?:{_tv})\s+(?:the\s+|a\s+|an\s+)?([a-z]+(?:\s+[a-z]+)?)\s+(?:to|for)\s+([A-Z][a-z]+)', re.I)
_transfer_A = re.compile(rf'([A-Z][a-z]+)\s+(?:{_tv})\s+([A-Z][a-z]+)\s+(?:the\s+|a\s+|an\s+)?([a-z]+(?:\s+[a-z]+)?)', re.I)
_negation_re = re.compile(r"\b(not|never|no longer|didn'?t|doesn'?t|don'?t|wasn'?t|weren'?t|won'?t|can'?t|cannot|couldn'?t)\b", re.I)

# ── Helpers ───────────────────────────────────────────────────────────────────

def has_negation(s, pos):    return bool(_negation_re.search(s[:pos]))
def clean_item(i):           return re.sub(r'[^\w\s]', '', i).strip().lower()

def get_offset(item, text, cache):
    k = item.lower()
    if k not in cache:
        pos = text.lower().find(k)
        if pos == -1:
            m = re.search(r'\b' + re.escape(k) + r'\b', text.lower())
            pos = m.start() if m else -1
        cache[k] = pos
    return cache[k]

def extract_character(sentence):
    s = re.sub(r'^(in|at|by|from|to|into|onto|inside|outside)\s+\w+[\s,]+', '', sentence, flags=re.I)
    for w in s.split():
        b = w.strip(".,!?;:'\"()")
        if b and b[0].isupper() and b.lower() not in NOT_A_CHARACTER and not b.isupper() and len(b) > 1:
            return b
    return None

def extract_object(m, sentence):
    after = re.sub(r'^[\s,.:;!?()\-]+', '', sentence[m.end():])
    while True:
        skip = re.match(r'^(' + '|'.join(SKIP_WORDS) + r')\s+', after, re.I)
        if skip: after = after[skip.end():]
        else: break
    nm = re.match(r'^([a-zA-Z]+(?:\s+[a-zA-Z]+){0,2})', after)
    if nm:
        words = [w for w in nm.group(1).lower().split()]
        while words and words[-1] in SKIP_WORDS | NOT_A_CHARACTER: words.pop()
        return ' '.join(words) if words else None
    return None

def resolve(obj, key, last_g, last_c, events):
    if obj in ANAPHORS:
        resolved = last_c.get(key) or last_g
        if not resolved:
            resolved = next((e["item"] for e in reversed(events) if e.get("item")), None)
        return resolved or obj
    return obj

# ── Core Processor ────────────────────────────────────────────────────────────

def process_story(story):
    inv    = defaultdict(dict)   # char -> {item: offset}
    events, results = [], []
    q_count, last_g = 0, None
    last_c  = defaultdict(lambda: None)
    text_full, cache = story["full_text"], {}

    def snap():     return {k: sorted(v.keys()) for k, v in inv.items()}
    def off(item):  return get_offset(item, text_full, cache)
    def add(key, item):
        for c in list(inv):
            if c != key and item in inv[c]: del inv[c][item]
        inv[key][item] = max(off(item), -1)
    def log(lid, txt, etype, char, item, **kw):
        events.append({"line_id": lid, "sentence": txt, "event_type": etype,
                        "character": char, "item": item, "inventory_snapshot": snap(), **kw})

    for sent in story["sentences"]:
        t, lid = sent["text"], sent["line_id"]

        # ── Question
        if sent["is_question"]:
            q_count += 1
            m = re.search(r'What (?:is|does) (\w+)', t, re.I)
            if not m: continue
            char, key = m.group(1), m.group(1).lower()
            held = sorted(inv.get(key, {}).keys())
            pred = " ".join(map(str, sorted(inv.get(key, {}).values()))) if held else "-1"

            exp_raw = sent["answer"]
            if exp_raw is None:        exp_offs = None
            elif exp_raw == "-1":      exp_offs = "-1"
            else:
                offs = sorted(o for i in exp_raw.split(",") if i.strip()
                              for o in [off(i.strip())] if o >= 0)
                exp_offs = " ".join(map(str, offs)) if offs else "-1"

            log(lid, t, "question", char, None)
            results.append({
                "story_id": story["story_id"], "question_id": q_count,
                "question_text": t, "character": char, "inventory_at_q": held,
                "predicted_answer": pred, "expected_answer": exp_raw,
                "expected_offsets": exp_offs,
                "is_correct": (pred == exp_offs) if exp_offs is not None else None,
                "events": [e.copy() for e in events], "full_text": text_full,
            })
            continue

        # ── Transfer (try pattern B first, then A)
        tm = _transfer_B.search(t) or _transfer_A.search(t)
        if tm:
            if _transfer_B.search(t):
                giver, item, receiver = tm.group(1), tm.group(2), tm.group(3)
            else:
                giver, item, receiver = tm.group(1), tm.group(3), tm.group(2)
            if not has_negation(t, tm.start()):
                obj = clean_item(item)
                if obj and obj not in NOT_A_CHARACTER:
                    gk, rk = giver.lower(), receiver.lower()
                    inv[gk].pop(obj, None)
                    inv[rk][obj] = max(off(obj), -1)
                    last_g = last_c[rk] = obj
                    log(lid, t, "transfer", giver, obj, receiver=receiver)
                    continue

        # ── Extract subject
        char = extract_character(t)
        if not char:
            log(lid, t, "none", None, None); continue
        key = char.lower()

        # ── Pickup
        pm = _pickup_re.search(t)
        if pm:
            if not has_negation(t, pm.start()):
                obj = extract_object(pm, t)
                if obj:
                    obj = resolve(clean_item(obj), key, last_g, last_c, events)
                    if obj and obj not in NOT_A_CHARACTER:
                        add(key, obj); last_g = last_c[key] = obj
            log(lid, t, "pickup" if not has_negation(t, pm.start()) else "negated_pickup", char, obj if not has_negation(t, pm.start()) else None)
            continue

        # ── Drop
        dm = _drop_re.search(t)
        if dm:
            if not has_negation(t, dm.start()):
                obj = extract_object(dm, t)
                if obj:
                    obj = resolve(clean_item(obj), key, last_g, last_c, events)
                    if obj in inv.get(key, {}):
                        del inv[key][obj]; last_g = last_c[key] = obj
                    elif obj in NOT_A_CHARACTER | SKIP_WORDS:
                        lh = last_c.get(key)
                        if lh and lh in inv.get(key, {}): del inv[key][lh]
                else:
                    lh = last_c.get(key)
                    if lh and lh in inv.get(key, {}): del inv[key][lh]
            log(lid, t, "drop" if not has_negation(t, dm.start()) else "negated_drop", char, obj if not has_negation(t, dm.start()) else None)
            continue

        log(lid, t, "move", char, None)

    return results

# ── Parser ────────────────────────────────────────────────────────────────────

def parse_file(path):
    stories, current, parts, sid, seen = [], [], [], 1, False
    with open(path, encoding="utf-8-sig") as f:
        for physical_line, raw in enumerate(f, 1):
            line = raw.rstrip("\n")
            if not line.strip(): continue
            m = re.match(r'^(\d+)\s+(.*)$', line)
            if not m:
                raise ValueError(f"Line {physical_line}: expected a number followed by story text.")
            num, content = int(m.group(1)), m.group(2)
            if not content.strip():
                raise ValueError(f"Line {physical_line}: story text cannot be empty.")
            if not seen and num != 1:
                raise ValueError("Each file must begin with story line 1.")
            if current and num != 1 and num <= current[-1]['line_id']:
                raise ValueError(f"Line {physical_line}: story line numbers must increase or restart at 1.")
            if num == 1 and seen:
                stories.append({"story_id": sid, "sentences": current, "full_text": "\n".join(parts)})
                sid += 1; current = []; parts = []
            seen = True
            is_q = content.strip().startswith("What")
            ans = None
            if is_q:
                spl = content.split("\t")
                content, ans = spl[0].strip(), (spl[1].strip() if len(spl) > 1 else None)
            current.append({"line_id": num, "text": content, "is_question": is_q, "answer": ans})
            parts.append(content)
    if current:
        stories.append({"story_id": sid, "sentences": current, "full_text": "\n".join(parts)})
    return stories

def run(path):
    results = []
    for s in parse_file(path): results.extend(process_story(s))
    return results

# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(results, error_csv=None):
    labelled = [r for r in results if r["expected_offsets"] is not None]
    if not labelled: print("No labelled data found."); return {}

    total   = len(labelled)
    correct = sum(1 for r in labelled if r["is_correct"])
    acc     = correct / total

    buckets = {"nothing": [], "single": [], "multi": []}
    for r in labelled:
        exp = r["expected_offsets"]
        buckets["nothing" if exp == "-1" else "multi" if " " in exp else "single"].append(r)

    bucket_stats = {n: {"total": len(v), "correct": (c := sum(1 for i in v if i["is_correct"])),
                        "accuracy": c / len(v) if v else 0} for n, v in buckets.items()}

    def etype(r):
        exp, pred = r["expected_offsets"], r["predicted_answer"]
        if pred == "-1" and exp != "-1": return "false_nothing"
        if pred != "-1" and exp == "-1": return "false_item"
        es, ps = set(exp.split()), set(pred.split())
        return "missing_item" if ps < es else "extra_item" if ps > es else "wrong_item"

    errors = [{"story_id": r["story_id"], "question_id": r["question_id"],
               "question_text": r["question_text"], "character": r["character"],
               "expected": r["expected_answer"], "expected_offsets": r["expected_offsets"],
               "predicted": r["predicted_answer"], "error_type": etype(r),
               "inventory_snapshot": r["events"][-1]["inventory_snapshot"] if r["events"] else {}}
              for r in labelled if not r["is_correct"]]

    if error_csv and errors:
        with open(error_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Story","Q#","Character","Question","Expected","Predicted","Error Type","Expected Offsets","Inventory"])
            for e in errors:
                inv_str = ", ".join(f"{k}:{','.join(v)}" for k, v in e["inventory_snapshot"].items())
                w.writerow([e["story_id"], e["question_id"], e["character"], e["question_text"],
                            e["expected"], e["predicted"], e["error_type"], e["expected_offsets"], inv_str])

    print(f"\n{'='*52}\n  Accuracy : {correct}/{total}  ({acc:.2%})\n{'='*52}")
    for n, s in bucket_stats.items():
        if s["total"]: print(f"  {n:>8}  {s['correct']:>4}/{s['total']}  ({s['accuracy']:.2%})")
    if errors:
        print(f"\n--- First {min(10,len(errors))} errors ---")
        for e in errors[:10]:
            print(f"  Story {e['story_id']} Q{e['question_id']} '{e['character']}':\n"
                  f"    Q: {e['question_text']}\n    Expected : {e['expected']} ({e['expected_offsets']})\n"
                  f"    Predicted: {e['predicted']}  [{e['error_type']}]\n"
                  f"    Inventory: {e['inventory_snapshot']}\n")
    else:
        print("No errors!")

    return {"accuracy": acc, "total": total, "correct": correct, "wrong": total - correct,
            "by_answer_type": bucket_stats, "error_details": errors}

# ── Submission Writer ─────────────────────────────────────────────────────────

def write_submission(results, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["textID", "sortedAnswerList"])
        for r in results: w.writerow([f"{r['story_id']}_{r['question_id']}", r["predicted_answer"]])
    print(f"Submission saved: {output_path}  ({len(results)} rows)")

    from pathlib import Path
    target = Path(output_path)
    readable = str(target.with_name(target.stem + "_readable.csv"))
    with open(readable, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Story_Q", "Character", "Predicted_Items", "Raw_Offsets"])
        for r in results:
            raw, ft = r["predicted_answer"], r.get("full_text", "")
            items = []
            if raw != "-1":
                for s in raw.split():
                    try:
                        o = int(s)
                        if 0 <= o < len(ft):
                            st = o
                            while st > 0 and ft[st-1].isalnum(): st -= 1
                            en = o
                            while en < len(ft) and ft[en].isalnum(): en += 1
                            items.append(ft[st:en])
                        else: items.append("[INVALID]")
                    except ValueError: items.append("[ERROR]")
            w.writerow([f"{r['story_id']}_{r['question_id']}", r["character"],
                        ", ".join(items) if items else "nothing", raw])
    print(f"Readable version saved: {readable}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nus-3244-setqa")
    p = argparse.ArgumentParser()
    p.add_argument("--train",     default=os.path.join(d, "train.txt"))
    p.add_argument("--test",      default=os.path.join(d, "test.txt"))
    p.add_argument("--output",    default="submission.csv")
    p.add_argument("--error_csv", default="errors_optimized.csv")
    a = p.parse_args()

    if os.path.isfile(a.train):
        print(f"Evaluating: {a.train}")
        summary = evaluate(run(a.train), error_csv=a.error_csv)
        if summary: print(f"Errors: {a.error_csv}")
    else:
        print(f"[INFO] train.txt not found: {a.train}")

    if os.path.isfile(a.test):
        print(f"Testing: {a.test}")
        write_submission(run(a.test), a.output)
    else:
        print(f"[ERROR] test.txt not found: {a.test}")

if __name__ == "__main__":
    main()
