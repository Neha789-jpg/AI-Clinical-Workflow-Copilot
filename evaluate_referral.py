# evaluate_referral.py
# Compares generate_referral() against what the real GP decided (evaluation/referral_gold_standard.csv,
# built from the clinicians' own plans in dataset 03). Reports accuracy / precision / recall / F1
# for the referral decision, and specialist agreement for true referrals.
#
# usage: python3 evaluate_referral.py            (all 57 PriMock cases, ~1 per minute on free tier)
#        python3 evaluate_referral.py --limit 10

import argparse
import time
import pandas as pd

from src.nlp.extractor import extract_entities
from src.documentation.generator import generate_soap
from src.workflow.referral import generate_referral

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--engine", default="auto", choices=["auto", "llm", "rules"])
args = parser.parse_args()

gold = pd.read_csv("evaluation/referral_gold_standard.csv")
consults = pd.read_excel("data/processed/01_Consultation_Transcription.xlsx").set_index("consultation_id")
if args.limit:
    gold = gold.head(args.limit)

rows = []
for g in gold.itertuples():
    transcript = str(consults.loc[g.consultation_id, "transcript"])
    ent = extract_entities(transcript, engine=args.engine)
    soap = generate_soap(transcript, ent)
    ref = generate_referral(transcript, ent, soap, engine=args.engine)

    pred = bool(ref["referral_needed"])
    gold_ref = bool(g.gold_referral)
    outcome = ("TP" if pred and gold_ref else "TN" if not pred and not gold_ref
               else "FP" if pred else "FN")
    ALIASES = {"stroke": ["neurology", "stroke", "tia"],
               "physiotherapy": ["physio", "musculoskeletal", "msk"]}
    spec_ok = None
    if gold_ref and pred:
        gs, ps = str(g.gold_specialist).lower(), str(ref["specialist"] or "").lower()
        words = [w for w in gs.replace("/", " ").split() if len(w) > 3]
        for k, v in ALIASES.items():
            if k in gs:
                words += v
        spec_ok = any(w in ps for w in words)

    rows.append({"consultation_id": g.consultation_id, "gold_referral": gold_ref,
                 "gold_specialist": g.gold_specialist, "predicted_referral": pred,
                 "predicted_specialist": ref["specialist"], "outcome": outcome,
                 "specialist_match": spec_ok, "note": g.note if isinstance(g.note, str) else ""})
    print(f"{g.consultation_id}  gold={gold_ref!s:<5} pred={pred!s:<5} {outcome}  {ref['specialist'] or ''}")
    if args.engine != "rules":
        time.sleep(8)

df = pd.DataFrame(rows)
df.to_csv("evaluation/results/referral_results.csv", index=False)

tp = (df.outcome == "TP").sum(); fp = (df.outcome == "FP").sum()
fn = (df.outcome == "FN").sum(); tn = (df.outcome == "TN").sum()
precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
print(f"\nn={len(df)}  TP={tp} FP={fp} FN={fn} TN={tn}")
print(f"accuracy={(tp + tn) / len(df):.2f}  precision={precision:.2f}  recall={recall:.2f}  f1={f1:.2f}")
if tp:
    print(f"specialist correct on {df.specialist_match.sum()}/{tp} true referrals")
print("saved evaluation/results/referral_results.csv")
