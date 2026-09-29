# evaluate_coding.py
# Feeds each clinician's own assessment line into suggest_code() and checks the result
# against evaluation/coding_gold_standard.csv. No LLM needed for the rules engine.
#   correct   - right code
#   wrong     - a real code but the wrong concept (the dangerous error)
#   missed    - concept is in the table but we returned no code
#   abstained - concept not in table and we correctly returned no code
#   invented  - concept not in table but we returned a code anyway
# usage: python3 evaluate_coding.py [--engine rules|llm]
import argparse
import pandas as pd
from src.workflow.coding import suggest_code

parser = argparse.ArgumentParser()
parser.add_argument("--engine", default="rules", choices=["rules", "llm"])
args = parser.parse_args()

gold = pd.read_csv("evaluation/coding_gold_standard.csv")
rows = []
for g in gold.itertuples():
    r = suggest_code(str(g.doctor_assessment), engine=args.engine)
    pred = r["snomed_description"]
    if g.in_table:
        outcome = "correct" if pred in str(g.expected_description).split("|") else ("missed" if pred is None else "wrong")
    else:
        outcome = "abstained" if pred is None else "invented"
    rows.append({"consultation_id": g.consultation_id, "doctor_assessment": g.doctor_assessment,
                 "expected": g.expected_description, "predicted": pred, "code": r["snomed_code"],
                 "confidence": r["confidence"], "outcome": outcome})
    print(f"{g.consultation_id}  {outcome:<9} {str(g.doctor_assessment)[:45]:<45} -> {pred}")

df = pd.DataFrame(rows)
df.to_csv(f"evaluation/results/coding_results_{args.engine}.csv", index=False)
c = df.outcome.value_counts()
n_in = gold.in_table.sum(); n_out = (~gold.in_table).sum()
print(f"\nengine={args.engine}  n={len(df)}")
print(f"in-table ({n_in}):  correct={c.get('correct',0)}  wrong={c.get('wrong',0)}  missed={c.get('missed',0)}"
      f"  -> accuracy {c.get('correct',0)/n_in:.2f}, wrong-code rate {c.get('wrong',0)/n_in:.2f}")
print(f"not in table ({n_out}): abstained={c.get('abstained',0)}  invented={c.get('invented',0)}")
