# verify_snomed.py
# Checks each code in data/reference/snomed_common.csv against a public
# SNOMED CT FHIR terminology server. Never ship a code that isn't OK here.

import time
import requests
import pandas as pd

FHIR = "https://r4.ontoserver.csiro.au/fhir/CodeSystem/$lookup"

df = pd.read_csv("data/reference/snomed_common.csv")

ok, bad = 0, 0
for _, row in df.iterrows():
    code = str(row["snomed_code"]).strip()
    try:
        r = requests.get(FHIR, params={"system": "http://snomed.info/sct", "code": code},
                         headers={"Accept": "application/fhir+json"}, timeout=20)
        data = r.json()
    except Exception as e:
        print(f"ERROR      {code:<12} {row['diagnosis']:<40} ({e})")
        bad += 1
        continue

    display = next((p.get("valueString") for p in data.get("parameter", [])
                    if p.get("name") == "display"), None)

    if r.status_code != 200 or not display:
        print(f"NOT FOUND  {code:<12} {row['diagnosis']}")
        bad += 1
    elif display.lower() == str(row["snomed_description"]).lower():
        print(f"OK         {code:<12} {display}")
        ok += 1
    else:
        print(f"MISMATCH   {code:<12} ours='{row['snomed_description']}'  official='{display}'")
        bad += 1
    time.sleep(0.3)

print(f"\n{ok} OK, {bad} need attention")