"""
Build Dataset 01 (Consultation / Transcription) and Dataset 03 (SOAP / Clinical
Documentation) from the three public datasets, using the exact column schema of
the placeholder Excel files.

Sources
  PriMock57   https://github.com/babylonhealth/primock57
  MTS-Dialog  https://github.com/abachaa/MTS-Dialog
  ACI-Bench   https://github.com/microsoft/clinical_visit_note_summarization_corpus

Usage
  pip install pandas openpyxl textgrid
  python build_datasets_01_03.py --raw data/raw --out data/processed

Expected layout under --raw (just clone / unzip the three repos there):
  data/raw/primock57/
  data/raw/MTS-Dialog/
  data/raw/clinical_visit_note_summarization_corpus/
"""
import argparse
import glob
import json
import os
import re

import pandas as pd

# --------------------------------------------------------------------------
# Output schemas (identical to the placeholder xlsx files, + source_id)
# --------------------------------------------------------------------------
COLS_01 = ["consultation_id", "patient_id", "doctor_id", "date", "specialty",
           "audio_path", "audio_duration_sec", "transcript", "symptoms",
           "medical_history", "allergies", "vital_signs", "source_type", "source_id"]
COLS_03 = ["consultation_id", "transcript", "subjective", "objective", "assessment",
           "plan", "consultation_summary", "patient_visit_note", "source_type", "source_id"]


def clean(t):
    if t is None or (isinstance(t, float) and pd.isna(t)):
        return ""
    t = str(t).replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def visit_note(s, o, a, p):
    parts = []
    if s: parts.append(f"Subjective: {s}")
    if o: parts.append(f"Objective: {o}")
    if a: parts.append(f"Assessment: {a}")
    if p: parts.append(f"Plan: {p}")
    return "\n".join(parts)


# --------------------------------------------------------------------------
# 1. PriMock57  (audio + TextGrid transcripts + clinician notes)
# --------------------------------------------------------------------------
def load_primock(root):
    import textgrid  # pip install textgrid

    def utterances(path, speaker):
        tg = textgrid.TextGrid()
        tg.read(path)
        out = []
        for tier in tg.tiers:
            for iv in tier.intervals:
                if iv.mark.strip():
                    txt = re.sub(r"</?UNSURE>|<UNIN/>|<INAUDIBLE_SPEECH/>", "", iv.mark)
                    out.append((iv.minTime, f"{speaker}: {clean(txt)}"))
        return out, tg.maxTime

    rows = []
    for doc_tg in sorted(glob.glob(os.path.join(root, "transcripts", "*_doctor.TextGrid"))):
        base = os.path.basename(doc_tg).replace("_doctor.TextGrid", "")  # day1_consultation01
        pat_tg = doc_tg.replace("_doctor.TextGrid", "_patient.TextGrid")
        note_json = os.path.join(root, "notes", base + ".json")
        if not (os.path.exists(pat_tg) and os.path.exists(note_json)):
            continue
        ud, dur_d = utterances(doc_tg, "Doctor")
        up, dur_p = utterances(pat_tg, "Patient")
        transcript = "\n".join(t for _, t in sorted(ud + up, key=lambda x: x[0]))
        note = json.load(open(note_json, encoding="utf-8"))
        note_txt = clean(note.get("note", ""))

        # Heuristic SOAP split of the clinician's free-text note
        # (PriMock notes are UK GP style: history ... PMH/DH/SH ... Imp: ... Plan: ...)
        m_imp = re.search(r"\b(Imp|Impression|Dx|Diagnosis)\s*:", note_txt, re.I)
        m_plan = re.search(r"\bPlan\s*:", note_txt, re.I)
        subj = note_txt
        assess = plan = ""
        if m_imp:
            subj = note_txt[:m_imp.start()].strip()
            assess = note_txt[m_imp.end():m_plan.start() if m_plan else None].strip()
        if m_plan:
            if not m_imp:
                subj = note_txt[:m_plan.start()].strip()
            plan = note_txt[m_plan.end():].strip()
        pmh = re.search(r"PMH\s*:\s*(.+)", note_txt)
        alg = re.search(r"(Allerg\w*|NKDA)\s*:?\s*(.*)", note_txt)

        day, num = re.findall(r"day(\d+)_consultation(\d+)", base)[0]
        rows.append(dict(
            source_id=base,
            doctor_id=f"PM_DR_day{day}",
            date="",
            specialty="General Practice",
            audio_path=f"primock57/audio/{base}_doctor.wav | primock57/audio/{base}_patient.wav",
            audio_duration_sec=round(max(dur_d, dur_p), 1),
            transcript=transcript,
            symptoms=clean(note.get("presenting_complaint", "")),
            medical_history=clean(pmh.group(1)) if pmh else "",
            allergies=clean(alg.group(2) or alg.group(1)) if alg else "",
            vital_signs="",
            subjective=subj,
            objective="",
            assessment=assess,
            plan=plan,
            consultation_summary="; ".join(note.get("highlights", [])),
            source_type="public_primock57",
        ))
    return rows


# --------------------------------------------------------------------------
# 2. MTS-Dialog  (short dialogue excerpt + one note section per row)
# --------------------------------------------------------------------------
MTS_SECTION_TO_SOAP = {
    "CC": "subjective", "GENHX": "subjective", "PASTMEDICALHX": "subjective",
    "FAM/SOCHX": "subjective", "ROS": "subjective", "ALLERGY": "subjective",
    "MEDICATIONS": "subjective", "PASTSURGICAL": "subjective", "GYNHX": "subjective",
    "IMMUNIZATIONS": "subjective", "OTHER_HISTORY": "subjective",
    "EXAM": "objective", "LABS": "objective", "IMAGING": "objective",
    "ASSESSMENT": "assessment", "DIAGNOSIS": "assessment",
    "PLAN": "plan", "DISPOSITION": "plan", "EDCOURSE": "plan", "PROCEDURES": "plan",
}


def load_mts(root):
    files = {
        "train": "Main-Dataset/MTS-Dialog-TrainingSet.csv",
        "valid": "Main-Dataset/MTS-Dialog-ValidationSet.csv",
        "test1": "Main-Dataset/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv",
        "test2": "Main-Dataset/MTS-Dialog-TestSet-2-MEDIQA-Sum-2023.csv",
    }
    rows = []
    for split, rel in files.items():
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            print("  skip missing", path)
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            sec = str(r["section_header"]).strip()
            field = MTS_SECTION_TO_SOAP.get(sec, "subjective")
            text = clean(r["section_text"])
            row = dict(
                source_id=f"mts_{split}_{r['ID']}",
                doctor_id="", date="", specialty="",
                audio_path="", audio_duration_sec="",
                transcript=clean(r["dialogue"]),
                symptoms=text if sec == "CC" else "",
                medical_history=text if sec in ("PASTMEDICALHX", "PASTSURGICAL", "FAM/SOCHX") else "",
                allergies=text if sec == "ALLERGY" else "",
                vital_signs="",
                subjective="", objective="", assessment="", plan="",
                consultation_summary=f"[{sec}] {text}",
                source_type="public_mts_dialog",
            )
            row[field] = text
            rows.append(row)
    return rows


# --------------------------------------------------------------------------
# 3. ACI-Bench  (full dialogue + full structured note + metadata)
# --------------------------------------------------------------------------
ACI_SPLITS = ["train", "valid", "clinicalnlp_taskB_test1",
              "clinicalnlp_taskC_test2", "clef_taskC_test3"]
SUBJ_SECTIONS = {"CHIEF COMPLAINT", "HISTORY OF PRESENT ILLNESS", "REVIEW OF SYSTEMS",
                 "PAST HISTORY", "MEDICAL HISTORY", "SOCIAL HISTORY", "FAMILY HISTORY",
                 "MEDICATIONS", "ALLERGIES", "CURRENT MEDICATIONS", "SURGICAL HISTORY"}
OBJ_SECTIONS = {"PHYSICAL EXAMINATION", "PHYSICAL EXAM", "VITALS REVIEWED", "VITALS",
                "RESULTS", "LABS", "IMAGING", "EXAM"}


def split_aci_note(note):
    """Split an ACI-Bench note into {SECTION NAME: text} using ALL-CAPS headers."""
    sections, current = {}, None
    for line in note.split("\n"):
        s = line.strip()
        if s and s.upper() == s and re.fullmatch(r"[A-Z][A-Z /&\-]+:?", s):
            current = s.rstrip(":")
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {k: clean("\n".join(v)) for k, v in sections.items()}


def load_aci(root):
    base = os.path.join(root, "data", "aci-bench", "challenge_data")
    rows = []
    for split in ACI_SPLITS:
        f = os.path.join(base, f"{split}.csv")
        fm = os.path.join(base, f"{split}_metadata.csv")
        if not os.path.exists(f):
            print("  skip missing", f)
            continue
        df = pd.read_csv(f)
        meta = pd.read_csv(fm) if os.path.exists(fm) else pd.DataFrame()
        if len(meta):
            df = df.merge(meta, on=["dataset", "encounter_id"], how="left")
        for _, r in df.iterrows():
            note = clean(r["note"])
            sec = split_aci_note(note)
            subj = "\n".join(sec[k] for k in sec if k in SUBJ_SECTIONS)
            obj = "\n".join(sec[k] for k in sec if k in OBJ_SECTIONS)
            ap = "\n".join(sec[k] for k in sec if "ASSESSMENT" in k or "PLAN" in k)
            vitals = "\n".join(sec[k] for k in sec if "VITAL" in k)
            cc = clean(r.get("cc", ""))
            second = clean(r.get("2nd_complaints", ""))
            rows.append(dict(
                source_id=f"aci_{split}_{r['encounter_id']}",
                doctor_id=clean(r.get("doctor_name", "")),
                date="", specialty="",
                audio_path="", audio_duration_sec="",
                transcript=clean(r["dialogue"]),
                symptoms="; ".join(x for x in [cc, second] if x),
                medical_history="\n".join(sec[k] for k in sec if "HISTORY" in k and "PRESENT" not in k),
                allergies=sec.get("ALLERGIES", ""),
                vital_signs=vitals,
                subjective=subj,
                objective=obj,
                assessment=ap,      # ACI notes combine Assessment + Plan in one section
                plan=ap,
                consultation_summary=sec.get("CHIEF COMPLAINT", cc),
                source_type="public_aci_bench",
                _age=r.get("patient_age", ""), _gender=r.get("patient_gender", ""),
            ))
    return rows


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for name, fn, sub in [("PriMock57", load_primock, "primock57"),
                          ("MTS-Dialog", load_mts, "MTS-Dialog"),
                          ("ACI-Bench", load_aci, "clinical_visit_note_summarization_corpus")]:
        p = os.path.join(a.raw, sub)
        if os.path.isdir(p):
            r = fn(p)
            print(f"{name}: {len(r)} consultations")
            rows += r
        else:
            print(f"{name}: folder not found at {p} — skipped")

    # Assign project IDs in the placeholder format
    for i, r in enumerate(rows, 1):
        r["consultation_id"] = f"C{i:05d}"
        r["patient_id"] = f"P{i:05d}"
        r["patient_visit_note"] = visit_note(r["subjective"], r["objective"],
                                             r["assessment"], r["plan"])

    df = pd.DataFrame(rows)
    d01 = df[COLS_01]
    d03 = df[COLS_03]

    d01.to_csv(os.path.join(a.out, "01_Consultation_Transcription.csv"), index=False)
    d03.to_csv(os.path.join(a.out, "03_SOAP_Clinical_Documentation.csv"), index=False)
    with pd.ExcelWriter(os.path.join(a.out, "01_Consultation_Transcription.xlsx")) as w:
        d01.to_excel(w, sheet_name="MVP_Dataset", index=False)
    with pd.ExcelWriter(os.path.join(a.out, "03_SOAP_Clinical_Documentation.xlsx")) as w:
        d03.to_excel(w, sheet_name="MVP_SOAP", index=False)

    # Demographics from ACI metadata → useful seed for Dataset 07 later
    if "_age" in df:
        demo = df[["consultation_id", "patient_id", "_age", "_gender", "source_type"]].rename(
            columns={"_age": "age", "_gender": "gender"})
        demo = demo[demo["age"].notna() & (demo["age"] != "")]
        demo.to_csv(os.path.join(a.out, "07_seed_demographics.csv"), index=False)

    print("\nTotal:", len(df))
    print(df["source_type"].value_counts().to_string())
    print("\nWritten to", a.out)


if __name__ == "__main__":
    main()
