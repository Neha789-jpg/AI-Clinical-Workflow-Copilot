import os
import json
import time
import pandas as pd
import random

from dotenv import load_dotenv
from openai import OpenAI


# ==========================================
# CONFIGURATION
# ==========================================

INPUT_FILE = "data/processed/01_Consultation_Transcription.xlsx"

OUTPUT_FILE = "evaluation/auto_gold_standard.json"

MODEL_NAME = "openai/gpt-oss-120b"


# ==========================================
# LOAD API KEY
# ==========================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")


client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_excel(INPUT_FILE)

print("Total consultations:", len(df))


# ==========================================
# LOAD EXISTING PROGRESS
# ==========================================

if os.path.exists(OUTPUT_FILE):

    with open(
        OUTPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        all_references = json.load(file)

else:

    all_references = {}


print(
    "Already processed:",
    len(all_references)
)


# ==========================================
# LLM ANNOTATION FUNCTION
# ==========================================

def generate_reference(transcript):

    prompt = f"""
You are a clinical data annotation assistant.

Read the following medical consultation transcript
and extract only information explicitly mentioned
in the transcript.

Do not invent information.
If a field is not mentioned, return an empty list
or null where appropriate.

Return ONLY valid JSON using this structure:

{{
    "patient_info": {{
        "age": null,
        "gender": null
    }},
    "symptoms": [
        {{
            "text": "",
            "duration": null,
            "severity": null,
            "status": "current"
        }}
    ],
    "diagnoses": [
        {{
            "text": "",
            "status": ""
        }}
    ],
    "current_medications": [],
    "prescribed": [
        {{
            "name": "",
            "dose": null,
            "frequency": null
        }}
    ],
    "investigations": [],
    "follow_up": [],
    "advice": [],
    "allergies": [],
    "vitals": {{
        "bp": null,
        "pulse": null,
        "temperature": null,
        "spo2": null
    }},
    "medical_history": [],
    "family_history": [],
    "social_history": [],
    "exposure_history": []
}}

Important rules:

1. Extract only information present in the transcript.
2. Do not make medical assumptions.
3. Keep symptoms separate from advice.
4. Keep diagnoses separate from symptoms.
5. Include resolved and suspected statuses when explicitly stated.
6. Do not add information just because it is medically likely.

TRANSCRIPT:

{transcript}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful clinical information "
                    "extraction assistant."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        response_format={
            "type": "json_object"
        }
    )

    content = response.choices[0].message.content

    return json.loads(content)


# ==========================================
# PROCESS ALL CONSULTATIONS WITH RETRIES
# ==========================================

for index, row in df.iterrows():

    consultation_id = str(
        row["consultation_id"]
    ).strip()

    transcript = row["transcript"]

    # Skip consultations already processed
    if consultation_id in all_references:

        print(
            f"[{index + 1}/{len(df)}] "
            f"Skipping {consultation_id}"
        )

        continue

    print(
        f"\n[{index + 1}/{len(df)}] "
        f"Processing {consultation_id}"
    )

    max_retries = 6
    attempt = 0

    while attempt < max_retries:

        try:

            reference = generate_reference(
                transcript
            )

            all_references[consultation_id] = reference

            # Save progress after every consultation
            with open(
                OUTPUT_FILE,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    all_references,
                    file,
                    indent=2,
                    ensure_ascii=False
                )

            print(
                f"Completed: {consultation_id}"
            )

            # Wait between successful requests
            time.sleep(15)

            break

        except Exception as error:

            attempt += 1

            error_text = str(error)

            if "429" in error_text:

                wait_time = 30 * attempt + random.randint(0, 10)

                print(
                    f"Rate limit reached for "
                    f"{consultation_id}."
                )

                print(
                    f"Retry {attempt}/{max_retries} "
                    f"in {wait_time} seconds..."
                )

                time.sleep(wait_time)

            else:

                print(
                    f"Error processing "
                    f"{consultation_id}: {error}"
                )

                break

    else:

        print(
            f"Failed after {max_retries} retries: "
            f"{consultation_id}"
        )

        continue

# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n===================================")

print("PROCESSING COMPLETED")

print(
    "Total references saved:",
    len(all_references)
)

print(
    "Output file:",
    OUTPUT_FILE
)

print("===================================")