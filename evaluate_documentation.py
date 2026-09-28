
import json
import os
import re
import pandas as pd

from openai import OpenAI

from src.nlp.extractor import extract_with_llm
from src.documentation.generator import generate_soap


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/01_Consultation_Transcription.xlsx"

OUTPUT_FILE = "evaluation/results/documentation_llm_results.csv"

MODEL = "openai/gpt-oss-120b"

# Set to None to evaluate all consultations.
# For testing, you can temporarily use a small list such as:
# TARGET_IDS = ["C00004", "C00008"]
TARGET_IDS = [
    "C00004",
    "C00008",
    "C00012",
    "C00015",
    "C00025",
    "C00030",
    "C00031",
    "C00034",
    "C00040",
    "C00045"
]

FORCE_REEVALUATE = True


# ============================================================
# LLM EVALUATOR PROMPT
# ============================================================

EVALUATOR_PROMPT = """
You are evaluating a clinical documentation system.

You are given:

1. The original consultation transcript.
2. A generated SOAP note based on that consultation.

Your task is to evaluate whether the generated SOAP note accurately
represents the information contained in the consultation transcript.

IMPORTANT RULES:

- Use ONLY information explicitly supported by the transcript.
- Do not assume information that is not present.
- Do not penalize the SOAP note for omitting information that is
  genuinely unimportant for clinical documentation.
- Do not expect exact wording.
- Paraphrases are acceptable if they preserve the meaning.
- A generated statement is a hallucination if the transcript does
  not support it.
- Missing important clinical information should reduce completeness.
- Information placed in the wrong SOAP section should reduce
  section correctness.
- Do not judge the medical quality of the doctor's decisions.
  Evaluate the documentation itself.

Evaluate these five dimensions:

1. COMPLETENESS
   How completely does the SOAP note capture the clinically relevant
   information from the transcript?

2. FAITHFULNESS
   How accurately does the SOAP note represent information that is
   actually supported by the transcript?

3. HALLUCINATION
   Does the SOAP note introduce information that is not supported
   by the transcript?

4. SECTION_CORRECTNESS
   Is information placed in the appropriate SOAP section
   (Subjective, Objective, Assessment, Plan)?

5. OVERALL_QUALITY
   Overall quality of the generated documentation as a faithful
   representation of the consultation.

Use a score from 0 to 1 for each dimension:

0.0 = very poor
0.25 = poor
0.5 = moderate
0.75 = good
1.0 = excellent

For HALLUCINATION, the score has the following meaning:

1.0 = no meaningful unsupported information
0.75 = very minor unsupported information
0.5 = some unsupported information
0.25 = substantial unsupported information
0.0 = extensive hallucination

Also identify:

- important_information_missing
- unsupported_information
- section_errors
- strengths
- weaknesses

Return ONLY valid JSON in exactly this format:

{
    "completeness": 0.0,
    "faithfulness": 0.0,
    "hallucination": 0.0,
    "section_correctness": 0.0,
    "overall_quality": 0.0,
    "important_information_missing": [],
    "unsupported_information": [],
    "section_errors": [],
    "strengths": [],
    "weaknesses": []
}
"""


# ============================================================
# GROQ CLIENT
# ============================================================

def get_llm_client():
    """
    Create the Groq OpenAI-compatible client.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. "
            "Make sure your .env file is loaded before running."
        )

    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )


# ============================================================
# SAFE JSON PARSING
# ============================================================

def parse_json_response(response_text):
    """
    Parse JSON returned by the LLM.

    Handles occasional markdown code fences or extra whitespace.
    """

    response_text = response_text.strip()

    # Remove markdown code fences if present
    response_text = re.sub(
        r"^```(?:json)?\s*",
        "",
        response_text,
        flags=re.IGNORECASE
    )

    response_text = re.sub(
        r"\s*```$",
        "",
        response_text
    )

    return json.loads(response_text)


# ============================================================
# SAFE LIST
# ============================================================

def safe_list(value):
    """
    Ensure list-valued evaluator fields are always stored as lists.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return value

    return [str(value)]


# ============================================================
# EVALUATE DOCUMENTATION
# ============================================================

def evaluate_documentation(
    client,
    transcript,
    soap_note
):
    """
    Ask the LLM to evaluate a generated SOAP note against
    the original consultation transcript.
    """

    prompt = f"""
{EVALUATOR_PROMPT}

============================================================
ORIGINAL CONSULTATION TRANSCRIPT
============================================================

{transcript}

============================================================
GENERATED SOAP NOTE
============================================================

SUBJECTIVE:
{soap_note.get("subjective", "")}

OBJECTIVE:
{soap_note.get("objective", "")}

ASSESSMENT:
{soap_note.get("assessment", "")}

PLAN:
{soap_note.get("plan", "")}

============================================================
END INPUT
============================================================

Return ONLY the required JSON object.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful clinical documentation evaluator. "
                    "Return valid JSON only."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    return parse_json_response(response.choices[0].message.content)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading consultation data...")

df = pd.read_excel(INPUT_FILE)

print("\nColumns in dataset:")
print(df.columns.tolist())

print(f"Total consultations in dataset: {len(df)}")


# ============================================================
# FILTER CONSULTATIONS
# ============================================================

if TARGET_IDS is not None:

    TARGET_IDS = [str(x) for x in TARGET_IDS]

    df["consultation_id"] = df["consultation_id"].astype(str)

    df = df[df["consultation_id"].isin(TARGET_IDS)]

    print(f"Evaluating selected consultations: {len(df)}")

else:

    print(f"Evaluating all consultations: {len(df)}")


# ============================================================
# PREPARE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


# ============================================================
# LOAD PREVIOUS RESULTS
# ============================================================

if os.path.exists(OUTPUT_FILE) and not FORCE_REEVALUATE:

    previous_df = pd.read_csv(OUTPUT_FILE)

    if not previous_df.empty:

        completed_ids = set(
            previous_df["consultation_id"]
            .astype(str)
            .tolist()
        )

        print(
            f"Found {len(completed_ids)} previously evaluated consultations."
        )

    else:

        completed_ids = set()

else:

    previous_df = pd.DataFrame()
    completed_ids = set()

    if FORCE_REEVALUATE:
        print("FORCE_REEVALUATE=True -> starting fresh evaluation.")


# ============================================================
# LLM CLIENT
# ============================================================

client = get_llm_client()


# ============================================================
# EVALUATION LOOP
# ============================================================

results = []

for _, row in df.iterrows():

    consultation_id = str(row["consultation_id"])

    # --------------------------------------------------------
    # Skip already completed consultations
    # --------------------------------------------------------

    if consultation_id in completed_ids:

        print(
            f"\nSkipping {consultation_id} "
            "(already evaluated)"
        )

        continue


    print("\n" + "=" * 70)
    print(f"Evaluating documentation: {consultation_id}")
    print("=" * 70)


    # --------------------------------------------------------
    # Get transcript
    # --------------------------------------------------------

    transcript = str(row["transcript"])


    try:

        # ----------------------------------------------------
        # Step 1: Extract clinical information
        # ----------------------------------------------------

        print("Running NLP extractor...")

        entities = extract_with_llm(transcript)

        print(json.dumps(entities, indent=2, ensure_ascii=False))


        # ----------------------------------------------------
        # Step 2: Generate SOAP documentation
        # ----------------------------------------------------

        print("Generating SOAP note...")

        soap_note = generate_soap(
            transcript,
            entities
        )


        # ----------------------------------------------------
        # Step 3: Evaluate SOAP note
        # ----------------------------------------------------

        print("Evaluating SOAP note with LLM...")

        evaluation = evaluate_documentation(
            client,
            transcript,
            soap_note
        )


        # ----------------------------------------------------
        # Extract scores
        # ----------------------------------------------------

        completeness = float(
            evaluation.get("completeness", 0)
        )

        faithfulness = float(
            evaluation.get("faithfulness", 0)
        )

        hallucination = float(
            evaluation.get("hallucination", 0)
        )

        section_correctness = float(
            evaluation.get("section_correctness", 0)
        )

        overall_quality = float(
            evaluation.get("overall_quality", 0)
        )


        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        result = {
            "consultation_id": consultation_id,

            "completeness": completeness,
            "faithfulness": faithfulness,
            "hallucination": hallucination,
            "section_correctness": section_correctness,
            "overall_quality": overall_quality,

            "important_information_missing": json.dumps(
                safe_list(
                    evaluation.get(
                        "important_information_missing"
                    )
                ),
                ensure_ascii=False
            ),

            "unsupported_information": json.dumps(
                safe_list(
                    evaluation.get(
                        "unsupported_information"
                    )
                ),
                ensure_ascii=False
            ),

            "section_errors": json.dumps(
                safe_list(
                    evaluation.get(
                        "section_errors"
                    )
                ),
                ensure_ascii=False
            ),

            "strengths": json.dumps(
                safe_list(
                    evaluation.get("strengths")
                ),
                ensure_ascii=False
            ),

            "weaknesses": json.dumps(
                safe_list(
                    evaluation.get("weaknesses")
                ),
                ensure_ascii=False
            ),

            "subjective": soap_note.get(
                "subjective",
                ""
            ),

            "objective": soap_note.get(
                "objective",
                ""
            ),

            "assessment": soap_note.get(
                "assessment",
                ""
            ),

            "plan": soap_note.get(
                "plan",
                ""
            )
        }


        results.append(result)


        # ----------------------------------------------------
        # Save after EVERY successful consultation
        # ----------------------------------------------------

        new_df = pd.DataFrame(results)

        if not previous_df.empty:

            combined_df = pd.concat(
                [
                    previous_df,
                    new_df
                ],
                ignore_index=True
            )

        else:

            combined_df = new_df


        combined_df.to_csv(
            OUTPUT_FILE,
            index=False
        )


        # ----------------------------------------------------
        # Print current result
        # ----------------------------------------------------

        print("\nScores:")

        print(
            f"Completeness:       {completeness:.2f}"
        )

        print(
            f"Faithfulness:       {faithfulness:.2f}"
        )

        print(
            f"Hallucination:      {hallucination:.2f}"
        )

        print(
            f"Section correctness: {section_correctness:.2f}"
        )

        print(
            f"Overall quality:    {overall_quality:.2f}"
        )


    except Exception as e:

        error_message = str(e)

        print(
            f"\nERROR while evaluating {consultation_id}:"
        )

        print(error_message)


        # ----------------------------------------------------
        # Stop on Groq rate limits
        # ----------------------------------------------------

        if (
            "429" in error_message
            or "rate limit" in error_message.lower()
            or "TPD" in error_message
            or "TPM" in error_message
        ):

            print(
                "\nGroq rate limit detected."
            )

            print(
                "Progress has already been saved."
            )

            break


# ============================================================
# FINAL RESULTS
# ============================================================

if os.path.exists(OUTPUT_FILE):

    final_df = pd.read_csv(OUTPUT_FILE)

    if not final_df.empty:

        print("\n" + "=" * 70)
        print("DOCUMENTATION EVALUATION SUMMARY")
        print("=" * 70)

        print(
            f"Consultations evaluated: "
            f"{len(final_df)}"
        )

        print(
            f"\nAverage completeness: "
            f"{final_df['completeness'].mean():.3f}"
        )

        print(
            f"Average faithfulness: "
            f"{final_df['faithfulness'].mean():.3f}"
        )

        print(
            f"Average hallucination score: "
            f"{final_df['hallucination'].mean():.3f}"
        )

        print(
            f"Average section correctness: "
            f"{final_df['section_correctness'].mean():.3f}"
        )

        print(
            f"Average overall quality: "
            f"{final_df['overall_quality'].mean():.3f}"
        )

        print(
            f"\nResults saved to:"
        )

        print(OUTPUT_FILE)

    else:

        print("\nNo evaluation results available yet.")

else:

    print("\nNo results file was created.")


