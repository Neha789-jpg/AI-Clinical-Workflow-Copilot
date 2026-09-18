def generate_subjective(entities):
    """
    Generate the Subjective section from extracted clinical information.
    Includes symptoms and separate medical, family, and social history.
    """

    symptoms = entities.get("symptoms", [])
    medical_history = entities.get("medical_history", [])
    family_history = entities.get("family_history", [])
    social_history = entities.get("social_history", [])
    current_medications = entities.get("current_medications", [])

    lines = []

    # Add symptoms
    if symptoms:
        lines.append("Symptoms:")

        for symptom in symptoms:

    # Handle symptoms returned as dictionaries
            if isinstance(symptom, dict):
               text = symptom.get("text")
               duration = symptom.get("duration")

    # Handle symptoms returned as strings
            else:
              text = symptom
              duration = None

            if not text:
              continue

            if duration:
              lines.append(f"- {text} for {duration}")
            else:
              lines.append(f"- {text}")

    else:
        lines.append("Symptoms: None documented.")

    # Add separate history sections
    history_sections = [
        ("Medical History", medical_history),
        ("Family History", family_history),
        ("Social History", social_history)
    ]

    for title, items in history_sections:
        lines.append("")
        lines.append(f"{title}:")

        if items:
            for item in items:
                lines.append(f"- {item}")
        else:
            lines.append("- None documented.")

    lines.append("")
    lines.append("Current Medications:")

    if current_medications:
        for medication in current_medications:
            if isinstance(medication, dict):
                name = medication.get("name")
                dose = medication.get("dose")
                frequency = medication.get("frequency")

                if not name:
                    continue

                medication_text = name

                if dose:
                    medication_text += f" - {dose}"

                if frequency:
                    medication_text += f", {frequency}"

                lines.append(f"- {medication_text}")
            else:
                lines.append(f"- {medication}")
    else:
        lines.append("- None documented.")

    return "\n".join(lines)

def generate_objective(entities):
    """
    Generate the Objective section from extracted clinical information.
    Includes vital signs.
    """

    vitals = entities.get("vitals", {})

    lines = []

    has_vitals = (
    vitals.get("bp")
    or vitals.get("pulse") is not None
    or vitals.get("temperature") is not None
    or vitals.get("spo2") is not None
)



    if has_vitals:
        lines.append("Vital Signs:")

        if vitals.get("bp"):
            lines.append(f"- BP: {vitals['bp']}")

        if vitals.get("pulse") is not None:
            lines.append(f"- Pulse: {vitals['pulse']} bpm")

        if vitals.get("temperature") is not None:
            lines.append(f"- Temperature: {vitals['temperature']}")

        if vitals.get("spo2") is not None:
            lines.append(f"- SpO2: {vitals['spo2']}%")

    if lines:
        return "\n".join(lines)

    return "No objective findings documented."


def generate_assessment(entities):
    """
    Generate the Assessment section from extracted diagnoses.
    """

    diagnoses = entities.get("diagnoses", [])

    if not diagnoses:
        return "No diagnosis documented."

    assessment_parts = []

    for diagnosis in diagnoses:

        if isinstance(diagnosis, dict):
            text = diagnosis.get("text")
            status = diagnosis.get("status")
        else:
            text = diagnosis
            status = None

        if text:
            if status == "suspected":
               if text.lower() == "pregnancy":
                assessment_parts.append("Pregnancy to be excluded")
               else:
                assessment_parts.append(f"Possible {text}")

            else:
                assessment_parts.append(text)

    if not assessment_parts:
        return "No diagnosis documented."

    return "Assessment: " + ", ".join(assessment_parts) + "."


def generate_plan(entities):
    """
    Generate the Plan section from extracted medications,
    investigations, follow-up instructions, and advice.
    """

    medications = entities.get("prescribed", [])
    investigations = entities.get("investigations", [])
    follow_up = entities.get("follow_up", [])
    advice = entities.get("advice", [])

    lines = []

    # --------------------------------------------------
    # Medications
    # --------------------------------------------------

    if medications:
        medication_lines = []

        for medication in medications:

            if isinstance(medication, dict):
                name = medication.get("name")
                dose = medication.get("dose")
                frequency = medication.get("frequency")
            else:
                name = medication
                dose = None
                frequency = None

            if not name:
                continue

            medication_text = name

            if dose:
                medication_text += f" - {dose}"

            if frequency:
                medication_text += f", {frequency}"

            medication_lines.append(f"- {medication_text}")

        if medication_lines:
            lines.append("Medications:")
            lines.extend(medication_lines)

    # --------------------------------------------------
    # Investigations / Tests
    # --------------------------------------------------

    if investigations:
        lines.append("Investigations/Tests:")

        for test in investigations:
            if test:
                lines.append(f"- {test}")

    # --------------------------------------------------
    # Follow-up
    # --------------------------------------------------

    if follow_up:
        lines.append("Follow-up:")

        for instruction in follow_up:
            if instruction:
                lines.append(f"- {instruction}")

    # --------------------------------------------------
    # Advice
    # --------------------------------------------------

    if advice:
        lines.append("Advice:")

        for instruction in advice:
            if instruction:
                lines.append(f"- {instruction}")

    # --------------------------------------------------
    # Final output
    # --------------------------------------------------

    if lines:
        return "\n".join(lines)

    return "No plan documented."

def generate_soap(transcript, entities):
    """
    Generate a structured SOAP note from a transcript
    and extracted clinical information.
    """

    subjective = generate_subjective(entities)
    objective = generate_objective(entities)
    assessment = generate_assessment(entities)
    plan = generate_plan(entities)

    return {
        "subjective": subjective,
        "objective": objective,
        "assessment": assessment,
        "plan": plan
    }