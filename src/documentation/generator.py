def generate_subjective(entities):
    """
    Generate the Subjective section from extracted clinical information.
    Includes symptoms, their duration, and relevant history.
    """

    symptoms = entities.get("symptoms", [])
    history = entities.get("history", [])

    lines = []

    # Add symptoms
    if symptoms:
        lines.append("Symptoms:")
        
        for symptom in symptoms:
            text = symptom.get("text")
            duration = symptom.get("duration")

            if not text:
                continue

            if duration:
                lines.append(f"- {text} for {duration}")
            else:
                lines.append(f"- {text}")

    else:
        lines.append("Symptoms: None documented.")

    # Add relevant history
    if history:
        lines.append("")
        lines.append("Relevant History:")

        for item in history:
            lines.append(f"- {item}")

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
        text = diagnosis.get("text")
        status = diagnosis.get("status")

        if text:
            if status == "suspected":
                assessment_parts.append(f"Possible {text}")
            else:
                assessment_parts.append(text)

    return "Assessment: " + ", ".join(assessment_parts) + "."


def generate_plan(entities):
    """
    Generate the Plan section from extracted medications.
    """

    medications = entities.get("medications", [])

    plan_parts = []

    for medication in medications:
        name = medication.get("name")
        dose = medication.get("dose")
        frequency = medication.get("frequency")

        if not name:
            continue

        medication_text = name

        if dose:
            medication_text += f" {dose}"

        if frequency:
            medication_text += f", {frequency}"

        plan_parts.append(medication_text)

    if plan_parts:
        return "Medications: " + "; ".join(plan_parts) + "."

    return "No medications documented."


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