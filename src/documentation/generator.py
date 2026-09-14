def generate_subjective(entities):
    symptoms = entities.get("symptoms", [])

    if not symptoms:
        return "No subjective symptoms documented."

    symptom_text = []

    for symptom in symptoms:
        text = symptom["text"]
        duration = symptom.get("duration")

        if duration:
            symptom_text.append(f"{text} for {duration}")
        else:
            symptom_text.append(text)

    return "Patient reports " + ", ".join(symptom_text) + "."

def generate_soap(transcript, entities):
    """
    Generate a structured SOAP note from a transcript
    and extracted clinical information.
    """

    subjective = generate_subjective(entities)
    objective = ""
    assessment = ""
    plan = ""

    return {
        "subjective": subjective,
        "objective": objective,
        "assessment": assessment,
        "plan": plan
    }