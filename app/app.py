import os
import tempfile

import streamlit as st

from src.transcription.transcriber import transcribe_audio
from src.nlp.extractor import extract_entities
from src.documentation.generator import generate_soap
from src.workflow.referral import generate_referral


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Clinical Workflow Copilot",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# CUSTOM STYLING
# =====================================================

st.markdown(
    """
<style>

.stApp {
    background-color: #f3f6fa;
}

/* Main content */

.block-container {
    max-width: 1400px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background-color: #172b49;
}

[data-testid="stSidebar"] * {
    color: #e5edf7;
}

[data-testid="stSidebar"] hr {
    border-color: #405570;
}

/* Header */

.app-header {
    background: #ffffff;
    border: 1px solid #e1e8f0;
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 28px;
    box-shadow: 0 4px 18px rgba(23, 43, 73, 0.05);
}

.app-tag {
    color: #147d78;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 10px;
}

.app-title {
    color: #172b49;
    font-size: 30px;
    font-weight: 700;
    letter-spacing: -0.8px;
    line-height: 1.3;
}

.app-subtitle {
    color: #718096;
    font-size: 14px;
    margin-top: 8px;
}

/* Section headings */

.section-heading {
    color: #172b49;
    font-size: 21px;
    font-weight: 650;
    margin-top: 14px;
    margin-bottom: 5px;
}

.section-description {
    color: #718096;
    font-size: 13px;
    margin-bottom: 18px;
}

/* Streamlit containers */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #ffffff;
    border: 1px solid #e1e8f0;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 3px 14px rgba(23, 43, 73, 0.035);
}

/* Card headings */

.card-title {
    color: #172b49;
    font-size: 16px;
    font-weight: 650;
    margin-bottom: 12px;
}

/* Information labels */

.info-label {
    color: #718096;
    font-size: 12px;
    margin-bottom: 4px;
}

.info-value {
    color: #263b55;
    font-size: 15px;
    font-weight: 600;
}

/* Metrics */

.metric-card {
    background-color: #ffffff;
    border: 1px solid #e1e8f0;
    border-radius: 11px;
    padding: 18px 20px;
    min-height: 90px;
    box-shadow: 0 3px 14px rgba(23, 43, 73, 0.035);
}

.metric-label {
    color: #718096;
    font-size: 12px;
    font-weight: 500;
}

.metric-value {
    color: #172b49;
    font-size: 21px;
    font-weight: 700;
    margin-top: 8px;
}

/* Buttons */

.stButton > button {
    background-color: #147d78;
    color: white;
    border: none;
    border-radius: 7px;
    min-height: 42px;
    font-size: 14px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #0f625e;
    color: white;
}

/* Tabs */

.stTabs [data-baseweb="tab-list"] {
    gap: 5px;
    background-color: #e6ecf3;
    padding: 6px;
    border-radius: 10px;
}

.stTabs [data-baseweb="tab"] {
    color: #607087;
    font-size: 13px;
    font-weight: 550;
    border-radius: 7px;
    padding: 10px 18px;
}

.stTabs [aria-selected="true"] {
    background-color: #ffffff;
    color: #172b49;
    box-shadow: 0 2px 6px rgba(23, 43, 73, 0.08);
}

/* File uploader */

[data-testid="stFileUploader"] {
    background-color: #ffffff;
    border: 1px dashed #b7c6d8;
    border-radius: 11px;
    padding: 10px;
}

/* Expander */

[data-testid="stExpander"] {
    background-color: #ffffff;
    border: 1px solid #e1e8f0;
    border-radius: 9px;
}

/* General text */

p, li {
    color: #425466;
    font-size: 14px;
    line-height: 1.7;
}

h1, h2, h3 {
    color: #172b49;
}

</style>
""",
    unsafe_allow_html=True
)


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def display_value(value):
    """
    Converts missing values into readable text.
    """

    if value is None or value == "" or value == []:
        return "Not available"

    return str(value)


def get_item_text(item):
    """
    Converts dictionary or string items into readable text.
    """

    if isinstance(item, str):
        return item

    if isinstance(item, dict):
        return (
            item.get("text")
            or item.get("name")
            or item.get("diagnosis")
            or item.get("medication")
            or str(item)
        )

    return str(item)


def card_title(title):
    """
    Displays a consistent heading inside a card.
    """

    st.markdown(
        f'<div class="card-title">{title}</div>',
        unsafe_allow_html=True
    )


def display_list(title, items):
    """
    Displays list-based clinical information.
    """

    card_title(title)

    if not items:
        st.caption("No information recorded.")
        return

    for item in items:
        st.markdown(f"• {get_item_text(item)}")


def display_patient_info(patient_info):
    """
    Displays patient details.
    """

    card_title("Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="info-label">Age</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="info-value">{display_value(patient_info.get("age"))}</div>',
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            '<div class="info-label">Gender</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="info-value">{display_value(patient_info.get("gender"))}</div>',
            unsafe_allow_html=True
        )


def display_symptoms(symptoms):
    """
    Displays symptoms with duration, severity, and status.
    """

    card_title("Symptoms")

    if not symptoms:
        st.caption("No symptoms identified.")
        return

    for symptom in symptoms:

        if isinstance(symptom, dict):

            symptom_name = symptom.get(
                "text",
                "Unknown symptom"
            )

            duration = symptom.get("duration")
            severity = symptom.get("severity")
            status = symptom.get("status")

            st.markdown(
                f"**{symptom_name.title()}**"
            )

            details = []

            if duration:
                details.append(f"Duration: {duration}")

            if severity:
                details.append(f"Severity: {severity}")

            if status:
                details.append(f"Status: {status}")

            if details:
                st.caption(" | ".join(details))

        else:
            st.markdown(f"• {symptom}")


def display_history(title, history):
    """
    Displays medical, family, or social history.
    """

    card_title(title)

    if not history:
        st.caption("No information recorded.")
        return

    if isinstance(history, list):

        for item in history:
            st.markdown(f"• {get_item_text(item)}")

    else:
        st.write(history)


def display_clinical_information(entities):
    """
    Displays extracted clinical information
    using proper Streamlit containers.
    """

    patient_info = entities.get(
        "patient_info",
        {}
    )

    # Patient information

    with st.container(border=True):
        display_patient_info(patient_info)

    # Symptoms

    with st.container(border=True):
        display_symptoms(
            entities.get("symptoms", [])
        )

    # Diagnoses and medications

    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):
            display_list(
                "Diagnoses",
                entities.get("diagnoses", [])
            )

    with col2:

        with st.container(border=True):

            medications = entities.get(
                "current_medications",
                entities.get("medications", [])
            )

            display_list(
                "Medications",
                medications
            )

    # Allergies and investigations

    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):
            display_list(
                "Allergies",
                entities.get("allergies", [])
            )

    with col2:

        with st.container(border=True):
            display_list(
                "Investigations",
                entities.get("investigations", [])
            )

    # History

    with st.container(border=True):

        display_history(
            "Medical History",
            entities.get("medical_history", [])
        )

        display_history(
            "Family History",
            entities.get("family_history", [])
        )

        display_history(
            "Social History",
            entities.get("social_history", [])
        )

    # Advice and follow-up

    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):
            display_list(
                "Clinical Advice",
                entities.get("advice", [])
            )

    with col2:

        with st.container(border=True):
            display_list(
                "Follow-up",
                entities.get("follow_up", [])
            )


def create_soap_text(soap):
    """
    Converts the SOAP dictionary into downloadable text.
    """

    return f"""
CLINICAL SOAP NOTE
==================

SUBJECTIVE
----------
{soap.get("subjective", "Not available")}


OBJECTIVE
---------
{soap.get("objective", "Not available")}


ASSESSMENT
----------
{soap.get("assessment", "Not available")}


PLAN
----
{soap.get("plan", "Not available")}
"""


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown(
        """
        <div style="font-size: 21px; font-weight: 700;">
        Clinical Copilot
        </div>

        <div style="font-size: 12px; color: #aebed2;">
        AI-assisted clinical documentation
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### Workflow")

    st.markdown(
        """
        **01**  Audio transcription

        **02**  Clinical information extraction

        **03**  SOAP note generation

        **04**  Referral assessment
        """
    )

    st.divider()

    st.markdown("### System Information")

    st.caption(
        "This prototype assists with documentation "
        "and should not replace professional clinical judgment."
    )


# =====================================================
# MAIN HEADER
# =====================================================

st.markdown(
    '<div class="app-header">'
    '<div class="app-tag">CLINICAL DOCUMENTATION PLATFORM</div>'
    '<div class="app-title">AI Clinical Workflow Copilot</div>'
    '<div class="app-subtitle">'
    'Convert consultation recordings into structured clinical '
    'information, SOAP documentation, and referral letters.'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)


# =====================================================
# AUDIO UPLOAD
# =====================================================

st.markdown(
    '<div class="section-heading">Consultation Recording</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Upload an audio recording to begin the documentation workflow.'
    '</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=[
        "wav",
        "mp3",
        "mpeg",
        "mp4",
        "m4a"
    ],
    label_visibility="collapsed"
)


if uploaded_file is not None:

    st.audio(
        uploaded_file,
        format=uploaded_file.type
    )

    st.write("")

    if st.button(
        "Process Consultation",
        use_container_width=True
    ):

        audio_path = None

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(
                uploaded_file.name
            )[1]
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            audio_path = temp_file.name

        try:

            # =================================================
            # TRANSCRIPTION
            # =================================================

            with st.spinner(
                "Transcribing consultation..."
            ):

                transcript = transcribe_audio(
                    audio_path
                )

            st.success(
                "Transcription completed."
            )

            # =================================================
            # CLINICAL INFORMATION EXTRACTION
            # =================================================

            with st.spinner(
                "Extracting clinical information..."
            ):

                entities = extract_entities(
                    transcript
                )

            st.success(
                "Clinical information extracted."
            )

            # =================================================
            # SOAP GENERATION
            # =================================================

            with st.spinner(
                "Generating SOAP note..."
            ):

                soap = generate_soap(
                    transcript,
                    entities
                )

            st.success(
                "SOAP note generated."
            )

            # =================================================
            # REFERRAL GENERATION
            # =================================================

            with st.spinner(
                "Assessing referral requirement..."
            ):

                referral = generate_referral(
                    transcript,
                    entities,
                    soap
                )

            st.success(
                "Referral assessment completed."
            )

            # =================================================
            # RESULTS HEADER
            # =================================================

            st.divider()

            st.markdown(
                '<div class="section-heading">Consultation Results</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-description">'
                'Review the generated documentation and extracted clinical information.'
                '</div>',
                unsafe_allow_html=True
            )

            # =================================================
            # SUMMARY METRICS
            # =================================================

            symptoms_count = len(
                entities.get("symptoms", [])
            )

            diagnoses_count = len(
                entities.get("diagnoses", [])
            )

            referral_status = (
                "Required"
                if referral.get("referral_needed")
                else "Not identified"
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">
                        Symptoms Identified
                        </div>
                        <div class="metric-value">
                        {symptoms_count}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">
                        Diagnoses Identified
                        </div>
                        <div class="metric-value">
                        {diagnoses_count}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col3:

                st.markdown(
                    """
                    <div class="metric-card">
                        <div class="metric-label">
                        Documentation
                        </div>
                        <div class="metric-value">
                        Generated
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col4:

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">
                        Referral Status
                        </div>
                        <div class="metric-value">
                        {referral_status}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.write("")

            # =================================================
            # TABS
            # =================================================

            tab1, tab2, tab3, tab4 = st.tabs(
                [
                    "Transcription",
                    "Clinical Information",
                    "SOAP Note",
                    "Referral"
                ]
            )

            # =================================================
            # TRANSCRIPTION TAB
            # =================================================

            with tab1:

                st.markdown(
                    '<div class="section-heading">'
                    'Consultation Transcription'
                    '</div>',
                    unsafe_allow_html=True
                )

                with st.container(border=True):

                    st.write(transcript)

                st.download_button(
                    label="Download Transcription",
                    data=transcript,
                    file_name="consultation_transcript.txt",
                    mime="text/plain"
                )

            # =================================================
            # CLINICAL INFORMATION TAB
            # =================================================

            with tab2:

                st.markdown(
                    '<div class="section-heading">'
                    'Clinical Information'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="section-description">'
                    'Structured information extracted from the consultation.'
                    '</div>',
                    unsafe_allow_html=True
                )

                display_clinical_information(
                    entities
                )

                with st.expander(
                    "View technical JSON output"
                ):

                    st.json(entities)

            # =================================================
            # SOAP NOTE TAB
            # =================================================

            with tab3:

                st.markdown(
                    '<div class="section-heading">'
                    'Clinical SOAP Note'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="section-description">'
                    'Automatically generated clinical documentation.'
                    '</div>',
                    unsafe_allow_html=True
                )

                soap_sections = {
                    "Subjective": soap.get(
                        "subjective",
                        "Not available"
                    ),
                    "Objective": soap.get(
                        "objective",
                        "Not available"
                    ),
                    "Assessment": soap.get(
                        "assessment",
                        "Not available"
                    ),
                    "Plan": soap.get(
                        "plan",
                        "Not available"
                    )
                }

                for section_name, section_content in soap_sections.items():

                    with st.container(border=True):

                        card_title(section_name)

                        st.write(
                            section_content
                        )

                soap_text = create_soap_text(
                    soap
                )

                st.download_button(
                    label="Download SOAP Note",
                    data=soap_text,
                    file_name="soap_note.txt",
                    mime="text/plain"
                )

            # =================================================
            # REFERRAL TAB
            # =================================================

            with tab4:

                st.markdown(
                    '<div class="section-heading">'
                    'Referral Assessment'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="section-description">'
                    'Review whether a referral was identified in the consultation.'
                    '</div>',
                    unsafe_allow_html=True
                )

                if referral.get(
                    "referral_needed",
                    False
                ):

                    st.success(
                        "Referral identified."
                    )

                    with st.container(border=True):

                        card_title("Specialist")

                        st.write(
                            referral.get(
                                "specialist",
                                "Not available"
                            )
                        )

                        card_title("Referral Reason")

                        st.write(
                            referral.get(
                                "referral_reason",
                                "Not available"
                            )
                        )

                    with st.container(border=True):

                        card_title(
                            "Generated Referral Letter"
                        )

                        referral_letter = referral.get(
                            "referral_letter",
                            ""
                        )

                        st.text_area(
                            "Referral letter",
                            referral_letter,
                            height=350,
                            label_visibility="collapsed"
                        )

                        st.download_button(
                            label="Download Referral Letter",
                            data=referral_letter,
                            file_name="referral_letter.txt",
                            mime="text/plain"
                        )

                else:

                    st.info(
                        "No referral was identified in this consultation."
                    )

        except Exception as error:

            st.error(
                "An error occurred while processing the consultation."
            )

            st.exception(error)

        finally:

            if (
                audio_path
                and os.path.exists(audio_path)
            ):

                os.remove(audio_path)