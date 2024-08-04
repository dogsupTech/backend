# routes/api.py
import logging
from datetime import datetime
from functools import wraps
from typing import List

from flask import Blueprint, request, jsonify, g
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from openai import OpenAI
from pydub import AudioSegment
from services.clinic import ClinicService
from services.vet import VetService
from werkzeug.utils import secure_filename
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
import openai
import os

api_bp = Blueprint('api', __name__)


def authorize(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        authorization_header = request.headers.get('Authorization')
        logging.info("Authorization header: %s", authorization_header)

        id_token = authorization_header.replace("Bearer ", "").strip() if authorization_header else None
        logging.info("Extracted ID token: %s", id_token)

        if not id_token:
            return jsonify({"error": "Authorization token is missing or invalid"}), 401

        user = VetService.get_user_by_token(id_token)
        if not user:
            return jsonify({"error": "User not found"}), 404

        g.user = user
        g.clinic_id = user['user']['clinic_id']
        return f(*args, **kwargs)

    return decorated_function


@api_bp.route('/clinics/users', methods=['GET'])
@authorize
def get_all_users():
    clinic_id = g.clinic_id

    try:
        users = VetService.get_all_users_for_clinic(clinic_id)
        if users:
            return jsonify(users), 200
        else:
            return jsonify({"error": "No users found for this clinic"}), 404
    except Exception as e:
        logging.error(f"Error fetching users for clinic {clinic_id}: {str(e)}")
        return jsonify({"error": "Error fetching users for clinic"}), 500


@api_bp.route('/consultations/<consultation_id>', methods=['PUT'])
@authorize
def edit_consultation(consultation_id):
    user = g.user
    clinic_id = user['user']['clinic_id']
    vet_email = user['user']['email']

    # Get the updated consultation data from the request
    updated_data = request.json
    logging.info("Updating consultation with data: %s", updated_data)

    try:
        # Fetch the existing consultation
        existing_consultation = VetService.get_consultation(clinic_id, vet_email, consultation_id)

        if not existing_consultation:
            return jsonify({"error": "Consultation not found"}), 404

        # Save the updated consultation
        VetService.update_consultation(clinic_id, vet_email, consultation_id, updated_data)

        return jsonify({"message": "Consultation updated successfully",
                        "consultation": {**existing_consultation, **updated_data}}), 200

    except Exception as e:
        logging.error(f"Error updating consultation: {str(e)}")
        return jsonify({"error": "Error updating consultation"}), 500


# endpoint to get all consultations for a vet
@api_bp.route('/consultations', methods=['GET'])
@authorize
def get_consultations():
    user = g.user
    clinic_id = user['user']['clinic_id']
    vet_email = user['user']['email']
    consultations = VetService.get_consultations(clinic_id, vet_email)
    return jsonify(consultations), 200


# endpoint to get a consultation by id
@api_bp.route('/consultations/<consultation_id>', methods=['GET'])
@authorize
def get_consultation(consultation_id):
    user = g.user
    clinic_id = user['user']['clinic_id']
    vet_email = user['user']['email']
    consultation = VetService.get_consultation(clinic_id, vet_email, consultation_id)
    return jsonify(consultation), 200


@api_bp.route('/clinics', methods=['POST'])
def create_clinic():
    data = request.json
    clinic_id = ClinicService.create(data)
    return jsonify({"message": "Clinic created", "clinic_id": clinic_id}), 201


@api_bp.route('/clinics/<clinic_id>/vets', methods=['POST'])
def create_vet(clinic_id):
    data = request.json
    vet_id = VetService.create(clinic_id, data)
    return jsonify({"message": "Vet created", "vet_id": vet_id}), 201


@api_bp.route('/me', methods=['GET'])
@authorize
def me():
    return jsonify(g.user), 200


@api_bp.route('/upload-consultation', methods=['POST'])
@authorize
def upload_consultation():
    logging.info("request.files: %s", request.files)
    file = get_uploaded_file(request)
    if file is None:
        logging.warning("No file part in the request.")
        return jsonify({"error": "No file part"}), 400

    name = request.form.get('name', 'Unnamed Consultation')
    language = request.form.get('language', 'swedish')  # Default to Swedish if not specified
    file_path = save_temp_file(file)
    logging.info("File saved to: %s", file_path)
    try:
        logging.info("Processing file: %s", file_path)
        chunks = chunk_audio(file_path)
        logging.info("Audio file chunked into %d parts.", len(chunks))

        transcriptions = transcribe_chunks(chunks)
        full_transcription = " ".join(transcriptions)
        logging.info("Full transcription completed. Length of transcription: %d characters", len(full_transcription))

        consultation = create_consultation_object(name, full_transcription, language)
        logging.info("Consultation object created: %s", consultation)

        # Get user info from the authorization token
        user = g.user
        clinic_id = user['user']['clinic_id']
        vet_email = user['user']['email']
        logging.info("Saving consultation for clinic_id: %s, vet_email: %s", clinic_id, vet_email)

        # Save the consultation
        consultation_id = VetService.save_consultation(clinic_id, vet_email, name, consultation)
        logging.info("Consultation successfully saved to database.")

        # Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info("Temporary file removed: %s", file_path)
        else:
            logging.warning("Temporary file not found for removal: %s", file_path)

        consultation['id'] = consultation_id
        return jsonify(consultation), 200
    except Exception as e:
        logging.error("Error processing consultation. Exception: %s", str(e))
        # Clean up the temporary file in case of error
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info("Temporary file removed after error: %s", file_path)
        return jsonify({"error": "Error processing consultation"}), 500


ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'wav', 'ogg', 'm4a'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_uploaded_file(request):
    if 'file' not in request.files:
        return None
    return request.files['file']


def save_temp_file(file):
    filename = secure_filename(file.filename)
    file_path = os.path.join("/tmp", filename)
    file.save(file_path)
    return file_path


def transcribe_chunks(chunks: List[AudioSegment]) -> List[str]:
    transcriptions = []
    for i, chunk in enumerate(chunks):
        chunk_path = os.path.join("/tmp", f"chunk_{i}.mp3")
        chunk.export(chunk_path, format="mp3")
        with open(chunk_path, "rb") as chunk_file:
            logging.info("Transcribing chunk %s", i)
            transcription = transcribe_audio(chunk_file)
            transcriptions.append(transcription)
    return transcriptions


def chunk_audio(file_path: str, chunk_length_ms: int = 60000) -> List[AudioSegment]:
    audio = AudioSegment.from_file(file_path)
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    return chunks


openai_client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY") 
    # api_key="My API Key",
)


def transcribe_audio(file):
    try:
        transcription = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=file
        )
        logging.info("Transcription: %s", transcription)
        return transcription.text
    except Exception as e:
        logging.error("Error transcribing audio: %s", e)
        raise e


openai.api_key = os.getenv('OPENAI_API_KEY')

# Hardcoded response schemas with English section names mapped to Swedish
RESPONSE_SCHEMAS = [
    ResponseSchema(name="General_Information_Visit_Reason",
                   description="The content for Visit Reason in General Information"),
    ResponseSchema(name="General_Information_Anamnesis",
                   description="The content for Anamnesis in General Information"),
    ResponseSchema(name="General_Information_Medical_History",
                   description="The content for Medical History in General Information"),
    ResponseSchema(name="General_Information_Dietary_Habits",
                   description="The content for Dietary Habits in General Information"),
    ResponseSchema(name="General_Information_Vaccination_Status",
                   description="The content for Vaccination Status in General Information"),
    ResponseSchema(name="General_Information_Previous_Surgeries_or_Treatments",
                   description="The content for Previous Surgeries or Treatments in General Information"),
    ResponseSchema(name="Physical_Examination_Temperature",
                   description="The content for Temperature in Physical Examination"),
    ResponseSchema(name="Physical_Examination_Heart_Rate",
                   description="The content for Heart Rate in Physical Examination"),
    ResponseSchema(name="Physical_Examination_Respiratory_Rate",
                   description="The content for Respiratory Rate in Physical Examination"),
    ResponseSchema(name="Physical_Examination_Weight",
                   description="The content for Weight in Physical Examination"),
    ResponseSchema(name="Physical_Examination_General_Appearance",
                   description="The content for General Appearance in Physical Examination"),
    ResponseSchema(name="Physical_Examination_Detailed_Examination_Notes",
                   description="The content for Detailed Examination Notes in Physical Examination"),
    ResponseSchema(name="Clinical_Notes_Subjective_Observations",
                   description="The content for Subjective Observations in Clinical Notes"),
    ResponseSchema(name="Clinical_Notes_Objective_Observations",
                   description="The content for Objective Observations in Clinical Notes"),
    ResponseSchema(name="Clinical_Notes_Assessment",
                   description="The content for Assessment in Clinical Notes"),
    ResponseSchema(name="Clinical_Notes_Plan",
                   description="The content for Plan in Clinical Notes"),
    ResponseSchema(name="Formatted_Transcription",
                   description="Formatted transcription in JSON format with roles identified as either 'Vet' or 'Patient'")
]

# Initialize the structured output parser
PARSER = StructuredOutputParser.from_response_schemas(RESPONSE_SCHEMAS)

# Create the prompt template
TEMPLATE = """You are an AI that excels at extracting key points and formatting text. Please identify and list the main points from the following veterinary consultation transcription.:

Extract the following sections and their subsections from the given text. Convert the extracted sections to veterinarian tonality. If a particular piece of information is not present, output "Not specified".

Sections:
1. General Information
   - Visit Reason
   - Anamnesis
   - Medical History
   - Dietary Habits
   - Vaccination Status
   - Previous Surgeries or Treatments

2. Physical Examination
   - Temperature
   - Heart Rate
   - Respiratory Rate
   - Weight
   - General Appearance
   - Detailed Examination Notes

3. Clinical Notes
   - Subjective Observations
   - Objective Observations
   - Assessment
   - Plan 
   
4. Formatted Transcription
  - Format the whole text with roles identified as either 'Vet' or 'Patient'. Output in the following JSON format:
    [
      {{"Vet": "Vet's dialogue"}},
      {{"Patient": "Patient's dialogue"}}
    ]

Text: {transcription}

Please provide your response in {language} with a veterinary journal tonality.

{format_instructions}
"""

PROMPT = ChatPromptTemplate.from_template(TEMPLATE)

# Initialize the language model
LLM = ChatOpenAI(model_name="gpt-4o", temperature=0)


# Function to organize the parsed output into structured sections
def organize_sections(parsed_output: dict) -> dict:
    return {
        "general_information": {
            "visit_reason": parsed_output.get("General_Information_Visit_Reason", "Not specified"),
            "anamnesis": parsed_output.get("General_Information_Anamnesis", "Not specified"),
            "medical_history": parsed_output.get("General_Information_Medical_History", "Not specified"),
            "dietary_habits": parsed_output.get("General_Information_Dietary_Habits", "Not specified"),
            "vaccination_status": parsed_output.get("General_Information_Vaccination_Status", "Not specified"),
            "previous_surgeries_or_treatments": parsed_output.get(
                "General_Information_Previous_Surgeries_or_Treatments", "Not specified")
        },
        "physical_examination": {
            "temperature": parsed_output.get("Physical_Examination_Temperature", "Not specified"),
            "heart_rate": parsed_output.get("Physical_Examination_Heart_Rate", "Not specified"),
            "respiratory_rate": parsed_output.get("Physical_Examination_Respiratory_Rate", "Not specified"),
            "weight": parsed_output.get("Physical_Examination_Weight", "Not specified"),
            "general_appearance": parsed_output.get("Physical_Examination_General_Appearance", "Not specified"),
            "detailed_examination_notes": parsed_output.get("Physical_Examination_Detailed_Examination_Notes",
                                                            "Not specified")
        },
        "clinical_notes": {
            "subjective_observations": parsed_output.get("Clinical_Notes_Subjective_Observations", "Not specified"),
            "objective_observations": parsed_output.get("Clinical_Notes_Objective_Observations", "Not specified"),
            "assessment": parsed_output.get("Clinical_Notes_Assessment", "Not specified"),
            "plan": parsed_output.get("Clinical_Notes_Plan", "Not specified")
        },
        "formatted_transcription": parsed_output.get("Formatted_Transcription", "Not specified")
    }


# Create the extraction chain using LLMChain
extraction_chain =  PROMPT | LLM | PARSER


def extract_sections(transcription: str, language: str = "swedish") -> dict:
    result = extraction_chain.invoke(
        {"transcription": transcription, "language": language, "format_instructions": PARSER.get_format_instructions()})
    logging.info("Extracted sections: %s", result)
    return organize_sections(result)


def create_consultation_object(name: str, full_transcription: str, language: str = "swedish") -> dict:
    sections = extract_sections(full_transcription, language)
    return {
        "name": name,
        "full_transcript": full_transcription,
        "sections": sections,
        "date": datetime.utcnow().isoformat()
    }
