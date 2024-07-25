# routes/api.py
import logging
from datetime import datetime
from functools import wraps
from typing import Dict, List

from flask import Blueprint, request, jsonify, g
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from openai import OpenAI
from pydub import AudioSegment
from services.clinic import ClinicService
from services.consultation import ConsultationService
from services.vet import VetService
from werkzeug.utils import secure_filename
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate

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
        return f(*args, **kwargs)

    return decorated_function


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


@api_bp.route('/clinics/<clinic_id>/vets/<vet_id>/consultations', methods=['POST'])
def create_consultation(clinic_id, vet_id):
    data = request.json
    consultation_id = ConsultationService.create(clinic_id, vet_id, data)
    return jsonify({"message": "Consultation created", "consultation_id": consultation_id}), 201


@api_bp.route('/me', methods=['GET'])
@authorize
def me():
    return jsonify(g.user), 200


ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'wav', 'ogg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@api_bp.route('/upload-consultation', methods=['POST'])
def upload_consultation():
    file = get_uploaded_file(request)
    if file is None:
        return jsonify({"error": "No file part"}), 400

    name = request.form.get('name', 'Unnamed Consultation')
    language = request.form.get('language', 'swedish')  # Default to Swedish if not specified
    file_path = save_temp_file(file)

    try:
        chunks = chunk_audio(file_path)
        transcriptions = transcribe_chunks(chunks)

        full_transcription = " ".join(transcriptions)
        logging.info("Full transcription completed.")

        consultation = create_consultation_object(name, full_transcription, language)
        logging.info("Consultation object created: %s", consultation)

        # Clean up the temporary file
        os.remove(file_path)

        return jsonify(consultation), 200
    except Exception as e:
        logging.error(f"Error processing consultation: {str(e)}")
        # Clean up the temporary file in case of error
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"error": "Error processing consultation"}), 500


def get_uploaded_file(request):
    if 'file' not in request.files:
        return None
    return request.files['file']


def save_temp_file(file):
    filename = secure_filename(file.filename)
    file_path = os.path.join("/tmp", filename)
    file.save(file_path)
    return file_path


def transcribe_chunks(chunks):
    transcriptions = []
    for i, chunk in enumerate(chunks):
        chunk_path = os.path.join("/tmp", f"chunk_{i}.mp3")
        chunk.export(chunk_path, format="mp3")
        with open(chunk_path, "rb") as chunk_file:
            logging.info("Transcribing chunk %s", i)
            transcription = transcribe_audio(chunk_file)
            transcriptions.append(transcription)
    return transcriptions


def chunk_audio(file_path, chunk_length_ms=60000):
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


import openai
import os

openai.api_key = os.getenv('OPENAI_API_KEY')

# Define sections and subsections
SECTIONS = {
    "Allmän information": [
        "Besöksorsak", "Anamnes", "Medicinsk historia", "Kostvanor",
        "Vaccinationsstatus", "Tidigare operationer eller behandlingar"
    ],
    "Fysisk undersökning": [
        "Temperatur", "Hjärtfrekvens", "Andningsfrekvens", "Vikt",
        "Allmänt utseende", "Detaljerade undersökningsanteckningar"
    ],
    "Kliniska anteckningar": [
        "Subjektiva observationer", "Objektiva observationer",
        "Bedömning", "Plan"
    ]
}

# Create response schemas for each subsection
RESPONSE_SCHEMAS = [
    ResponseSchema(name=f"{section}_{subsection.replace(' ', '_')}",
                   description=f"The content for {subsection} in {section}")
    for section, subsections in SECTIONS.items()
    for subsection in subsections
]

# Initialize the structured output parser
PARSER = StructuredOutputParser.from_response_schemas(RESPONSE_SCHEMAS)

# Create the prompt template
TEMPLATE = """You are an AI that excels at extracting key points from text. Please identify and list the main points from the following text:

Extract the following sections and their subsections from the given text. If a particular piece of information is not present, output "Not specified".

Sections:
1. Allmän information
   - Besöksorsak
   - Anamnes
   - Medicinsk historia
   - Kostvanor
   - Vaccinationsstatus
   - Tidigare operationer eller behandlingar

2. Fysisk undersökning
   - Temperatur
   - Hjärtfrekvens
   - Andningsfrekvens
   - Vikt
   - Allmänt utseende
   - Detaljerade undersökningsanteckningar

3. Kliniska anteckningar
   - Subjektiva observationer
   - Objektiva observationer
   - Bedömning
   - Plan

Text: {transcription}

Please provide your response in {language}.

{format_instructions}
"""

PROMPT = ChatPromptTemplate.from_template(TEMPLATE)

# Initialize the language model
LLM = ChatOpenAI(model_name="gpt-4", temperature=0)


# Function to organize the parsed output into structured sections
def organize_sections(parsed_output: dict) -> dict:
    return {
        section: {
            subsection.replace('_', ' '): parsed_output.get(f"{section}_{subsection.replace(' ', '_')}",
                                                            "Not specified")
            for subsection in subsections
        }
        for section, subsections in SECTIONS.items()
    }


# Create the extraction chain using LLMChain
extraction_chain =  PROMPT | LLM | PARSER


def extract_sections(transcription: str, language: str = "swedish") -> dict:
    logging.info("PRASE", PARSER)
    result = extraction_chain.invoke(
        {"transcription": transcription, "language": language, "format_instructions": PARSER.get_format_instructions()})
    logging.info("Extracted sections: %s", result)
    return result


def create_consultation_object(name: str, full_transcription: str, language: str = "swedish") -> dict:
    sections = extract_sections(full_transcription, language)
    return {
        "name": name,
        "full_transcript": full_transcription,
        "sections": sections,
        "date": datetime.utcnow().isoformat()
    }
