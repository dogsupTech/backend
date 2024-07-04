import uuid
from datetime import datetime
import openai
from openai import OpenAI
from pydub.utils import which
from dotenv import load_dotenv
from flask import Flask, request, jsonify, Response, g
from flask_cors import CORS
import os
import logging
import cloudinary
import cloudinary.uploader
import cloudinary.api

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydub import AudioSegment
from werkzeug.utils import secure_filename

from services.account import AccountService, UserSaveError, Patient, Intake
from services.llm import LLM
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from functools import wraps

app = Flask(__name__)
CORS(app)


def create_or_load_vector_store():
    if os.path.exists('llm_faiss_index'):
        # Pass the OpenAIEmbeddings instance when loading the local FAISS index
        return FAISS.load_local('llm_faiss_index', OpenAIEmbeddings(), allow_dangerous_deserialization=True)

    # if not exists, create the FAISS index from the documents and embeddings and save it
    facialExpressionLoader = PyPDFLoader('./services/FacialExpressionsDogs.pdf')
    pages = facialExpressionLoader.load()  # Set the chunk size and overlap for splitting the text from the documents

    # split thje text
    chunk_size = 1000  # Number of characters in each text chunk
    chunk_overlap = 100  # Number of overlapping characters between consecutive chunks
    # Initialize the text splitter with default separators if they're not set
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap, length_function=len)
    texts = text_splitter.split_documents(pages)
    # Create the FAISS index from the documents and embeddings
    vector_db = FAISS.from_documents(texts, OpenAIEmbeddings())
    vector_db.save_local('llm_faiss_index')
    return vector_db


os.environ['OPENAI_API_KEY'] = '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'
llm = LLM(model_name="gpt-4o", api_key=os.environ['OPENAI_API_KEY'], vector_db=create_or_load_vector_store())

# Set up logging to a file
log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format, handlers=[
    logging.FileHandler("app.log"),
    logging.StreamHandler()
])

cred = credentials.Certificate('./serviceAccountKey.json')
fbApp = firebase_admin.initialize_app(cred)
db = firestore.client()

load_dotenv()

account_service = AccountService(db)


def check_request_count(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = g.user
        if user.request_count <= 0:
            return jsonify({"error": "Insufficient request count"}), 403
        response = f(*args, **kwargs)
        # Decrement request count after the response is processed
        account_service.decrement_request_count(user.uid)
        return response

    return decorated_function


def authorize(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        authorization_header = request.headers.get('Authorization')
        logging.info("Authorization header: %s", authorization_header)

        id_token = authorization_header.replace("Bearer ", "").strip() if authorization_header else None
        logging.info("Extracted ID token: %s", id_token)

        if not id_token:
            return jsonify({"error": "Authorization token is missing or invalid"}), 401

        user = account_service.get_saved_user(id_token)
        if not user:
            return jsonify({"error": "User not found"}), 404

        g.user = user
        return f(*args, **kwargs)

    return decorated_function


# Configuration
CLOUDINARY_CLOUD_NAME = os.getenv('CLOUDINARY_CLOUD_NAME')
CLOUDINARY_API_KEY = os.getenv('CLOUDINARY_API_KEY')
CLOUDINARY_API_SECRET = os.getenv('CLOUDINARY_API_SECRET')

if not CLOUDINARY_CLOUD_NAME or not CLOUDINARY_API_KEY or not CLOUDINARY_API_SECRET:
    raise ValueError("Cloudinary credentials must be set in environment variables")

cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET
)


@app.route('/chat', methods=['POST'])
@authorize
@check_request_count
def chat_endpoint():
    user_input = request.form.get('input')
    file = request.files.get('file')
    cloudinary_url = None

    if file:
        try:
            upload_result = cloudinary.uploader.upload(file)
            cloudinary_url = upload_result['secure_url']
            logging.info("Uploaded to Cloudinary: %s", cloudinary_url)
        except Exception as e:
            logging.error("Error uploading to Cloudinary: %s", e)
            return jsonify({"error": "Failed to upload to Cloudinary"}), 500  # Log the incoming request

    # Check if the request contains user input
    if user_input is None:
        return jsonify({"error": "Input is missing"}), 400

    user = g.user

    # Pass the user's dog to stream_openai_chat
    chat_response = llm.stream_openai_chat(user.dog, user_input, user.uid, image_url=cloudinary_url)

    # Return the chat response as a text event stream
    return Response(chat_response, content_type='text/event-stream')


@app.route('/me', methods=['GET'])
@authorize
def get_me():
    logging.info("GET /me")
    user = g.user
    return jsonify({"user": user.to_dict()}), 200


@app.route('/upload-pdf', methods=['POST'])
@authorize
def upload_pdf():
    file = request.files.get('file')
    title = request.form.get('title')
    author = request.form.get('author')
    cloudinary_url = None

    if not file:
        logging.error("No file part in the request")
        return jsonify({"error": "No file provided"}), 400

    if not title or not author:
        logging.error("Missing paper details")
        return jsonify({"error": "Missing paper details"}), 400

    try:
        # Upload file to Cloudinary
        upload_result = cloudinary.uploader.upload(file)
        cloudinary_url = upload_result['secure_url']
        logging.info("Uploaded to Cloudinary: %s", cloudinary_url)

        if not cloudinary_url:
            logging.error("Failed to obtain Cloudinary URL after upload")
            return jsonify({"error": "Failed to obtain Cloudinary URL"}), 500

    except Exception as e:
        logging.error("Error uploading to Cloudinary: %s", e)
        return jsonify({"error": "Failed to upload to Cloudinary"}), 500

    try:
        summary = llm.generate_summary(file)
    except Exception as e:
        logging.error("Error processing PDF and generating summary: %s", e)
        return jsonify({"error": "Failed to process PDF and generate summary"}), 500

    user = g.user  # Assuming this is necessary for authorization or further processing
    try:
        account_service.save_paper(user.uid, title, author, summary, cloudinary_url)
    except UserSaveError as e:
        logging.error("Error saving paper to Firestore: %s", e)
        return jsonify({"error": "Failed to save paper"}), 500

    # Return a successful response with the Cloudinary URL and summary
    return jsonify({"cloudinary_url": cloudinary_url, "summary": summary}), 200


@app.route('/saved-papers', methods=['GET'])
@authorize
def get_saved_papers():
    user = g.user
    try:
        papers = account_service.get_papers(user.uid)
        return jsonify({"papers": papers}), 200
    except Exception as e:
        logging.error("Error retrieving saved papers: %s", e)
        return jsonify({"error": "Failed to retrieve saved papers"}), 500


@app.route('/all-papers', methods=['GET'])
def get_all_papers():
    try:
        papers = account_service.get_all_papers()
        return jsonify({"papers": papers}), 200
    except Exception as e:
        logging.error("Error retrieving all papers: %s", e)
        return jsonify({"error": "Failed to retrieve all papers"}), 500


@app.route('/new-patient', methods=['POST'])
@authorize
def add_patient():
    try:
        # Extract the patient data from the request
        data = request.json
        patient_name = data.get('name')

        if not patient_name:
            return jsonify({"error": "Invalid request: 'name' is required"}), 400

        # Create a Patient object
        patient = Patient(name=patient_name,
                          created_at=datetime.utcnow(),
                          updated_at=datetime.utcnow(),
                          uid=str(uuid.uuid4()))

        # Save the patient using AccountService
        account_service.save_patient(g.user.uid, patient)

        return jsonify({"message": "Patient added successfully"}), 200
    except Exception as e:
        logging.error("Error adding patient: %s", e)
        return jsonify({"error": "Failed to add patient"}), 500


@app.route('/update-patient', methods=['POST'])
@authorize
def update_patient():
    try:
        # Extract the patient data from the request
        data = request.json
        patient_id = data.get('patient_id')
        patient_name = data.get('name')

        if not patient_id or not patient_name:
            return jsonify({"error": "Invalid request: 'patient_id' and 'name' are required"}), 400

        # Create a Patient object
        patient = Patient(name=patient_name, updated_at=datetime.utcnow())

        # Update the patient using AccountService
        account_service.save_patient(g.user.uid, patient)

        return jsonify({"message": "Patient updated successfully"}), 200
    except Exception as e:
        logging.error("Error updating patient: %s", e)
        return jsonify({"error": "Failed to update patient"}), 500


@app.route('/patients', methods=['GET'])
@authorize
def get_all_patients():
    try:
        patients = account_service.get_all_patients(g.user.uid)
        return jsonify({"patients": patients}), 200
    except Exception as e:
        logging.error("Error getting patients: %s", e)
        return jsonify({"error": "Failed to get patients"}), 500


@app.route('/evals', methods=['GET'])
def run_all_evals():
    try:
        dataset = "ds-infectious_disease"  # Replace with your actual dataset name
        experiment_prefix = "ds-infectious_disease-eval"  # Replace with your actual experiment prefix

        results = llm.run_evals(dataset=dataset, experiment_prefix=experiment_prefix)
        return jsonify({"run": "ok"}), 200
    
    except Exception as e:
        logging.error("Error running evals: %s", e)
        return jsonify({"error": "Failed to run evals"}), 500
    
    
openai.api_key = os.getenv("OPENAI_API_KEY")
AudioSegment.converter = which("ffmpeg")


@app.route('/intake', methods=['POST'])
@authorize
def intake():
    try:
        patient_id = request.form.get('patient_id')
        file = request.files.get('file')

        if not patient_id or not file:
            return jsonify({"error": "Invalid request: 'patient_id' and 'file' are required"}), 400

        # Save the file temporarily
        filename = secure_filename(file.filename)
        file_path = os.path.join("/tmp", filename)
        file.save(file_path)

        # Split the file into chunks if it's larger than 25MB
        chunks = chunk_audio(file_path)

        # Transcribe each chunk
        transcriptions = []
        for i, chunk in enumerate(chunks):
            chunk_path = os.path.join("/tmp", f"chunk_{i}.mp3")
            chunk.export(chunk_path, format="mp3")
            with open(chunk_path, "rb") as chunk_file:
                logging.info("Transcribing chunk %s", i)
                transcription = transcribe_audio(chunk_file)
                transcriptions.append(transcription)

        full_transcription = " ".join(transcriptions)
        logging.info("Full transcription completed.")
        # Process the transcription
        final_notes, sections = processDictation(full_transcription)

        # Analyze sentiment and generate abstract summary
        try:
            intake = Intake(
                date=datetime.utcnow(),
                transcription=full_transcription,
                notes=final_notes,
                preparation_intro=sections.get('Preparation and Introduction'),
                history_taking=sections.get('History Taking'),
                physical_exam=sections.get('Physical Examination'),
                diagnostic_testing=sections.get('Diagnostic Testing'),
                diagnosis=sections.get('Diagnosis'),
                treatment_plan=sections.get('Treatment Plan'),
                client_education=sections.get('Client Education and Instructions'),
                conclusion=sections.get('Conclusion')
            )
            logging.info("Intake: %s", intake.to_dict())
            account_service.save_intake(g.user.uid, patient_id, intake)
            logging.info("Intake saved for patient %s", patient_id)
        except Exception as e:
            logging.error("Error saving intake: %s", e)
            return jsonify({"error": "Failed to save intake"}), 500

        return jsonify({
            "message": "Intake file uploaded, transcribed, and associated successfully",
            "transcription": full_transcription,
            "notes": final_notes,
            "sections": sections
        }), 200

    except Exception as e:
        logging.error("Error processing intake: %s", e)
        return jsonify({"error": "Failed to process intake"}), 500


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


def processDictation(transcription):
    sections = organizeIntoSections(transcription)
    summarizedSections = summarizeKeyPoints(sections)
    structuredTemplate = formatIntoTemplate(summarizedSections)
    finalNotes = reviewAndHighlight(structuredTemplate)
    return finalNotes, sections


def organizeIntoSections(text):
    sections = {
        "Preparation and Introduction": extractSection(text, "describe the preparation and introduction of the consultation"),
        "History Taking": extractSection(text, "extract the history taking part of the consultation"),
        "Physical Examination": extractSection(text, "describe the physical examination part of the consultation"),
        "Diagnostic Testing": extractSection(text, "extract any diagnostic tests mentioned"),
        "Diagnosis": extractSection(text, "describe the diagnosis given during the consultation"),
        "Treatment Plan": extractSection(text, "extract the treatment plan discussed"),
        "Client Education and Instructions": extractSection(text, "describe the client education and instructions provided"),
        "Conclusion": extractSection(text, "summarize the conclusion of the consultation")
    }
    return sections

def extractSection(text, sectionDescription):
    response = openai_client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": f"You are an AI that extracts specific sections from a veterinary consultation transcript. Please extract the section that {sectionDescription}:"
            },
            {
                "role": "user",
                "content": text
            }
        ]
    )
    return response.choices[0].message.content.strip()


def callOpenAISummarizeAPI(text):
    response = openai_client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are an AI skilled in summarizing texts. Please provide a concise summary of the following content:"
            },
            {
                "role": "user",
                "content": text
            }
        ]
    )
    return response.choices[0].message.content.strip()

def generate_key_points_extraction(transcription):
    response = openai_client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are an AI that excels at extracting key points from text. Please identify and list the main points from the following text:"
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response.choices[0].message.content

# Function to format the summarized sections into a structured template
def formatIntoTemplate(sections):
    template = f"""
    **Preparation and Introduction:**
    {sections['Preparation and Introduction']}
    
    **History Taking:**
    {sections['History Taking']}
    
    **Physical Examination:**
    {sections['Physical Examination']}
    
    **Diagnostic Testing:**
    {sections['Diagnostic Testing']}
    
    **Diagnosis:**
    {sections['Diagnosis']}
    
    **Treatment Plan:**
    {sections['Treatment Plan']}
    
    **Client Education and Instructions:**
    {sections['Client Education and Instructions']}
    
    **Conclusion:**
    {sections['Conclusion']}
    """
    return template



# Function to review and highlight important information using OpenAI API
def reviewAndHighlight(text):
    highlightedTemplate = callOpenAIHighlightAPI(text)
    return highlightedTemplate

def callOpenAIHighlightAPI(text):
    response = openai_client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are an AI specialized in highlighting critical information. Please highlight important information in the following content without removing any information:"
            },
            {
                "role": "user",
                "content": text
            }
        ]
    )
    return response.choices[0].message.content.strip()


def summarizeKeyPoints(sections):
    for key, value in sections.items():
        sections[key] = callOpenAISummarizeAPI(value)
    return sections


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8081))  # Default to 8081 if no PORT env var is set
    app.run(host='0.0.0.0', port=port, debug=True)
