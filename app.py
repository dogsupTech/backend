import io

import PyPDF2
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

from services.account import AccountService
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
    except Exception as e:
        logging.error("Error saving paper to Firestore: %s", e)
        return jsonify({"error": "Failed to save paper"}), 500

    # Return a successful response with the Cloudinary URL and summary
    return jsonify({"cloudinary_url": cloudinary_url, "summary": summary}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8081))  # Default to 8081 if no PORT env var is set
    app.run(host='0.0.0.0', port=port, debug=True)
