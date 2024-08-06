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

from services_old.account import AccountService, UserSaveError, Patient, Intake
from services_old.llm import LLM
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
    facialExpressionLoader = PyPDFLoader('services_old/FacialExpressionsDogs.pdf')
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



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8081))  # Default to 8081 if no PORT env var is set
    app.run(host='0.0.0.0', port=port, debug=True)
