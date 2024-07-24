# routes/api.py
import logging
from functools import wraps

from flask import Blueprint, request, jsonify, g

from services.clinic import ClinicService
from services.consultation import ConsultationService
from services.vet import VetService

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
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    name = request.form.get('name', 'Unnamed Consultation')

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        logging.info("Uploading file: %s, with name: %s", file.filename, name)
        # Here you would typically save the file and do something with the name
        # For example:
        # filename = secure_filename(file.filename)
        # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # save_consultation_name(name, filename)  # You'd need to implement this function
        return jsonify({"message": "File uploaded successfully", "filename": file.filename, "name": name}), 201
    else:
        return jsonify({"error": "File type not allowed"}), 400
