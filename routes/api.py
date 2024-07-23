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

