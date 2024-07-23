# routes/api.py
from flask import Blueprint, request, jsonify

from services.clinic import ClinicService
from services.consultation import ConsultationService
from services.vet import VetService

api_bp = Blueprint('api', __name__)

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
