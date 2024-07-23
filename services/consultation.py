# services/consultation.py
from firebase_admin import firestore
from uuid import uuid4

class ConsultationService:
    @staticmethod
    def create(clinic_id, vet_id, data):
        db = firestore.client()
        consultation_id = str(uuid4())
        new_consultation = {
            "date": data.get("date"),
            "petId": data.get("petId"),
            "reason": data.get("reason"),
            "diagnosis": data.get("diagnosis"),
            "treatment": data.get("treatment")
        }
        db.collection('clinics').document(clinic_id).collection('vets').document(vet_id).collection('consultations').document(consultation_id).set(new_consultation)
        return consultation_id
