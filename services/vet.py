# services/vet.py
from firebase_admin import firestore
from uuid import uuid4

class VetService:
    @staticmethod
    def create(clinic_id, data):
        db = firestore.client()
        vet_id = str(uuid4())
        new_vet = {
            "name": data.get("name"),
            "email": data.get("email"),
            "phone": data.get("phone"),
            "address": data.get("address"),
            "specialization": data.get("specialization")
        }
        db.collection('clinics').document(clinic_id).collection('vets').document(vet_id).set(new_vet)
        return vet_id
