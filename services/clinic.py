# services/clinic.py
from firebase_admin import firestore
from uuid import uuid4

class ClinicService:
    @staticmethod
    def create(data):
        db = firestore.client()
        clinic_id = str(uuid4())
        new_clinic = {
            "name": data.get("name"),
            "location": data.get("location"),
            "contactInfo": data.get("contactInfo")
        }
        db.collection('clinics').document(clinic_id).set(new_clinic)
        return clinic_id
