# services/vet.py
from firebase_admin import firestore, auth
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
        
        # Create a new user in Firebase Authentication
        user = auth.create_user(
            email=data.get("email"),
            email_verified=False,
            password=data.get("password"),  # Ensure password is in the data
            display_name=data.get("name"),
            disabled=False
        )
        return vet_id
