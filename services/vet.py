from firebase_admin import firestore, auth
import logging

class VetService:
    @staticmethod
    def create(clinic_id, data):
        db = firestore.client()
        vet_email = data.get("email")
        new_vet = {
            "name": data.get("name"),
            "email": vet_email,
        }
        db.collection('clinics').document(clinic_id).collection('vets').document(vet_email).set(new_vet)

        # Create a new user in Firebase Authentication
        user = auth.create_user(
            email=vet_email,
            password=data.get("password"),  # Ensure password is in the data
        )

        # Map user ID to clinic ID directly
        db.collection('user_clinic_mapping').document(user.uid).set({"clinic_id": clinic_id})

        return vet_email

    @staticmethod
    def get_user_by_token(id_token):
        try:
            decoded_token = auth.verify_id_token(id_token)
            uid = decoded_token['uid']
            user = auth.get_user(uid)
            vet_email = user.email

            db = firestore.client()
            # Get clinic ID from user_clinic_mapping
            clinic_ref = db.collection('user_clinic_mapping').document(uid)
            clinic_doc = clinic_ref.get()

            if not clinic_doc.exists:
                raise Exception(f"Clinic mapping for user with UID {uid} not found")

            clinic_id = clinic_doc.to_dict().get("clinic_id")

            # Get vet data from the clinic/vets collection
            vet_ref = db.collection('clinics').document(clinic_id).collection('vets').document(vet_email)
            vet_doc = vet_ref.get()

            if not vet_doc.exists:
                raise Exception(f"Vet with email {vet_email} not found in clinic {clinic_id}")

            vet_data = vet_doc.to_dict()
            vet_data['clinic_id'] = clinic_id

            # Get clinic data
            clinic_ref = db.collection('clinics').document(clinic_id)
            clinic_doc = clinic_ref.get()
            if not clinic_doc.exists:
                raise Exception(f"Clinic with ID {clinic_id} not found")

            clinic_data = clinic_doc.to_dict()

            vet_data['clinic'] = clinic_data

            return {"user": vet_data}
        except Exception as e:
            logging.error("Error verifying ID token or fetching vet data: %s", e)
            return None
