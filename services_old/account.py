import datetime
import logging
from uuid import uuid4
from firebase_admin import firestore
from firebase_admin import auth
from components.llm import Dog


class Intake:
    def __init__(self, date, transcription, notes, preparation_intro=None, history_taking=None, physical_exam=None, diagnostic_testing=None, diagnosis=None, treatment_plan=None, client_education=None, conclusion=None, file_url=None):
        self.date = date
        self.transcription = transcription
        self.notes = notes
        self.preparation_intro = preparation_intro
        self.history_taking = history_taking
        self.physical_exam = physical_exam
        self.diagnostic_testing = diagnostic_testing
        self.diagnosis = diagnosis
        self.treatment_plan = treatment_plan
        self.client_education = client_education
        self.conclusion = conclusion
        self.file_url = file_url

    def to_dict(self):
        return {
            "date": self.date.isoformat(),
            "transcription": self.transcription,
            "notes": self.notes,
            "preparation_intro": self.preparation_intro,
            "history_taking": self.history_taking,
            "physical_exam": self.physical_exam,
            "diagnostic_testing": self.diagnostic_testing,
            "diagnosis": self.diagnosis,
            "treatment_plan": self.treatment_plan,
            "client_education": self.client_education,
            "conclusion": self.conclusion,
            "file_url": self.file_url,
        }

    def __repr__(self):
        return (f"Intake(date={self.date}, transcription={self.transcription}, notes={self.notes}, "
                f"preparation_intro={self.preparation_intro}, history_taking={self.history_taking}, "
                f"physical_exam={self.physical_exam}, diagnostic_testing={self.diagnostic_testing}, "
                f"diagnosis={self.diagnosis}, treatment_plan={self.treatment_plan}, "
                f"client_education={self.client_education}, conclusion={self.conclusion}, "
                f"file_url={self.file_url})")


class User:
    def __init__(self, email: str, uid: str, email_verified: bool,
                 updated_at: datetime.datetime, vet_ai_waitlist: bool, vet_ai_is_white_listed: bool,
                 dog: Dog = None, roles: list = None, request_count: int = 0, paper_ids: list = None, **kwargs):
        self.email = email
        self.uid = uid
        self.email_verified = email_verified
        self.updated_at = updated_at
        self.vet_ai_waitlist = vet_ai_waitlist
        self.vet_ai_is_white_listed = vet_ai_is_white_listed
        self.dog = dog
        self.roles = roles if roles else ['user']
        self.request_count = request_count
        self.paper_ids = paper_ids if paper_ids else []

    def __repr__(self):
        return (f"User(email={self.email}, uid={self.uid},"
                f"email_verified={self.email_verified}, updated_at={self.updated_at}, "
                f"vet_ai_waitlist={self.vet_ai_waitlist}, vet_ai_is_white_listed={self.vet_ai_is_white_listed}, "
                f"dog={self.dog}, roles={self.roles}, request_count={self.request_count}, paper_ids={self.paper_ids})")

    def to_dict(self):
        return {
            "email": self.email,
            "uid": self.uid,
            "email_verified": self.email_verified,
            "updated_at": self.updated_at.isoformat(),
            "vet_ai_waitlist": self.vet_ai_waitlist,
            "vet_ai_is_white_listed": self.vet_ai_is_white_listed,
            "dog": self.dog.to_dict() if self.dog else None,
            "roles": self.roles,
            "request_count": self.request_count,
            "paper_ids": self.paper_ids
        }


class Patient:
    def __init__(self, name: str, created_at: datetime, updated_at: datetime, uid: str):
        self.name = name
        self.created_at = created_at
        self.updated_at = updated_at
        self.uid = uid

    def to_dict(self):
        return {
            "name": self.name,
            "uid": self.uid,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    def __repr__(self):
        return f"Patient(name={self.name}, created_at={self.created_at}, updated_at={self.updated_at}), uid={self.uid})"


class Clinic:
    def __init__(self, name: str, address: str, phone: str, email: str, website: str):
        self.name = name
        self.address = address
        self.phone = phone
        self.email = email
        self.website = website
    
    def to_dict(self):
        return {
            "name": self.name,
            "address": self.address,
            "phone": self.phone,
            "email": self.email,
            "website": self.website
        }
    
    def create_consultation(self, patient: Patient, intake: Intake):
        pass
    

class ClinicService:
    def __init__(self, db):
        self._firestore = db

    def save_clinic(self, clinic: Clinic):
        try:
            logging.info(f"Saving clinic: {clinic.name}")
            clinic_data = clinic.to_dict()
            self._firestore.collection('clinics').document(clinic.name).set(clinic_data)
            logging.info(f"Clinic saved successfully: {clinic.name}")
        except Exception as error:
            logging.error('Error saving clinic:', error)
            raise ClinicSaveError() from error
        

class AccountService:
    def __init__(self, db):
        self._firestore = db
        self._user_requests_per_month = 100
        self._research_requests_per_month = 1000

    def get_saved_user(self, id_token):
        try:
            decoded_token = auth.verify_id_token(id_token)
            uid = decoded_token['uid']
            user_ref = self._firestore.collection('users').document(uid)
            user_doc = user_ref.get()
            if user_doc.exists:
                user_data = user_doc.to_dict()
                logging.info("User data: %s", user_data)

                # Check if 'roles' exists, if not, set it to ['user'] and set the request count 
                if 'roles' not in user_data:
                    user_data['roles'] = ['user']
                if 'request_count' not in user_data:
                    user_data['request_count'] = self._user_requests_per_month

                user_ref.update({'roles': user_data['roles'], 'request_count': user_data['request_count']})

                dog_data = user_data.pop('dog', None)
                dog = Dog(**dog_data) if dog_data else None

                # TODO: needs to be fixed
                user_data['updated_at'] = user_data.pop('updatedAt')  # Change the key here
                user = User(dog=dog, **user_data)  # Pass the user_data dictionary with updated keys
                return user
            return None
        except Exception as e:
            logging.error("Authorization failed: %s", str(e))
            return None

    def decrement_request_count(self, uid):
        try:
            user_ref = self._firestore.collection('users').document(uid)
            user_doc = user_ref.get()
            if user_doc.exists:
                current_count = user_doc.get('request_count')
                if current_count > 0:
                    user_ref.update({'request_count': current_count - 1})
                    logging.info(f"Decremented request count for user {uid} to {current_count - 1}")
        except Exception as e:
            logging.error(f"Failed to decrement request count for user {uid}: {str(e)}")

    def save_paper(self, uid: str, title: str, author: str, summary: str, pdf_url: str):
        try:
            logging.info(f"Saving paper for user: {uid}")
            paper_id = str(uuid4())
            paper_data = {
                "title": title,
                "author": author,
                "summary": summary,
                "url": pdf_url,
                "saved_by": uid,
                "created_at": firestore.SERVER_TIMESTAMP  # Correct usage
            }
            self._firestore.collection('papers').document(paper_id).set(paper_data)

            # Update the user's paper_ids
            user_ref = self._firestore.collection('users').document(uid)
            user_ref.update({"paper_ids": firestore.ArrayUnion([paper_id])})

            logging.info(f"Paper saved successfully for user: {uid}")
        except Exception as error:
            logging.error('Error saving paper:', error)
            raise UserSaveError() from error

    def get_papers(self, uid: str):
        try:
            user_ref = self._firestore.collection('users').document(uid)
            user_doc = user_ref.get()
            if user_doc.exists:
                user_data = user_doc.to_dict()
                paper_ids = user_data.get('paper_ids', [])

                papers = []
                for paper_id in paper_ids:
                    paper_ref = self._firestore.collection('papers').document(paper_id)
                    paper_doc = paper_ref.get()
                    if paper_doc.exists:
                        papers.append(paper_doc.to_dict())

                return papers
            return []
        except Exception as error:
            logging.error('Error getting papers:', error)
            return []

    def get_all_papers(self):
        try:
            papers_ref = self._firestore.collection('papers')
            papers = papers_ref.stream()
            all_papers = [paper.to_dict() for paper in papers]
            return all_papers
        except Exception as error:
            logging.error('Error getting all papers:', error)
            return []

    def save_patient(self, uid: str, patient: Patient):
        try:
            logging.info(f"Saving patient for user: {uid}")
            patient_data = patient.to_dict()
            logging.info("Patient data: %s", patient_data)
            # Save the patient to Firestore under the user's patients collection
            self._firestore.collection('users').document(uid).collection('patients').document(patient.uid).set(
                patient_data)

            logging.info(f"Patient saved successfully for user: {uid}")
        except Exception as error:
            logging.error('Error saving patient:', error)
            raise UserSaveError() from error

    def get_all_patients(self, uid: str):
        try:
            logging.info(f"Retrieving all patients for user: {uid}")
            patients_ref = self._firestore.collection('users').document(uid).collection('patients')
            patients = patients_ref.stream()
            patients_list = []

            for patient in patients:
                patient_data = patient.to_dict()
                patient_id = patient.id

                # Retrieve intakes for each patient
                intakes_ref = self._firestore.collection('users').document(uid).collection('patients').document(patient_id).collection('intakes')
                intakes = intakes_ref.stream()
                intakes_list = [intake.to_dict() for intake in intakes]

                patient_data['intakes'] = intakes_list
                patients_list.append(patient_data)

            logging.info(f"Retrieved {len(patients_list)} patients for user: {uid}")
            return patients_list
        except Exception as error:
            logging.error('Error retrieving patients:', error)
            raise UserSaveError() from error

    def save_intake(self, userId: str, patientId: str, intake: Intake):
        try:
            logging.info(f"Saving intake for user: {userId} and patient: {patientId}")
            intake_data = intake.to_dict()
            logging.info("Intake data: %s", intake_data)
            # Save the intake to Firestore under the user's patients collection
            self._firestore.collection('users').document(userId).collection('patients').document(patientId).collection('intakes').document().set(
                intake_data)
            logging.info(f"Intake saved successfully for user: {userId}")
        except Exception as error:
            logging.error('Error saving intake:', error)
            raise UserSaveError() from error
        
        
class UserSaveError(Exception):
    """Raised when there is an error saving the user to Firestore."""
    pass


class ClinicSaveError(Exception):
    """Raised when there is an error saving the clinic to Firestore."""
    pass
