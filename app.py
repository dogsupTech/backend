import datetime
from firebase_admin import auth
from flask import Flask, request, jsonify, Response, g
from flask_cors import CORS
import os
import logging
from components.llm import LLM
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from functools import wraps

app = Flask(__name__)
CORS(app)

os.environ['OPENAI_API_KEY'] = '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'

llm = LLM(model_name="gpt-3.5-turbo", api_key=os.environ['OPENAI_API_KEY'])

log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

cred = credentials.Certificate('./serviceAccountKey.json')
fbApp = firebase_admin.initialize_app(cred)
db = firestore.client()


class Dog:
    def __init__(self, birthDate: str, dogName: str, selectedBreed: str, sex: str):
        self.birth_date = datetime.datetime.strptime(birthDate, '%Y-%m-%dT%H:%M:%S.%fZ')
        self.name = dogName
        self.breed = selectedBreed
        self.sex = sex

    @property
    def age(self) -> int:
        today = datetime.date.today()
        age = today.year - self.birth_date.year - (
            (today.month, today.day) < (self.birth_date.month, self.birth_date.day))
        return age

    def __repr__(self):
        return f"Dog(name={self.name}, sex={self.sex}, breed={self.breed}, birth_date={self.birth_date}, age={self.age})"

    def to_dict(self):
        return {
            "birthDate": self.birth_date.isoformat(),
            "dogName": self.name,
            "selectedBreed": self.breed,
            "sex": self.sex
        }


class User:
    def __init__(self, email: str, uid: str, emailActivationToken: str, email_verified: bool,
                 updatedAt: datetime.datetime,
                 vet_ai_waitlist: bool, vet_ai_is_white_listed: bool, dog: Dog):
        self.email = email
        self.uid = uid
        self.emailActivationToken = emailActivationToken
        self.email_verified = email_verified
        self.updatedAt = updatedAt
        self.vet_ai_waitlist = vet_ai_waitlist
        self.vet_ai_is_white_listed = vet_ai_is_white_listed
        self.dog = dog

    def __repr__(self):
        return f"User(email={self.email}, uid={self.uid}, emailActivationToken={self.emailActivationToken}, email_verified={self.email_verified}, updatedAt={self.updatedAt}, vet_ai_waitlist={self.vet_ai_waitlist}, vet_ai_is_white_listed={self.vet_ai_is_white_listed}, dog={self.dog})"

    def to_dict(self):
        return {
            "email": self.email,
            "uid": self.uid,
            "emailActivationToken": self.emailActivationToken,
            "email_verified": self.email_verified,
            "updatedAt": self.updatedAt.isoformat(),
            "vet_ai_waitlist": self.vet_ai_waitlist,
            "vet_ai_is_white_listed": self.vet_ai_is_white_listed,
            "dog": self.dog.to_dict() if self.dog else None
        }


class AccountService:
    def __init__(self, db):
        self._firestore = db

    def get_saved_user(self, id_token):
        try:
            decoded_token = auth.verify_id_token(id_token)
            uid = decoded_token['uid']
            user_ref = self._firestore.collection('users').document(uid)
            user_doc = user_ref.get()
            if user_doc.exists:
                user_data = user_doc.to_dict()
                dog_data = user_data.pop('dog', None)
                dog = Dog(**dog_data) if dog_data else None
                updatedAt = user_data.pop('updatedAt')
                user = User(updatedAt=updatedAt, dog=dog, **user_data)
                return user
            return None
        except Exception as e:
            logging.error("Authorization failed: %s", str(e))
            return None


account_service = AccountService(db)


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


@app.route('/chat', methods=['POST'])
@authorize
def chat_endpoint():
    data = request.json
    user_input = data.get("input")

    # Log the incoming request
    logging.info("Received request: %s", data)

    # Check if the request contains user input
    if user_input is None:
        return jsonify({"error": "Input is missing"}), 400

    user = g.user

    # Pass the user's dog to stream_openai_chat
    chat_response = llm.stream_openai_chat(user.dog, user_input, user.uid)

    # Return the chat response as a text event stream
    return Response(chat_response, content_type='text/event-stream')


@app.route('/me', methods=['GET'])
@authorize
def get_me():
    logging.info("GET /me")
    user = g.user
    return jsonify({"user": user.to_dict()}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8081))  # Default to 8081 if no PORT env var is set
    app.run(host='0.0.0.0', port=port, debug=True)
