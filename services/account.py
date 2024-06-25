import datetime
import logging

from firebase_admin import auth


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
                 updatedAt: datetime.datetime, vet_ai_waitlist: bool, vet_ai_is_white_listed: bool,
                 dog: Dog, roles: list = None, request_count: int = 0):
        self.email = email
        self.uid = uid
        self.emailActivationToken = emailActivationToken
        self.email_verified = email_verified
        self.updatedAt = updatedAt
        self.vet_ai_waitlist = vet_ai_waitlist
        self.vet_ai_is_white_listed = vet_ai_is_white_listed
        self.dog = dog
        self.roles = roles if roles else ['user']
        self.request_count = request_count

    def __repr__(self):
        return (f"User(email={self.email}, uid={self.uid}, emailActivationToken={self.emailActivationToken}, "
                f"email_verified={self.email_verified}, updatedAt={self.updatedAt}, "
                f"vet_ai_waitlist={self.vet_ai_waitlist}, vet_ai_is_white_listed={self.vet_ai_is_white_listed}, "
                f"dog={self.dog}, roles={self.roles}, request_count={self.request_count})")

    def to_dict(self):
        return {
            "email": self.email,
            "uid": self.uid,
            "emailActivationToken": self.emailActivationToken,
            "email_verified": self.email_verified,
            "updatedAt": self.updatedAt.isoformat(),
            "vet_ai_waitlist": self.vet_ai_waitlist,
            "vet_ai_is_white_listed": self.vet_ai_is_white_listed,
            "dog": self.dog.to_dict() if self.dog else None,
            "roles": self.roles,
            "request_count": self.request_count
        }


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
                updatedAt = user_data.pop('updatedAt')
                user = User(updatedAt=updatedAt, dog=dog, **user_data)
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

