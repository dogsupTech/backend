import datetime
import logging


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

