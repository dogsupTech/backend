# app.py
import os
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from firebase_admin import credentials, initialize_app

from routes.api import api_bp
from services_old.account import AccountService
load_dotenv()


def create_app():
    app = Flask(__name__)
    CORS(app)

    # Firebase setup
    cred = credentials.Certificate('./serviceAccountKey.json')
    firebase_app = initialize_app(cred)

    # OpenAI setup
    # os.environ['OPENAI_API_KEY'] = 'your-openai-api-key'
    # vector_db = create_or_load_vector_store()
    # app.llm = LLM(model_name="gpt-4", api_key=os.environ['OPENAI_API_KEY'], vector_db=vector_db)
    
    # Services setup
    app.account_service = AccountService(firebase_app)

    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api/v1')

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 8081))
    app.run(host='0.0.0.0', port=port, debug=True)
