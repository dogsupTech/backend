import os
import logging
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from firebase_admin import credentials, initialize_app

from routes.api import api_bp

load_dotenv()


def create_app():
    app = Flask(__name__)
    CORS(app)

    # Firebase setup
    cred = credentials.Certificate('./serviceAccountKey.json')
    firebase_app = initialize_app(cred)

    # OpenAI setup
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    # Set up logging to a file
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ])

    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api/v1')

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 8081))
    app.run(host='0.0.0.0', port=port, debug=True)
