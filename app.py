from langchain_openai import ChatOpenAI

from flask import Flask, request, jsonify, Response
from flask_cors import CORS, cross_origin
import os
import logging

from components.llm import LLM, Dog

app = Flask(__name__)
CORS(app)

os.environ['OPENAI_API_KEY'] = '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'

llm = LLM(model_name="gpt-3.5-turbo", api_key=os.environ['OPENAI_API_KEY'])


# Configure logging to output to console
logging.basicConfig(
    level=logging.INFO,  # Log level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log message format
)

# Add a StreamHandler to the root logger to log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set the log level for console output
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)



@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data = request.json
    user_input = data.get("input")
    dog_details = data.get("dog")

    # Log the incoming request
    logging.info("Received request: %s", data)

    # Check if the request contains both user input and dog details
    if user_input is None:
        return jsonify({"error": "Input is missing"}), 400
    if dog_details is None:
        return jsonify({"error": "Dog details are missing"}), 400

    # Validate and process dog details
    try:
        # Correctly map dog details to Dog class attributes
        new_dog = Dog(
            name=dog_details.get('dogName'),
            sex=dog_details.get('sex'),
            breed=dog_details.get('selectedBreed'),
            birth_date=dog_details.get('birthDate')
        )
        logging.info("Created new dog: %s", vars(new_dog))
    except Exception as e:
        logging.error("Error creating dog: %s", str(e))
        return jsonify({"error": str(e)}), 400


    # Process user input
    chat_response = llm.stream_openai_chat(new_dog, user_input)

    # Return the chat response as a text event stream
    return Response(chat_response, content_type='text/event-stream')


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8081))  # Default to 8080 if no PORT env var is set
    app.run(host='0.0.0.0', port=port, debug=True)

# @app.route('/docqna', methods=["POST"])
# def process_claim():
#     try:
#         input_json = request.get_json(force=True)
#         query = input_json.get("query")
#         if not query:
#             return jsonify({"error": "No query provided"}), 400
# 
#         output = get_question_answering_answer(query)
#         return jsonify(output)
#     except Exception as e:
#         logging.error("Error processing claim: %s", str(e))
#         return jsonify({"error": str(e)}), 500
# 
# 
# def get_question_answering_answer(query):
#     try:
#         facial_expression_embeddings = create_or_load_facial_expression_embeddings()
#         chunk_docs = get_relevant_docs(query, facial_expression_embeddings)
#         results = get_qa_chain({"input_documents": chunk_docs, "question": query})
#         logging.info("Model output before parsing: %s", results["output_text"])
#         text_reference = "".join(doc.page_content for doc in results["input_documents"])
#         output = {"Answer": results["output_text"], "Reference": text_reference}
#         return output
# 
#     except Exception as e:
#         logging.error("Failed to parse output: %s", str(e))
#         return {"error": f"Could not parse output: {e}"}
