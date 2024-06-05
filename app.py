from langchain_openai import ChatOpenAI

from flask import Flask, request, jsonify, Response
from flask_cors import CORS, cross_origin
import os

from components.llm import LLM

app = Flask(__name__)
CORS(app)

os.environ['OPENAI_API_KEY'] = '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'

llm = LLM(model_name="gpt-3.5-turbo", api_key=os.environ['OPENAI_API_KEY'])


@app.route('/chat', methods=['POST'])
def chat_endpoint():
    user_input = request.json.get("input")
    return Response(llm.stream_openai_chat(user_input), content_type='text/event-stream')


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Default to 8080 if no PORT env var is set
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
