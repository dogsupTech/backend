from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import os
import logging

# Importing necessary modules from langchain
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.output_parsers import RegexParser

app = Flask(__name__)
CORS(
    app)  # This enables CORS for all domains on all routes. For more granularity, use the @cross_origin decorator or configure CORS more tightly.

os.environ['OPENAI_API_KEY'] = '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'

documents_path = './annika/*.pdf'

# Initialize the loader with the PyPDFLoader to handle PDF files
loader = DirectoryLoader('docs', glob=documents_path, loader_cls=PyPDFLoader)
documents = loader.load()

# Set the chunk size and overlap for splitting the text from the documents
chunk_size = 1000  # Number of characters in each text chunk
chunk_overlap = 100  # Number of overlapping characters between consecutive chunks

# Initialize the text splitter with default separators if they're not set
separators = ['.', '!', '?']  # Default sentence-ending punctuation marks
text_splitter = RecursiveCharacterTextSplitter(separators=separators, chunk_size=chunk_size,
                                               chunk_overlap=chunk_overlap, length_function=len)
texts = text_splitter.split_documents(documents)

openai_embeddings = OpenAIEmbeddings()


# Function to create or load embeddings
def create_or_load_embeddings():
    if os.path.exists('llm_faiss_index'):
        # Pass the OpenAIEmbeddings instance when loading the local FAISS index
        return FAISS.load_local('llm_faiss_index', openai_embeddings, allow_dangerous_deserialization=True)
    else:
        embeddings = FAISS.from_documents(texts, openai_embeddings)
        embeddings.save_local('llm_faiss_index')
        return embeddings


embeddings = create_or_load_embeddings()

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

This should be in the following format:

Question: [question here]
Helpful Answer: [answer here]
Score: [score between 0 and 100]

Begin!

Context:
---------
{context}
---------
Question: {question}
Helpful Answer:"""
output_parser = RegexParser(
    regex=r"(.*?)\nScore: (.*)",
    output_keys=["answer", "score"],
)

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"],
    output_parser=output_parser
)

# Update to use a MapRerankDocumentsChain with a RegexParser
chain = load_qa_chain(
    llm=OpenAI(temperature=0),
    chain_type="map_rerank",
    return_intermediate_steps=True,
    prompt=PROMPT,
)
# Configure basic logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Existing imports and initializations...
# Configure basic logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


@app.route('/docqna', methods=["POST"])
@cross_origin()  # Allows CORS specifically for this route
def process_claim():
    try:
        input_json = request.get_json(force=True)
        query = input_json.get("query")
        if not query:
            return jsonify({"error": "No query provided"}), 400

        output = getanswer(query)
        return jsonify(output)
    except Exception as e:
        logging.error("Error processing claim: %s", str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/hello')
@cross_origin()  # Allows CORS specifically for this route
def hello():
    return jsonify({"message": "Hello World!"})


def getanswer(query):
    try:
        relevant_chunks = embeddings.similarity_search_with_score(query, k=2)
        chunk_docs = [chunk[0] for chunk in relevant_chunks]
        logging.info("Documents to process: %s", chunk_docs)

        results = chain({"input_documents": chunk_docs, "question": query})
        logging.info("Model output before parsing: %s", results["output_text"])

        text_reference = "".join(doc.page_content for doc in results["input_documents"])
        output = {"Answer": results["output_text"], "Reference": text_reference}
        return output
    except Exception as e:
        logging.error("Failed to parse output: %s", str(e))
        return {"error": f"Could not parse output: {e}"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Default to 8080 if no PORT env var is set
    app.run(host='0.0.0.0', port=port, debug=False)
