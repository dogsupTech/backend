import logging
import os

from langchain.chains.question_answering import load_qa_chain
from langchain.output_parsers import RegexParser
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ['OPENAI_API_KEY'] = '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'


# Function to create or load embeddings
def create_or_load_facial_expression_embeddings():
    openai_embeddings = OpenAIEmbeddings()
    if os.path.exists('facial_expression_faiss_index'):
        # Pass the OpenAIEmbeddings instance when loading the local FAISS index
        return FAISS.load_local('facial_expression_faiss_index', openai_embeddings,
                                allow_dangerous_deserialization=True)
    else:
        documents_path = '../annika/*.pdf'
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
        
        e = FAISS.from_documents(texts, openai_embeddings)
        e.save_local('llm_faiss_index')
        return e


def get_relevant_docs(query, embeddings):
    relevant_chunks = embeddings.similarity_search_with_score(query, k=2)
    chunk_docs = [chunk[0] for chunk in relevant_chunks]
    logging.info("Documents to process: %s", chunk_docs)
    return chunk_docs


def get_qa_chain():
    embeddings = create_or_load_facial_expression_embeddings()

    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the 
    answer, just say that you don't know, don't try to make up an answer.

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
    qa_chain_chain = load_qa_chain(
        llm=OpenAI(temperature=0),
        chain_type="map_rerank",
        return_intermediate_steps=True,
        prompt=PROMPT,
    )

    return qa_chain_chain


