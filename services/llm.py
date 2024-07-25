import datetime
import logging
from typing import Optional

from langchain.prompts import MessagesPlaceholder
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langsmith import evaluate
from langsmith.evaluation import LangChainStringEvaluator
from pydantic import json

chat_history = ChatMessageHistory()


class LLM:
    def __init__(self, model_name: str, api_key: str, vector_db: VectorStore):
        self.model_name = model_name
        self.api_key = api_key
        self.chat_model = ChatOpenAI(model=self.model_name, api_key=self.api_key)
        self.embeddings = OpenAIEmbeddings()
        self.vector_db = vector_db
        self.retriever = self.vector_db.as_retriever()

    def create_history_aware_retriever(self):
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question which might reference context in the chat history, "
            "formulate a standalone question which can be understood without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        return create_history_aware_retriever(
            self.chat_model, self.retriever, contextualize_q_prompt
        )

    def create_qa_chain(self):
        qa_system_prompt = (
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        return create_stuff_documents_chain(self.chat_model, qa_prompt)

    def get_relevant_docs(self, query):
        relevant_chunks = self.vector_db.similarity_search_with_score(query, k=1)
        logging.info("Relevant chunks: %s", relevant_chunks)
        chunk_docs = [chunk[0] for chunk in relevant_chunks]
        logging.info("Documents to process: %s", chunk_docs)
        return chunk_docs

    def generate_summary(self, file):
        return "summary of the file"
    
    def run_evals(self, dataset: str, experiment_prefix: str):
        # System template
        system_template = (
            "You are an expert in Diagnostic Imaging and Radiology for small animals. Answer questions about diagnostic imaging and radiology."
        )

        # Creating the ChatPromptTemplate with system and messages
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_template),
                ("user", "{Question}"),
                # MessagesPlaceholder(variable_name="messages"),
            ]
        )

        output_parser = StrOutputParser()

        # Create the evaluation chain
        chain = prompt | self.chat_model | output_parser

        # Define evaluators
        evaluators = [
            LangChainStringEvaluator("cot_qa"),
        ]

        # Perform evaluation
        results = evaluate(
            chain.invoke,
            data=dataset,
            evaluators=evaluators,
            experiment_prefix=experiment_prefix,
        )

        logging.info("Results: %s", results)
        return results


