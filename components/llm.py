import datetime
import logging

from langchain.prompts import MessagesPlaceholder
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

chat_history = ChatMessageHistory()
LANGCHAIN_TRACING_V2 = True
LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_API_KEY = "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"


class Dog:
    def __init__(self, name: str, sex: str, breed: str, birth_date: str):
        self.name = name
        self.sex = sex
        self.breed = breed
        self.birth_date = datetime.datetime.strptime(birth_date, '%Y-%m-%dT%H:%M:%S.%fZ')

    @property
    def age(self) -> int:
        today = datetime.date.today()
        age = today.year - self.birth_date.year - (
            (today.month, today.day) < (self.birth_date.month, self.birth_date.day))
        return age

    def __repr__(self):
        return f"Dog(name={self.name}, sex={self.sex}, breed={self.breed}, birth_date={self.birth_date}, age={self.age})"


class LLM:
    def __init__(self, model_name: str, api_key: str, vector_db: VectorStore):
        self.model_name = model_name
        self.api_key = api_key
        self.chat_model = ChatOpenAI(model=self.model_name, api_key=self.api_key)
        self.embeddings = OpenAIEmbeddings()
        self.vector_db = vector_db

    def stream_openai_chat(self, dog: Dog, question: str, user_id: str):
        logging.info("eme",self.get_relevant_docs(question))

        system_template = (
            "You are an expert in dog behavior with a deep understanding of canine psychology, training techniques, "
            "and behavioral science. Answer questions about dog behavior with practical advice and insights based on your expertise. "
            "The dog's details are as follows: - Name: {dog_name} - Birth Date: {dog_birth_date} - Age: {dog_age} - Breed: {dog_breed} "
            "- Sex: {dog_sex}. Please make the answer personalized for {dog_name}."
            "If the question is not about dog behavior or you don't know the answer, say so. Use three sentences maximum and keep the answer concise."
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_template),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        chain = prompt | self.chat_model
        # Log the state of the chain
        logging.info("Chain: %s", chain)

        chat_history.add_user_message(question)

        # Start streaming the response
        response_stream = chain.stream({
            "dog_name": dog.name,
            "dog_birth_date": dog.birth_date.strftime('%Y-%m-%d'),
            "dog_age": dog.age,
            "dog_breed": dog.breed,
            "dog_sex": dog.sex,
            "question": question,
            "messages": chat_history.messages,
        },
            config={"metadata": {"user_id": user_id}},
        )

        # Log the response chunks as they are received
        response = ""
        for chunk in response_stream:
            if chunk.content:
                response += chunk.content
                yield f"{chunk.content}"

        # Add the response to the chat history
        chat_history.add_ai_message(response)

    def get_relevant_docs(self, query):
        relevant_chunks = self.vector_db.similarity_search_with_score(query, k=1)
        logging.info("Relevant chunks: %s", relevant_chunks)
        chunk_docs = [chunk[0] for chunk in relevant_chunks]
        logging.info("Documents to process: %s", chunk_docs)
        return chunk_docs

