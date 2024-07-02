import datetime
import logging
from typing import Optional

from langchain.prompts import MessagesPlaceholder
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

chat_history = ChatMessageHistory()


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

    def stream_openai_chat(self, dog: Dog, question: str, user_id: str, image_url: Optional[str] = None):
        # retriever
        relevant_docs = self.vector_db.as_retriever().invoke(question)

        # retriever
        system_template = (
            "You are an expert in dog behavior with a deep understanding of canine psychology, training techniques, "
            "and behavioral science. Answer questions about dog behavior with practical advice and insights based on your expertise. "
            "The dog's details are as follows: - Name: {dog_name} - Birth Date: {dog_birth_date} - Age: {dog_age} - Breed: {dog_breed} "
            "- Sex: {dog_sex}. Please make the answer personalized for {dog_name}."
            "If the question is not about dog behavior or you don't know the answer, say so. Use three sentences maximum and keep the answer concise."
            "Base the answer on the dog's details and this context: {context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_template),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        chain = prompt | self.chat_model

        # Add the image URL to the message if provided
        if image_url:
            image_message = [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                }
            ]
            chat_history.add_user_message(image_message)
        else:
            image_message = [{"type": "text", "text": question}]

        chat_history.add_user_message(image_message)

        # Start streaming the response
        response_stream = chain.stream({
            "context": relevant_docs[0].page_content,
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

    def generate_summary(self, file):
        return "summary of the file"
    
    


