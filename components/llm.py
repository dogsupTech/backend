import datetime
import logging
from typing import List

from flask import g
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
from langchain.chains.question_answering import load_qa_chain
from langchain.output_parsers import RegexParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory

demo_ephemeral_chat_history = ChatMessageHistory()
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


# multimodality
# change prompts
# upload pdfs

class LLM:
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        self.chat_model = ChatOpenAI(model=self.model_name, api_key=self.api_key)
        self.embeddings = None

    def stream_openai_chat(self, dog: Dog, question: str, user_id:str):
        # embed the query 
        embeddings = self.create_or_load_facial_expression_embeddings()

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

        demo_ephemeral_chat_history.add_user_message(question)

        # Start streaming the response
        response_stream = chain.stream({
            "dog_name": dog.name,
            "dog_birth_date": dog.birth_date.strftime('%Y-%m-%d'),
            "dog_age": dog.age,
            "dog_breed": dog.breed,
            "dog_sex": dog.sex,
            "question": question,
            "messages": demo_ephemeral_chat_history.messages,
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
        demo_ephemeral_chat_history.add_ai_message(response)

    def create_or_load_facial_expression_embeddings(self) -> List[List[float]]:
        # load the pdf
        facialExpressionLoader = PyPDFLoader('./components/FacialExpressionsDogs.pdf')
        pages = facialExpressionLoader.load()
        # logging.info("Pages: %s", pages)

        # split the pages into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )

        splitted_pages = text_splitter.split_documents(pages)
        logging.info("length of chunks to embed: %s", len(splitted_pages))

        # convert splitted pages to list of strings
        texts = [page.page_content for page in splitted_pages]

        # embed the chunks
        embedding_model = OpenAIEmbeddings()
        embeddings = embedding_model.embed_documents(texts)
        logging.info("type of embeddings: %s", type(embeddings))
        return embeddings

        # if self.embeddings is None:
        #     openai_embeddings = OpenAIEmbeddings()
        #     if os.path.exists('facial_expression_faiss_index'):
        #         # Pass the OpenAIEmbeddings instance when loading the local FAISS index
        #         self.embeddings = FAISS.load_local('facial_expression_faiss_index', openai_embeddings,
        #                                            allow_dangerous_deserialization=True)
        #     else:
        #         documents_path = '../annika/*.pdf'
        #         # Initialize the loader with the PyPDFLoader to handle PDF files
        #         loader = DirectoryLoader('docs', glob=documents_path, loader_cls=PyPDFLoader)
        #         documents = loader.load()
        # 
        #         # Set the chunk size and overlap for splitting the text from the documents
        #         chunk_size = 1000  # Number of characters in each text chunk
        #         chunk_overlap = 100  # Number of overlapping characters between consecutive chunks
        # 
        #         # Initialize the text splitter with default separators if they're not set
        #         separators = ['.', '!', '?']  # Default sentence-ending punctuation marks
        #         text_splitter = RecursiveCharacterTextSplitter(separators=separators, chunk_size=chunk_size,
        #                                                        chunk_overlap=chunk_overlap, length_function=len)
        #         texts = text_splitter.split_documents(documents)
        # 
        #         self.embeddings = FAISS.from_documents(texts, openai_embeddings)
        #         self.embeddings.save_local('llm_faiss_index')
        #         
        # return self.embeddings

    def get_relevant_docs(self, query):
        embeddings = self.create_or_load_facial_expression_embeddings()
        relevant_chunks = embeddings.similarity_search_with_score(query, k=2)
        chunk_docs = [chunk[0] for chunk in relevant_chunks]
        logging.info("Documents to process: %s", chunk_docs)
        return chunk_docs

    def get_qa_chain(self):
        embeddings = self.create_or_load_facial_expression_embeddings()

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
            llm=self.chat_model,
            chain_type="map_rerank",
            return_intermediate_steps=True,
            prompt=PROMPT,
        )

        return qa_chain_chain
