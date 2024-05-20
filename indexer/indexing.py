import asyncio
import json
import os
import re
from abc import ABC, abstractmethod
from lib.data_models import IndexerInput

import asyncpg
# import docx2txt
# import fitz
import pandas as pd
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.pgvector import PGVector

from lib.document_collection import (DocumentCollection, DocumentFormat,
                                     DocumentSourceFile)
from lib.kafka_utils import KafkaConsumer
from lib.storage import LocalStorage


load_dotenv()


kafka_bootstrap_servers = os.getenv('KAFKA_BROKER')
kafka_topic = os.getenv('KAFKA_CONSUMER_TOPIC')
print("kafka_bootstrap_servers", kafka_bootstrap_servers)
print("kafka", kafka_topic)
consumer = KafkaConsumer.from_env_vars(group_id='test_grp_id', auto_offset_reset='latest')


def docx_to_text_converter(docx_file_path):
    text = docx2txt.process(docx_file_path)
    return text


def pdf_to_text_converter(pdf_file_path):
    doc = fitz.open(pdf_file_path)
    content = "\n"
    for page in doc:
        text = page.get_text("text", textpage=None, sort=False)
        text = re.sub(r'\n\d+\s*\n', '\n', text)
        content += text
    return content


class TextConverter:
    async def textify(self, filename: str, doc_collection: DocumentCollection) -> str:
        file_path = doc_collection.local_file_path(filename)
        if filename.endswith(".pdf"):
            content = pdf_to_text_converter(file_path)
        elif filename.endswith(".docx"):
            content = docx_to_text_converter(file_path)
        else:
            with open(file_path, "r") as f:
                content = f.read()

        # remove multiple new lines between paras
        regex = r"(?<!\n\s)\n(?!\n| \n)"
        content = re.sub(regex, "", content)

        content = repr(content)[1:-1]
        content = bytes(content, "utf-8")
        print("TYPE OF CONTENT", type(content))
        await doc_collection.write_file(filename, content, DocumentFormat.TEXT)
        # await doc_collection.public_url(filename, DocumentFormat.TEXT)
        return content


class Indexer(ABC):
    @abstractmethod
    async def index(self, document_collection: DocumentCollection):
        pass


class LangchainIndexer(Indexer):
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=4 * 1024, chunk_overlap=0, separators=["\n\n", "\n", ".", ""]
        )
        self.sub_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3.5 * 1024, chunk_overlap=0, separators=["\n\n", "\n", ".", ""]
        )
        self.db_name = os.getenv("POSTGRES_DATABASE_NAME")
        self.db_user = os.getenv("POSTGRES_DATABASE_USERNAME")
        self.db_password = os.getenv("POSTGRES_DATABASE_PASSWORD")
        self.db_host = os.getenv("POSTGRES_DATABASE_HOST")
        self.db_port = os.getenv("POSTGRES_DATABASE_PORT")
        self.db_url = f"postgresql+psycopg2://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        self.text_converter = TextConverter()

    def process_xslx_files(self, xslx_file_paths, counter=0):
        chunks = []
        pattern = r"(\d+)\((\d+)\)"
        file_path = xslx_file_paths[0]
        default_columns = [
            "BNS Section No.",
            "BNS Section Title",
            "BNS Sub-Section",
            "BNS Section Content",
            "BNS Consequence/Punishment",
            "IPC Section No.",
            "IPC Section Title",
            "IPC Section Content",
            "IPC Consequence/Punishment",
            "Remarks/Comments",
        ]

        dataframe = pd.read_excel(file_path, usecols=default_columns)
        dataframe.columns = default_columns
        dataframe.dropna(axis=0, how="all", inplace=True)
        dataframe = dataframe.fillna("")

        for index in range(len(dataframe)):
            row = dataframe.iloc[index]
            bns_1 = str(row["BNS Section No."]).replace("S. ", "")
            bns_2 = row["BNS Section Title"]
            bns_3 = str(row["BNS Sub-Section"]).replace("S. ", "")
            bns_4 = row["BNS Section Content"]
            bns_5 = row["BNS Consequence/Punishment"]

            ipc_1 = str(row["IPC Section No."]).replace("S. ", "")
            ipc_2 = row["IPC Section Title"]
            ipc_3 = row["IPC Section Content"]
            ipc_4 = row["IPC Consequence/Punishment"]

            bns = ""
            if bns_3 == "" or bns_3 == "NA" or bns_3 is None:
                bns += "BNS Section No: " + bns_1
                metadata_bns_section_number = bns_1
                metadata_bns_sub_section = ""
            else:
                bns += "BNS Section No: " + bns_3
                matches = re.match(pattern, bns_3)
                if matches:
                    metadata_bns_section_number = matches.group(1)
                    metadata_bns_sub_section = matches.group(2)

            bns += "\nBNS Section Title: " + bns_2
            bns += "\nBNS Section Content: " + bns_4
            bns += "\nBNS Section Consequence/Punishment: " + bns_5

            bns += "\n\n"
            if (
                (ipc_1 == "" or ipc_1 == "NA")
                and (ipc_2 == "" or ipc_2 == "NA")
                and (ipc_3 == "" or ipc_3 == "NA")
            ):
                bns += "There is no equivalent IPC section for the given BNS section."
            else:
                bns += "\nIPC Section No: " + ipc_1
                bns += "\nIPC Section Title: " + ipc_2
                bns += "\nIPC Section Content: " + ipc_3
                bns += "\nIPC Section Consequence/Punishment: " + ipc_4
                bns += "\nRemarks/Comments: " + row["Remarks/Comments"]

            metadata = {
                "source": str(counter),
                "bns_section_no": metadata_bns_section_number,
                "bns_section_title": bns_2,
                "bns_sub_section": metadata_bns_sub_section,
                "ipc_section_no": ipc_1,
                "ipc_section_title": ipc_2,
            }
            chunks.append(Document(page_content=bns, metadata=metadata))
            counter += 1

        return chunks

    async def create_pg_vector_index_if_not_exists(self):
        # Establish an asynchronous connection to the database
        print("Inside create_pg_vector_index_if_not_exists")
        connection = await asyncpg.connect(
            host=self.db_host,
            user=self.db_user,
            password=self.db_password,
            database=self.db_name,
            port=self.db_port
        )
        try:
            # Create an asynchronous cursor object to interact with the database
            async with connection.transaction():
                # Execute the alter table query
                await connection.execute("ALTER TABLE langchain_pg_embedding ALTER COLUMN embedding TYPE vector(1536)")
                await connection.execute("CREATE INDEX IF NOT EXISTS langchain_embeddings_hnsw ON langchain_pg_embedding USING hnsw (embedding vector_cosine_ops)")
        finally:
            # Close the connection
            await connection.close()

    # async def index(self, collection_name: str, files: list[str]):
    async def index(self, indexer_input: IndexerInput):
        # document_collection = DocumentCollection(collection_name,
        #                                          LocalStorage(os.environ["DOCUMENT_LOCAL_STORAGE_PATH"]),
        #                                          LocalStorage(os.environ["DOCUMENT_LOCAL_STORAGE_PATH"]))
        
        xslx_file_paths = []
        source_files = []
        print("Inside Indexer-1")
        file_size = 0.0
        print('files', indexer_input.files)

        for file in indexer_input.files:
            file_path = os.path.join(os.environ["DOCUMENT_LOCAL_STORAGE_PATH"], file)
            print('file_path', file_path)
            file_size += os.path.getsize(file_path)
            if file.endswith(".xlsx"):
                xslx_file_paths.append(file_path)
            else:
                file_reader = open(file_path, "rb")
                print("FILE_PATH:", file_path)
                print("FILE NAME:", file)
                source_files.append(DocumentSourceFile(file, file_reader))

        # await document_collection.init_from_files(source_files)
        # async for filename in document_collection.list_files():
        #     await self.text_converter.textify(filename, document_collection)

        # collection_name = document_collection.id
        source_chunks = []
        counter = 0
        # async for filename in document_collection.list_files():
        #     content = await document_collection.read_file(filename, DocumentFormat.TEXT)
        #     # public_text_url = await document_collection.public_url(filename,
        #     #                                                   DocumentFormat.TEXT)
        #     content = content.decode('utf-8')
        #     content = content.replace("\\n", "\n")
        #     for chunk in self.splitter.split_text(content):
        #         new_metadata = {
        #             "source": str(counter),
        #             "document_name": filename,
        #         }
        #         source_chunks.append(
        #             Document(page_content=chunk, metadata=new_metadata)
        #         )
        #         counter += 1
        xslx_source_chunks = self.process_xslx_files(xslx_file_paths, counter)
        source_chunks.extend(xslx_source_chunks)
        # try:
        if os.environ["OPENAI_API_TYPE"] == "azure":
            embeddings = OpenAIEmbeddings(client="", deployment=os.environ["OPENAI_EMBEDDINGS_DEPLOYMENT"])
        else:
            embeddings = OpenAIEmbeddings(client="")
        # create a collection in pgvector database
        db = PGVector.from_documents(
                embedding=embeddings,
                documents=source_chunks,
                collection_name=indexer_input.collection_name,
                connection_string=self.db_url,
                pre_delete_collection=True  # delete collection if it already exists
            )
        print("Embeddings have been created after dropping old collection if it was found.")
        await self.create_pg_vector_index_if_not_exists()
        # except openai.error.RateLimitError as e:
        #     raise ServiceUnavailableException(
        #         f"OpenAI API request exceeded rate limit: {e}"
        #     )
        # except (openai.error.APIError, openai.error.ServiceUnavailableError):
        #     raise ServiceUnavailableException(
        #         "Server is overloaded or unable to answer your request at the moment."
        #         " Please try again later"
        #     )
        # except Exception as e:
        #     raise InternalServerException(e.__str__())
        print("Indexer done")


langchain_indexer = LangchainIndexer()
while True:
    # will block until message is received
    msg = consumer.receive_message(kafka_topic)
    print("msg", msg)
    data = json.loads(msg)
    indexer_input = IndexerInput(**data)


    asyncio.run(langchain_indexer.index(indexer_input))
