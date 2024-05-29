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
            "BNSS Section No.",
            "BNSS Section Title",
            "BNSS Sub-Section",
            "BNSS Section Content",
            "CrPC Section No.",
            "CrPC Section Title",
            "CrPC Sub-Section",
            "CrPC Section Content",
            "Remarks/Comments",
        ]

        dataframe = pd.read_excel(file_path, usecols=default_columns)
        dataframe.columns = default_columns
        # dataframe.dropna(subset=["BNSS Section No."], inplace=True)
        dataframe.dropna(axis=0, how="all", inplace=True)
        dataframe = dataframe.fillna("")

        for index in range(len(dataframe)):
            row = dataframe.iloc[index]
            bnss_1 = str(row["BNSS Section No."]).replace("S. ", "")
            bnss_2 = row["BNSS Section Title"]
            bnss_3 = str(row["BNSS Sub-Section"]).replace("S. ", "")
            bnss_4 = row["BNSS Section Content"]

            crpc_1 = str(row["CrPC Section No."]).replace("S. ", "")
            crpc_2 = row["CrPC Section Title"]
            crpc_3 = str(row["CrPC Sub-Section"]).replace("S. ", "")
            crpc_4 = row["CrPC Section Content"]

            bnss = ""
            if bnss_3 == "" or bnss_3 == "NA" or bnss_3 is None:
                bnss += "*BNSS Section No:* " + bnss_1
                metadata_bnss_section_number = bnss_1
                metadata_bnss_sub_section = ""
            else:
                bnss += "*BNSS Section No:* " + bnss_3
                matches = re.match(pattern, bnss_3)
                if matches:
                    metadata_bnss_section_number = matches.group(1)
                    metadata_bnss_sub_section = matches.group(2)

            bnss += "\n*BNSS Section Title:* " + bnss_2
            bnss += "\n*BNSS Section Content:* \n" + bnss_4

            bnss += "\n\n"
            if (
                (crpc_1 == "" or crpc_1 == "NA")
                and (crpc_2 == "" or crpc_2 == "NA")
                and (crpc_3 == "" or crpc_3 == "NA")
                and (crpc_4 == "" or crpc_4 == "NA")
            ):
                bnss += (
                    "There is no equivalent CrPC section for the given BNSS section."
                )

            elif crpc_3 == "" or crpc_3 == "NA" or crpc_3 is None:
                bnss += "\n*CrPC Section No:* " + crpc_1
                metadata_crpc_section_number = crpc_1
                metadata_crpc_sub_section = ""

            else:
                bnss += "\n*CrPC Section No:* " + crpc_3
                matches = re.match(pattern, crpc_3)
                if matches:
                    metadata_crpc_section_number = matches.group(1)
                    metadata_crpc_sub_section = matches.group(2)

            bnss += "\n*CrPC Section Title:* " + crpc_2
            bnss += "\n*CrPC Section Content:* \n" + crpc_4
            bnss += "\n*Remarks/Comments:* " + row["Remarks/Comments"]

            metadata = {
                "source": str(counter),
                "bnss_section_no": metadata_bnss_section_number,
                "bnss_section_title": bnss_2,
                "bnss_sub_section": metadata_bnss_sub_section,
                "crpc_section_no": metadata_crpc_section_number,
                "crpc_sub_section": metadata_crpc_sub_section,
                "crpc_section_title": crpc_2,
                # "category": "",
            }
            chunks.append(Document(page_content=bnss, metadata=metadata))
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
