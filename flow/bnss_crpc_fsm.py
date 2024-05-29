import json
from typing import Any, Dict, Type

from jb_manager_bot import (
    AbstractFSM,
    FSMOutput,
    MessageData,
    MessageType,
    Status,
)
from jb_manager_bot.parsers import OptionParser
from jb_manager_bot.parsers.utils import LLMManager


class BNSS_CRPC(AbstractFSM):
    states = [
        "zero",
        "select_language",
        "ask_for_question",
        "extract_category",
        "fetch_answer",
        "generate_response",
        "ask_another_question",
        "end",
    ]

    transitions = [
        {"trigger": "next", "source": "zero", "dest": "select_language"},
        {"trigger": "next", "source": "select_language", "dest": "ask_for_question"},
        {"trigger": "next", "source": "ask_for_question", "dest": "extract_category"},
        {
            "trigger": "next",
            "source": "extract_category",
            "dest": "fetch_answer",
        },
        {"trigger": "next", "source": "fetch_answer", "dest": "generate_response"},
        {
            "trigger": "next",
            "source": "generate_response",
            "dest": "ask_another_question",
        },
        {
            "trigger": "next",
            "source": "ask_another_question",
            "dest": "extract_category",
        }
    ]

    conditions = {}
    output_variables = set()

    def __init__(self, send_message: callable, credentials: Dict[str, Any] = None):

        if credentials is None:
            credentials = {}

        self.credentials = {}
        self.credentials["OPENAI_API_KEY"] = credentials.get("OPENAI_API_KEY")
        if not self.credentials["OPENAI_API_KEY"]:
            raise ValueError("Missing credential: OPENAI_API_KEY")
        self.credentials["AZURE_OPENAI_API_KEY"] = None
        self.credentials["AZURE_OPENAI_API_KEY"] = None
        self.credentials["AZURE_OPENAI_API_VERSION"] = None
        self.credentials["AZURE_OPENAI_API_ENDPOINT"] = None

        self.plugins: Dict[str, AbstractFSM] = {}
        super().__init__(send_message=send_message)

    def send_output(self, message):
        self.send_message(FSMOutput(message_data=MessageData(body=message)))

    def on_enter_select_language(self):
        self.status = Status.WAIT_FOR_ME
        self.send_message(
            FSMOutput(
                message_data=MessageData(
                    body="Hi there! Welcome to BNSS-CrPC Bot!\nPlease choose a language"
                ),
                type=MessageType.TEXT,
                dialog="language",
                dest="channel",
            )
        )
        self.status = Status.WAIT_FOR_USER_INPUT

    def on_enter_ask_for_question(self):
        self.status = Status.WAIT_FOR_ME
        self.send_output("Please ask your query")
        self.status = Status.WAIT_FOR_USER_INPUT

    def on_exit_ask_for_question(self):
        self.variables["query"] = self.current_input

    def on_enter_extract_category(self):
        self.status = Status.WAIT_FOR_ME
        self.variables["query"] = self.current_input
        self.variables["metadata"] = {}
        out = LLMManager.llm(
            messages=[
                LLMManager.sm(
                    f"""
                  You are an helpful assistant who helps with extracting the BNSS section and sub section number or CrPC section and sub section number from the given string. Do not make up any section numbers. Return the answer only as a python dictionary in the following format:

If any of "bnss_section_no", "bnss_sub_section", "crpc_section_no", "crpc_sub_section" is null, do not include them in the dictionary.
{{
    "bnss_section_no": <extracted-bnss-section-number>,
    "bnss_sub_section": <extracted-bnss-sub-section-number>,
    "crpc_section_no": <extracted-crpc-section-number>,
    "crpc_sub_section": <extracted-crpc-sub-section-number>,
}}
    [Query]
    {self.variables["query"]}
    """
                ),
                LLMManager.um(f"User: {self.variables['query']}\nBot: "),
            ],
            openai_api_key=self.credentials["OPENAI_API_KEY"],
            azure_openai_api_key=self.credentials["AZURE_OPENAI_API_KEY"],
            azure_openai_api_version=self.credentials["AZURE_OPENAI_API_VERSION"],
            azure_endpoint=self.credentials["AZURE_OPENAI_API_ENDPOINT"],
            # response_format={"type": "json_object"},
            model="gpt-3.5-turbo",
        )
        dictionary = {}
        out = json.loads(out)
        for key in out:
            if out[key] is not None:
                dictionary[key] = out[key]

        self.variables["category"] = dictionary
        self.status = Status.MOVE_FORWARD

    def on_enter_fetch_answer(self):
        self.status = Status.WAIT_FOR_ME
        self.send_message(
            FSMOutput(
                message_data=MessageData(body=self.variables["query"]),
                dest="rag",
            )
        )
        self.status = Status.WAIT_FOR_USER_INPUT

    def on_enter_generate_response(self):
        self.status = Status.WAIT_FOR_ME
        chunks = self.current_input
        chunks = json.loads(chunks)["chunks"]
        knowledge = "\n".join([row["chunk"] for row in chunks])

        result = LLMManager.llm(
            messages=[
                LLMManager.sm(
                    f"""
You are a helpful assistant who helps with answering questions. Given the user's query [query]{self.variables["query"]}, you should display the corresponding answer from the given chunk knowledge in the exact format. Understand if the user inputs query is under the Bharatiya Nyaya Sanhita (BNS) or the older Indian Penal Code (IPC), as specified in your knowledge base and clearly display the respective answer. Do not make up or paraphrase any answers. Ensure that the output is presented in string format.
If the query {self.variables["query"]} has CrPC section information, you have to display BNSS details equivalent of given CrPC and vice versa. 
    
Example 1:
    User: What is CrPC section 354 (assault on women) under the BNSS?

Since asked for the relevant sections of BNSS, you should give BNSS Section number, BNSS Section Title, BNSS Section Content, Remarks. Be specific to the BNSS, do not give the CrPC details.

Example 2:
User : What is BNSS Section 4 under CrPC?

Since the user's query asked for the relevant sections of the CrPC, you should give CrPC Section number,CrPC Sub Section, CrPC Section Title, CrPC Section Content, Remarks. Be specific to CrPC, do not give the BNSS details.
 [Knowledge]
    {knowledge}  
                """
                ),
                LLMManager.um(self.current_input),
            ],
            openai_api_key=self.credentials["OPENAI_API_KEY"],
            azure_openai_api_key=self.credentials["AZURE_OPENAI_API_KEY"],
            azure_openai_api_version=self.credentials["AZURE_OPENAI_API_VERSION"],
            azure_endpoint=self.credentials["AZURE_OPENAI_API_ENDPOINT"],
            model="gpt-3.5-turbo",
        )
        self.send_output(result)
        self.status = Status.MOVE_FORWARD

    def on_enter_ask_another_question(self):
        self.status = Status.WAIT_FOR_ME
        self.status = Status.WAIT_FOR_USER_INPUT
