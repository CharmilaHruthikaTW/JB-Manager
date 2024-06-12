import json
from typing import Any, Dict, Type

from jb_manager_bot import (
    AbstractFSM,
    FSMOutput,
    MessageData,
    MessageType,
    Status,
)
from jb_manager_bot.parsers.utils import LLMManager


class BNSS(AbstractFSM):
    states = [
        "zero",
        "select_language",
        "ask_for_question",
        "extract_section",
        "fetch_answer",
        "generate_response",
        "ask_another_question",
        "end",
    ]

    transitions = [
        {"trigger": "next", "source": "zero", "dest": "select_language"},
        {"trigger": "next", "source": "select_language", "dest": "ask_for_question"},
        {
            "trigger": "next",
            "source": "ask_for_question",
            "dest": "categorise",
            "conditions": "is_not_categorised",
        },
        {
            "trigger": "next",
            "source": "ask_for_question",
            "dest": "extract_section",
            "conditions": "is_categorised",
        },
        {
            "trigger": "next",
            "source": "categorise",
            "dest": "category_response",
        },
        {"trigger": "next", "source": "category_response", "dest": "extract_section"},
        {
            "trigger": "next",
            "source": "extract_section",
            "dest": "fetch_answer",
        },
        {
            "trigger": "next",
            "source": "fetch_answer",
            "dest": "generate_response",
        },
        {
            "trigger": "next",
            "source": "generate_response",
            "dest": "ask_another_question",
        },
        {
            "trigger": "next",
            "source": "ask_another_question",
            "dest": "extract_section",
            "conditions": "is_categorised",
        },
        {
            "trigger": "next",
            "source": "ask_another_question",
            "dest": "categorise",
            "conditions": "is_not_categorised",
        },
    ]

    conditions = {"is_categorised", "is_not_categorised"}
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

    def is_categorised(self):
        return self.contains_keywords() is True

    def is_not_categorised(self):
        return self.contains_keywords() is False

    def contains_keywords(self):
        query = self.current_input
        keywords = ["BNS", "IPC", "BNSS", "CrPC"]
        query_lower = query.lower()  # To ensure case-insensitive matching
        return any(keyword.lower() in query_lower for keyword in keywords)

    def on_enter_select_language(self):
        self.status = Status.WAIT_FOR_ME
        self.send_message(
            FSMOutput(
                message_data=MessageData(body="Hey there! Welcome to BNSS-CrPC Bot!"),
                type=MessageType.TEXT,
                dialog="language",
                dest="channel",
            )
        )
        self.status = Status.WAIT_FOR_USER_INPUT

    def on_enter_ask_for_question(self):
        self.status = Status.WAIT_FOR_ME
        self.send_output("Proceed to ask your question")
        self.variables["history"] = []
        self.status = Status.WAIT_FOR_USER_INPUT

    def on_exit_ask_for_question(self):
        self.variables["query"] = self.current_input
        self.variables["history"].append(
            {"name": "User", "message": self.variables["query"]}
        )

    def on_enter_categorise(self):
        self.status = Status.WAIT_FOR_ME
        out = LLMManager.llm(
            messages=[
                LLMManager.sm(
                    f"""
                        You are an assistant who answers questions related to laws under the categories BNS, IPC, BNSS, and CrPC. BNS is the newer set of laws under IPC, and BNSS is the newer set of laws under CrPC. 
                        Your task is to understand the user's query and determine if it belongs to the BNS-IPC collection or the BNSS-CrPC collection.
                        If there isn't adequate information to categorize the query into one of the two collections, generate a simple, meaningful follow-up question.
                        

                        Your follow-up question must be concise and aimed at helping you categorize the query correctly.

                        If there isn't enough information to categorize and none of the terms BNS, BNSS, IPC, or CrPC are mentioned, generate a follow-up question.
                        
                        Example follow-up questions:

                        Are you asking about the laws under BNS or IPC?
                        Could you please specify if your query relates to BNSS or CrPC?
                        Can you clarify whether your question is about BNS or BNSS?
                        Which specific law are you referring to: IPC or CrPC?
                        Please indicate if your query concerns BNS, IPC, BNSS, or CrPC.

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

        self.variables["history"].append({"name": "Bot", "message": out})

        self.send_output(out)
        self.status = Status.MOVE_FORWARD

    def on_enter_category_response(self):
        self.status = Status.WAIT_FOR_ME
        self.status = Status.WAIT_FOR_USER_INPUT

    def on_exit_category_response(self):
        self.variables["history"].append(
            {"name": "User", "message": self.current_input}
        )

    def on_enter_extract_section(self):
        self.status = Status.WAIT_FOR_ME
        self.variables["metadata"] = {}
        out = LLMManager.llm(
            messages=[
                LLMManager.sm(
                    f"""
            You are a helpful assistant who helps with extracting the BNS Section and sub section number or IPC section and sub section number or BNSS section and sub section number or CrPC section and sub section number from the given query and chat history. 
            Do not make up any section numbers. Do not get confused between each term.
            Return the answer only as a python dictionary in the following format:

            If any of "bnss_section_no", "bnss_sub_section", "crpc_section_no", "crpc_sub_section", "bns_section_no", "bns_sub_section", "ipc_section_no", "ipc_sub_section" is null, do not include them in the dictionary.
            {{
                "bnss_section_no": <extracted-bnss-section-number>,
                "bnss_sub_section": <extracted-bnss-sub-section-number>,
                "crpc_section_no": <extracted-crpc-section-number>,
                "crpc_sub_section": <extracted-crpc-sub-section-number>,
                "bns_section_no": <extracted-bns-section-number>,
                "bns_sub_section": <extracted-bns-sub-section-number>,
                "ipc_section_no": <extracted-ipc-section-number>,
                "ipc_sub_section": <extracted-ipc-sub-section-number>,
            }}

                [Query]
                {self.variables["query"]}
                [Chat History]
                {self.variables["history"]}
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

        self.variables["metadata"] = dictionary
        self.status = Status.MOVE_FORWARD

    def on_enter_fetch_answer(self):
        self.status = Status.WAIT_FOR_ME
        self.send_message(
            FSMOutput(
                message_data=MessageData(body=self.variables["query"]),
                dest="rag",
                metadata=self.variables["metadata"],
            )
        )
        self.status = Status.WAIT_FOR_USER_INPUT

    def on_enter_generate_response(self):
        self.status = Status.WAIT_FOR_ME
        chunks = self.current_input
        chunks = json.loads(chunks)["chunks"]
        knowledge = "\n".join([row["chunk"] for row in chunks])

        if len(chunks) == 0:
            self.cb(
                FSMOutput(
                    text="Sorry, I don't have information about this. Please try again with a different query.\n"
                )
            )
            self.status = Status.MOVE_FORWARD
        else:

            result = LLMManager.llm(
                messages=[
                    LLMManager.sm(
                        f"""
                    You are a helpful assistant who helps with answering questions.
                    Given the user's query [query]{self.variables["query"]}, you should display the corresponding answer from the given chunk knowledge in the exact format.
                    Understand if the user query is under the BNSS or the older CrPC act, as specified in your knowledge chunks and clearly display the respective answer.
                    If there are multiple relevant knowledge chunks, display all of them.
                    Do not make up or paraphrase any answers. Ensure that the output is presented in string format.
                    
                    [query] {self.variables["query"]}
                    [Knowledge] {knowledge}  
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
        self.variables["history"] = []
        self.status = Status.WAIT_FOR_USER_INPUT

    def on_exit_ask_another_question(self):
        self.variables["query"] = self.current_input
        self.variables["history"].append(
            {"name": "User", "message": self.variables["query"]}
        )
