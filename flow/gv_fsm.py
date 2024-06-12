import json
from typing import Any, Dict, Type

from jb_manager_bot import (
    AbstractFSM,
    FSMOutput,
    MessageData,
    MessageType,
    Status,
    OptionsListType,
)
from jb_manager_bot.parsers.utils import LLMManager


class GV(AbstractFSM):
    states = [
        "zero",
        "select_language",
        "ask_for_question",
        "extract_category",
        "fetch_answer",
        "generate_response",
        "feedback",
        "register_query",
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
            "conditions": "is_in_scope",
        },
        {
            "trigger": "next",
            "source": "extract_category",
            "dest": "register_query",
            "conditions": "is_not_in_scope",
        },
        {
            "trigger": "next",
            "source": "fetch_answer",
            "dest": "generate_response",
        },
        {
            "trigger": "next",
            "source": "generate_response",
            "dest": "feedback",
        },
        {
            "trigger": "next",
            "source": "feedback",
            "dest": "ask_another_question",
            "conditions": "is_satisfied",
        },
        {
            "trigger": "next",
            "source": "feedback",
            "dest": "register_query",
            "conditions": "is_not_satisfied",
        },
        {
            "trigger": "next",
            "source": "register_query",
            "dest": "ask_another_question",
        },
    ]

    conditions = {"is_in_scope", "is_not_in_scope", "is_satisfied", "is_not_satisfied"}
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

    def is_satisfied(self):
        return self.current_input == "1"

    def is_not_satisfied(self):
        return self.current_input == "2"

    def is_in_scope(self):
        return self.variables["category"] is True

    def is_not_in_scope(self):
        return self.variables["category"] is False

    def on_enter_select_language(self):
        self.status = Status.WAIT_FOR_ME
        self.send_message(
            FSMOutput(
                message_data=MessageData(body="Hey there! Welcome to Gram Vaani Bot!"),
                type=MessageType.TEXT,
                dialog="language",
                dest="channel",
            )
        )
        self.status = Status.WAIT_FOR_USER_INPUT

    def on_enter_ask_for_question(self):
        self.status = Status.WAIT_FOR_ME
        self.send_output("Proceed to ask your question regarding provident fund, etc")
        self.status = Status.WAIT_FOR_USER_INPUT

    def on_exit_ask_for_question(self):
        self.variables["query"] = self.current_input

    def on_enter_extract_category(self):
        self.status = Status.WAIT_FOR_ME
        out = LLMManager.llm(
            messages=[
                LLMManager.sm(
                    f"""
            You are a helpful assistant specialized in answering queries related to EPF (Employees' Provident Fund), Provident Fund, Pensions, NREGA (National Rural Employment Guarantee Act), and similar topics. You will provide accurate and relevant information based on the given context.
Return true if the query is under any of the following domains : EPF (Employees' Provident Fund), Provident Fund, Pensions, NREGA (National Rural Employment Guarantee Act), and similar topics , else return false.

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

        self.variables["category"] = bool(out)
        self.status = Status.MOVE_FORWARD

    def on_enter_fetch_answer(self):
        self.status = Status.WAIT_FOR_ME
        self.send_message(
            FSMOutput(
                message_data=MessageData(body=self.variables["query"]),
                dest="rag",
                metadata={},
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
You are an helpful assistant who helps with answering questions. Given the user's query, you should display the corresponding answer from the given chunk knowledge in the exact format. Do not make up or paraphrase any answers. Ensure that the output is presented in string format.
                    
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
        self.variables["answer"] = result
        self.send_output(result)
        self.status = Status.MOVE_FORWARD

    def on_enter_feedback(self):
        self.status = Status.WAIT_FOR_ME
        self.send_message(
            FSMOutput(
                message_data=MessageData(body="Are you satisfied with the answer?"),
                type=MessageType.INTERACTIVE,
                options_list=[
                    OptionsListType(id="1", title="Yes"),
                    OptionsListType(id="2", title="No"),
                ],
            )
        )
        self.status = Status.WAIT_FOR_USER_INPUT

    def on_enter_register_query(self):
        self.status = Status.WAIT_FOR_ME
        self.variables["review_query"] = self.variables["query"]
        self.send_message(
            FSMOutput(
                message_data=MessageData(
                    body="Registering your query for review, will get back to you soon."
                ),
            )
        )
        self.status = Status.MOVE_FORWARD

    def on_enter_ask_another_question(self):
        self.status = Status.WAIT_FOR_ME
        self.status = Status.WAIT_FOR_USER_INPUT

    def on_exit_ask_another_question(self):
        self.variables["query"] = self.current_input
