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


class Maker(AbstractFSM):
    states = [
        "zero",
        "select_language",
        "ask_for_question",
        "fetch_answer",
        "generate_response",
        "ask_another_question",
        "end",
    ]

    transitions = [
        {"trigger": "next", "source": "zero", "dest": "select_language"},
        {"trigger": "next", "source": "select_language", "dest": "ask_for_question"},
        {"trigger": "next", "source": "ask_for_question", "dest": "fetch_answer"},
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
            "dest": "fetch_answer",
        },
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
        self.credentials["AZURE_OPENAI_API_KEY"] = credentials.get(
            "AZURE_OPENAI_API_KEY"
        )
        self.credentials["AZURE_OPENAI_API_VERSION"] = credentials.get(
            "AZURE_OPENAI_API_VERSION"
        )
        self.credentials["AZURE_OPENAI_API_ENDPOINT"] = credentials.get(
            "AZURE_OPENAI_ENDPOINT"
        )

        self.plugins: Dict[str, AbstractFSM] = {}
        super().__init__(send_message=send_message)

    def send_output(self, message):
        self.send_message(FSMOutput(message_data=MessageData(body=message)))

    def on_enter_select_language(self):
        self.status = Status.WAIT_FOR_ME
        self.send_message(
            FSMOutput(
                message_data=MessageData(
                    body="Hey there! Welcome to MakeForJustice Bot!"
                ),
                type=MessageType.TEXT,
                dialog="language",
                dest="channel",
            )
        )
        self.status = Status.WAIT_FOR_USER_INPUT

    def on_enter_ask_for_question(self):
        self.status = Status.WAIT_FOR_ME
        self.send_output("Proceed to ask your query")
        self.status = Status.WAIT_FOR_USER_INPUT

    def on_exit_ask_for_question(self):
        self.variables["query"] = self.current_input

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
You are an helpful assistant who helps with answering questions. Given the user's query, you should display the corresponding answer from the given chunk knowledge. Ensure that the output is presented in string format.
                    
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

    def on_enter_ask_another_question(self):
        self.status = Status.WAIT_FOR_ME
        self.status = Status.WAIT_FOR_USER_INPUT

    def on_exit_ask_another_question(self):
        self.variables["query"] = self.current_input
