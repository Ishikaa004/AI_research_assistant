import json
import os

from langchain_core.messages import HumanMessage, AIMessage


class ConversationMemory:

    def __init__(self, file_path="data/chat_history.json"):

        self.file_path = file_path

        self.history = []

        self.load()


    # ==========================================
    # Add user message
    # ==========================================

    def add_user_message(self, message):

        self.history.append(
            HumanMessage(
                content=message
            )
        )

        self.save()


    # ==========================================
    # Add AI message
    # ==========================================

    def add_ai_message(self, message):

        self.history.append(
            AIMessage(
                content=message
            )
        )

        self.save()


    # ==========================================
    # Get conversation history
    # ==========================================

    def get_history(self):

        return self.history


    # ==========================================
    # Save history to JSON
    # ==========================================

    def save(self):

        os.makedirs(
            os.path.dirname(self.file_path),
            exist_ok=True
        )

        data = []

        for message in self.history:

            if isinstance(
                message,
                HumanMessage
            ):

                message_type = "human"

            else:

                message_type = "ai"


            data.append(
                {
                    "type": message_type,
                    "content": message.content
                }
            )


        with open(
            self.file_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False
            )


    # ==========================================
    # Load history from JSON
    # ==========================================

    def load(self):

        if not os.path.exists(
            self.file_path
        ):

            return


        try:

            with open(
                self.file_path,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)


            for message in data:

                if message["type"] == "human":

                    self.history.append(
                        HumanMessage(
                            content=message["content"]
                        )
                    )

                elif message["type"] == "ai":

                    self.history.append(
                        AIMessage(
                            content=message["content"]
                        )
                    )


        except (
            json.JSONDecodeError,
            KeyError,
            TypeError
        ):

            print(
                "Warning: Could not load chat history."
            )


    # ==========================================
    # Clear history
    # ==========================================

    def clear(self):

        self.history = []

        self.save()