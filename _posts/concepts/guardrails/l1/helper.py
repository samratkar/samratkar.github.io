import os
from typing import List, Tuple

import ipywidgets as widgets
import numpy as np
from dotenv import find_dotenv, load_dotenv
from guardrails import Guard, settings
from IPython.display import display
from sentence_transformers import SentenceTransformer

# these expect to find a .env file at the directory above the lesson.
# the format for that file is (without the comment)#API_KEYNAME=AStringThatIsTheLongAPIKeyFromSomeService


def load_env():
    _ = load_dotenv(find_dotenv())


def get_openai_api_key():
    load_env()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_api_key


def get_guardrails_api_key():
    load_env()
    guardrails_api_key = os.getenv("GUARDRAILS_API_KEY")
    return guardrails_api_key


class ChatWidget:
    def __init__(self, client=None, guard_name=None, system_message=None):
        """
        A widget for handling chat interactions.

        Parameters
        ----------
        client : object
            The OpenAI client object to use for generating responses.
        system_message : str, optional
            An optional system message to initialize the chat with.
        """
        self.chat_logs = []
        self.messages = []
        if system_message:
            self.messages.append({"role": "assistant", "content": system_message})
        # self.main_output = widgets.Output()

        self.text_input = widgets.Textarea(
            value="",
            placeholder="Type something then click the blue submit button.",
            disabled=False,
            continuous_update=False,
            layout=widgets.Layout(width="400px", height="75px"),
            form="chatform",
        )

        self.text_input.observe(self.handle_submit, names="value")
        self.submit_button = widgets.Button(
            form="chatform",
            icon="paper-plane",
            button_style="primary",
            type="submit",
            layout=widgets.Layout(width="40px", margin_y="auto"),
        )

        action_bar = widgets.HBox(
            [
                self.text_input,
                widgets.VBox(
                    [self.submit_button],
                    layout=widgets.Layout(justify_content="center", margin_y="auto"),
                ),
            ],
            layout=widgets.Layout(
                justify_content="center", width="480px", padding_y="10px"
            ),
        )

        self.chat_box = widgets.VBox(
            [], layout=widgets.Layout(max_height="300px", overflow_y="auto")
        )
        self.main_container = widgets.VBox(
            [self.chat_box, action_bar],
            layout=widgets.Layout(width="505px", justify_content="center"),
        )
        self._client = client
        self._guard_name = guard_name

    @property
    def client(self):
        return self._client

    @client.setter
    def client(self, value):
        self._client = value

    def reset(self):
        self.chat_logs = []
        self.messages = []
        self.chat_box.children = self.chat_logs

    def create_msg_widget(self, type, content, is_error=False):
        """Utility function to create a message widget based on the type"""
        common_style = """
            padding: 8px;
            margin: 2px 0;
            border-radius: 5px;
            width: fit-content;
            max-width: 70%;
            word-wrap: break-word;
            white-space: pre-wrap;
            overflow-wrap: break-word;
            line-height: 1.4;
        """

        if type == "user":
            style = (
                common_style
                + "justify-content: flex-end; background-color: #f0f0f0; float: right;"
            )
        elif type == "bot":
            style = common_style + "justify-content: flex-start;"
        else:
            raise ValueError("Type must be either 'user' or 'bot'")

        html_content = f'<div style="{style}">{content}</div>'
        return widgets.HTML(html_content)

    def update_chat_box(self, user_msg, bot_msg, error=False):
        user_widget = self.create_msg_widget("user", user_msg)
        bot_widget = self.create_msg_widget("bot", bot_msg, error)
        self.chat_logs.extend([user_widget, bot_widget])
        self.chat_box.children = self.chat_logs

    def show_loading(self, message):
        user_widget = self.create_msg_widget("user", message)
        loading_widget = self.create_msg_widget("bot", "Thinking...")
        loading_chat_logs = self.chat_logs.copy()
        loading_chat_logs.extend([user_widget, loading_widget])
        self.chat_box.children = loading_chat_logs

    def handle_submit(self, change):
        if (
            change["type"] == "change"
            and change["name"] == "value"
            and change["new"] != ""
        ):
            user_msg = change["new"]
            self.show_loading(user_msg)
            change["owner"].value = ""

            # self.remove_loading()
            query_messages = self.messages.copy()
            query_messages.append({"role": "user", "content": user_msg})

            # get the bot response
            error = False
            try:
                bot_message = self.bot_response_generator(query_messages)

                # write the user msg and bot response back in to message history
                self.messages.append({"role": "user", "content": user_msg})
                self.messages.append({"role": "assistant", "content": bot_message})

            except Exception as e:
                print(e)

                # we don't write here cuz it's errors
                bot_message = str(e)
                error = True

            # Clear the input after submission
            self.update_chat_box(user_msg, bot_message, error)

    def display(self):
        display(self.main_container)

    def bot_response_generator(self, message_history):
        if self.client:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=message_history,
                seed=42,
                temperature=0.0,
            )
            bot_msg = response.choices[0].message.content
            return bot_msg
        else:
            settings.use_server = True
            response = Guard(name=self._guard_name)(
                model="gpt-3.5-turbo",
                messages=message_history,
            )

            settings.use_server = False
            return response.validated_output


def chunk_markdown_files(directory):
    chunks = []

    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            # Split content into lines
            lines = content.split("\n")

            title = os.path.splitext(filename)[0]
            current_h1 = ""
            current_h2 = ""
            current_content = []

            for line in lines:
                if line.startswith("# "):
                    # New h1 header
                    if current_content:
                        chunks.append(
                            format_chunk(title, current_h1, current_h2, current_content)
                        )
                    current_h1 = line[2:].strip()
                    current_h2 = ""
                    current_content = []
                elif line.startswith("## "):
                    # New h2 header
                    if current_content:
                        chunks.append(
                            format_chunk(title, current_h1, current_h2, current_content)
                        )
                    current_h2 = line[3:].strip()
                    current_content = []
                else:
                    # Content (including h3 headers and list items)
                    current_content.append(line)

            # Add the last chunk
            if current_content:
                chunks.append(
                    format_chunk(title, current_h1, current_h2, current_content)
                )

    return chunks


def format_chunk(title, h1, h2, content):
    section_info = f"{h1}/{h2}" if h2 else h1
    content_text = "\n".join(content).strip()
    return f"Title: {title}\nSection: {section_info}\n{content_text}"


class SimpleVectorDB:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = []
        self.strings = []

    def add_strings(self, strings: List[str]):
        new_embeddings = self.model.encode(strings)
        self.embeddings.extend(new_embeddings)
        self.strings.extend(strings)

    def query(
        self, query_string: str, k: int, threshold: float
    ) -> List[Tuple[str, float]]:
        query_embedding = self.model.encode([query_string])[0]

        if not self.embeddings:
            return []

        embeddings_array = np.array(self.embeddings)

        # Calculate cosine similarities
        similarities = np.dot(embeddings_array, query_embedding) / (
            np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_embedding)
        )

        # Convert similarities to distances (1 - similarity)
        distances = 1 - similarities
        # distances = similarities

        # Sort indices by distance
        sorted_indices = np.argsort(distances)

        results = []
        for idx in sorted_indices:
            if distances[idx] < threshold and len(results) < k:
                results.append((self.strings[idx], float(distances[idx])))
            else:
                break

        results.reverse()

        return results

    @classmethod
    def from_files(cls, directory: str):
        chunks = chunk_markdown_files(directory)
        db = cls()
        db.add_strings(chunks)
        return db


class RAGChatWidget(ChatWidget):
    def __init__(
        self,
        client=None,
        guard_name=None,
        system_message=None,
        vector_db=None,
        # data_directory="shared_data/",
    ):
        super().__init__(
            client=client, guard_name=guard_name, system_message=system_message
        )
        self.vector_db = vector_db
        # self.data_directory = data_directory

        # self.hydrate_vector_db()

    def hydrate_vector_db(self):
        chunks = chunk_markdown_files(self.data_directory)
        self.vector_db.add_strings(chunks)

    def bot_response_generator(self, message_history, context=None):
        if self.client:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=message_history,
                seed=42,
                temperature=0.0,
            )
            bot_msg = response.choices[0].message.content
            return bot_msg
        else:
            # Context is a list of touples, we want to map down to the first value in the tuples
            sources = [c[0] for c in context]
            settings.use_server = True

            response = Guard(name=self._guard_name)(
                model="gpt-3.5-turbo",
                messages=message_history,
                metadata={"sources": sources, "chunk_strategy": "sentence"},
            )

            settings.use_server = False

            return response.validated_output

    def handle_submit(self, change):
        if (
            change["type"] == "change"
            and change["name"] == "value"
            and change["new"] != ""
        ):
            # extract message user sent
            user_msg = change["new"]
            self.show_loading(user_msg)
            # Clear the input after submission
            change["owner"].value = ""

            context = self.retrieve(user_msg, k=3)

            # do retrieval, add to message history
            augmented_user_msg = self.retrieval_augmentation(user_msg, context)
            error = False
            query_messages = self.messages.copy()
            query_messages.append({"role": "user", "content": augmented_user_msg})
            # get the bot response
            try:
                bot_message = self.bot_response_generator(
                    query_messages, context=context
                )

                # write the user msg and bot response back in to message history
                self.messages.append({"role": "user", "content": augmented_user_msg})
                self.messages.append({"role": "assistant", "content": bot_message})

            except Exception as e:
                print(e)

                # we don't write here cuz it's errors
                bot_message = e.body['detail']
                error = True

            # We show user_msg here instead of the augmented_user_msg to hide the retrieval
            self.update_chat_box(user_msg, bot_message, error)

    def retrieve(self, user_msg, k=1, threshold=0.9):
        retrieval = self.vector_db.query(user_msg, k=k, threshold=threshold)
        retrieved_ctx = ""
        for idx, (ctx, _) in enumerate(retrieval):
            retrieved_ctx += f"# Context {idx + 1}:\n{ctx}\n\n"
        # return retrieval
        return retrieved_ctx

    def retrieval_augmentation(self, user_msg, retrieval):
        augmented_user_msg = f"""\n
Use this context to help answer the question:

{retrieval}

User message:
{user_msg}
"""

        return augmented_user_msg


# Example usage
if __name__ == "__main__":
    directory = "shared_data/"
    chunks = chunk_markdown_files(directory)
    for chunk in chunks:
        print(chunk)
        print("-" * 50)
