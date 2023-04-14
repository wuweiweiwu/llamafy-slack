import os
import re
import time
import json

from sqlalchemy import (
    MetaData,
    Table,
    create_engine,
    inspect,
    select,
    text,
    Inspector,
    Engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError
from sqlalchemy.schema import CreateTable
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
import openai

# Load default environment variables (.env)
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")

openai.api_key = OPENAI_API_KEY


def format_index(index: dict) -> str:
    return (
        f'Name: {index["name"]}, Unique: {index["unique"]},'
        f' Columns: {str(index["column_names"])}'
    )


def get_table_indexes(inspector: Inspector, table: Table) -> str:
    indexes = inspector.get_indexes(table.name)
    indexes_formatted = "\n".join(map(format_index, indexes))
    return f"Table Indexes:\n{indexes_formatted}"


def get_sample_rows(engine: Engine, table: Table) -> str:
    sample_rows_in_table_info = 3

    # build the select command
    command = select(table).limit(sample_rows_in_table_info)

    # save the columns in string format
    columns_str = "\t".join([col.name for col in table.columns])

    try:
        # get the sample rows
        with engine.connect() as connection:
            sample_rows = connection.execute(command)
            # shorten values in the sample rows
            sample_rows = list(map(lambda ls: [str(i)[:100] for i in ls], sample_rows))

        # save the sample rows in string format
        sample_rows_str = "\n".join(["\t".join(row) for row in sample_rows])

    # in some dialects when there are no rows in the table a
    # 'ProgrammingError' is returned
    except ProgrammingError:
        sample_rows_str = ""

    return (
        f"{sample_rows_in_table_info} rows from {table.name} table:\n"
        f"{columns_str}\n"
        f"{sample_rows_str}"
    )


def get_table_info():
    """
    Get table info from the database
    https://github.com/hwchase17/langchain/blob/634358db5e9d0f091c66c82b8ed1379ec6531f88/langchain/sql_database.py#L128
    """
    engine = create_engine("sqlite:///./chinook.db")
    inspector = inspect(engine)
    metadata = MetaData()
    metadata.reflect(bind=engine)

    table_names = inspector.get_table_names()
    dialect = "sqlite"

    meta_tables = [
        tbl
        for tbl in metadata.sorted_tables
        if tbl.name in set(table_names)
        and not (dialect == "sqlite" and tbl.name.startswith("sqlite_"))
    ]

    tables = []
    for table in meta_tables:
        # add create table command
        create_table = str(CreateTable(table).compile(engine))
        table_info = f"{create_table.rstrip()}"

        # extra info
        table_info += "\n\n/*"
        table_info += f"\n{get_table_indexes(inspector,table)}\n"
        table_info += f"\n{get_sample_rows(engine,table)}\n"
        table_info += "*/"

        tables.append(table_info)

    final_str = "\n\n".join(tables)
    return final_str


def openai_call(
    prompt: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.5,
    max_tokens: int = 100,
):
    while True:
        try:
            # Use chat completion API
            messages = [{"role": "system", "content": prompt}]
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                stop=None,
            )

            return response.choices[0].message.content.strip()
        except openai.error.RateLimitError:
            print(
                "The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again."
            )
            time.sleep(10)  # Wait 10 seconds and try again


# Initializes your app with your bot token and socket mode handler
app = App(token=SLACK_BOT_TOKEN)


# Listens to incoming messages that contain "hello"
@app.event("app_mention")
def handle_mentions(event, client, say):
    thread_ts = event.get("thread_ts", None) or event["ts"]
    prompt = re.sub("\\s<@[^, ]*|^<@[^, ]*", "", event["text"])

    # say() sends a message to the channel where the event was triggered
    say(
        # blocks=[
        #     {
        #         "type": "section",
        #         "text": {"type": "mrkdwn", "text": f"Hey there <@{event['user']}>!"},
        #         "accessory": {
        #             "type": "button",
        #             "text": {"type": "plain_text", "text": "Click Me"},
        #             "action_id": "button_click",
        #             "value": thread_ts,
        #         },
        #     }
        # ],
        text=prompt,
        thread_ts=thread_ts,
    )


@app.event("message")
def handle_message_events(body, logger):
    logger.info(body)


# Start your app
if __name__ == "__main__":
    print(get_table_info())
    # SocketModeHandler(app, SLACK_APP_TOKEN).start()
