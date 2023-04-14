import os
import re
import time
import json
from typing import Dict, List
from collections import OrderedDict

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

# global engine
# needs to be dynamic in the future
ENGINE = create_engine("sqlite:///./chinook.db")

DIALECT = "sqlite"


def get_open_ai_completion(
    messages: List[Dict],
    model: str = "gpt-3.5-turbo",
    temperature: float = 0,
):
    while True:
        try:
            # Use chat completion API
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )

            return response.choices[0]
        except openai.error.RateLimitError:
            print(
                "The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again."
            )
            time.sleep(10)  # Wait 10 seconds and try again


# def format_index(index: dict) -> str:
#     return (
#         f'Name: {index["name"]}, Unique: {index["unique"]},'
#         f' Columns: {str(index["column_names"])}'
#     )


def extract_text_from_markdown(text: str) -> str:
    matches = re.findall(r"```([\s\S]+?)```", text)
    if matches:
        return matches[0]
    return text


def extract_sql_query_from_message(text: str) -> Dict:
    try:
        data = json.loads(text)
    except Exception as e:
        print("e: ", e)
        raise e

    if data.get("MissingData"):
        return data

    sql = data["SQL"]

    return {"SQL": sql}


# def get_table_indexes(table: Table) -> str:
#     inspector = inspect(ENGINE)
#     indexes = inspector.get_indexes(table.name)
#     indexes_formatted = "\n".join(map(format_index, indexes))
#     return f"Table Indexes:\n{indexes_formatted}"


# def get_sample_rows(table: Table) -> str:
#     sample_rows_in_table_info = 3

#     # build the select command
#     command = select(table).limit(sample_rows_in_table_info)

#     # save the columns in string format
#     columns_str = "\t".join([col.name for col in table.columns])

#     try:
#         # get the sample rows
#         with ENGINE.connect() as connection:
#             sample_rows = connection.execute(command)
#             # shorten values in the sample rows
#             sample_rows = list(map(lambda ls: [str(i)[:100] for i in ls], sample_rows))

#         # save the sample rows in string format
#         sample_rows_str = "\n".join(["\t".join(row) for row in sample_rows])

#     # in some dialects when there are no rows in the table a
#     # 'ProgrammingError' is returned
#     except ProgrammingError:
#         sample_rows_str = ""

#     return (
#         f"{sample_rows_in_table_info} rows from {table.name} table:\n"
#         f"{columns_str}\n"
#         f"{sample_rows_str}"
#     )


def get_table_info(table_names: List[str] = None) -> str:
    """
    Get table info from the database
    CREATE statements
    https://github.com/hwchase17/langchain/blob/634358db5e9d0f091c66c82b8ed1379ec6531f88/langchain/sql_database.py#L128
    """
    inspector = inspect(ENGINE)
    metadata = MetaData()
    metadata.reflect(bind=ENGINE)

    if not table_names:
        table_names = inspector.get_table_names()

    meta_tables = [
        tbl
        for tbl in metadata.sorted_tables
        if tbl.name in set(table_names)
        and not (DIALECT == "sqlite" and tbl.name.startswith("sqlite_"))
    ]

    tables = []
    for table in meta_tables:
        # add create table command
        create_table = str(CreateTable(table).compile(ENGINE))
        table_info = f"{create_table.rstrip()}"

        # extra info
        # table_info += "\n\n/*"
        # table_info += f"\n{get_table_indexes(table)}\n"
        # table_info += f"\n{get_sample_rows(table)}\n"
        # table_info += "*/"

        tables.append(table_info)

    final_str = "\n\n".join(tables)
    return final_str

    # return "\n\n".join(table_names)


def execute_sql(sql_query: str) -> Dict:
    with ENGINE.connect() as connection:
        with connection.begin():
            sql_text = text(sql_query)
            result = connection.execute(sql_text)

        column_names = list(result.keys())
        rows = [list(r) for r in result.all()]

        results = []
        for row in rows:
            result = OrderedDict()
            for i, column_name in enumerate(column_names):
                result[column_name] = row[i]
            results.append(result)

        return {
            "column_names": column_names,
            "results": results,
        }


# def get_table_selection_messages():
#     """
#     system messages and few shot examples

#     """
#     # default_messages = [
#     #     {
#     #         "role": "system",
#     #         "content": (
#     #             "You are a helpful assistant for identifying relevant SQL tables to use for answering a natural language query."
#     #             ' You respond in JSON format with your answer in a field named "tables" which is a list of strings.'
#     #             " Respond with an empty list if you cannot identify any relevant tables."
#     #             # " Write your answer in markdown format."
#     #             "\n"
#     #             "The following are descriptions of available tables and enums:\n"
#     #             "---------------------\n" + get_table_info() + "---------------------\n"
#     #         ),
#     #     }
#     # ]

#     # default_messages.extend([

#     # ])
#     return []


def get_table_selection_prompt(
    natural_language_query: str,
) -> str:
    return f"""
You are an expert data scientist.
Return a JSON object with relevant SQL tables for answering the following natural language query:
---------------
{natural_language_query}
---------------
Respond in JSON format with your answer in a field named \"tables\" which is a list of strings.
Respond with an empty list if you cannot identify any relevant tables.
Write your answer in markdown format.

The following are the scripts that created the tables:
---------------------
{get_table_info()} 
--------------------- 

in your answer, provide the following information:

- <one to two sentence comment explaining what tables can be relevant goes here>
- <for each table identified, comment double checking the table is in the schema above along with what the first column in the table is or (none) if it doesn't exist. be careful that any tables suggested were actually above>
- <if any tables were incorrectly identified, make a note here about what tables from the schema should actually be used if any>
- the markdown formatted like this:
```
<json of the tables>
```

Thanks!
"""


def get_relevant_tables(natural_language_query: str) -> List[str]:
    """
    Identify relevant tables for answering a natural language query via LM
    """

    # messages = get_table_selection_messages().copy()
    messages = []

    prompt = get_table_selection_prompt(
        natural_language_query=natural_language_query,
    )
    messages.append({"role": "user", "content": prompt})

    assistant_message = get_open_ai_completion(
        messages=messages,
        model="gpt-3.5-turbo",
    )["message"]["content"]

    print(assistant_message)

    tables_json_str = extract_text_from_markdown(assistant_message)
    tables = json.loads(tables_json_str).get("tables")

    return tables


# def get_rephrase_prompt(
#     natural_language_query: str,
#     table_info: str,
# ):
#     return f"""
# Let's start by rephrasing the query to be more analytical. Use the schema context to rephrase the user question in a way that leads to optimal query results: {natural_language_query}
# The following are schemas of tables you can query as well as sample rows in that table:
# ---------------------
# {table_info}
# ---------------------

# Do not include any of the table names in the query.
# Ask the natural language query the way a data analyst, with knowledge of these tables, would.
# """


def get_retry_prompt(natural_language_query: str, table_info: str) -> str:
    return f"""You are an expert and empathetic database engineer that is generating correct read-only {DIALECT} query to answer the following question/command: {natural_language_query}

We already created the tables in the database with the following CREATE TABLE code:
---------------------
{table_info}
---------------------

Ensure to include which table each column is from (table.column)
Use CTE format for computing subqueries.

Provide a properly formatted JSON object with the following information. Ensure to escape any special characters so it can be parsed as JSON.

{{
    "Schema": "<1 to 2 sentences about the tables/columns/enums above to use>",
    "Applicability": "<1 to 2 sentences about which columns and enums are relevant, or which ones are missing>",
    "SQL": "<your query>"
}}

However, if the tables don't contain all the required data (e.g. the column isn't there or there aren't relevant enums), instead return a JSON object with just: 

{{
    "Schema": "<1 to 2 sentences about the tables/columns/enums above to use>",
    "Applicability": "<1 to 2 sentences about which columns and enums are relevant, or which ones are missing>",
    "MissingData": "<1 to 2 sentences about what data is missing>"
}}

However, if a query can be close enough to the intent of the question/command, generate the SQL that gets it instead of returning MissingData.
"""


def get_try_again_prompt_with_error(error_message: str) -> str:
    return (
        "Try again. "
        f"Only respond with valid {DIALECT}. Write your answer in JSON. "
        f"The {DIALECT} query you just generated resulted in the following error message:\n"
        f"{error_message}"
        "Check the table schema and ensure that the columns for the table exist and will provide the expected results."
    )


def text_to_sql_with_retry(
    natural_language_query: str,
    table_names: List[str],
    k=3,
):
    """
    Tries to take a natural language query and generate valid SQL to answer it K times
    """

    table_info = get_table_info(table_names)

    # ask the assistant to rephrase before generating the query
    # rephrase = [{
    #     "role": "user",
    #     "content": make_rephrase_msg_with_schema_and_warnings().format(
    #         natural_language_query=natural_language_query,
    #         schemas=schemas
    #         )
    # }]
    # rephrased_query = get_assistant_message(rephrase)["message"]["content"]
    # print(f'[REPHRASED_QUERY] {rephrased_query}')
    # natural_language_query=rephrased_query

    content = get_retry_prompt(natural_language_query, table_info)

    # messages = make_default_messages(schemas, scope)
    messages = []
    messages.append({"role": "user", "content": content})

    assistant_message = None

    for _ in range(k):
        sql_query_data = {}
        try:
            assistant_message = get_open_ai_completion(
                messages,
                model="gpt-3.5-turbo",
            )["message"]["content"]

            print(assistant_message)

            sql_query_data = extract_sql_query_from_message(assistant_message)

            if sql_query_data.get("MissingData"):
                return {"MissingData": sql_query_data["MissingData"]}, ""

            sql_query = sql_query_data["SQL"]

            response = execute_sql(sql_query)

            # Generated SQL query did not produce exception. Return result
            return response, sql_query

        except Exception as e:
            print(f"Failed to execute sql query {sql_query} with {e}")

            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message["message"]["content"],
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": get_try_again_prompt_with_error(str(e)),
                }
            )

    print(f"Could not generate {DIALECT} query after {k} tries.")

    return None, None


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


# @app.event("message")
# def handle_message_events(body, logger):
#     logger.info(body)


# Start your app
if __name__ == "__main__":
    # print(get_table_info())

    question = "Who are the top 3 best selling artists?"

    # tables = get_relevant_tables(question)

    tables = ["invoices", "invoice_items", "tracks", "albums"]

    result, sql_query = text_to_sql_with_retry(question, tables)

    # SocketModeHandler(app, SLACK_APP_TOKEN).start()
