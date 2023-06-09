import os
import re
import time
import json
from typing import Dict, List, Any, Callable
from collections import OrderedDict
from datetime import datetime, date
from logging import Logger
from bson import ObjectId

from sqlalchemy import (
    MetaData,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.schema import CreateTable
from slack_bolt import App, BoltResponse
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
import openai
import vl_convert as vlc
from cloudinary.uploader import upload
import cloudinary
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Load default environment variables (.env)
load_dotenv()

# logging.basicConfig(level=logging.DEBUG)

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

CLOUDINARY_KEY = os.environ.get("CLOUDINARY_KEY")
CLOUDINARY_SECRET = os.environ.get("CLOUDINARY_SECRET")

ATLAS_URI = os.environ.get("ATLAS_URI")

assert SLACK_APP_TOKEN
assert SLACK_BOT_TOKEN
assert SLACK_SIGNING_SECRET
assert OPENAI_API_KEY
assert CLOUDINARY_KEY
assert CLOUDINARY_SECRET
assert ATLAS_URI

cloudinary.config(
    cloud_name="deqqzvauj",
    api_key=CLOUDINARY_KEY,
    api_secret=CLOUDINARY_SECRET,
    secure=True,
)

openai.api_key = OPENAI_API_KEY


# Initializes your app with your bot token and socket mode handler
app = App(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)

mongo_client = MongoClient(ATLAS_URI, server_api=ServerApi("1"))
llamafy_db = mongo_client.llamafy
questions_collection = llamafy_db.questions

# global engine
# needs to be dynamic in the future
example_db_engine = create_engine("sqlite:///./chinook.db")
example_db_dialect = example_db_engine.dialect.name

example_color_palette = ["#D4AFB9", "#D1CFE2", "#9CADCE", "#7EC4CF", "#52B2CF"]


sql_generation_few_shots = [
    {
        "user": "Which neighborhood had the most crime in 2021?",
        "assistant": "SELECT neighborhood, COUNT(*) as num_crimes \\nFROM sf_crime_incidents\\nWHERE occurred >= '2021-01-01' AND occurred < '2022-01-01'\\nGROUP BY neighborhood\\nORDER BY num_crimes DESC NULLS LAST\\nLIMIT 1;",
    },
    {
        "user": "What are the largest 3 neighborhoods?",
        "assistant": "SELECT stp1.neighborhood, sum(tract_population) as total_population\\nFROM sf_total_pop_by_census_tract stp1\\nGROUP BY stp1.neighborhood\\nORDER BY total_population DESC NULLS LAST\\nLIMIT 3;",
    },
    {
        "user": "where are the places with the most poop per capita?",
        "assistant": "WITH total_population AS (\\nSELECT neighborhood, tract_population\\nFROM sf_total_pop_by_census_tract\\n),\\nincident_counts AS (\\nSELECT\\nsfi.neighborhood,\\nCOUNT(*) as poop_count\\nFROM\\nsf_311_incidents sfi\\nWHERE\\nsfi.incident_type = 'Feces/Urine'\\nGROUP BY\\nsfi.neighborhood\\n),\\nneighborhood_stats AS (\\nSELECT\\nic.neighborhood,\\nSUM(tp.tract_population) as total_population,\\nic.poop_count\\nFROM\\nincident_counts ic\\nJOIN total_population tp ON ic.neighborhood = tp.neighborhood\\nGROUP BY\\nic.neighborhood, ic.poop_count\\n)\\nSELECT\\nneighborhood,\\ntotal_population,\\npoop_count,\\npoop_count / NULLIF(total_population, 0) as poop_per_capita\\nFROM\\nneighborhood_stats\\nORDER BY\\npoop_per_capita DESC NULLS LAST;",
    },
    {
        "user": "Where is the most violent crime by percentage of crime",
        "assistant": "SELECT neighborhood, \\n       100.0 * COUNT(CASE WHEN incident_type IN ('Aggravated Assault', 'Arson', 'Assault', 'Manslaughter', 'Robbery', 'Sexual Offense', 'Homicide') THEN 1 END) / COUNT(*) as violent_crime_percentage\\nFROM sf_crime_incidents\\nWHERE neighborhood IS NOT NULL\\nGROUP BY neighborhood\\nORDER BY violent_crime_percentage DESC NULLS LAST;",
    },
    {
        "user": "What are the top 5 neighborhoods with the most encampments per capita?",
        "assistant": "WITH total_population AS (\\nSELECT neighborhood, SUM(tract_population) AS total_population\\nFROM sf_total_pop_by_census_tract\\nGROUP BY neighborhood\\n),\\nencampments AS (\\nSELECT\\nsfi.neighborhood,\\ntp.total_population AS total_pop,\\nCOUNT(*) AS num_encampments\\nFROM\\nsf_311_incidents sfi\\nJOIN total_population tp ON sfi.neighborhood = tp.neighborhood\\nWHERE\\nsfi.incident_type = 'Homeless Encampments'\\nGROUP BY\\nsfi.neighborhood, tp.total_population\\n)\\nSELECT\\nneighborhood,\\ntotal_pop,\\nnum_encampments,\\nnum_encampments / NULLIF(total_pop, 0) AS encampments_per_capita\\nFROM\\nencampments\\nWHERE\\ntotal_pop > 0\\nORDER BY\\nencampments_per_capita DESC NULLS LAST\\nLIMIT 5;",
    },
    {
        "user": "Give me the number of poop incidents per capita in the Mission neighborhood",
        "assistant": "WITH stp AS (\\n    SELECT neighborhood, SUM(tract_population) AS total_population\\n    FROM sf_total_pop_by_census_tract\\n    GROUP BY neighborhood\\n), incidents_per_neighborhood AS (\\n    SELECT sfi.neighborhood, COUNT(*) AS num_incidents\\n    FROM sf_311_incidents sfi\\n    WHERE sfi.neighborhood = 'Mission' AND sfi.incident_type = 'Feces/Urine'\\n    GROUP BY sfi.neighborhood\\n)\\nSELECT\\n    incidents_per_neighborhood.neighborhood,\\n    incidents_per_neighborhood.num_incidents,\\n    incidents_per_neighborhood.num_incidents / NULLIF(SUM(stp.total_population), 0) AS incidents_per_capita\\nFROM stp\\nJOIN incidents_per_neighborhood ON stp.neighborhood = incidents_per_neighborhood.neighborhood\\nGROUP BY incidents_per_neighborhood.neighborhood, incidents_per_neighborhood.num_incidents;\\n",
    },
    {
        "user": "3 neighborhoods with the highest female to male ratio",
        "assistant": "WITH total_population AS (\\n    SELECT neighborhood, SUM(male_population + female_population) AS total_pop\\n    FROM sf_sex_by_census_tract\\n    GROUP BY neighborhood\\n), female_to_male_ratio AS (\\n    SELECT \\n        sfs.neighborhood, \\n        SUM(sfs.female_population) / NULLIF(SUM(sfs.male_population), 0) AS ratio\\n    FROM sf_sex_by_census_tract sfs\\n    JOIN total_population tp ON sfs.neighborhood = tp.neighborhood\\n    GROUP BY sfs.neighborhood\\n)\\nSELECT \\n    neighborhood, \\n    ratio\\nFROM female_to_male_ratio\\nORDER BY ratio DESC NULLS LAST\\nLIMIT 3;",
    },
    {
        "user": "which 5 neighborhoods had the most drug violations involving heroin?",
        "assistant": "WITH heroin_violations AS (\\n    SELECT neighborhood, COUNT(*) AS num_heroin_violations\\n    FROM sf_crime_incidents\\n    WHERE incident_type = 'Drug Violation' AND description ILIKE '%heroin%'\\n    GROUP BY neighborhood\\n)\\nSELECT neighborhood, num_heroin_violations\\nFROM heroin_violations\\nORDER BY num_heroin_violations DESC NULLS LAST\\nLIMIT 5;",
    },
    {
        "user": "How many crimes were related to guns?",
        "assistant": "SELECT COUNT(*) as num_gun_crimes\\nFROM sf_crime_incidents\\nWHERE description ~* '\\\\m(gun|firearm)\\\\M';",
    },
    {
        "user": "How many crimes had ties to knives?",
        "assistant": "SELECT COUNT(*) as num_gun_crimes\\nFROM sf_crime_incidents\\nWHERE description ~* '\\\\m(knife|stabbing)\\\\M';",
    },
]


class NotReadOnlyException(Exception):
    pass


def get_openai_completion(
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


def extract_text_from_markdown_code_block(text: str) -> str:
    matches = re.findall(r"```([\s\S]+?)```", text)
    if matches:
        return matches[0]
    return text


def extract_json_str_from_markdown_code_block(text: str) -> str:
    matches = re.findall(r"```([\s\S]+?)```", text)

    if matches:
        code_str = matches[0]
        match = re.search(r"(?i)json\s+(.*)", code_str, re.DOTALL)
        if match:
            code_str = match.group(1)
    else:
        code_str = text

    return code_str


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
    inspector = inspect(example_db_engine)
    metadata = MetaData()
    metadata.reflect(bind=example_db_engine)

    if not table_names:
        table_names = inspector.get_table_names()

    meta_tables = [
        tbl
        for tbl in metadata.sorted_tables
        if tbl.name in set(table_names)
        and not (example_db_dialect == "sqlite" and tbl.name.startswith("sqlite_"))
    ]

    tables = []
    for table in meta_tables:
        # add create table command
        create_table = str(CreateTable(table).compile(example_db_engine))
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
    if not is_read_only_query(sql_query):
        raise NotReadOnlyException("Only read-only queries are allowed.")

    with example_db_engine.connect() as connection:
        connection = connection.execution_options(postgresql_readonly=True)
        with connection.begin():
            sql_text = text(sql_query)
            result = connection.execute(sql_text)

        # # dump to csv
        # outfile = open("mydump.csv", "w")
        # outcsv = csv.writer(outfile)

        # # dump column titles (optional)
        # outcsv.writerow(list(result.keys()))
        # # dump rows
        # outcsv.writerows(result.fetchall())

        # outfile.close()

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


def is_read_only_query(sql_query: str):
    """
    Checks if the given SQL query string is read-only.
    Returns True if the query is read-only, False otherwise.
    """
    # List of SQL statements that modify data in the database
    modifying_statements = [
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "CREATE",
        "ALTER",
        "GRANT",
        "TRUNCATE",
        "LOCK TABLES",
        "UNLOCK TABLES",
    ]

    # Check if the query contains any modifying statements
    for statement in modifying_statements:
        if not sql_query or statement in sql_query.upper():
            return False

    # If no modifying statements are found, the query is read-only
    return True


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

    assistant_message = get_openai_completion(
        messages=messages,
        model="gpt-3.5-turbo",
    )["message"]["content"]

    print(assistant_message)

    tables_json_str = extract_text_from_markdown_code_block(assistant_message)
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


def get_sql_generation_prompt(natural_language_query: str, table_info: str) -> str:
    return f"""You are an expert and empathetic database engineer that is generating correct read-only {example_db_dialect} query to answer the following question/command: {natural_language_query}

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


def get_sql_generation_try_again_prompt_with_error(error_message: str) -> str:
    return (
        "Try again. "
        f"Only respond with valid {example_db_dialect}. Write your answer in JSON. "
        f"The {example_db_dialect} query you just generated resulted in the following error message:\n"
        f"{error_message}"
        "Check the table schema and ensure that the columns for the table exist and will provide the expected results."
    )


def generate_and_execute_sql(
    natural_language_query: str,
    table_names: List[str],
    max_retries=3,
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

    content = get_sql_generation_prompt(natural_language_query, table_info)

    # messages = make_default_messages(schemas, scope)
    messages = []

    # add few shots
    for example in sql_generation_few_shots:
        messages.append({"role": "user", "content": example["user"]})
        messages.append({"role": "assistant", "content": example["assistant"]})

    messages.append({"role": "user", "content": content})

    assistant_message = None

    for _ in range(max_retries):
        sql_query_data = {}

        try:
            assistant_message = get_openai_completion(
                messages,
                model="gpt-3.5-turbo",
            )[
                "message"
            ]["content"]

            print(assistant_message)

            sql_query_data = extract_sql_query_from_message(assistant_message)

            if sql_query_data.get("MissingData"):
                return {"MissingData": sql_query_data["MissingData"]}, ""

            sql_query = sql_query_data["SQL"]

            response = execute_sql(sql_query)

            # Generated SQL query did not produce exception. Return result
            return response, sql_query

        except Exception as e:
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message,
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": get_sql_generation_try_again_prompt_with_error(str(e)),
                }
            )

    print(
        f"Could not generate {example_db_dialect} query after {max_retries} tries.")

    return None, None


def get_conversational_answer_messages(
    natural_language_query: str,
    context,
) -> List[Any]:
    return [
        {
            "role": "system",
            "content": f"""
You are a helpful assistant for answering questions based on relevant data.
If you don't know the answer, say "Sorry! I don't have enough information to answer this question. Please rephrase your question and try again.", don't try to make up an answer.
If you do know the answer, provide a short answer that is correct and concise.
Make sure to format your answer so that it is easier to read.
The following is relevant data for answering the question:
----------------
{context}
""",
        },
        {
            "role": "user",
            "content": f"Provide an answer for the following question: {natural_language_query}",
        },
    ]


def get_conversational_answer(
    natural_language_query: str,
    context: str,
) -> str:
    messages = get_conversational_answer_messages(
        natural_language_query, context)

    assistant_message = get_openai_completion(
        messages,
        model="gpt-3.5-turbo",
    )[
        "message"
    ]["content"]

    return assistant_message


def get_visualization_messages(data: str):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a data visualization expert tasked with generating syntactically correct Vega-Lite specs that are best for visualizing the given data."
                " Make sure that ALL axis titles are human-readable and not snake_case or camelCase."
                f" Make sure only following colors are used: {', '.join(example_color_palette)}."
                " Write responses in markdown format."
            ),
        },
        {
            "role": "user",
            "content": (
                "Generate a syntactically correct Vega-Lite spec to best visualize the given data."
                "\n\n"
                f"{data}"
            ),
        },
    ]

    return messages


def get_visualization_json_spec(data: str):
    messages = get_visualization_messages(data)

    assistant_message = get_openai_completion(messages)["message"]["content"]

    print(assistant_message)

    vega_str = extract_json_str_from_markdown_code_block(assistant_message)

    vega_json = json.loads(vega_str)

    return vega_json


def get_visualization_image_url(spec: Any):
    png_data = vlc.vegalite_to_png(vl_spec=spec, scale=2)
    res = upload(png_data)

    return res["url"]


def get_sql_complexity_messages(sql_query: str):
    return [
        {
            "role": "system",
            "content": f"""
You are a SQL expert tasked with analyzing the complexity of a SQL query. Rate the complexity of the provided query on a scale of 1 to 10 and justify your reasoning.
Respond in JSON format with your answer in a field named \"complexity\" which is a integer.
in your answer, provide the following information:

- <1 or 2 sentences explaining the reasoning behind the score>
- the markdown formatted like this:
```
<json of the complexity>
```
""",
        },
        {
            "role": "user",
            "content": f"Rate the complexity of the following query on a scale of 1 to 10 and justify your reasoning.\n\n```\n{sql_query}\n```",
        },
    ]


def get_sql_complexity(sql_query: str) -> int:
    messages = get_sql_complexity_messages(sql_query)

    assistant_message = get_openai_completion(messages)["message"]["content"]

    print(assistant_message)

    json_str = extract_json_str_from_markdown_code_block(assistant_message)

    return json.loads(json_str)["complexity"]


@app.event("app_mention")
def handle_mentions(event, client, say):
    thread_ts = event.get("thread_ts", None) or event["ts"]
    question = re.sub("\\s<@[^, ]*|^<@[^, ]*", "", event["text"])

    # print(question)

    tables = get_relevant_tables(question)
    result, sql_query = generate_and_execute_sql(question, tables)

    # if result["MissingData"]:
    #     print(result["MissingData"])
    #     say(
    #         blocks=[
    #             {
    #                 "type": "section",
    #                 "text": {
    #                     "type": "plain_text",
    #                     "text": result["MissingData"],
    #                 },
    #             },
    #         ],
    #         thread_ts=thread_ts,
    #     )
    #     return

    data = json.dumps(result["results"], indent=2)

    # print(data)

    #     context = f"""
    # tables queried: {tables}
    # sql query: {sql_query}
    # columns: {result["column_names"]}
    # data: {data}
    # """

    answer = get_conversational_answer(question, data)

    spec = get_visualization_json_spec(data)
    url = get_visualization_image_url(spec)

    # say() sends a message to the channel where the event was triggered
    say(
        blocks=[
            {
                "type": "section",
                "text": {
                    "type": "plain_text",
                    "text": answer,
                },
            },
            {
                "type": "image",
                "image_url": url,
                "alt_text": "visualization",
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"_Source_: {example_db_engine.url.database}\n_Date_: {date.today()}",
                    }
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "Would you like an analyst to check my work?",
                },
                "accessory": {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Yes",
                    },
                    # "value": "click_me_123",
                    # "url": "https://google.com",
                    "action_id": "button-action",
                },
            },
        ],
        text=answer,
        thread_ts=thread_ts,
    )


def build_home_view(user_id):
    # get all questions
    user_questions = questions_collection.find({"user_id": user_id})

    blocks = []

    if (
        user_questions
        and questions_collection.count_documents({"user_id": user_id}) > 0
    ):
        for question in user_questions:
            answer = ""
            match question["status"]:
                case "pending":
                    answer = "I am working on your question. Please check back later."
                case "completed":
                    answer = question["answer"]
                case "error":
                    answer = "I was not able to answer your question. Please try again."

            created_at_str = datetime.fromtimestamp(
                int(question["created_at"])
            ).strftime("%m/%d/%Y")

            # TODO: should only show the overflow menu if the question is completed
            # when its pending we should show a cancel button
            blocks.extend(
                [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*{question['question']}*\n{answer}",
                        },
                        "accessory": {
                            "type": "overflow",
                            "options": [
                                {
                                    "text": {"type": "plain_text", "text": "View data"},
                                    "value": str(question["_id"]),
                                },
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "Delete",
                                    },
                                    "value": str(question["_id"]),
                                },
                            ],
                            "action_id": "question_overflow",
                        },
                    },
                ],
            )

            if question["status"] == "completed":
                # duration = int(question["answered_at"]) - \
                #     int(question["created_at"])
                # duration_str = f"{duration % 60} seconds"

                verification_status = ""
                match question["verification_status"]:
                    case "not_verified":
                        verification_status = "*:white_circle: Not verified*"
                    case "pending":
                        verification_status = (
                            "*:large_orange_circle: Pending verification by <@U053A2RUYQ5>*"
                        )
                    case "verified":
                        verification_status = "*:green_circle: Verified by <@U053A2RUYQ5>*"
                    case "rejected":
                        verification_status = "*:red_circle: Rejected by <@U053A2RUYQ5>*"

                blocks.extend(
                    [
                        {
                            "type": "context",
                            "elements": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Date*: {created_at_str}\n*Source*: {example_db_engine.url.database}",
                                }
                            ],
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": verification_status
                            }
                        },
                        {
                            "type": "actions",
                            "elements": [
                                {
                                    "type": "button",
                                    "text": {
                                        "type": "plain_text",
                                        "text": "Verify",
                                    },
                                    "style": "primary",
                                },
                                {
                                    "type": "button",
                                    "text": {
                                        "type": "plain_text",
                                        "text": "Visualize",
                                    },
                                }
                            ],
                        },
                    ]
                )
            else:
                blocks.extend(
                    [
                        {
                            "type": "context",
                            "elements": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Date*: {created_at_str}",
                                }
                            ],
                        },
                        {
                            "type": "actions",
                            "elements": [
                                {
                                    "type": "button",
                                    "text": {
                                        "type": "plain_text",
                                        "text": "Cancel",
                                    },
                                    "style": "danger",
                                },
                            ],
                        },
                    ]
                )

    else:
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "plain_text",
                    "text": "You haven't asked any questions yet!",
                },
            },
        )

    return {
        "type": "home",
        "callback_id": "home_view",
        # body of the view
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "Here's what you can do with Llamafy:",
                },
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "Ask a question",
                        },
                        "style": "primary",
                        # "value": "ask_question",
                        "action_id": "open_question_modal",
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "View recent questions",
                        },
                        # "value": "view_recent_questions",
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "Settings",
                        },
                        # "value": "help",
                    },
                ],
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "*Your Questions*"},
            },
            {"type": "divider"},
            *blocks,
        ],
    }


@app.event("app_home_opened")
def update_home_tab(client, event, logger):
    try:
        client.views_publish(
            user_id=event["user"],
            view=build_home_view(user_id=event["user"]),
        )

    except Exception as e:
        logger.error(f"Error publishing home tab: {e}")


@app.action("open_question_modal")
def open_question_modal(ack, body, client):
    # Acknowledge the command request
    ack()

    # Call views_open with the built-in client
    client.views_open(
        # Pass a valid trigger_id within 3 seconds of receiving it
        trigger_id=body["trigger_id"],
        # View payload
        view={
            "type": "modal",
            # View identifier
            "callback_id": "view_1",
            "title": {"type": "plain_text", "text": "Llamafy"},
            "submit": {"type": "plain_text", "text": "Submit"},
            "close": {"type": "plain_text", "text": "Cancel"},
            "blocks": [
                {
                    "type": "input",
                    "block_id": "input_c",
                    "label": {"type": "plain_text", "text": "What is your question?"},
                    "element": {
                        "type": "plain_text_input",
                        "action_id": "question_input",
                        "multiline": False,
                        "placeholder": {
                            "type": "plain_text",
                            "text": "What was retention in Q4 2021?",
                        },
                    },
                },
                {
                    "type": "input",
                    "optional": True,
                    "block_id": "input_d",
                    "label": {"type": "plain_text", "text": "Additional context:"},
                    "element": {
                        "type": "plain_text_input",
                        "action_id": "context_input",
                        "multiline": True,
                        "placeholder": {
                            "type": "plain_text",
                            "text": "Any additional information that you think would be helpful for answering your question.",
                        },
                    },
                },
            ],
        },
    )


@app.action("question_overflow")
def handle_question_overflow(ack, respond, logger, context, body, client):
    user_id = body["user"]["id"]

    question_id = body["actions"][0]["selected_option"]["value"] or None
    action = body["actions"][0]["selected_option"]["text"]["text"] or None

    ack()

    if action == "Delete":
        questions_collection.delete_one({"_id": ObjectId(question_id)})

    client.views_publish(
        user_id=user_id,
        view=build_home_view(user_id=user_id),
    )


# Handle a view_submission request
@app.view("view_1")
def handle_submission(ack, body, client, view, logger):
    question = view["state"]["values"]["input_c"]["question_input"]["value"]
    context = view["state"]["values"]["input_d"]["context_input"]["value"] or ""

    user_id = body["user"]["id"]

    # Validate the inputs
    errors = {}
    if question is not None and len(question) <= 5:
        errors["input_c"] = "The value must be longer than 5 characters"
    if len(errors) > 0:
        ack(response_action="errors", errors=errors)
        return

    ack()

    created_question = questions_collection.insert_one(
        {
            "question": question,
            "answer": None,
            "context": context,
            "user_id": user_id,
            "status": "pending",
            "verification_status": "not_verified",
            "created_at": datetime.now().timestamp(),
            "answered_at": None,
            "sql_query": None,
            "data": None,
            "visualization_image_url": None,
            "visualization_json": None,
        }
    )

    client.views_publish(
        user_id=user_id,
        view=build_home_view(user_id=user_id),
    )

    # how do i do this async?

    # generate answer
    tables = get_relevant_tables(question)
    result, sql_query = generate_and_execute_sql(question, tables)
    data = json.dumps(result["results"], indent=2)
    answer = get_conversational_answer(question, data)

    # update in mongo
    updated_question = questions_collection.find_one_and_update(
        {"_id": created_question.inserted_id},
        {
            "$set": {
                "answer": answer,
                "status": "completed",
                "sql_query": sql_query,
                "data": result["results"],
                "answered_at": datetime.now().timestamp(),
            }
        },
        new=True,
    )

    client.views_publish(
        user_id=user_id,
        view=build_home_view(user_id=user_id),
    )


@app.middleware  # or app.use(log_request)
def log_request(
    logger: Logger, body: dict, next: Callable[[], BoltResponse]
) -> BoltResponse:
    logger.debug(body)
    return next()


# Start your app
if __name__ == "__main__":
    # question = "Who are the top 3 best selling artists?"

    # tables = get_relevant_tables(question)

    # result, sql_query = generate_and_execute_sql(question, tables)

    # data = json.dumps(result["results"], indent=2)

    # print(data)

    # execute_sql(
    #     "WITH track_sales AS (\n    SELECT TrackId, SUM(Quantity) AS total_sales\n    FROM invoice_items\n    GROUP BY TrackId\n)\nSELECT artists.Name AS artist_name, SUM(track_sales.total_sales) AS total_sales\nFROM artists\nJOIN albums ON artists.ArtistId = albums.ArtistId\nJOIN tracks ON albums.AlbumId = tracks.AlbumId\nJOIN track_sales ON tracks.TrackId = track_sales.TrackId\nGROUP BY artists.Name\nORDER BY total_sales DESC\nLIMIT 3;"
    # )

    # print(get_conversational_answer(question, data))

    # spec = get_visualization_json_spec(data)

    # print(get_visualization_image_url(spec))

    # get_visualization_image_url(
    #     {
    #         "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    #         "data": {
    #             "values": [
    #                 {"artist_name": "Iron Maiden", "total_sales": 138.5999999999998},
    #                 {"artist_name": "U2", "total_sales": 105.92999999999982},
    #                 {"artist_name": "Metallica", "total_sales": 90.0899999999999},
    #             ]
    #         },
    #         "mark": "bar",
    #         "encoding": {
    #             "x": {"field": "artist_name", "type": "nominal"},
    #             "y": {"field": "total_sales", "type": "quantitative"},
    #         },
    #     }
    # )

    SocketModeHandler(app, SLACK_APP_TOKEN).start()
