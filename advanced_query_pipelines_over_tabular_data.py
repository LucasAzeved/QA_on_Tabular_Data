# %% [markdown]
# # Query Pipeline over Pandas DataFrames
# 
# This is a simple example that builds a query pipeline that can perform structured operations over a Pandas DataFrame to satisfy a user query, using LLMs to infer the set of operations.
# 
# This can be treated as the "from-scratch" version of our `PandasQueryEngine`.

# %%
# !pip install llama-index==0.9.45.post1 arize-phoenix==2.2.1 pyvis

# %%
from llama_index.query_pipeline import (
    QueryPipeline as QP, 
    Link, 
    InputComponent, 
)
from llama_index.query_engine.pandas import PandasInstructionParser
from llama_index.llms import OpenAI
from llama_index.prompts import PromptTemplate

# %%
import os

os.environ["OPENAI_API_KEY"] = "<OPENAI_API_KEY>"

# %% [markdown]
# ## Download Data
# 
# Here we load the Titanic CSV dataset.

# %%
# !wget 'https://raw.githubusercontent.com/jerryjliu/llama_index/main/docs/examples/data/csv/titanic_train.csv' -O 'titanic_train.csv'

# %%
import pandas as pd

df = pd.read_csv("./titanic_train.csv")
df

# %% [markdown]
# ## Define Modules
# 
# Here we define the set of modules:
# 1. Pandas prompt to infer pandas instructions from user query
# 2. Pandas output parser to execute pandas instructions on dataframe, get back dataframe
# 3. Response synthesis prompt to synthesize a final response given the dataframe
# 4. LLM
# 
# The pandas output parser specifically is designed to safely execute Python code. It includes a lot of safety checks that may be annoying to write from scratch. This includes only importing from a set of approved modules (e.g. no modules that would alter the file system like `os`), and also making sure that no private/dunder methods are being called.

# %%
instruction_str = (
    "1. Convert the query to executable Python code using Pandas.\n"
    "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
    "3. The code should represent a solution to the query.\n"
    "4. PRINT ONLY THE EXPRESSION.\n"
    "5. Do not quote the expression.\n"
)

pandas_prompt_str = (
    "You are working with a pandas dataframe in Python.\n"
    "The name of the dataframe is `df`.\n"
    "This is the result of `print(df.head())`:\n"
    "{df_str}\n\n"
    "Follow these instructions:\n"
    "{instruction_str}\n"
    "Query: {query_str}\n\n"
    "Expression:"
)
response_synthesis_prompt_str = (
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n\n"
    "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
    "Pandas Output: {pandas_output}\n\n"
    "Response: "
)

pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
    instruction_str=instruction_str, df_str=df.head(5)
)
pandas_output_parser = PandasInstructionParser(df)
response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)
llm = OpenAI(model="gpt-3.5-turbo")

# %% [markdown]
# ## Build Query Pipeline
# 
# Looks like this:
# input query_str -> pandas_prompt -> llm1 -> pandas_output_parser -> response_synthesis_prompt -> llm2
# 
# Additional connections to response_synthesis_prompt: llm1 -> pandas_instructions, and pandas_output_parser -> pandas_output.

# %%
qp = QP(
    modules={
        "input": InputComponent(),
        "pandas_prompt": pandas_prompt,
        "llm1": llm,
        "pandas_output_parser": pandas_output_parser,
        "response_synthesis_prompt": response_synthesis_prompt,
        "llm2": llm,
    },
    verbose=True,
)
qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
qp.add_links(
    [
        Link("input", "response_synthesis_prompt", dest_key="query_str"),
        Link(
            "llm1", "response_synthesis_prompt", dest_key="pandas_instructions"
        ),
        Link(
            "pandas_output_parser",
            "response_synthesis_prompt",
            dest_key="pandas_output",
        ),
    ]
)
# add link from response synthesis prompt to llm2
qp.add_link("response_synthesis_prompt", "llm2")

# %%
from pyvis.network import Network

net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(qp.dag)
net.show("text2sql_dag.html")

# %% [markdown]
# ## Run Query

# %%
response = qp.run(
    query_str="What is the correlation between survival and age?",
)

# %%
print(response.message.content)

# %% [markdown]
# # Query Pipeline for Advanced Text-to-SQL
# 
# In this guide we show you how to setup a text-to-SQL pipeline over your data with our [query pipeline](https://docs.llamaindex.ai/en/stable/module_guides/querying/pipeline/root.html) syntax.
# 
# This gives you flexibility to enhance text-to-SQL with additional techniques. We show these in the below sections:
# 1. **Query-Time Table Retrieval**: Dynamically retrieve relevant tables in the text-to-SQL prompt.
# 2. **Query-Time Sample Row retrieval**: Embed/Index each row, and dynamically retrieve example rows for each table in the text-to-SQL prompt.
# 
# Our out-of-the box pipelines include our `NLSQLTableQueryEngine` and `SQLTableRetrieverQueryEngine`. (if you want to check out our text-to-SQL guide using these modules, take a look [here](https://docs.llamaindex.ai/en/stable/examples/index_structs/struct_indices/SQLIndexDemo.html)). This guide implements an advanced version of those modules, giving you the utmost flexibility to apply this to your own setting.

# %% [markdown]
# ## Load and Ingest Data
# 
# 
# ### Load Data
# We use the [WikiTableQuestions dataset](https://ppasupat.github.io/WikiTableQuestions/) (Pasupat and Liang 2015) as our test dataset.
# 
# We go through all the csv's in one folder, store each in a sqlite database (we will then build an object index over each table schema).

# %%
# !wget "https://github.com/ppasupat/WikiTableQuestions/releases/download/v1.0.2/WikiTableQuestions-1.0.2-compact.zip" -O data.zip
# !unzip -o data.zip

# %%
import pandas as pd
from pathlib import Path

data_dir = Path("./WikiTableQuestions/csv/200-csv")
csv_files = sorted([f for f in data_dir.glob("*.csv")])
dfs = []
for csv_file in csv_files:
    print(f"processing file: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
        dfs.append(df)
    except Exception as e:
        print(f"Error parsing {csv_file}: {str(e)}")

# %% [markdown]
# ### Extract Table Name and Summary from each Table
# 
# Here we use gpt-3.5 to extract a table name (with underscores) and summary from each table with our Pydantic program.

# %%
# tableinfo_dir = "WikiTableQuestions_TableInfo"
# !mkdir {tableinfo_dir}

# %%
from llama_index.program import LLMTextCompletionProgram
from llama_index.bridge.pydantic import BaseModel, Field
from llama_index.llms import OpenAI


class TableInfo(BaseModel):
    """Information regarding a structured table."""

    table_name: str = Field(
        ..., description="table name (must be underscores and NO spaces)"
    )
    table_summary: str = Field(
        ..., description="short, concise summary/caption of the table"
    )


prompt_str = """\
Give me a summary of the table with the following JSON format.

- The table name must be unique to the table and describe it while being concise.
- Do NOT output a generic table name (e.g. table, my_table).

Do NOT make the table name one of the following: {exclude_table_name_list}

Table:
{table_str}

Summary: """

program = LLMTextCompletionProgram.from_defaults(
    output_cls=TableInfo,
    llm=OpenAI(model="gpt-3.5-turbo"),
    prompt_template_str=prompt_str,
)

# %%
import json


def _get_tableinfo_with_index(idx: int) -> str:
    results_gen = Path(tableinfo_dir).glob(f"{idx}_*")
    results_list = list(results_gen)
    if len(results_list) == 0:
        return None
    elif len(results_list) == 1:
        path = results_list[0]
        return TableInfo.parse_file(path)
    else:
        raise ValueError(
            f"More than one file matching index: {list(results_gen)}"
        )


table_names = set()
table_infos = []
for idx, df in enumerate(dfs):
    table_info = _get_tableinfo_with_index(idx)
    if table_info:
        table_infos.append(table_info)
    else:
        while True:
            df_str = df.head(10).to_csv()
            table_info = program(
                table_str=df_str,
                exclude_table_name_list=str(list(table_names)),
            )
            table_name = table_info.table_name
            print(f"Processed table: {table_name}")
            if table_name not in table_names:
                table_names.add(table_name)
                break
            else:
                # try again
                print(f"Table name {table_name} already exists, trying again.")
                pass

        out_file = f"{tableinfo_dir}/{idx}_{table_name}.json"
        json.dump(table_info.dict(), open(out_file, "w"))
    table_infos.append(table_info)

# %% [markdown]
# ### Put Data in SQL Database
# 
# We use `sqlalchemy`, a popular SQL database toolkit, to load all the tables.

# %%
# put data into sqlite db
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
)
import re


# Function to create a sanitized column name
def sanitize_column_name(col_name):
    # Remove special characters and replace spaces with underscores
    return re.sub(r"\W+", "_", col_name)


# Function to create a table from a DataFrame using SQLAlchemy
def create_table_from_dataframe(
    df: pd.DataFrame, table_name: str, engine, metadata_obj
):
    # Sanitize column names
    sanitized_columns = {col: sanitize_column_name(col) for col in df.columns}
    df = df.rename(columns=sanitized_columns)

    # Dynamically create columns based on DataFrame columns and data types
    columns = [
        Column(col, String if dtype == "object" else Integer)
        for col, dtype in zip(df.columns, df.dtypes)
    ]

    # Create a table with the defined columns
    table = Table(table_name, metadata_obj, *columns)

    # Create the table in the database
    metadata_obj.create_all(engine)

    # Insert data from DataFrame into the table
    with engine.connect() as conn:
        for _, row in df.iterrows():
            insert_stmt = table.insert().values(**row.to_dict())
            conn.execute(insert_stmt)
        conn.commit()


engine = create_engine("sqlite:///:memory:")
metadata_obj = MetaData()
for idx, df in enumerate(dfs):
    tableinfo = _get_tableinfo_with_index(idx)
    print(f"Creating table: {tableinfo.table_name}")
    create_table_from_dataframe(df, tableinfo.table_name, engine, metadata_obj)

# %%
# setup Arize Phoenix for logging/observability
import phoenix as px
import llama_index

px.launch_app()
llama_index.set_global_handler("arize_phoenix")

# %% [markdown]
# ## Advanced Capability 1: Text-to-SQL with Query-Time Table Retrieval.
# 
# We now show you how to setup an e2e text-to-SQL with table retrieval.
# 
# ### Define Modules
# 
# Here we define the core modules.
# 1. Object index + retriever to store table schemas
# 2. SQLDatabase object to connect to the above tables + SQLRetriever.
# 3. Text-to-SQL Prompt
# 4. Response synthesis Prompt
# 5. LLM

# %% [markdown]
# Object index, retriever, SQLDatabase

# %%
from llama_index.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index import SQLDatabase, VectorStoreIndex

sql_database = SQLDatabase(engine)

table_node_mapping = SQLTableNodeMapping(sql_database)
table_schema_objs = [
    SQLTableSchema(table_name=t.table_name, context_str=t.table_summary)
    for t in table_infos
]  # add a SQLTableSchema for each table

obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex,
)
obj_retriever = obj_index.as_retriever(similarity_top_k=3)

# %% [markdown]
# SQLRetriever + Table Parser

# %%
from llama_index.retrievers import SQLRetriever
from typing import List
from llama_index.query_pipeline import FnComponent

sql_retriever = SQLRetriever(sql_database)


def get_table_context_str(table_schema_objs: List[SQLTableSchema]):
    """Get table context string."""
    context_strs = []
    for table_schema_obj in table_schema_objs:
        table_info = sql_database.get_single_table_info(
            table_schema_obj.table_name
        )
        if table_schema_obj.context_str:
            table_opt_context = " The table description is: "
            table_opt_context += table_schema_obj.context_str
            table_info += table_opt_context

        context_strs.append(table_info)
    return "\n\n".join(context_strs)


table_parser_component = FnComponent(fn=get_table_context_str)

# %% [markdown]
# Text-to-SQL Prompt + Output Parser

# %%
from llama_index.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from llama_index.prompts import PromptTemplate
from llama_index.query_pipeline import FnComponent
from llama_index.llms import ChatResponse


def parse_response_to_sql(response: ChatResponse) -> str:
    """Parse response to SQL."""
    response = response.message.content
    sql_query_start = response.find("SQLQuery:")
    if sql_query_start != -1:
        response = response[sql_query_start:]
        # TODO: move to removeprefix after Python 3.9+
        if response.startswith("SQLQuery:"):
            response = response[len("SQLQuery:") :]
    sql_result_start = response.find("SQLResult:")
    if sql_result_start != -1:
        response = response[:sql_result_start]
    return response.strip().strip("```").strip()


sql_parser_component = FnComponent(fn=parse_response_to_sql)

text2sql_prompt = DEFAULT_TEXT_TO_SQL_PROMPT.partial_format(
    dialect=engine.dialect.name
)
print(text2sql_prompt.template)

# %% [markdown]
# Response Synthesis Prompt

# %%
response_synthesis_prompt_str = (
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n"
    "SQL: {sql_query}\n"
    "SQL Response: {context_str}\n"
    "Response: "
)
response_synthesis_prompt = PromptTemplate(
    response_synthesis_prompt_str,
)

# %%
llm = OpenAI(model="gpt-3.5-turbo")

# %% [markdown]
# ### Define Query Pipeline
# 
# Now that the components are in place, let's define the query pipeline!

# %%
from llama_index.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
    CustomQueryComponent,
)

qp = QP(
    modules={
        "input": InputComponent(),
        "table_retriever": obj_retriever,
        "table_output_parser": table_parser_component,
        "text2sql_prompt": text2sql_prompt,
        "text2sql_llm": llm,
        "sql_output_parser": sql_parser_component,
        "sql_retriever": sql_retriever,
        "response_synthesis_prompt": response_synthesis_prompt,
        "response_synthesis_llm": llm,
    },
    verbose=True,
)

# %%
qp.add_chain(["input", "table_retriever", "table_output_parser"])
qp.add_link("input", "text2sql_prompt", dest_key="query_str")
qp.add_link("table_output_parser", "text2sql_prompt", dest_key="schema")
qp.add_chain(
    ["text2sql_prompt", "text2sql_llm", "sql_output_parser", "sql_retriever"]
)
qp.add_link(
    "sql_output_parser", "response_synthesis_prompt", dest_key="sql_query"
)
qp.add_link(
    "sql_retriever", "response_synthesis_prompt", dest_key="context_str"
)
qp.add_link("input", "response_synthesis_prompt", dest_key="query_str")
qp.add_link("response_synthesis_prompt", "response_synthesis_llm")

# %% [markdown]
# ### Visualize Query Pipeline
# 
# A really nice property of the query pipeline syntax is you can easily visualize it in a graph via networkx.

# %%
from pyvis.network import Network

net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(qp.dag)
net.show("text2sql_dag.html")

# %% [markdown]
# ### Run Some Queries!
# 
# Now we're ready to run some queries across this entire pipeline.

# %%
response = qp.run(
    query="What was the year that The Notorious B.I.G was signed to Bad Boy?"
)
print(str(response))

# %%
response = qp.run(query="Who won best director in the 1972 academy awards")
print(str(response))

# %%
response = qp.run(query="What was the term of Pasquale Preziosa?")
print(str(response))

# %% [markdown]
# ## 2. Advanced Capability 2: Text-to-SQL with Query-Time Row Retrieval (along with Table Retrieval)
# 
# One problem in the previous example is that if the user asks a query that asks for "The Notorious BIG" but the artist is stored as "The Notorious B.I.G", then the generated SELECT statement will likely not return any matches.
# 
# We can alleviate this problem by fetching a small number of example rows per table. A naive option would be to just take the first k rows. Instead, we embed, index, and retrieve k relevant rows given the user query to give the text-to-SQL LLM the most contextually relevant information for SQL generation.
# 
# We now extend our query pipeline.

# %%
from llama_index.query_pipeline import QueryPipeline as QP
from llama_index.service_context import ServiceContext

qp = QP(verbose=True)
# NOTE: service context will be deprecated in v0.10 (though will still be backwards compatible)
service_context = ServiceContext.from_defaults(callback_manager=qp.callback_manager)

# %% [markdown]
# ### Index Each Table
# 
# We embed/index the rows of each table, resulting in one index per table.

# %%
from llama_index import VectorStoreIndex, load_index_from_storage
from sqlalchemy import text
from llama_index.schema import TextNode
from llama_index.storage import StorageContext
import os
from pathlib import Path
from typing import Dict


def index_all_tables(
    sql_database: SQLDatabase, table_index_dir: str = "table_index_dir"
) -> Dict[str, VectorStoreIndex]:
    """Index all tables."""
    if not Path(table_index_dir).exists():
        os.makedirs(table_index_dir)

    vector_index_dict = {}
    engine = sql_database.engine
    for table_name in sql_database.get_usable_table_names():
        print(f"Indexing rows in table: {table_name}")
        if not os.path.exists(f"{table_index_dir}/{table_name}"):
            # get all rows from table
            with engine.connect() as conn:
                cursor = conn.execute(text(f'SELECT * FROM "{table_name}"'))
                result = cursor.fetchall()
                row_tups = []
                for row in result:
                    row_tups.append(tuple(row))

            # index each row, put into vector store index
            nodes = [TextNode(text=str(t)) for t in row_tups]

            # put into vector store index (use OpenAIEmbeddings by default)
            index = VectorStoreIndex(nodes, service_context=service_context)

            # save index
            index.set_index_id("vector_index")
            index.storage_context.persist(f"{table_index_dir}/{table_name}")
        else:
            # rebuild storage context
            storage_context = StorageContext.from_defaults(
                persist_dir=f"{table_index_dir}/{table_name}"
            )
            # load index
            index = load_index_from_storage(
                storage_context, index_id="vector_index", service_context=service_context
            )
        vector_index_dict[table_name] = index

    return vector_index_dict


vector_index_dict = index_all_tables(sql_database)

# %%
test_retriever = vector_index_dict["Bad_Boy_Artists"].as_retriever(
    similarity_top_k=1
)
nodes = test_retriever.retrieve("P. Diddy")
print(nodes[0].get_content())

# %% [markdown]
# ### Define Expanded Table Parser Component
# 
# We expand the capability of our `table_parser_component` to not only return the relevant table schemas, but also return relevant rows per table schema.
# 
# It now takes in both `table_schema_objs` (output of table retriever), but also the original `query_str` which will then be used for vector retrieval of relevant rows.

# %%
from llama_index.retrievers import SQLRetriever
from typing import List
from llama_index.query_pipeline import FnComponent

sql_retriever = SQLRetriever(sql_database)


def get_table_context_and_rows_str(
    query_str: str, table_schema_objs: List[SQLTableSchema]
):
    """Get table context string."""
    context_strs = []
    for table_schema_obj in table_schema_objs:
        # first append table info + additional context
        table_info = sql_database.get_single_table_info(
            table_schema_obj.table_name
        )
        if table_schema_obj.context_str:
            table_opt_context = " The table description is: "
            table_opt_context += table_schema_obj.context_str
            table_info += table_opt_context

        # also lookup vector index to return relevant table rows
        vector_retriever = vector_index_dict[
            table_schema_obj.table_name
        ].as_retriever(similarity_top_k=2)
        relevant_nodes = vector_retriever.retrieve(query_str)
        if len(relevant_nodes) > 0:
            table_row_context = "\nHere are some relevant example rows (values in the same order as columns above)\n"
            for node in relevant_nodes:
                table_row_context += str(node.get_content()) + "\n"
            table_info += table_row_context

        context_strs.append(table_info)
    return "\n\n".join(context_strs)


table_parser_component = FnComponent(fn=get_table_context_and_rows_str)

# %% [markdown]
# ### Define Expanded Query Pipeline
# 
# This looks similar to the query pipeline in section 1, but with an upgraded table_parser_component.

# %%
from llama_index.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
    CustomQueryComponent,
)

qp.add_modules({
    "input": InputComponent(),
    "table_retriever": obj_retriever,
    "table_output_parser": table_parser_component,
    "text2sql_prompt": text2sql_prompt,
    "text2sql_llm": llm,
    "sql_output_parser": sql_parser_component,
    "sql_retriever": sql_retriever,
    "response_synthesis_prompt": response_synthesis_prompt,
    "response_synthesis_llm": llm,
})

# %%
qp.add_link("input", "table_retriever")
qp.add_link("input", "table_output_parser", dest_key="query_str")
qp.add_link(
    "table_retriever", "table_output_parser", dest_key="table_schema_objs"
)
qp.add_link("input", "text2sql_prompt", dest_key="query_str")
qp.add_link("table_output_parser", "text2sql_prompt", dest_key="schema")
qp.add_chain(
    ["text2sql_prompt", "text2sql_llm", "sql_output_parser", "sql_retriever"]
)
qp.add_link(
    "sql_output_parser", "response_synthesis_prompt", dest_key="sql_query"
)
qp.add_link(
    "sql_retriever", "response_synthesis_prompt", dest_key="context_str"
)
qp.add_link("input", "response_synthesis_prompt", dest_key="query_str")
qp.add_link("response_synthesis_prompt", "response_synthesis_llm")

# %%
from pyvis.network import Network

net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(qp.dag)
net.show("text2sql_dag.html")

# %% [markdown]
# ### Run Some Queries
# 
# We can now ask about relevant entries even if it doesn't exactly match the entry in the database.

# %%
response = qp.run(
    query="What was the year that The Notorious BIG was signed to Bad Boy?"
)
print(str(response))

# %%



