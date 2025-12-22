import os
from langchain.chat_models import init_chat_model
import requests, pathlib
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from dotenv import load_dotenv


load_dotenv(verbose=True)

url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
local_path = pathlib.Path("./Chinook.db")
if local_path.exists():
    print(f"{local_path} already exists. skipping download.")
else:
    respose = requests.get(url)
    if respose.status_code == 200:
        with open(local_path, "wb") as f:
            f.write(respose.content)
        print(f"Downloaded {local_path}.")
    else:
        print(f"Failed to download {url}. Status code: {respose.status_code}")

db = SQLDatabase.from_uri(f"sqlite:///{local_path}")
print(f"Dialect: {db.dialect}")
print(f"Avaliable tables:{db.get_usable_table_names()}")
print(f'Sample output: {db.run("SELECT * FROM Artist LIMIT 5;")}')
model = init_chat_model(
    "gpt-4o-mini",
    temperature=0,
    timeout=30,
    max_tokens=1000,
)
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()
for tool in tools:
    print(f"Tool name: {tool.name}, description: {tool.description}\n")
system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.
1
You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
    dialect=db.dialect,
    top_k=5,
)

agent = create_agent(
    model,
    tools,
    system_prompt=system_prompt,
    middleware=[HumanInTheLoopMiddleware(
        interrupt_on={"sql_db_query": True},
        description_prefix="Tool execution pending approval:",
    )],
    checkpointer=InMemorySaver(),
)
config = {"configurable": {"thread_id": "1"}}

question = "Which genre on average has the longest tracks?"
for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    config,
    stream_mode="values",
):
    print(f"Step keys: {step.keys()}")
    if "__interrupt__" in step:
        print("-------INTERRUPT DETECTED---------------")
        interrupt = step["__interrupt__"][0]
        for request in interrupt.value["action_requests"]:
            print(request["description"])
    elif "messages" in step:
        print(f"messages: {step['messages']}")
        step["messages"][-1].pretty_print()
    else:
        pass
