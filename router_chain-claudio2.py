import os
from dotenv import find_dotenv, load_dotenv
import openai
#from langchain.chat_models import ChatOpenAI
from langchain_openai.llms import OpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain, SequentialChain

os.system('cls' if os.name == 'nt' else 'clear')

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

#==== Using OpenAI Chat API =======
llm_model = "gpt-3.5-turbo"
#chat = OpenAI(temperature=0.0, model=llm_model)
from langchain_openai import ChatOpenAI
chat = ChatOpenAI(model="gpt-4o-mini")


biology_template = """You are a very smart biology professor. 
You are great at answering questions about biology in a concise and easy to understand manner. 
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{input}"""


math_template = """You are a very good mathematician. You are great at answering math questions. 
You are so good because you are able to break down hard problems into their component parts, 
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{input}"""

astronomy_template = """You are a very good astronomer. You are great at answering astronomy questions. 
You are so good because you are able to break down hard problems into their component parts, 
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{input}"""

travel_agent_template = """You are a very good travel agent with a large amount
of knowledge when it comes to getting people the best deals and recommendations
for travel, vacations, flights and world's best destinations for vacation. 
You are great at answering travel, vacation, flights, transportation, tourist guides questions. 
You are so good because you are able to break down hard problems into their component parts, 
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{input}"""

prompt_infos = [
    {
        "name": "Biology",
        "description": "Good for answering Biology related questions",
        "prompt_template": biology_template
    },
    {
        "name": "math",
        "description": "Good for answering math questions",
        "prompt_template": math_template,
    },
    {
        "name": "astronomy",
        "description": "Good for answering astronomy questions",
        "prompt_template": astronomy_template,
    },
    {
        "name": "travel_agent",
        "description": "Good for answering travel, tourism and vacation questions",
        "prompt_template": travel_agent_template,
    },
]

# destination_chains = {}
# for info in prompt_infos:
#     name = info["name"]
#     prompt_template = info["prompt_template"]
#     prompt = ChatPromptTemplate.from_template(template=prompt_template)
#     chain = prompt | chat
#     destination_chains[name] = chain

#print(f"Destination chains: {destination_chains}")
#print(f"Destination chains: {list(destination_chains.keys())}")

# Setup the default chain  
default_prompt = PromptTemplate.from_template("{input}")
default_chain = default_prompt | chat

from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
# deprecated
#from langchain.chains.router import MultiPromptChain
from langgraph.graph import END, START, StateGraph

# NEW ##############
from langchain_core.output_parsers import StrOutputParser
prompt_biology = ChatPromptTemplate.from_template(template=biology_template)
prompt_math = ChatPromptTemplate.from_template(template=math_template)
prompt_astronomy = ChatPromptTemplate.from_template(template=astronomy_template)
prompt_travel_agent = ChatPromptTemplate.from_template(template=travel_agent_template)

chain_biology = prompt_biology | chat | StrOutputParser()
chain_math = prompt_math | chat | StrOutputParser()
chain_astronomy = prompt_astronomy | chat | StrOutputParser()
chain_travel_agent = prompt_travel_agent | chat | StrOutputParser()
####################

# destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
# destinations_str = "\n".join(destinations)

# router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)

# router_prompt = PromptTemplate(
#     template=router_template,
#     input_variables=["input"],
#     output_parser=RouterOutputParser()
# )
 
# router_chain = LLMRouterChain.from_llm(
#     llm=chat,
#     prompt=router_prompt,
    
# ) 

# NEW ##############
from typing import Literal
from typing_extensions import TypedDict

route_system = "Route the user's query to either the biology or math or astronomy or travel agent expert."
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", route_system),
        ("human", "{input}"),
    ]
)

class RouteQuery(TypedDict):
    """Route query to destination expert."""
    destination: Literal["biology", "math", "astronomy", "travel_agent"]


route_chain = route_prompt | chat.with_structured_output(RouteQuery)
####################

# deprecated
# chain = MultiPromptChain(
#     router_chain=router_chain,
#     destination_chains=destination_chains,
#     default_chain=default_chain,
#     verbose=True
# )

# NEW ##############
from langchain_core.runnables import RunnableConfig

# For LangGraph, we will define the state of the graph to hold the query,
# destination, and final answer.
class State(TypedDict):
    query: str
    destination: RouteQuery
    answer: str


# We define functions for each node, including routing the query:
async def route_query(state: State, config: RunnableConfig):
    destination = await route_chain.ainvoke(state["query"], config)
    return {"destination": destination}


# And one node for each prompt
async def prompt_biology(state: State, config: RunnableConfig):
    return {"answer": await chain_biology.ainvoke(state["query"], config)}

async def prompt_math(state: State, config: RunnableConfig):
    return {"answer": await chain_math.ainvoke(state["query"], config)}

async def prompt_astronomy(state: State, config: RunnableConfig):
    return {"answer": await chain_astronomy.ainvoke(state["query"], config)}

async def prompt_travel_agent(state: State, config: RunnableConfig):
    return {"answer": await chain_travel_agent.ainvoke(state["query"], config)}


# We then define logic that selects the prompt based on the classification
def select_node(state: State) -> Literal["prompt_biology", "prompt_math", "prompt_astronomy", "prompt_travel_agent"]:
    if state["destination"] == "biology":
        return "prompt_biology"
    if state["destination"] == "math":
        return "prompt_math"
    if state["destination"] == "astronomy":
        return "prompt_astronomy"
    if state["destination"] == "travel_agent":
        return "prompt_travel_agent"
    #raise ValueError(f"Unknown destination {state['destination']}")

graph = StateGraph(State)
graph.add_node("route_query", route_query)
graph.add_node("prompt_biology", prompt_biology)
graph.add_node("prompt_math", prompt_math)
graph.add_node("prompt_astronomy", prompt_astronomy)
graph.add_node("prompt_travel_agent", prompt_travel_agent)

graph.add_edge(START, "route_query")
graph.add_conditional_edges("route_query", select_node)
graph.add_edge("prompt_biology", END)
graph.add_edge("prompt_math", END)
graph.add_edge("prompt_astronomy", END)
graph.add_edge("prompt_travel_agent", END)
app = graph.compile()
####################

# Test
#response = chain.run("I need to go to Kenya for vacation, a family of four. Can you help me plan this trip?")
#response = chain.run("How old as the stars?")
#print(response)

import asyncio

async def main():
    print("====== now invoking async function... ======")
    state = await app.ainvoke({"query": "what planets are in the solar system?"})
    print(state["destination"])
    print(state["answer"])
    print("====== done ======")

if __name__ == "__main__":
    asyncio.run(main())

print("====== now quitting ======")
quit()

