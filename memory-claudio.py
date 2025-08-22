import os
from dotenv import find_dotenv, load_dotenv
import openai
# was: from langchain.chat_models import ChatOpenAI
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

# deprecated - was: from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

os.system('cls' if os.name == 'nt' else 'clear')

#==== Using OpenAI Chat API =======
llm_model = "gpt-4o-mini"

llm = ChatOpenAI(temperature=0.6, 
                 model=llm_model)

#print("====== now quitting ======")
#quit()

#print(llm.invoke("My name is Claudio. What is yours?"))
#print(llm.invoke("Great!  What's my name?")) # we have memory issues!

#=================================
# this is deprecated
#=================================
# How to solve llms memory issues?
#memory = ConversationBufferMemory()
#conversation = ConversationChain(
#    llm=llm,
#    memory=memory,
#    verbose=True
#)

#conversation.predict(input="Hello there, I am Paulo")
#conversation.predict(input="Why is the sky blue?")
#conversation.predict(input="If phenomenon called Rayleigh didn't exist, what color would the sky be?")
#conversation.predict(input="What's my name?")
#print(f"Memory ===> {memory.buffer} <====")

#=================================
# this is ok as per langchain 0.3.x
#=================================
model = ChatOpenAI(model="gpt-4o-mini")

workflow = StateGraph(state_schema=MessagesState)

# Define the function that calls the model
def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    # Update message history with response:
    return {"messages": response}

# Define the node and edge
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add simple in-memory checkpointer
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}

output = app.invoke(
    {"messages": [HumanMessage(content="My name is Claudio. What is yours?")]},
    config #={"configurable": {"thread_id": "1"}},
)

output["messages"][-1].pretty_print()  # output contains all messages in state

output = app.invoke(
    {"messages": [HumanMessage(content="Great!  What's my name?")]},
    config #={"configurable": {"thread_id": "1"}},
)

output["messages"][-1].pretty_print()  # output contains all messages in state

# print(memory.load_memory_variables({}))








