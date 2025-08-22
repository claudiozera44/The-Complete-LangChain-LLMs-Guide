import os
import openai
from dotenv import find_dotenv, load_dotenv
# was: from langchain.llms import OpenAI
from langchain_openai.llms import OpenAI
# was: from langchain.chat_models import ChatOpenAI
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

llm_model = "gpt-3.5-turbo"

prompt = "How old is the Universe"
messages = [{"role": "user", "content": prompt}]

llm = OpenAI(temperature=0.7)
chat_model = ChatOpenAI(temperature=0.7)

# -------------------------------------------------------
# with OpenAI
# -------------------------------------------------------
# was: llm.predict
print(llm.invoke("What is the weather in London"))

# -------------------------------------------------------
# with langchain
# -------------------------------------------------------
# was: chat_model.predict
#print(chat_model.invoke("What is the weather in WA DC"))
# was: chat_model.predict_messages
print(chat_model.invoke(messages))