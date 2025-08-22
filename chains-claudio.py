import os
import openai
from dotenv import find_dotenv, load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

os.system('cls' if os.name == 'nt' else 'clear')

#==== Using OpenAI Chat API =======
llm_model = "gpt-4o-mini"

chat = ChatOpenAI(temperature=0.9, model=llm_model)
llm = OpenAI(temperature=0.7)
prompt = PromptTemplate.from_template("How do you say good morning in {language}")
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

print(chain.invoke({"language": "German"}))

print("====== now quitting ======")
quit()
