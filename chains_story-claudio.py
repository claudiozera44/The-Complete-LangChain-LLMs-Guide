import os
import openai
from dotenv import find_dotenv, load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.globals import set_verbose

set_verbose(True)

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

os.system('cls' if os.name == 'nt' else 'clear')

#==== Using OpenAI Chat API =======
llm_model = "gpt-4o-mini"

# chat = ChatOpenAI(temperature=0.9, model=llm_model)
llm = OpenAI(temperature=0.7)

template = """ 
 As a children's book writer, please come up with a simple and short (90 words)
 lullaby based on the location
 {location}
 and the main character {name}
 
 STORY:
"""

prompt = PromptTemplate.from_template(template)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser
story = chain.invoke({"location": "Zanzibar", "name": "Maya"})

print(story)
#print(story['text'])

print("====== now quitting ======")
quit()