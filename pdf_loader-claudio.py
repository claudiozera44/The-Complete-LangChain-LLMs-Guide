import os
from dotenv import find_dotenv, load_dotenv
import openai
#from langchain.chat_models import ChatOpenAI
from langchain_openai.llms import OpenAI
from langchain.llms import OpenAI

from langchain.document_loaders import PyPDFLoader

os.system('cls' if os.name == 'nt' else 'clear')

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

#==== Using OpenAI Chat API =======
llm_model = "gpt-3.5-turbo"

#llm = ChatOpenAI(temperature=0.0, model=llm_model) 
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")

### pip install pypdf

loader = PyPDFLoader("./langchain-course-code/data/react-paper.pdf")
pages = loader.load()

#print(len(pages))

page = pages[0]
#print(pages)
#print(page.page_content[0:700]) # first 700 characters on the page
print(page.metadata)


print("====== now quitting ======")
quit()
