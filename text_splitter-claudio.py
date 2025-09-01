import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain.text_splitter import CharacterTextSplitter

os.system('cls' if os.name == 'nt' else 'clear')

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

# 1. CharacterTextSplitter
with open("./langchain-course-code/data/i-have-a-dream.txt") as paper:
    speech = paper.read()
    
text_splitter = CharacterTextSplitter(
    
    chunk_size = 100,
    chunk_overlap = 20,
    length_function = len
)

#print("====== now quitting ======")
#quit()

texts = text_splitter.create_documents([speech])
print(texts[0])



