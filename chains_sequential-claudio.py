import os
import openai
from dotenv import find_dotenv, load_dotenv
from operator import itemgetter
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

from langchain.chains import LLMChain, SequentialChain

os.system('cls' if os.name == 'nt' else 'clear')

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

#==== Using OpenAI Chat API =======
llm_model = "gpt-4o-mini"

# chat = ChatOpenAI(temperature=0.9, model=llm_model)
model = OpenAI(temperature=0.7, model=llm_model)

template_story = """ 
 As a children's book writer, please come up with a simple and short (90 words)
 lullaby based on the location
 {location}
 and the main character {name}
 
 STORY:
"""

prompt_story = PromptTemplate.from_template(template_story)
output_parser = StrOutputParser()

chain_story = prompt_story | model | output_parser

#chain_story = LLMChain(llm=open_ai, prompt=prompt, 
#                       output_key="story",
#                       verbose=True)

print(chain_story.invoke({"location": "Zanzibar", "name": "Maya"}))
# print(story['text'])

# ======= Sequential Chain =====
# chain to translate the story to Portuguese
template_translate = """
Translate the {story} into {language}.  Make sure 
the language is simple and fun.

TRANSLATION:
"""

prompt_translate = PromptTemplate.from_template(template_translate)

chain_translate = prompt_translate | model | output_parser

# ==== Create the Sequential Chain ===
overall_chain = ({
    "story": chain_story,
    "language": itemgetter("language")
#    "input_description": "Provide the location, name of the character, and language for translation.",
#    "output_description": "Returns the original story and its translation.",
    }
| chain_translate)

response = overall_chain.invoke({
    "location": "Magical", 
    "name": "Karyna",
    "language": "Russian"
})

print(response)
print(f"English Version ====> { response[0]} \n \n")
print(f"Translated Version ====> { response[1]}")


print("====== now quitting ======")
quit()