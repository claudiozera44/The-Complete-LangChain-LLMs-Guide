import os
import openai
from dotenv import find_dotenv, load_dotenv
# from operator import itemgetter
# from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.llms import OpenAI
from langchain.prompts import PromptTemplate
# from langchain.schema.output_parser import StrOutputParser

os.system('cls' if os.name == 'nt' else 'clear')

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

#==== Using OpenAI Chat API =======
llm_model = "gpt-4o-mini"

# chat = ChatOpenAI(temperature=0.9, model=llm_model)
model = OpenAI(temperature=0.7, model=llm_model)

from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# first chain
first_prompt = PromptTemplate.from_template(
    "How much is {a} + {b}. Answer only witht the result\nResult: "
)
first_chain = first_prompt | model | StrOutputParser()

# second_chain
second_prompt = PromptTemplate.from_template(
    "How much is {a} times {c}"
)
second_chain = second_prompt | model | StrOutputParser()

# all together

# only return the result from second_chain
complete_chain_with_results_only_from_second_chain = ({
        "c": first_chain, # save result in variable 'c'
        "a": itemgetter("a")  # get variable a for next chain
        }
    | second_chain # run second_chain with a and c as input
    )

# also with the intermediate results
complete_chain_with_results_from_both_chains = ({
        "a": itemgetter("a"),
        "b": itemgetter("b"),
        "c": first_chain
        }
    | RunnablePassthrough.assign(d=second_chain)
    )

result1 = complete_chain_with_results_only_from_second_chain.invoke({'a': 2, 'b':3})
result2 = complete_chain_with_results_from_both_chains.invoke({'a': 2, 'b':3})

print(result1)
print("------")
print(result2)
print("====== now quitting ======")
quit()