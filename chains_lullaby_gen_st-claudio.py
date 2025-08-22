import os
import openai
from dotenv import find_dotenv, load_dotenv
from operator import itemgetter
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
import streamlit as st 

os.system('cls' if os.name == 'nt' else 'clear')

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

#==== Using OpenAI Chat API =======
llm_model = "gpt-4o-mini"

# chat = ChatOpenAI(temperature=0.9, model=llm_model)
model = OpenAI(temperature=0.7)

def generate_lullaby(location, name, language):
    
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
        "location": itemgetter("location"),
        "name": itemgetter("name"),
        "language": itemgetter("language")
    #    "input_description": "Provide the location, name of the character, and language for translation.",
    #    "output_description": "Returns the original story and its translation.",
        }
    | chain_translate)

    response = overall_chain.invoke({"location": location, 
                            "name": name,
                            "language": language
                            })
        
    return response

def main():
    st.set_page_config(page_title="Generate Children's Lullaby",
                       layout="centered")
    st.title("Let AI Write and Translate a Lullaby for You 📖")
    st.header("Get Started...")
    
    location_input = st.text_input(label="Where is the story set?", value="Zanzibar")
    main_character_input = st.text_input(label="What's the main charater's name", value="Maya")
    language_input = st.text_input(label="Translate the story into...", value="Italian")
    
    submit_button = st.button("Submit")
    if location_input and main_character_input and language_input:
        if submit_button:
            with st.spinner("Generating lullaby..."):
                response = generate_lullaby(location=location_input,
                                            name= main_character_input,
                                            language=language_input)
                
                with st.expander("English Version"):
                    #st.write(response['story'])
                    st.write(response[0])
                with st.expander(f"{language_input} Version"):
                    #st.write(response['translated'])
                    st.write(response)
                
            st.success("Lullaby Successfully Generated!")

 #Invoking main function
if __name__ == '__main__':
    main()  