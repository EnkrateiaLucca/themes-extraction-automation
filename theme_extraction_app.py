import streamlit as st
import openai
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
import json
import os
import requests
from io import BytesIO
import base64


local = st.sidebar.checkbox("Local mode", True)

if local:
    openai_api_key = os.environ["OPENAI_API_KEY"]
else:
    openai_api_key = st.sidebar.text_input("ENTER YOUR OPENAI_API_KEY, see here for more info: https://platform.openai.com/docs/quickstart?context=python")
    
client = OpenAI(api_key=openai_api_key)

def get_response(prompt_question):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "system", "content": "You are a helpful research and programming assistant"},
                  {"role": "user", "content": prompt_question}]
    )
    
    return response.choices[0].message.content


def extract_pdf_raw_contents(pdfpath):
    """Extracts the raw text contents from a pdf file or from a URL."""
    raw_contents = ""
    
    if pdfpath.startswith('http://') or pdfpath.startswith('https://'):
        response = requests.get(pdfpath)
        file = BytesIO(response.content)
        loader = PyPDFLoader(file)
        documents = loader.load()
        for doc in documents:
            raw_contents += doc.page_content + "\n"
    else:
        loader = PyPDFLoader(pdfpath)
        documents = loader.load()
        for doc in documents:
            raw_contents += doc.page_content + "\n"
    
    return raw_contents

def extract_pdf_from_deployed(pdf_file):
    base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">' 
    st.markdown(pdf_display, unsafe_allow_html=True)
    
    return pdf_display


@st.cache_data
def get_model_tags():
    models = openai.models.list()
    models_data = models.data
    return [m.id for m in models_data if "gpt" in m.id]
    
        
    

theme = {
    "alias": {
        "radii": {
            "radius_100": "4px",
            "radius_200": "8px",
            "radius_300": "999px",
        },
        "background_100": "#fcf2eb",
        "background_200": "#fadfca",
        "background_300": "#ffdbbf",
        "background": {
            "contrast_100": "#f29750",
            "contrast_200": "#ed6b06",
        },
        "text_100": "#ed6b06",
        "text_200": "#261782",
        "text_300": "#ffffff",
        "border_100": "#ed6b06",
        "border": {
            "contrast_100": "#a2a2a2",
            "contrast_200": "#5e5851",
        },
        "states": {
            "success": "#ed6b06",  # have to be Hex because of hexToRGB()
            "neutral": "#fcf2eb",
            "warning": "#fc9a4e",
            "danger": "#db4223",
        },
    },
        "gradients": {
        # Merge existing gradients with new ones
        # **theme_gradients,
        "variant1": "linear-gradient(0deg, #120949, #120949)",
        "variant2": "linear-gradient(0deg, #FFFFFF 19.85%, #FFF 118.93%)",  # Secondary button
        "variant3": "linear-gradient(to right, #ed6b06, #ed6b06)",
    },
}


class Radii(BaseModel):
    radius_100: Optional[str] = Field(..., description="Radius size of 100")
    radius_200: Optional[str] = Field(..., description="Radius size of 200")
    radius_300: Optional[str] = Field(..., description="Radius size of 300")

class Background(BaseModel):
    contrast_100: Optional[str] = Field(..., description="Background contrast level 100")
    contrast_200: Optional[str] = Field(..., description="Background contrast level 200")

class Border(BaseModel):
    contrast_100: Optional[str] = Field(..., description="Border contrast level 100")
    contrast_200: Optional[str] = Field(..., description="Border contrast level 200")

class States(BaseModel):
    success: Optional[str] = Field(..., description="Color for success state")
    neutral: Optional[str] = Field(..., description="Color for neutral state")
    warning: Optional[str] = Field(..., description="Color for warning state")
    danger: Optional[str] = Field(..., description="Color for danger state")

class Gradient(BaseModel):
    variant1: Optional[str] = Field(..., description="Gradient variant 1")
    variant2: Optional[str] = Field(..., description="Gradient variant 2")
    variant3: Optional[str] = Field(..., description="Gradient variant 3")

class Alias(BaseModel):
    radii: Optional[Radii] = Field(..., description="Radii specifications")
    background_100: Optional[str] = Field(..., description="Primary background color")
    background_200: Optional[str] = Field(..., description="Secondary background color")
    background_300: Optional[str] = Field(..., description="Tertiary background color")
    background: Optional[Background] = Field(..., description="Background contrast specifications")
    text_100: Optional[str] = Field(..., description="Primary text color")
    text_200: Optional[str] = Field(..., description="Secondary text color")
    text_300: Optional[str] = Field(..., description="Tertiary text color")
    border_100: Optional[str] = Field(..., description="Primary border color")
    border: Optional[Border] = Field(..., description="Border contrast specifications")
    states: Optional[States] = Field(..., description="State colors")
    gradients: Optional[Gradient] = Field(..., description="Gradient specifications")

class Theme(BaseModel):
    alias: Optional[Alias] = Field(..., description="Alias specifications and semantic meanings")




# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)



st.title("GPT-Theme-Extractor app")

st.header("GPT-Theme-Extractor app")


model = st.sidebar.selectbox("Select a model", get_model_tags())

system_prompt = st.sidebar.text_area("System prompt", "You are an expert extraction algorithm for design themes from pdf presentations.\
            Only extract relevant information from the pdf output that relates to the creation of the appropriate alias object for styling a page.\
            You will never return a null value for any section. If you do not know the value of an attribute asked to extract,\
            return the best option for the context for that attribute's value.")

pdfpath = st.text_input("Enter the path to the PDF file (related to root folder in the app github)")

if pdfpath:

    llm = ChatOpenAI(model=model, temperature=0)

    output_parser = JsonOutputParser(pydantic_object=Theme)

    format_instructions = output_parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt
            ),
            # Please see the how-to about improving performance with
            # reference examples.
            # MessagesPlaceholder('examples'),
            ("human", "{text}"),
        ],
    )

    runnable = prompt | llm.with_structured_output(schema=Theme)


    pdf_raw_contents = st.text_area("PDF raw contents (you can edit)", extract_pdf_raw_contents(pdfpath))

    if st.button("Extract theme"):
        output = runnable.invoke({"text": pdf_raw_contents})
        output_json = output.json()
        st.json(output_json)
        if st.button("Save theme to file"):
            with open('alias-style.json', 'w+') as f:
                json.dump(output_json, f)
    