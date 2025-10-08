# from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAI,ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
os.environ['LANGCHAIN_PROJECT']='#2 SEQENTIAL LLM'
load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model = GoogleGenerativeAI(model="gemini-2.5-flash")

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

config = {
    'tags': ['llm app', 'Report Generation', 'Summarization'],
    'metadata': {
        'model': 'gemini-2.5-flash',
        'parser': 'stroutputparser'
    }
}


result = chain.invoke({'topic': 'Unemployment in India'},config=config)

print(result)
