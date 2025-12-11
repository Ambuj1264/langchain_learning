from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import  PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()


model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, max_tokens=100)

# 1st prompt -> deatails report 
template1= PromptTemplate(
    template="""
    what is a details report on {topic}
    """,
    input_variables=["topic"]
)

# 2nd prompt ->  summery
template2 = PromptTemplate(
    template="""
    what is a summary on the following text.  \n {text}
    """,
    input_variables=["text"]
)

prompt1 = template1.invoke({"topic":"AI"})
result1 =model.invoke(prompt1)

prompt2 = template2.invoke({"text":result1.content})
result2 = model.invoke(prompt2)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic":"AI"})
print(result)



# result = model.invoke("What is the capital of France?")
# print(result.content)