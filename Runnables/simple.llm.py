from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# create the prompt template

prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
    Write a 100 word about {topic}
    """
)

# define the input 
topic =  input("Enter the topic: ")

# format the prompt manually using PromptTemplate
fromatted_prompt = prompt.format(topic=topic)

# generate the response
response = llm.invoke(fromatted_prompt)

print(response.content)