from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template= "Gerenate short and simple notes from the following facts text\n {text}",
    input_variables=["text"],

)
prompt2 = PromptTemplate(
    template="Generate 5 short question answer from the following text \n {text}",
    input_variables=["text"],
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n notes -> {notes} \n quiz -> {quiz}",
    input_variables=["notes", "quiz"],
)
model1 = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
model2 = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
# model2 = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

parser = StrOutputParser()

parrellel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "quiz": prompt2 | model2 | parser,
})

merge_chain = prompt3 | model1 | parser

final_chain = parrellel_chain | merge_chain

print(final_chain.invoke({"text": "Okay, okay. So, thank you. Right now, it's 12.30. And, like, we have, like, I can tell you that I wanted to tell, I wanted to create a Hacking the Top conversation of every day. Right now, I started the conversation with you as well. And right now, it's 12.30. So, signing off for today. But, I'm really glad to meet you on the next day, so that I can talk to you and improve my communication skills"}))