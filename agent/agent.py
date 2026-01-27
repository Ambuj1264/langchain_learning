from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langchain import hub
from dotenv import load_dotenv

load_dotenv()

# Tool
search_tool = DuckDuckGoSearchRun()

# LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Pull ReAct prompt
prompt = hub.pull("hwchase17/react")

# Create ReAct agent
react_agent = create_react_agent(
    llm=llm,
    prompt=prompt,
    tools=[search_tool]
)

# Agent Executor
agent_executor = AgentExecutor(
    agent=react_agent,
    tools=[search_tool],
    verbose=True
)

# Invoke agent
agent_executor.invoke({
    "input": "What is the top news in Noida today?"
})
