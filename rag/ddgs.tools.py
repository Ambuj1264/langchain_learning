from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

result = search_tool.invoke("tell me the news about the wpl and give the point table list ?")

print(result)

