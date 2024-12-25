import os
from tools.weather import get_weather
from langchain_community.tools.tavily_search import TavilySearchResults
from ChatGLM3 import GLM3, ChatGLM

os.environ['TAVILY_API_KEY'] = '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'

search = TavilySearchResults(max_results=2)
tools = [search]

model_path = r'D:\projects\chatglm.cpp\models\chatglm-ggml.bin'

llm = GLM3(model_path)
model = ChatGLM(llm)

model = model.bind_tools(tools)
print(model.invoke('请使用搜索工具 tavily_search_results_json 告诉我: 今天旧金山的天气如何？'))
