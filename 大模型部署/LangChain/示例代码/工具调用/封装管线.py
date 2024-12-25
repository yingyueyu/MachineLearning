from langchain_core.messages import SystemMessage, HumanMessage, ChatMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import chain

from model import get_model
from tools.weather import get_weather

# 获取模型
model = get_model()

# 制作工具列表
tools = [get_weather]

# 绑定工具
model = model.bind_tools(tools)

history = [
    # AIMessage(content='我的名字叫张三。。。。。')
]

# ChatPromptTemplate: 提示词模板
prompt = ChatPromptTemplate.from_messages([
    ('system', '你是人工智能助手，名字叫小明。'),
    # MessagesPlaceholder 消息占位符
    MessagesPlaceholder(variable_name='history'),
    # {input} 这是一个模板语法，input 在此处是一个变量
    ('user', '{input}')
])


# 是否调用工具
@chain
def is_call_tool(message):
    history.append(message)
    if len(message.tool_calls) > 0:
        # 走调用工具的流程
        return call_tool_pipeline
    else:
        return message


# 调用工具
@chain
def call_tool(message):
    print('调用工具')
    # 获取工具名称
    tool_name = message.tool_calls[0]['name']
    # 获取模型参数
    args = message.tool_calls[0]['args']
    # 获取工具
    tool_call = eval(tool_name)
    # 调用工具
    result = tool_call.invoke(args)
    return result


# 处理工具调用的结果
@chain
def result_handler(result):
    history.append(SystemMessage(content=str(result)))
    # return [HumanMessage(content=user_input), *history]
    messages = prompt.invoke({'input': '', 'history': history})
    messages.messages.pop()
    return messages


# 调用工具的流程
call_tool_pipeline = call_tool | result_handler | model

# prompt: 这是一个提示词模板，用于创建提示词
pipeline = prompt | model | is_call_tool

user_input = '重庆现在天气如何？'
# result = pipeline.invoke({'input': '你叫什么名字？', 'history': history})
result = pipeline.invoke({'input': user_input, 'history': history})

print(result)

# 此处的 pipeline 我们也可以称为 agent 代理
# 我们此处创建的代理叫做 Action Agent，特点就是前一个节点的输出丢给下一个节点作为输入
