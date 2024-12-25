# 流程步骤
# 1. 使用 @tool 创建工具
# 2. 创建 ChatOpenAI 模型
# 3. model.bind_tools 绑定工具
# 4. 用户提问
# 5. 模型思考用哪个工具
# 6. 获取工具，并使用 .invoke() 调用工具
# 7. 使用 SystemMessage 告诉模型工具调用的结果
# 8. 模型总结输出
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import chain

from model import get_model
from tools.weather import get_weather

# 获取模型
model = get_model()

# 制作工具列表
tools = [get_weather]

# 绑定工具
# 告诉ai你可以用什么工具
# .bind_tools 来自于 BaseChatModel
model = model.bind_tools(tools)

# 消息历史
history = []

# 用户输入的消息
input_message = ''

# 工具调用所产生的消息列表
tool_messages = [
    # HumanMessage(content='今天天气如何？')
    # AIMessage(content='```python\ntool_call()```')
    # SystemMessage(content='晴天 24°')
]


# 准备输入
# 该节点用于记录用户输入
@chain
def prepare_input(message: str):
    global input_message, history, tool_messages
    # 缓存输入
    input_message = message
    # 清空工具调用的缓存
    tool_messages.clear()
    return {'input': input_message, 'history': history}


prompt = ChatPromptTemplate.from_messages([
    ('system', '你是人工智能助手，名叫小明。'),
    MessagesPlaceholder(variable_name='history'),
    ('user', '{input}')
])


# 判断是否调用工具，控制程序流程
@chain
def is_call_tool(message):
    global input_message
    if len(message.tool_calls) > 0:
        # 调用工具
        return call_tool_pipeline
    else:
        # 若不调用工具，则直接追加消息到历史会话
        history.append(HumanMessage(content=input_message))
        history.append(message)
        return message


# 调用工具
@chain
def call_tool(message):
    global input_message, tool_messages

    # 获取工具名称
    tool_name = message.tool_calls[0]['name']
    # 获取模型参数
    args = message.tool_calls[0]['args']
    # 获取工具
    tool_call = eval(tool_name)
    # 调用工具
    result = tool_call.invoke(args)

    # 追加用户的提问
    tool_messages.append(HumanMessage(content=input_message))
    # 追加AI思考的工具调用的结果
    tool_messages.append(message)
    # 追加工具调用的结果
    tool_messages.append(SystemMessage(content=str(result)))

    return tool_messages


# 追加消息到历史会话
@chain
def append_message_to_history(message):
    global history, input_message
    # 先添加用户的提问
    history.append(HumanMessage(content=input_message))
    # 再添加模型回答的结果
    history.append(message)
    return message


call_tool_pipeline = (
        call_tool
        | model
        | append_message_to_history
)

agent = (
        prepare_input
        | prompt
        | model
        | is_call_tool
)

# result = agent.invoke('你叫什么名字？')
# print(result)
# result = agent.invoke('重庆现在天气如何？')
# print(result)

if __name__ == '__main__':
    while True:
        inp = input('user: ')
        result = agent.invoke(inp)
        print(result)
