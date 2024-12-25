# 流程步骤
# 1. 使用 @tool 创建工具
# 2. 创建 ChatOpenAI 模型
# 3. model.bind_tools 绑定工具
# 4. 用户提问
# 5. 模型思考用哪个工具
# 6. 获取工具，并使用 .invoke() 调用工具
# 7. 使用 SystemMessage 告诉模型工具调用的结果
# 8. 模型总结输出
from langchain_core.messages import SystemMessage, HumanMessage, ChatMessage

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

messages = [
    SystemMessage(content='你是人工智能助手，请回答用户的所有问题'),
    HumanMessage(content='现在重庆天气如何？'),
]

# 调用模型
message = model.invoke(messages)
messages.append(message)
print(message)

# 判断是否调用工具
if len(message.tool_calls) > 0:
    print('调用工具')
    # 获取工具名称
    tool_name = message.tool_calls[0]['name']
    # 获取模型参数
    args = message.tool_calls[0]['args']
    # 获取工具
    tool_call = eval(tool_name)
    # 调用工具
    # 使用 .invoke() 调用工具，用于兼容 LangChain 官方的工具
    result = tool_call.invoke(args)
    print(result)
    # 因为langchain不支持 role=observation 这个角色，所以我们使用系统角色来返回工具结果
    # 使用 str() 将字典转换成字符串，否则大模型服务器不识别
    messages.append(SystemMessage(content=str(result)))
    # 总结工具调用的结果
    message = model.invoke(messages)
    print(message)
else:
    print('未调用工具')
