# 各类消息的作用: https://python.langchain.com/v0.2/docs/how_to/custom_chat_model/#messages

# 知识点
# 使用 langchain_openai.ChatOpenAI 创建模型
# Message 类型
# 了解 Runnable 类型的 .invoke() .stream()
# Runnable 是 langchain 实现链式调用的基础，链式调用的每个节点必须是 Runnable 类型


# 导入聊天模型客户端
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ChatMessage

# xinference
# chatglm.cpp

# 创建 openai 客户端
model = ChatOpenAI(
    # 模型名称
    model='default',
    # 服务器的基础路径
    base_url='http://127.0.0.1:8000/v1',
    # api 服务器的权限 key
    # 此处我们使用自己的 api 服务器，所以无需验证权限，此处可以随便写
    api_key='EMPTY',
    temperature=0.7,
    top_p=0.8
)

# 创建消息
messages = [
    # 系统消息
    SystemMessage(content='你是人工智能助手，你的名字叫张三，今年3岁了。'),
    # 用户消息
    HumanMessage(content='你今年多大了？'),
    # AI消息
    AIMessage(content='我今年3岁了。'),
    # 聊天消息
    ChatMessage(role='user', content='你叫什么名字？')
]

# 通过客户端访问大模型

# 同步调用大模型
# message = model.invoke(messages)
# print(message)
# messages.append(message)

# 同步但流式输出
response = model.stream(messages)
print(response)  # 生成器
chunks = None
for chunk in response:
    # 此处的 chunk 为 AIMessageChunk 类
    print(chunk)
    if chunks is None:
        chunks = chunk
    else:
        # AIMessageChunk 类 可以直接累加
        chunks += chunk
print()
print(chunks)
messages.append(chunks)
