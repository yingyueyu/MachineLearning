from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI

# 创建客户端
# model 是一个 Runnable
# 第一个节点
model = ChatOpenAI(
    model='llama3-3b',
    base_url='http://192.168.128.134:9997/v1',
    api_key='EMPTY',
    temperature=0.7,
    top_p=0.8
)

# 聊天信息
messages = [
    SystemMessage(content='你是人工智能助手，你叫张三。'),
    HumanMessage(content='你叫什么名字？')
]


@chain
def node2(message):
    messages.append(message)
    return messages


def node3(_messages):
    result = ''
    for message in _messages:
        # 判断不同的消息类型，输出不同的文本
        if isinstance(message, SystemMessage):
            result += f'System: {message.content}\n'
        elif isinstance(message, HumanMessage):
            result += f'User: {message.content}\n'
        elif isinstance(message, AIMessage):
            result += f'AI: {message.content}\n'
    return result


# class Node2(Runnable):
#     def invoke(self):
#         pass
#
#     def stream(self):
#         pass
#
# node2 = Node2()


# pipeline = model.pipe(node2).pipe(node3)
pipeline = model | node2 | node3

result = pipeline.invoke(messages)
print(result)
