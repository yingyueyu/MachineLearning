import chatglm_cpp
from chatglm_cpp import ChatMessage

model = chatglm_cpp.Pipeline(r'D:\projects\chatglm3-ggml.bin')

message = model.chat([
    ChatMessage(role=ChatMessage.ROLE_SYSTEM, content='你是一个人工智能助手，你的名字叫做"张三"'),
    ChatMessage(role=ChatMessage.ROLE_USER, content='你好，请问你叫什么名字？')
])

print(message)
