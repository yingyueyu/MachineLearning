import chatglm_cpp
from chatglm_cpp import ChatMessage

generation_kwargs = dict(
    # 最大接收 token 长度
    max_length=2048,
    # 最大生成 token 长度
    max_new_tokens=256,
    # 最大上下文长度
    max_context_length=1024,
    # 是否进行采样
    # 当 do_sample 为 True 时，top_k、top_p、temperature、repetition_penalty 参数生效
    do_sample=True,
    # top_k 采样参数
    top_k=2,
    # top_p 采样参数
    top_p=0.8,
    # 温度
    # 描述模型的稳定程度，温度越高模型越不稳定
    temperature=1.,
    # 惩罚重复的标记序列
    repetition_penalty=1.,
    # 是否进行流式输出
    stream=True
)

model = chatglm_cpp.Pipeline('d:/projects/chatglm3-ggml.bin')

# 用于记录所有的对话
messages = [
    ChatMessage(role='system', content='你是一个聊天机器人，请用中文回答用户的问题。请在每一句回复中都赞美一下用户。'),
]

print('你好，现在你正在和 AI 交流，请你敞开心扉。')

# 对话循环
while True:
    # 获取用户输入
    message = input('用户: ')
    # 追加用户消息
    messages.append(ChatMessage(role='user', content=message))
    print('AI: ', end='')
    chunks = []
    for chunk in model.chat(messages, **generation_kwargs):
        print(chunk.content, end='')
        chunks.append(chunk)
    # 融合 AI 的回复消息
    message = model.merge_streaming_messages(chunks)
    # 追加 AI 消息
    messages.append(message)
    print()
