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

messages = [
    ChatMessage(role=ChatMessage.ROLE_SYSTEM, content='你是一个 python 代码生成器，请你按照用户要求生成代码。'),
    ChatMessage(role=ChatMessage.ROLE_USER, content='请打印 0~100 的质数')
]

chunks = []
for chunk in model.chat(messages, **generation_kwargs):
    print(chunk.content, end='')
    chunks.append(chunk)

print()

# 融合AI的回复
message = model.merge_streaming_messages(chunks)
# print(message)

if '```python' in message.content:
    # 提取代码
    code = message.content.split('```python')[1].split('```')[0]
    print(code)
    # 将字符串 code 当作脚本运行
    exec(code)
else:
    print('没有代码')
