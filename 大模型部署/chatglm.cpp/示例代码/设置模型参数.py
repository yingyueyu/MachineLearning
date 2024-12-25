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

model = chatglm_cpp.Pipeline(r'D:\projects\chatglm3-ggml.bin')

messages = [
    ChatMessage(role=ChatMessage.ROLE_SYSTEM, content='请你帮助用户完成用户的文章，你是一个文章续写助手。'),
    ChatMessage(role=ChatMessage.ROLE_USER, content='请帮我续写200字的文章，请以：“从前有座山，山里有座庙，”开头')
]

# 非流式的调用方法
# message = model.chat(messages, **generation_kwargs)
# print(message)

# 流式传输
# 模型将返回生成器
gen = model.chat(messages, **generation_kwargs)
chunks = []
for chunk in gen:
    print(chunk.content, end='')
    chunks.append(chunk)
# 合并流式传输的 chunks
print(model.merge_streaming_messages(chunks))
