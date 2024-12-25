import chatglm_cpp
from chatglm_cpp import ChatMessage
# 先导入工具
from search import search
# 再导入元数据
from register_tool import tool_meta_datas

print(tool_meta_datas)

generation_kwargs = dict(
    max_length=2048,
    max_new_tokens=1024,
    max_context_length=1024,
    do_sample=True,
    top_k=0,
    top_p=1.,
    temperature=.7,
    repetition_penalty=1.,
    stream=True
)

model = chatglm_cpp.Pipeline('d:/projects/chatglm3-ggml.bin')

messages = [
    ChatMessage(role=ChatMessage.ROLE_SYSTEM,
                content=f'Answer the following questions as best as you can. You have access to the following tools:\n{tool_meta_datas}'),
    ChatMessage(role=ChatMessage.ROLE_USER, content='请帮我上网搜索最近的新闻')
]

chunks = []
for chunk in model.chat(messages, **generation_kwargs):
    print(chunk.content, end='')
    chunks.append(chunk)
# 融合消息
message = model.merge_streaming_messages(chunks)
messages.append(message)

# 判断是否调用工具
if len(message.tool_calls) > 0:
    # 获取工具名称
    tool_name = message.tool_calls[0].function.name
    # 获取工具函数
    tool_call = eval(tool_name)
    # 执行工具
    results = eval(message.tool_calls[0].function.arguments)
    print(results)
    # 将工具调用结果告诉ai
    messages.append(ChatMessage(role=ChatMessage.ROLE_OBSERVATION, content=results))
    # 调用Ai，总结新闻
    chunks = []
    for chunk in model.chat(messages, **generation_kwargs):
        print(chunk.content, end='')
        chunks.append(chunk)
    messages.append(model.merge_streaming_messages(chunks))
