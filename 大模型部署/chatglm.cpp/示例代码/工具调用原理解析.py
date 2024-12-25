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
    stream=False
)

model = chatglm_cpp.Pipeline('d:/projects/chatglm3-ggml.bin')


def add(a, b):
    return a + b


messages = [
    ChatMessage(role=ChatMessage.ROLE_SYSTEM,
                content='Answer the following questions as best as you can. You have access to the following tools:\n'
                        '{ "add": {"name": "add", "description": "加法运算", "params": [{"name": "a", "description": "第一个加数", "type": "float", "required": true}, {"name": "b", "description": "第二个加数", "type": "float", "required": true}]} }'),
    ChatMessage(role=ChatMessage.ROLE_USER, content='256 + 128 = ?')
]

message = model.chat(messages, **generation_kwargs)
print(message)
# 保存 AI 调用工具的痕迹
messages.append(message)

# 判断 AI 是否需要调用工具
if len(message.tool_calls) > 0:
    # 将要调用的函数名赋值给 tool_call
    tool_call = eval(message.tool_calls[0].function.name)
    # 运行表达式
    result = eval(message.tool_calls[0].function.arguments)
    # 告诉AI工具调用的结果
    messages.append(ChatMessage(role=ChatMessage.ROLE_OBSERVATION, content=str(result)))
    # 让AI总结工具调用的结果
    message = model.chat(messages, **generation_kwargs)
    print(message)
