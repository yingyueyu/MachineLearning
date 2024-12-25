from config import config
import chatglm_cpp
from chatglm_cpp import ChatMessage
from jinja2 import Template

# model = chatglm_cpp.Pipeline("models/chatglm3-ggml-q8_0.bin")
model = chatglm_cpp.Pipeline("models/chatglm-ggml.bin")

# 对话参数
generation_kwargs = dict(
    max_length=2048,  # 最大总长度，包括提示和输出
    max_new_tokens=-1,  # 生成的最大标记数，忽略提示标记的数量
    max_context_length=512,  # 最大上下文长度
    do_sample=0.95 > 0,  # 是否开启采样策略，开启的话才会进行 top-k 和 top-p 采样
    top_k=0,  # top-k 采样，选择概率最高的 k 个样本，并在之中随机一个结果
    top_p=1.,  # top-p 采样，随机选取概率和超过 top-p 的样本，并在这些样本中随机一个结果
    temperature=0.7,  # 温度，模型输出的稳定程度，温度越高越不稳定
    repetition_penalty=1,  # 惩罚重复的标记序列
    stream=False,  # 是否流式输出
)

messages = []

# 构造系统消息
system_requires = '\n'.join(config['system_requires'])
print(system_requires)
messages.append(ChatMessage(role=ChatMessage.ROLE_SYSTEM, content=system_requires))


# chunks = []
# for chunk in model.chat(messages, **generation_kwargs):
#     print(chunk.content, sep="", end="", flush=True)
#     chunks.append(chunk)
#
# model.merge_streaming_messages(chunks)


def generate_content():
    params = {}

    for k, v in config['requires'].items():
        print(k, v)
        results = []
        for line in v['suffix']:
            prompt = v['prefix'] + line
            msg = [*messages, ChatMessage(role=ChatMessage.ROLE_USER, content=prompt)]
            result = model.chat(msg, **generation_kwargs)
            print(result)
            results.append(result.content)
        params[k] = results

    print(params)

    template = Template(config['template'])
    content = template.render(**params)
    print(content)
    content += config['sign']

    return content
