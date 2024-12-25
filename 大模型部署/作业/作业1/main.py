import chatglm_cpp

import random

from chatglm_cpp import ChatMessage

nums = 5

names = ['张三', '李四', '老王', '王麻子', '小明']
students = []

for i in range(nums):
    students.append({
        'name': random.choice(names) + str(i),
        'sex': '男' if random.random() > 0.5 else '女',
        'score': random.randint(0, 100)
    })

print(students)
print(len(str(students)))

generation_kwargs = dict(
    # 最大接收 token 长度
    max_length=8186,
    # 最大生成 token 长度
    max_new_tokens=4096,
    # 最大上下文长度
    max_context_length=4096,
    # 是否进行采样
    # 当 do_sample 为 True 时，top_k、top_p、temperature、repetition_penalty 参数生效
    do_sample=True,
    # top_k 采样参数
    top_k=0,
    # top_p 采样参数
    top_p=0.8,
    # 温度
    # 描述模型的稳定程度，温度越高模型越不稳定
    temperature=0.5,
    # 惩罚重复的标记序列
    repetition_penalty=1.,
    # 是否进行流式输出
    stream=True
)

model = chatglm_cpp.Pipeline('d:/projects/chatglm3-ggml.bin')

messages = [
    ChatMessage(role=ChatMessage.ROLE_SYSTEM, content='你是一个 python 代码生成器，请你按照用户要求生成代码。'),
    ChatMessage(role=ChatMessage.ROLE_SYSTEM, content='请你直接给出 "```python" 开头的代码'),
    ChatMessage(role=ChatMessage.ROLE_USER,
                content=f'请根据我提供的数据，使用 pandas 库生成一个 GBK 编码格式的 .csv 文件，并统计所有人的分数，将“总分”插入到表格最后一行；数据如下: \n{students}')
]

chunks = []
for chunk in model.chat(messages, **generation_kwargs):
    chunks.append(chunk)
    print(chunk.content, end='')
print()
message = model.merge_streaming_messages(chunks)

# 截取代码部分
code = message.content.split('```python')[1].split('```')[0]
print('code:')
print(code)
exec(code)
