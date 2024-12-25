# 让 AI 参与到应用程序中，需要思考把什么环节交给 AI 完成
# 把整个过程都交给 ai，这样做整个程序将非常不可控

import random

import chatglm_cpp
from chatglm_cpp import ChatMessage
from register_tool import tool_meta_datas, goal_num_compare

generation_kwargs = dict(
    max_length=2048,
    max_new_tokens=1024,
    max_context_length=1024,
    do_sample=True,
    top_k=0,
    top_p=0.,
    temperature=0.0,
    repetition_penalty=1.,
    stream=False
)

model = chatglm_cpp.Pipeline('d:/projects/chatglm4-ggml.bin')

messages = [
    ChatMessage(role=ChatMessage.ROLE_SYSTEM,
                content=f'Answer the following questions as best as you can. You have access to the following tools:\n{tool_meta_datas}'),
    # 让 AI 完成 goal 和 num 数字的比较
    # 我们的程序提供 goal 和 num 这两个数字
    ChatMessage(role=ChatMessage.ROLE_SYSTEM, content='系统需要你根据如下规则，进行回答：\n'
                                                      '你只能回答 “太大了” “太小了” “大了” “小了” “回答正确” 中的一句\n'
                                                      '请你调用工具 goal_num_compare 获取比较结果 result\n'
                                                      '若 result > 10 回答 “太大了”\n'
                                                      '若 result < -10 回答 “太小了”\n'
                                                      '若 result < 10 且 result > 0  回答 “大了”\n'
                                                      '若 result > -10 且 result < 0 回答 “小了”\n'
                                                      '若 result == 0 回答 “回答正确”'
                )
]

# 开始游戏

# 生成随机数
goal = str(random.randint(0, 100))
print(goal)

print('游戏开始')

# 用户猜数的次数
user_nums = 0
# AI猜数的次数
AI_nums = 0

while True:
    num = input('请猜数: ')
    user_nums += 1

    # 克隆提示词 并 添加提示词
    _msg = [*messages, ChatMessage(role=ChatMessage.ROLE_SYSTEM, content=f'goal={goal}; num={num}')]

    # 调用AI，判别结果
    message = model.chat(_msg, **generation_kwargs)
    print(message)
    _msg.append(message)

    # 判断是否调用工具
    if len(message.tool_calls) > 0:
        tool_call = goal_num_compare
        # 执行工具
        result = eval(message.tool_calls[0].function.arguments)
        print(f'result={result}')
        _msg.append(ChatMessage(role=ChatMessage.ROLE_OBSERVATION, content=f'result={result}'))
        # 调用AI回答问题
        message = model.chat(_msg, **generation_kwargs)
        print(message)

        # 判断是否该结束循环
        if message.content == '回答正确':
            break
    else:
        raise SystemError('AI 未调用工具')
