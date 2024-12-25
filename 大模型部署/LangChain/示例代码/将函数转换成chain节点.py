from langchain_core.runnables import chain


# 使用装饰器将普通函数转换为 Runnable
@chain
def chain1(input):
    return f"chain1: {input['a'] + input['b']}"


@chain
def chain2(input):
    return f'chain2: {input}'


@chain
def chain3(input):
    return f'chain3: {input}'


@chain
def chain4(input):
    return f'chain4: {input}'


@chain
def chain5(input):
    return f'chain5: {input}'


# 可以使用 lambda 表达式来代替 Rannable
# lambda 表达式在 langchain 中多用于流程控制，根据输入 x 判断下一个节点是哪个
pipeline = chain1 | chain2 | chain3 | (lambda x: chain4 if len(x) > 100 else chain5)

result = pipeline.invoke({'a': 3, 'b': 4})

print(result)
