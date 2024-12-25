# Runnable 是 langchain 实现链式调用的基础，链式调用的每个节点必须是 Runnable 类型
from langchain_core.runnables import Runnable


class Chain1(Runnable):
    # 大部分情况 input 为字典，a: 第一个数 b: 第二个数，例如 input={'a': 1, 'b': 2}
    def invoke(self, input, config=None, **kwargs):
        print('chain1')
        return f"chain1: {input['a'] + input['b']}"

    def stream(self, input, config=None, **kwargs):
        return self.invoke(input, config, **kwargs)


class Chain2(Runnable):
    def invoke(self, input, config=None, **kwargs):
        print('chain2')
        return f'chain2: {input}'

    def stream(self, input, config=None, **kwargs):
        return self.invoke(input, config, **kwargs)


class Chain3(Runnable):
    def invoke(self, input, config=None, **kwargs):
        print('chain3')
        return f'chain3: {input}'

    def stream(self, input, config=None, **kwargs):
        return self.invoke(input, config, **kwargs)


# 竖线 | 语法称为 LCEL (LangChain Expression Language) 语法
# 竖线 | 的含义，将左侧表达式的结果作为右侧表达式的输入
# pipeline: 管线
# pipeline = Chain1() | Chain2() | Chain3()

# 使用 pipe 来串联管线
pipeline = Chain1().pipe(Chain2()).pipe(Chain3())

# 调用管线
result = pipeline.invoke({'a': 1, 'b': 2})

print(f'result: {result}')
