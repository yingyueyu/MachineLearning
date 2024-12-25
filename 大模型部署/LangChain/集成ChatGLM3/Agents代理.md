# Agents代理

[官方教学](https://python.langchain.com/v0.2/docs/tutorials/agents/)

[自定义代理文档](https://python.langchain.com/v0.1/docs/modules/agents/how_to/custom_agent)

## 什么是代理？

大语言模型中 LLM 本质上只能处理文本，若想让模型执行一个操作（调用函数），那么就需要用到代理

所以代理就是一个结合了 LLM 并能够执行一些操作的工具，人们也称它为 **智能体**

> ==把我们之前调用工具的课程代码进行封装，就是一个会查天气的代理==

**代理的作用就是拿来执行工具**

## 关于 langchain 代理的槽点

langchain 的 Agent 功能内部**过度设计**，**高度抽象**、**高度封装**，**极其复杂**，使用 langchain 支持的商业模型将非常的方便

但是一旦使用 langchain 不支持的模型时将无比的痛苦

结论:

- 自定义模型不需要使用 `langchain` 的 `Agent` 功能
- 自定义模型可以自行封装 `Agent`
- `langchain` `Agent` 的相关用法不重要，但是 `Agent` 的思想重要

## 官方支持的模型使用代理


3. 创建代理
4. 创建代理执行器


5. 执行代理执行器

官方例子中 3、4 步融合成了一句话 `create_react_agent`

```python
# Import relevant functionality
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from ChatGLM3 import get_model

os.environ['TAVILY_API_KEY'] = 'xxx'

# 记忆服务
memory = MemorySaver()
# 这个是官方支持的模型
# model = ChatAnthropic(model_name="claude-3-sonnet-20240229")
model = get_model()
# 这是搜索引擎工具
search = TavilySearchResults(max_results=2)
tools = [search]
# 创建代理执行器
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# 使用代理时可以指定代理运行的线程 id
config = {"configurable": {"thread_id": "abc123"}}

# 执行代理
result = agent_executor.invoke(
    {"messages": [HumanMessage(content="你好，我是露露，我生活的城市是重庆(city_code=500000)")]}, config
)
print(result)
print("----")
```

## 自定义代理(使用 AgentExecutor)(不建议)

自定义代理的话，就需要自行实现以下内容:

3. 创建代理
4. 创建代理执行器

### 创建代理

假设已经获取了模型和工具

```python
model = get_model()
tools = [get_adcode, get_weather]
# chat_history 用于记录历史对话数据
MEMORY_KEY = "chat_history"
chat_history: List[BaseMessage] = []
# 绑定工具
model = model.bind_tools(tools)
```

接下来创建提示词模板，因为我们使用的是 OpenAI 的服务器接口，所以模板格式是固定的，是和服务器约定好的结构

```python
prompt = ChatPromptTemplate.from_messages(
    [
        # 系统提示
        (
            "system",
            "You are very powerful assistant, but don't know current events",
        ),
        # 用于存放历史信息
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        # 用户输入
        ("user", "{input}"),
        # 代理会在此处自动设置数据
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
```

然后创建代理

```python
# 可以看到代理本质上就是一个 chain
agent = (
        # 处理输入，构造一个提示词模板 prompt 需要的参数字典
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
            "chat_history": lambda x: x["chat_history"],
        }
        # 将上个节点的字典作为参数调用提示词模板
        | prompt
        # 将上个节点提示词模板返回的提示词作为参数，调用模型
        # 此处必须使用 lambda 表达式调用模型，否则会丢失 tool_calls 信息
        # 因为代理执行器会根据节点类型不同采取不同行为
        | (lambda x: model.invoke(x))
        # 自定义 Runnable 用于处理模型思考结果
        | ToolCallRunnable()
)
```

值得注意的是 `ToolCallRunnable`，这是一个自定义 `Runnable` 对象，我们调用工具的逻辑放在此处

### 实现 `ToolCallRunnable`

> Tip: 可以使用 @langchain_core.runnables.chain 装饰器，将任意函数转换成 Runnable 实例

`ToolCallRunnable` 的完整代码如下:

```python
class ToolCallRunnable(Runnable):
    def invoke(self, input, config=None, **kwargs):
        return_value = [input]
        if len(input.tool_calls) > 0:
            try:
                tool_result = {}
                for tool_call in input.tool_calls:
                    tool = [tool for tool in tools if tool.name == tool_call['name']]
                    _tool = tool[0] if len(tool) > 0 else None
                    tool = lambda **_kwargs: _tool.invoke(input=_kwargs) if isinstance(_tool, BaseTool) else _tool.func
                    result = tool(**tool_call['args'])
                    tool_result[tool_call['name']] = result
                return_value.append(SystemMessage(
                    content=f'已收到工具调用的结果，对应 tool_name 和调用结果如下: \n{tool_result}。\n请根据工具结果回答用户的问题。',
                    response_metadata={'call_tool': True}
                ))
            except:
                return_value.append(SystemMessage(
                    content=f'系统调用工具失败',
                    response_metadata={'call_tool': True}
                ))
        return AgentFinish(return_values={'messages': return_value}, log=str({'tool_calls': input.tool_calls}))
```

#### 重点分析:

##### 1. 判断是否 AI 认为要调用工具

```python
if len(input.tool_calls) > 0:
```

`input` 是模型返回的 `AIMessage` 实例，其中包含一个 `tool_call` 属性，该列表代表要调用的工具

##### 2. 获取工具

```python
tool = lambda **_kwargs: _tool.invoke(input=_kwargs) if isinstance(_tool, BaseTool) else _tool.func
```

因为使用 `@tool` 装饰器的自定义工具和基于 `BaseTool` 类的工具，调用方式有些不同，所以此处做了判断，并兼容了两种类型的工具调用

##### 3. 返回数据

我们返回 `AgentFinish` 告诉代理执行器代理执行结束了，类似的还有 `AgentStep` `AgentAction` 等

```python
return AgentFinish(return_values={'messages': return_value}, log=str({'tool_calls': input.tool_calls}))
```

==返回数据若是 `SystemMessage` 则包含元数据 `{call_tool: True}` 标志着已调用工具==

### 创建代理执行器

```python
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

**至此代理基本完成，但有些问题:**

当我们的代理访问了工具的话，此处返回的是调用工具的结果，而非模型整理后的自然语言输出

所以我们还要让模型总结工具结果并输出

### 调用工具

调用代理执行器既能执行代理，但是代理执行器内部是不会记录历史信息的，所以调用完后，还需要记录历史

```python
inp = "请问重庆的 adcode 高德地区编码是多少？"
result = agent_executor.invoke({"input": inp, "chat_history": chat_history})
chat_history.extend([
    HumanMessage(content=inp),
    *result['messages']
])
```

### 总结工具结果并输出

若代理调用了工具，则需要总结成自然语言，如下:

```python
# 判断是否需要 AI 总结工具响应的结果
if isinstance(chat_history[-1], SystemMessage) and 'call_tool' in chat_history[-1].response_metadata:
    # 再次调用模型，让他总结结果
    result = model.invoke(chat_history)
    chat_history.append(result)
```
