[toc]

# 集成ChatGLM3

[参考](https://zhipu-ai.feishu.cn/wiki/X5shwBPOBiDWyNkwZ6xcd33lnRe)

之前搭建的[简单应用](./搭建一个简单应用.md)是利用的 API 服务器；实际上我们可以从代码层面更深度的集成 ChatGLM3

## 为什么要深度集成 ChatGLM3

### 原因一: LangChain 原生支持模型少

目前 LangChain 仅支持流行的大模型，例如: ChatGPT，Berd，Claude

### 原因二: 不支持的模型功能受限

受支持的模型可以非常简单的使用工具 tool 和创建代理 agent

但是，若使用不受支持的模型，这些功能都不能直接使用

### 原因三: 提示词格式不同

标准 LangChain 的提示词格式，并不与 ChatGLM 直接兼容

## 大致流程

1. 继承 `LLM` 类，实现一个大预言模型，作为基础服务提供工具
2. 继承 `BaseChatModel` 类，实现一个聊天任务模型，作为聊天工具
3. 继承 `Runnable[LanguageModelInput, BaseMessage]` 作为实现 `BaseChatModel` 中 `bind_tools` 的返回值，该 `Runnable` 作为 `chain` 链式调用的最外层包装器

### 1. 创建基础服务 LLM

这一步可以跳过，直接使用 `chatglm.cpp` 的 `API` 服务器，如下:

```python
llm = ChatGLM(endpoint_url="http://127.0.0.1:8000", max_token=2048, top_p=0.7, temperature=0.95, with_history=False)
```

亦可以深度集成 `ChatGLM3` 项目或 `chatglm.cpp` 项目

这里以集成 `chatglm.cpp` 项目为例

创建 `GLM` 类，并集成 `LLM`，如下:

```python
class GLM3(LLM):
    generation_kwargs: dict = dict(
        max_length=2048,  # 最大总长度，包括提示和输出
        max_new_tokens=-1,  # 生成的最大标记数，忽略提示标记的数量
        max_context_length=512,  # 最大上下文长度
        do_sample=0.95 > 0,  # 是否开启采样策略，开启的话才会进行 top-k 和 top-p 采样
        top_k=0,  # top-k 采样，选择概率最高的 k 个样本，并在之中随机一个结果
        top_p=0.7,  # top-p 采样，随机选取概率和超过 top-p 的样本，并在这些样本中随机一个结果
        temperature=0.95,  # 温度，模型输出的稳定程度，温度越高越不稳定
        repetition_penalty=1,  # 惩罚重复的标记序列
        stream=False,  # 是否流式输出
    )

    model: object = None
    tool_calls: List[Any] = None

    # model_path: 指向模型文件的路径，如: D:\chatglm.cpp\model\chatglm-ggml.bin
    def __init__(self, model_path):
        super().__init__()
        self.load_model(model_path)

    def _llm_type(self) -> str:
        return 'ChatGLM3'

    def load_model(self, model_path):
        self.model = chatglm_cpp.Pipeline(model_path)

    def _call(self, prompt: str, history=None, stop=None, run_manager=None) -> str:
        if history is None:
            history = []
        if prompt.strip() != '':
            history.append(ChatMessage(role='user', content=prompt.split('Human:')[1].strip()))
        result = self.model.chat(history, **self.generation_kwargs)
        history.append(result)
        self.tool_calls = None if len(result.tool_calls) == 0 else result.tool_calls
        return result.content
```

#### 重点分析:

##### 1. _llm_type

必须实现该方法，这是 `langchain` 接口的要求，返回模型名称即可

##### 2. load_model

实现一个加载模型的方法，模型引擎不一定使用 `chatglm.cpp` 也可以使用 `transformers` 或 `vllm`

##### 3. _call

必须实现 `_call` 方法，该方法内需要实现整个模型推理的逻辑，如下:

```python
# prompt: 提示词，该提示词由 langchain 处理过
# history: 历史信息，由大模型引擎决定格式，此处的格式为 chatglm.cpp 的历史信息格式
# stop=None, run_manager=None: 这两个是 langchain 接口参数，可以不填值，但是必须声明在函数签名
def _call(self, prompt: str, history=None, stop=None, run_manager=None) -> str:
    if history is None:
        history = []
    if prompt.strip() != '':
        # 由于提示词被 langchain 处理过，此处提取源文本，去掉 langchain 添加的 Human:
        history.append(ChatMessage(role='user', content=prompt.split('Human:')[1].strip()))
    # 调用 chatglm.cpp 模型
    result = self.model.chat(history, **self.generation_kwargs)
    history.append(result)
    # 判断是否调用工具
    # 若要调用工具，这里会保存调用工具的元数据 result.tool_calls
    self.tool_calls = None if len(result.tool_calls) == 0 else result.tool_calls
    return result.content
```

### 2. 创建聊天任务模型

基于基础大模型服务 `GLM`，我们对它进行封装，得到一个适应对话任务的模型 `ChatGLM`，如下:

```python
class ChatGLM(BaseChatModel):
    llm: Any = None
    model_name: str = 'ChatGLM3'
    history: List[ChatMessage] = []
    tools_func: dict[str, Any] = {}

    def __init__(self, model_path):
        super().__init__()
        self.llm = GLM3(model_path)

    def _generate(
            self,
            messages,
            stop=None,
            run_manager=None,
            **kwargs: Any,
    ) -> ChatResult:
        if messages is None:
            messages = ''
        tokens = self.llm.invoke(messages, history=self.history)
        message = AIMessage(
            content=tokens,
            call_tool=self.llm.tool_calls is not None
        )
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def bind_tools(
            self,
            tools,
            **kwargs,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        tools_func = {}
        tools_meta = []
        for tool in tools:
            schema = tool.args_schema.schema()
            tools_meta.append({
                'name': tool.name,
                'description': tool.description,
                'parameters': [{
                    'name': k,
                    'description': v['description'],
                    'type': v['type'],
                    'required': k in schema['required']
                } for k, v in schema['properties'].items()]
            })
            tools_func[tool.name] = tool.func

        self.tools_func = tools_func

        self.history = [
            ChatMessage(role="system",
                        content=f"Answer the following questions as best as you can. You have access to the following tools: \n{str(tools_meta)}"),
        ]

        runnable = ChatGLM3Runnable(self, tools_meta)

        return runnable

    @property
    def _llm_type(self) -> str:
        return "echoing-chat-model-advanced"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
        }
```

内容比较多，我们关注以下重点

#### 重点分析:

##### 1. 实现父类要求的属性

这里的属性是根绝 `langchain` 的接口要求，必须实现的属性

```python
@property
def _llm_type(self) -> str:
    return "echoing-chat-model-advanced"

@property
def _identifying_params(self) -> Dict[str, Any]:
    return {
        "model_name": self.model_name,
    }
```

##### 2. 实现 _generate 方法

`_generate` 方法是生成对话的方法，必须实现，如下:

```python
# messages: 消息文本
def _generate(
        self,
        messages,
        stop=None,
        run_manager=None,
        **kwargs: Any,
) -> ChatResult:
    if messages is None:
        messages = ''
    # 调用时，需要把聊天的历史信息传给 LLM
    tokens = self.llm.invoke(messages, history=self.history)
    # 构造返回值
    message = AIMessage(
        content=tokens,
        # call_tool: 这是人为增加的元数据，用来区分是否需要调用工具
        response_metadata={"call_tool": self.llm.tool_calls is not None}
    )
    generation = ChatGeneration(message=message)
    return ChatResult(generations=[generation])
```

==**注意:** 此处的 `call_tool` 参数是人为添加的，用来区分是否需要调用工具的元数据==

##### 3. 实现 bind_tools 方法

我们不仅需要使用 ChatGLM 模型进行聊天，更希望它能够调用工具，为了能够调用工具，此处需要实现 bind_tools，**这一步尤其重要**

完整代码如下:

```python
def bind_tools(
            self,
            tools,
            **kwargs,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
    tools_func = {}
    tools_meta = []
    for tool in tools:
        schema = tool.args_schema.schema()
        tools_meta.append({
            'name': tool.name,
            'description': tool.description,
            'parameters': [{
                'name': k,
                'description': v['description'],
                'type': v['type'],
                'required': 'required' in schema and k in schema['required']
            } for k, v in schema['properties'].items()]
        })
        tools_func[tool.name] = lambda **kwargs: tool.invoke(input=kwargs) if isinstance(tool, BaseTool) else tool.func

    self.tools_func = tools_func

    self.history = [
        ChatMessage(role="system",
                    content=f"Answer the following questions as best as you can. You have access to the following tools: \n{str(tools_meta)}"),
    ]

    runnable = ChatGLM3Runnable(self, tools_meta)

    return runnable
```

参数解析如下:

- tools: 工具函数列表，每个函数需要使用 `@tool` 装饰器进行装饰

详细步骤如下:

创建保存可调用工具函数和工具元数据的变量

```python
# 用于保存工具函数
# key: 函数名 value: 工具函数
tools_func = {}
# 用于保存工具元数据，这些元数据为 AI 描述了这些工具的用途和参数
tools_meta = []
```

其中元数据的格式如下:

```json
// tools_meta
[
    // 一个工具的元数据如下:
    {
        "name": "工具名称",
        "description": "工具的描述信息",
        // 工具函数的参数
        "parameters": [
            {
                "name": "参数名",
                "description": "参数描述",
                "type": "类型",
                "required": "是否必填"
            }
        ]
    },
    {
        // 其他工具
    }
]
```

然后解析参数 `tools`，如下:

```python
# 循环 tools 列表
for tool in tools:
    # 获取每个工具的元数据结构
    schema = tool.args_schema.schema()
    # 解析元数据并保存到 tools_meta 中
    tools_meta.append({
        'name': tool.name,
        'description': tool.description,
        'parameters': [{
            'name': k,
            'description': v['description'],
            'type': v['type'],
            'required': 'required' in schema and k in schema['required']
        } for k, v in schema['properties'].items()]
    })
    # 保存工具函数
    tools_func[tool.name] = lambda **kwargs: tool.invoke(input=kwargs) if isinstance(tool, BaseTool) else tool.func
```

值得注意的是，为了兼容 `langchain` 的其他自带工具，我们这里需要判断该工具是否是 `langchain_core.tools.BaseTool` 类的实例，并作出不同的处理，如下:

```python
# BaseTool 类的工具，都提供一个 invoke 方法供我们调用
tools_func[tool.name] = lambda **kwargs: tool.invoke(input=kwargs) if isinstance(tool, BaseTool) else tool.func
```

然后为 `chatglm.cpp` 添加系统消息:

```python
self.history = [
    # 按照 ChatGLM 的逻辑，需要添加如下提示词，告诉 AI 我们提供了哪些工具可以使用
    # ChatGLM 会通过工具思维，自行思考调用哪个工具
    ChatMessage(role="system",
                content=f"Answer the following questions as best as you can. You have access to the following tools: \n{str(tools_meta)}"),
]
```

最后 `bind_tools` 需要返回一个 `Runnable` 的对象，我们需要自己实现一个该对象

### 3. 创建 Runnable[LanguageModelInput, BaseMessage] 子类

`Runnable` 是 `langchain` 中的接口，是一个链式调用的包装器，假设我们有 3 个 `Runnable` 实例，分别名为 `A` `B` `C`，那么我们可以使用 `LCEL` 语法来链式调用，如: `A | B | C`

该语法会把 `A` 返回值传入 `B`，`B` 返回值再传入 `C`，最后得到 `C` 的返回值

这种行为的背后，实际是由 `langchain` 自动调用了 `A` `B` `C` 三个 `Runnable` 的 `invoke` 方法而实现的，所以上述 `LCEL` 表达式可以转换为:

```python
result = A.invoke()
result = B.invoke(result)
result = C.invoke(result)
```

所以每个 `Runnable` 我们可以理解为一个 **中间件**

我们要创建一个用于判断**是否要调用工具**和**调用工具**的**中间件**，具体代码如下:

```python
class ChatGLM3Runnable(Runnable[LanguageModelInput, BaseMessage], ABC):
    def __init__(self, chat_model, tools):
        super().__init__()
        self.chat_model = chat_model
        self.tools = tools

    def invoke(
            self, input: Input, config=None, **kwargs: Any
    ) -> Output:
        result = self.chat_model.invoke(input)
        if result.call_tool:
            tool_result = {}
            for tool in self.chat_model.llm.tool_calls:
                tool_call = self.chat_model.tools_func[tool.function.name]
                call_result = eval(tool.function.arguments)
                tool_result[tool.function.name] = call_result
            self.chat_model.history.append(ChatMessage(role=ChatMessage.ROLE_OBSERVATION,
                                                       content=f'对应 tool_name 和调用结果如下: \n{str(tool_result)}'))
            result = self.chat_model.invoke('')

        return result
```

#### 重点分析:

##### 1. 初始化

初始化时主要是需要传入一些重要的参数

```python
# chat_model: 我们创建的聊天任务模型 ChatGLM 的实例
# tools: 之前解析好的 tools_meta 工具元数据
def __init__(self, chat_model, tools):
    super().__init__()
    self.chat_model = chat_model
    self.tools = tools
```

##### 2. 调用工具流程

先要根据之前 ChatGLM 中添加的元数据 `call_tool` 判断是否需要调用工具

```python
if result.response_metadata['call_tool']:
```

若要调用工具，则走工具调用流程

```python
# 准备一个空字典，用于保存工具的返回值
tool_result = {}
# 循环调用多个工具
for tool in self.chat_model.llm.tool_calls:
    # 保存要调用的函数引用
    tool_call = self.chat_model.tools_func[tool.function.name]
    # 将字符串 tool.function.arguments 作为脚本调用
    call_result = eval(tool.function.arguments)
    # 将返回值保存下来
    tool_result[tool.function.name] = call_result
    # 添加一条 role 为 observation 的历史消息，代表工具的返回结果
    self.chat_model.history.append(ChatMessage(role=ChatMessage.ROLE_OBSERVATION,
                                            content=f'对应 tool_name 和调用结果如下: \n{str(tool_result)}'))
# 带着工具的返回结果，再次调用大模型进行推理
result = self.chat_model.invoke('')
```

这里**值得注意的有两点**:

1. `tool.function.arguments` 该字符串大概长这样 `tool_call(city_code="500000")`，仔细观察发现，这就是一个调用函数 `tool_call` 的 `python` 代码，并且参数已经填进去了，所以此处我们直接获取 `tool_call` 的引用，并使用 `call_result = eval(tool.function.arguments)` 进行调用
2. 因为 `ChatGLM` 聊天任务模型内维护了一个 `history` 列表，所以，当我们新增一个工具调用结果 `ChatMessage(role=ChatMessage.ROLE_OBSERVATION, content='...')`，无需再传入任何参数，直接再次调用聊天模型就可以了，如下: `result = self.chat_model.invoke('')`