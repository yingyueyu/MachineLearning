from abc import ABC
from typing import Any, Dict, List, Iterator

import chatglm_cpp
from chatglm_cpp import ChatMessage
from langchain.llms.base import LLM
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, BaseMessageChunk
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.runnables import Runnable
from langchain_core.runnables.utils import Input, Output
from langchain_core.tools import BaseTool
from tools.weather import tools


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

    def stream(
            self,
            input,
            config=None,
            *,
            stop=None,
            **kwargs: Any,
    ) -> Iterator[str]:
        pass


class ChatGLM(BaseChatModel):
    llm: Any = None
    model_name: str = 'ChatGLM3'
    history: List[ChatMessage] = []
    tools_func: dict[str, Any] = {}

    def __init__(self, model_path_or_model):
        super().__init__()
        if isinstance(model_path_or_model, str):
            self.llm = GLM3(model_path_or_model)
        else:
            self.llm = model_path_or_model

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

    def stream(
            self,
            input,
            config=None,
            *,
            stop=None,
            **kwargs: Any,
    ) -> Iterator[BaseMessageChunk]:
        pass

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
            tools_func[tool.name] = lambda **kwargs: tool.invoke(input=kwargs) if isinstance(tool,
                                                                                             BaseTool) else tool.func

        self.tools_func = tools_func

        self.history = [
            ChatMessage(role="system",
                        content=f"Answer the following questions as best as you can. You have access to the following tools: \n{str(tools_meta)}"),
        ]

        runnable = ChatGLM3Runnable(self, tools_meta)

        return runnable

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "echoing-chat-model-advanced"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model_name,
        }


class ChatGLM3Runnable(Runnable[LanguageModelInput, BaseMessage], ABC):
    def __init__(self, chat_model, tools):
        super().__init__()
        self.chat_model = chat_model
        self.tools = tools

    def invoke(
            self, input: Input, config=None, **kwargs: Any
    ) -> Output:
        result = self.chat_model.invoke(input)
        # self.chat_model.history.append(ChatMessage(role=ChatMessage.ROLE_ASSISTANT, content=result.content))
        if result.call_tool:
            tool_result = {}
            for tool in self.chat_model.llm.tool_calls:
                tool_call = self.chat_model.tools_func[tool.function.name]
                call_result = eval(tool.function.arguments)
                tool_result[tool.function.name] = call_result
            self.chat_model.history.append(ChatMessage(role=ChatMessage.ROLE_OBSERVATION,
                                                       content=f'对应 tool_name 和调用结果如下: \n{str(tool_result)}'))
            result = self.chat_model.invoke('')
            # self.chat_model.history.append(ChatMessage(role=ChatMessage.ROLE_ASSISTANT, content=result.content))

        return result

    def stream(
            self,
            input,
            config=None,
            **kwargs,
    ) -> Iterator[Output]:
        pass


if __name__ == '__main__':
    model_path = r'D:\projects\chatglm.cpp\models\chatglm-ggml.bin'

    # llm = GLM3(model_path)
    # print(llm.invoke('你好'))

    model = ChatGLM(model_path)
    model = model.bind_tools(tools)
    # print(model.invoke('你好'))
    print(model.invoke([
        HumanMessage(content='你好'),
        AIMessage(content='你好啊，请问有什么可以帮助你的吗？'),
        HumanMessage(content='请问 1 + 1 等于几？')
    ]))
    # print(model.invoke('今天重庆(city_code: 500000)天气怎么样？'))
    # print(model.invoke('可以重述下重庆天气吗？'))
