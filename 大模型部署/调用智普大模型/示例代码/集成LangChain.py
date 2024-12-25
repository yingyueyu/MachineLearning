from typing import Optional, Any, Iterator

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import RunnableConfig

from GLMClient import GLMClient


class GLM(LLM):
    model: GLMClient = None

    def __init__(self, api_key, api_secret, model_id):
        super().__init__()
        self.model = GLMClient(api_key, api_secret, model_id)

    def _llm_type(self) -> str:
        return 'ChatGLM3'

    def _call(
            self,
            prompt: str,
            stop: Optional[list[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        return self.model.stream_sync(prompt)

    def stream(
            self,
            input: LanguageModelInput,
            config: Optional[RunnableConfig] = None,
            *,
            stop: Optional[list[str]] = None,
            **kwargs: Any,
    ) -> Iterator[str]:
        return self.model.stream(input)


if __name__ == '__main__':
    llm = GLM(
        api_key='97d00a9ed520c78f',
        api_secret='a32de9b20d8c8d7ed7f3f4e48476f5c4',
        model_id='67235cc6e04c28d5cba57868'
    )

    for chunk in llm.stream('你好'):
        print(chunk, end='')
