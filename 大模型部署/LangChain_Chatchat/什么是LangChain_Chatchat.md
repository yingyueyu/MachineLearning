# 什么是LangChain_Chatchat

[项目地址](https://github.com/chatchat-space/Langchain-Chatchat)

Langchain-Chatchat（原Langchain-ChatGLM）基于 Langchain 与 ChatGLM, Qwen 与 Llama 等语言模型的 RAG 与 Agent 应用 | Langchain-Chatchat (formerly langchain-ChatGLM), local knowledge based LLM (like ChatGLM, Qwen and Llama) RAG and Agent app with langchain

[原理视频](https://www.bilibili.com/video/BV13M4y1e7cN/?share_source=copy_web&vd_source=e6c5aafe684f30fbe41925d61ca6d514)

> **注意:** 自 0.3.0 起，该项目不支持直接加载模型，而是需要使用推理引擎框架，例如: Xinference、Ollama、LocalAI、FastChat、One API 等，但是这些推理引擎部署框架都不支持 chatglm.cpp 的 ggml 模型，所以此处我们选择降级

## 对这个框架的定位

他是用来搭建聊天应用服务的，该项目有以下特点

- 自带 API 接口文档
- 自动接入模型
- 支持知识库和工具调用
