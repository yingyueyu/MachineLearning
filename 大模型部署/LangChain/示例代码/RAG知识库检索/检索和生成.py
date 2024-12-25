from typing import Annotated

from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool

from model import get_model
from BGE_Embed import BGE_Embed

model = get_model()
embedding = BGE_Embed()

# 加载知识库
vectorstore = Chroma(
    embedding_function=embedding,
    persist_directory='./chroma_db'
)

# 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

result = retriever.invoke('刘备的结拜兄弟都有哪些？')
# print(result)
# print(len(result))
# exit()


def doc_formatter(docs):
    return '\n\n'.join([doc.page_content for doc in docs])


# 此处 {context} 代表的是检索结果
system_prompt = (
    "你是一个负责问答任务的助手。"
    "使用以下检索到的上下文来回答问题。"
    "如果你不知道答案，就说你不知道。"
    "最多使用三句话，并保持回答简洁。"
    "\n\n"
    "{context}"
)

# 创建提示词模板
prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    ('user', '{input}')
])

# RunnablePassthrough() 代表 agent 的输入，原封不动的会被替换到 RunnablePassthrough()
# StrOutputParser() 字符串输出转换器，将模型输出的 AIMessage 对象转换成字符串
agent = (
        {'context': retriever | doc_formatter, 'input': RunnablePassthrough()}
        | prompt
        | model
        # | (lambda x: x.content)
        | StrOutputParser()
)

print(agent.invoke('刘备的结拜兄弟都有哪些？'))


# 封装工具
@tool
def sanguo_lib(query: Annotated[str, '查询字符串']):
    """搜索三国相关历史数据"""
    return agent.invoke(query)
