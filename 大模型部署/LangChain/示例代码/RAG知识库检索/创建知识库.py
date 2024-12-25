from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from BGE_Embed import BGE_Embed

docs = TextLoader('./docs/三国演义.txt', encoding='GBK').load()
# print(docs)

# 获取文本字符串
txt = docs[0].page_content

# 创建文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    # 断句的符号
    separators=[
        "\n\n",  # 段落换行
        "\n",  # 单换行
        ".",  # 英文句号
        "?",  # 英文问号
        "!",  # 英文感叹号
        "。",  # 中文句号
        "？",  # 中文问号
        "！",  # 中文感叹号
        # "；",  # 中文分号
        # ";",  # 英文分号
        # "：",  # 中文冒号
        # ":",  # 英文冒号
    ],
    # 文本块的长度
    chunk_size=300,
    # 两文本块间重叠的长度
    chunk_overlap=100
)

# 分割文档，获取文本块
chunks = text_splitter.split_documents(docs)

# 加载词嵌入模型
embedding = BGE_Embed()

# 创建向量仓库
vectorstore = Chroma.from_documents(
    # 文档
    documents=chunks,
    # 词嵌入模型
    embedding=embedding,
    # 保存的目录
    persist_directory='./chroma_db'
)
