# 安装 chatglm-cpp[api]
# pip install chatglm-cpp[api]

# 启动服务
# set MODEL=./models/chatglm-ggml.bin && uvicorn chatglm_cpp.openai_api:app --host 127.0.0.1 --port 8000

# 安装 openai 库
# pip install openai


import openai

# 创建 Client 客户端 实例
client = openai.Client(
    # 开发者密钥，需要随便填写一个
    api_key='token123',
    # 访问请求时的基础网络路径
    base_url='http://127.0.0.1:8000/v1'
)

# api 文档: https://platform.openai.com/docs/api-reference/chat/create
# 向服务器发起请求
response = client.chat.completions.create(
    # 访问 default 模型
    model='default',
    # 消息
    messages=[
        {"role": "system", "content": "你是一个人工智能助手叫做张三"},
        {"role": "user", "content": "你叫什么名字？"}
    ],
    # 因为我们使用的是openai服务器的模拟器
    # 所以并非所有参数都有效
    # 是否流式输出
    stream=True
)

print(response)
for chunk in response:
    print(chunk)
