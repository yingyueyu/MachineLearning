import sys
from typing import Annotated

sys.path.append(r'D:\projects\学校\课程笔记\cq_ai_240701\大模型部署\chatglm.cpp\示例代码')

from register_tool import register_tool
from tavily import TavilyClient
from pprint import pformat


@register_tool
def search(query: Annotated[str, '查询关键字', True]):
    """搜索引擎"""
    # 1. 创建客户端工具
    client = TavilyClient(api_key='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')

    # 2. 搜索
    response = client.search(query, max_results=3)

    # print(pformat(response))
    return str(response)


if __name__ == '__main__':
    # from duckduckgo_search import DDGS
    #
    # results = DDGS().text("黑神话悟空", max_results=3)
    # print(pformat(results))
    print(pformat(search('小米su7')))
