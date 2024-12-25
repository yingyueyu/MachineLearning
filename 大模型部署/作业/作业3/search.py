from typing import Annotated

from register_tool import register_tool
from tavily import TavilyClient

# 搜索引擎客户端
client = TavilyClient(api_key='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')


@register_tool
def search(query: Annotated[str, '查询关键字', True]):
    """上网搜索信息"""
    response = client.search(query, max_results=5)
    return str(response['results'])
