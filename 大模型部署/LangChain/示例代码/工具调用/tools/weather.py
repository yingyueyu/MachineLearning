import json
from typing import Annotated

import requests
from langchain_core.tools import tool


# 使用 @tool 注册工具
@tool
def get_weather(city: Annotated[str, '城市名']):
    """查询城市的实时天气"""
    # 请求网络
    response = requests.get(f'https://api.seniverse.com/v3/weather/now.json?key=SqXOlV6wp9XtP_zqZ&location={city}')
    # 判断是否请求成功
    if response.status_code == 200:
        result = json.loads(response.text)
        return result['results'][0]['now']
    else:
        return '未查到天气信息'
