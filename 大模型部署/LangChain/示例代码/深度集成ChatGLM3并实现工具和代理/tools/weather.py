# from typing_extensions import Annotated
from typing import Annotated

import requests
from langchain_core.tools import tool


@tool
def get_weather(city_code: Annotated[str, '城市行政编码']='500000'):
    """
    获取city_code对应的天气信息
    """

    response = requests.get(
        f"https://restapi.amap.com/v3/weather/weatherInfo?key=fda1796339b20df0263dc620d14ebe13&city={city_code}")
    if response.status_code == 200:
        data = response.json()
        return str(data)
    else:
        return '天气预报 api 访问失败'


# print(get_weather.name)
# print(get_weather.description)
# print(get_weather.args)
# print(get_weather.args_schema.schema())

tools = [get_weather]
