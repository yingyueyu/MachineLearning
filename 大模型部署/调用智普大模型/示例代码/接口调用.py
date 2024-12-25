import requests
import json

base_url = 'https://chatglm.cn/chatglm/assistant-api/v1'


# get_token: 申请 token 权限的接口
def get_token():
    api_key = '97d00a9ed520c78f'
    api_secret = 'a32de9b20d8c8d7ed7f3f4e48476f5c4'

    # 请求服务器，收到一个 response 响应对象
    response = requests.post(base_url + '/get_token', json={'api_key': api_key, 'api_secret': api_secret})

    # 判断网络请求是否正常收到响应
    if response.status_code == 200:
        # 不是流式输出的结果，通常可以使用 response.json() 来获取响应参数
        with open('token.json', 'w') as file:
            json.dump(response.json(), file)
    else:
        print(response.json())
        raise RuntimeError('token 请求失败')


# 流式输出
def stream(prompt: str):
    # 请求时需要添加 stream=True
    response = requests.post(base_url + '/stream',
                             headers={
                                 'Authorization': f'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJmcmVzaCI6ZmFsc2UsImlhdCI6MTczMDQ0NzIwOSwianRpIjoiYmI5NjdiOGMtNjViMS00Mjg1LTliODItNGIxOGEyNGQwZWZmIiwidHlwZSI6ImFjY2VzcyIsInN1YiI6IkFQSV82NjdkZmNkMjlmZjBmNmM0ZTVkOGE4YmNfNjU5NGQ3MjUiLCJuYmYiOjE3MzA0NDcyMDksImV4cCI6MTczMTMxMTIwOSwidWlkIjoiNjcyMzVhNzY5ZjkyOTJhMmU1MmQyZjhjIiwidXBsYXRmb3JtIjoiIiwiYXBpX3JvbGUiOiJkZXZlbG9wZXIiLCJyb2xlcyI6WyJhdXRoZWRfdXNlciJdfQ.uIVsdbrj9JE49Aw-mfUcoKHg2qlOhM9yGTuNNzwwAeQ'},
                             json={
                                 'assistant_id': '67235cc6e04c28d5cba57868',
                                 'prompt': prompt
                             }, stream=True)

    if response.status_code == 200:
        # 缓存上一次返回的文本
        last_words = None

        # decode_unicode: 解码 unicode 编码
        # http 协议中会自动对特殊字符或中文进行 unicode 编码，所以此处需要解码
        for chunk in response.iter_content(None, decode_unicode=True):
            # print(chunk)
            json_data = json.loads(chunk.split('event:message\ndata: ')[1])
            # print(json_data)
            if 'message' in json_data:
                message = json_data['message']
                if 'content' in message:
                    content = message['content']
                    if 'text' in content:
                        text = content['text']
                        if last_words is None:
                            yield text
                        else:
                            # 截取并返回新增文本
                            yield text.split(last_words)[1]
                        last_words = text

    else:
        raise RuntimeError('请求异常')


# 同步调用接口
def stream_sync(prompt: str):
    # 请求时需要添加 stream=True
    response = requests.post(base_url + '/stream_sync',
                             headers={
                                 'Authorization': f'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJmcmVzaCI6ZmFsc2UsImlhdCI6MTczMDQ0NzIwOSwianRpIjoiYmI5NjdiOGMtNjViMS00Mjg1LTliODItNGIxOGEyNGQwZWZmIiwidHlwZSI6ImFjY2VzcyIsInN1YiI6IkFQSV82NjdkZmNkMjlmZjBmNmM0ZTVkOGE4YmNfNjU5NGQ3MjUiLCJuYmYiOjE3MzA0NDcyMDksImV4cCI6MTczMTMxMTIwOSwidWlkIjoiNjcyMzVhNzY5ZjkyOTJhMmU1MmQyZjhjIiwidXBsYXRmb3JtIjoiIiwiYXBpX3JvbGUiOiJkZXZlbG9wZXIiLCJyb2xlcyI6WyJhdXRoZWRfdXNlciJdfQ.uIVsdbrj9JE49Aw-mfUcoKHg2qlOhM9yGTuNNzwwAeQ'},
                             json={
                                 'assistant_id': '67235cc6e04c28d5cba57868',
                                 'prompt': prompt
                             })

    if response.status_code == 200:
        response_data = response.json()
        return response_data['result']['output'][0]['content'][0]['text']
    else:
        print(response.json())
        raise RuntimeError('请求异常')


if __name__ == '__main__':
    # get_token()
    # for chunk in stream('你好'):
    #     print(chunk)
    print(stream_sync("你好"))
