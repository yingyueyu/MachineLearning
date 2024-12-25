# 智谱大模型客户端
import json
import os.path
import time

import requests


class GLMClient:
    base_url = 'https://chatglm.cn/chatglm/assistant-api/v1'
    # 权限令牌
    token = None

    def __init__(self,
                 api_key,
                 api_secret,
                 model_id,  # 模型的 id
                 base_url=None,
                 ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.model_id = model_id
        if base_url is not None:
            self.base_url = base_url
        self._get_token()

    def _get_token(self):
        # 先尝试访问本地token
        # 判断是否存在token文件
        if os.path.exists('token.json'):
            with open('token.json', 'r') as file:
                json_data = json.load(file)
            # 验证是否过期
            # 获取当前系统时间
            now = time.time()
            if now < json_data['result']['token_expires']:
                # 未过期
                self.token = json_data['result']['access_token']
                print('获取缓存')
                return

        # 请求服务器，收到一个 response 响应对象
        response = requests.post(self.base_url + '/get_token',
                                 json={'api_key': self.api_key, 'api_secret': self.api_secret})

        # 判断网络请求是否正常收到响应
        if response.status_code == 200:
            json_data = response.json()
            # 不是流式输出的结果，通常可以使用 response.json() 来获取响应参数
            with open('token.json', 'w') as file:
                json.dump(json_data, file)
            self.token = json_data['result']['access_token']
        else:
            print(response.json())
            raise RuntimeError('token 请求失败')

    def stream(self, prompt: str):
        # 请求时需要添加 stream=True
        response = requests.post(self.base_url + '/stream',
                                 headers={
                                     'Authorization': f'Bearer {self.token}'},
                                 json={
                                     'assistant_id': self.model_id,
                                     'prompt': prompt
                                 }, stream=True)

        last_txt = None
        expired = False
        for line in response.iter_content(None, decode_unicode=True):
            json_txt = line.split('event:message\ndata:')[1].strip()
            json_data = json.loads(json_txt)

            if response.status_code == 200:
                message = json_data['message']
                if 'content' in message:
                    content = message['content']
                    if 'text' in content:
                        if last_txt is not None:
                            yield content['text'].split(last_txt)[1]
                        else:
                            yield content['text']
                        last_txt = content['text']
            elif response.status_code == 403 and json_data['status'] == 10018:
                # 无权限或 token 过期
                expired = True
                break
            else:
                print(response.text)
                raise RuntimeError('请求失败')

        if expired:
            # 1 秒后重试
            time.sleep(1)
            self._get_token()
            return self.stream(prompt)

    def stream_sync(self, prompt: str):
        # 请求时需要添加 stream=True
        response = requests.post(self.base_url + '/stream_sync',
                                 headers={
                                     'Authorization': f'Bearer {self.token}'},
                                 json={
                                     'assistant_id': self.model_id,
                                     'prompt': prompt
                                 })

        response_data = response.json()
        if response.status_code == 200:
            return response_data['result']['output'][0]['content'][0]['text']
        elif response.status_code == 403 and response_data['status'] == 10018:
            # token 过期
            time.sleep(1)
            self._get_token()
            return self.stream_sync(prompt)
        else:
            print(response.json())
            raise RuntimeError('请求异常')


if __name__ == '__main__':
    client = GLMClient(
        api_key='97d00a9ed520c78f',
        api_secret='a32de9b20d8c8d7ed7f3f4e48476f5c4',
        model_id='67235cc6e04c28d5cba57868'
    )

    # print(client.stream_sync('你好'))
    for chunk in client.stream('你好'):
        print(chunk, end='')
