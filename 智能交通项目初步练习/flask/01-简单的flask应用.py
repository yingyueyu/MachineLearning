from flask import Flask

app = Flask(__name__)


# http://localhost:9000/
# http://192.168.124.161:9000/
# http://公网IP:9000/
# https://baidu.com
# https(s safe 安全协议) baidu.com (域名 解决公网IP不足)  端口为什么不需要：ngnix 服务器代理
@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>" # 网页文件


if __name__ == '__main__':
    # app.run("localhost", 9000)
    app.run("192.168.124.161",9000)
    app.debug = True # 在控制台查看服务器的运行状态
