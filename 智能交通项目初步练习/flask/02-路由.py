from flask import Flask

app = Flask(__name__)


@app.route("/a")  # http://192.168.124.161:9000/a
def hello_world1():
    return "<h1>Hello, World! A</p>"


@app.route("/b")  # http://192.168.124.161:9000/b
def hello_world2():
    return "<p>Hello, World! B</p>"


if __name__ == '__main__':
    app.run("192.168.124.161", 9000)
    app.debug = True  # 在控制台查看服务器的运行状态
