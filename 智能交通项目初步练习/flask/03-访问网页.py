from flask import Flask,render_template

app = Flask(__name__)


@app.route("/")  # http://192.168.124.161:9000/
def hello_world():
    # 此处会返回一个网页
    return render_template("index.html")


if __name__ == '__main__':
    app.run("192.168.124.161", 9000)
    app.debug = True  # 在控制台查看服务器的运行状态
