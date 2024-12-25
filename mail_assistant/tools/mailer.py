import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
from typing import Annotated, List, Union, Optional
from config import config

from langchain_core.tools import tool


mail_conf = config['email_common']


@tool
def send_mail(
        sender_name: Annotated[str, '发送人名称'],
        sender_email: Annotated[str, '发送人邮箱'],
        password: Annotated[str, '发送人邮箱密码'],
        to: Annotated[List[str], '收件人邮箱'],
        subject: Annotated[str, '主题'],
        content: Annotated[str, '内容'],
        cc: Annotated[Optional[List[str]], '抄送邮箱'] = None
):
    """
    发送邮件
    """

    to_emails = to
    cc_emails = cc if cc is not None else []
    # bcc_emails = ["bcc_recipient1@example.com", "bcc_recipient2@example.com"]

    # 创建一个带有标题和正文的多部分邮件对象
    message = MIMEMultipart()
    message["From"] = formataddr((sender_name, sender_email))  # 添加发件人名字和邮箱
    message["To"] = ", ".join(to_emails)  # 多个收件人用逗号分隔
    message["Cc"] = ", ".join(cc_emails)  # 多个抄送人用逗号分隔
    message["Subject"] = subject

    # 邮件正文内容
    body = content
    # message.attach(MIMEText(body, "plain"))
    message.attach(MIMEText(body, "html"))

    # 收件人列表，包含所有 To、Cc 和 Bcc 的收件人
    recipients = to_emails + cc_emails  # + bcc_emails

    # 连接到邮件服务器（这里以 Gmail 为例）
    smtp_server = mail_conf["server"]
    # port = 587  # For starttls
    # port = 465  # For SSL
    port = mail_conf["port"]  # 未加密

    # 启动一个与邮件服务器的连接
    server = smtplib.SMTP(smtp_server, port)
    server.login(sender_email, password)  # 登录发件邮箱
    try:
        # 发送邮件，收件人列表包括 To、Cc 和 Bcc
        server.sendmail(sender_email, recipients, message.as_string())
        return True
    except:
        # 此处 sendmail 未报错就是发送成功
        return False


@tool
def now(city_name: Annotated[str, '城市名称']):
    """获取城市的实时时间"""
    import time
    return f'{city_name} 当前时间是: {time.time()}'


if __name__ == '__main__':
    print(send_mail.func(
        sender_name='露露',
        sender_email='shampoo6@163.com',
        to=['454714691@qq.com'],
        subject='测试',
        content='<h1>测试</h1>',
        cc=['luxf_cq@hqyj.com']
    ))
