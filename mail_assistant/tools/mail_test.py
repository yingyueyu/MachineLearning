import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr

# 发件人信息（包括名字）
sender_name = "露露"
sender_email = "shampoo6@163.com"
to_emails = ["454714691@qq.com"]
cc_emails = ["luxf_cq@hqyj.com"]
# bcc_emails = ["bcc_recipient1@example.com", "bcc_recipient2@example.com"]
password = "YSZ7e3AWqHFDuwh7"  # 使用第三方应用密码来保证安全性

# 创建一个带有标题和正文的多部分邮件对象
message = MIMEMultipart()
message["From"] = formataddr((sender_name, sender_email))  # 添加发件人名字和邮箱
message["To"] = ", ".join(to_emails)  # 多个收件人用逗号分隔
message["Cc"] = ", ".join(cc_emails)  # 多个抄送人用逗号分隔
message["Subject"] = "测试邮件发送"

# 邮件正文内容
body = "<h1>This is a test email sent from Python with a sender name!</h1>"
# message.attach(MIMEText(body, "plain"))
message.attach(MIMEText(body, "html"))

# 收件人列表，包含所有 To、Cc 和 Bcc 的收件人
recipients = to_emails + cc_emails  # + bcc_emails

# 连接到邮件服务器（这里以 Gmail 为例）
smtp_server = "smtp.163.com"
# port = 587  # For starttls
# port = 465  # For SSL
port = 25  # 未加密

# 启动一个与邮件服务器的连接
server = smtplib.SMTP(smtp_server, port)
server.login(sender_email, password)  # 登录发件邮箱
# 发送邮件，收件人列表包括 To、Cc 和 Bcc
server.sendmail(sender_email, recipients, message.as_string())
# 此处 sendmail 未报错就是发送成功
