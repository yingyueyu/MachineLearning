from generate import generate_content
from tools.mailer import send_mail
from config import config
from datetime import datetime

content = generate_content()
print(content)

email_conf = config['email']

current_time = datetime.now()
dt = current_time.strftime(' %Y%m%d')
email_conf['subject'] += dt

print(send_mail.func(content=content, **email_conf))
