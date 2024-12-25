# < img
# class ="file-image" role="presentation"
# src = "https://shampoo6.github.io/img/qmd-cq.jpg"
# style = "cursor: pointer;" >

config = {
    "email_common": {
        "server": "smtp.163.com",
        "port": 25,

    },
    "email": {
        "sender_name": "陆宪甫",
        "sender_email": "shampoo6@163.com",
        "password": "xxxxxxxxxxx",
        "to": ["454714691@qq.com"],
        "subject": "重庆中心-教学部-陆宪甫-工作日报",
        "cc": ["2721445883@qq.com"]
    },
    # 生成内容的要求（系统级别要求）
    "system_requires": [
        "请你帮助我完成工作日报的填写工作",  # 交代任务
        "我是一名老师，我每天会完成教学工作",  # 交代行业
        "生成的总字数在 20 字以内",  # 限制字数
        "不要生成时间",
        "不要生成代码",  # 限制生成代码
    ],
    # 生成文本的内容要求
    # key: jinja 模版中的变量名
    "requires": {
        # "today_works": "请帮我编写今天的工作内容汇报，生成的内容和以下工作相关:\n"
        #                "langchain教学、langchain-chatchat教学、chatglm教学",
        "today_works": {
            "prefix": "请帮我编写 20 字的今天的工作汇报，与以下内容相关: ",
            "suffix": [
                "langchain教学",
                "langchain-chatchat教学",
                "chatglm教学"
            ]
        },
        # "tomorrow_works": "请帮我生成明天的工作内容汇报，汇报内容和以下工作相关:\n"
        #                   "langchain教学、xinference教学、ubuntu系统教学",
        "tomorrow_works": {
            "prefix": "请帮我编写 20 字的明天的工作计划，与以下内容相关: ",
            "suffix": [
                "langchain教学",
                "xinference教学",
                "ubuntu系统教学"
            ]
        },
    },
    # 邮件内容模版
    "template": """<p>余老师，你好：</p>
<p>以下是今天的工作总结及明天工作计划：</p>
<p>今天工作总结：</p>
<ul>
  {% for work in today_works %}
    <li>{{ work }}</li>
  {% endfor %}
</ul>
<p>明天工作计划：</p>
<ul>
  {% for work in tomorrow_works %}
    <li>{{ work }}</li>
  {% endfor %}
</ul>""",
    "sign": """<p>顺颂商祺！</p>
    <p><br></p>
    <p><strong>陆宪甫&nbsp;&nbsp;|&nbsp;教学部</strong></p>
    <p><br></p>
    <p>
        <img src="https://gitee.com/shampoo6/cq_ai_240701/raw/master/qmd-cq.jpg"/>
    </p>
    <p><br></p>
    <p>公司地址：重庆市渝中区虎头岩总部城A区四号楼12楼</p>
    <p>手机号码：17783683002</p>
    <p>座机号码：023-61966978</p>
    <p>咨询热线：400-611-6270</p>
    <p>电子邮件：<a href="mailto:luxf_cq@hqyj.com" target="_blank">luxf_cq@hqyj.com</a></p>
    <p><strong>集团官网：</strong><a href="about:blank" target="_blank">www.hqyj.com</a></p>
    <p>创客学院：<a href="about:blank" target="_blank">www.makeru.com.cn</a></p>
    <p>研发中心：<a href="about:blank" target="_blank">www.fsdev.com.cn</a></p>
    <p><br></p>
    <p><a href="http://bj.hqyj.com/" target="_blank">北京</a>·<a href="http://sh.hqyj.com/" target="_blank">上海</a>·<a
            href="http://sz.hqyj.com/" target="_blank">深圳</a>·<a href="http://cd.hqyj.com/" target="_blank">成都</a>·<a
            href="http://nj.hqyj.com/" target="_blank">南京</a>·<a href="http://wh.hqyj.com/" target="_blank">武汉</a>·<a
            href="http://xa.hqyj.com/" target="_blank">西安</a>·<a href="http://gz.hqyj.com/" target="_blank">广州</a>·<a
            href="http://sy.hqyj.com/" target="_blank">沈阳</a>·<a href="http://jn.hqyj.com/" target="_blank">济南</a>·<a
            href="http://cq.hqyj.com/" target="_blank">重庆</a>·<a href="http://cs.hqyj.com/" target="_blank">长沙</a>·<a
            href="http://cq.hqyj.com/" target="_blank">重庆</a>·<a href="http://hz.hqyj.com/" target="_blank">杭州</a></p>
    <p><br></p>"""
}
