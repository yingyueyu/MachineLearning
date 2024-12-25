from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import chain, RunnablePassthrough
from langchain.agents import AgentExecutor

from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from tools.mailer import send_mail, now


def create_email_agent(llm):
    tools = [send_mail]
    # tools = [now]

    model = llm.bind_tools(tools)

    # tools_meta = [{
    #     'name': tool.name,
    #     'description': tool.description,
    #     'parameters':
    # } for tool in tools]

    tools_meta = []
    for tool in tools:
        schema = tool.args_schema.schema()
        param = [{
            'name': k,
            'description': v['description'],
            'type': v['type'] if 'type' in v else v['anyOf'][0]['type'],
            'required': k in schema['required']
        } for k, v in schema['properties'].items()]
        tools_meta.append({
            'name': tool.name,
            'description': tool.description,
            'parameters': param
        })

    prompt = ChatPromptTemplate.from_messages([
        (
            'system',
            "Answer the following questions as best as you can. You have access to the following tools: \n{tools_meta}"
        ),
        (
            'user',
            '我想要发送邮件，对应发送邮件的信息如下:\n'
            # 'mail_data={mail_data}'
            '发送人是 {sender_name}\n'
            '发送人邮箱是 {sender_email}\n'
            '收件人是 {to}\n'
            '主题是 {subject}\n'
            '内容是 {content}\n'
            '抄送邮箱是 {cc}\n'
        ),
        # MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    @chain
    def call_model(input):
        print(input)
        message = model.invoke(input)
        print(message)
        return message

    @chain
    def model_result_processor(input):
        print(input)
        return input

    # 发送邮件代理
    send_mail_agent = (
        # {
        #     # "sender_name": lambda x: x["sender_name"],
        #     # "sender_email": lambda x: x["sender_email"],
        #     # "to": lambda x: x["to"],
        #     # "subject": lambda x: x["subject"],
        #     # "content": lambda x: x["content"],
        #     # "cc": lambda x: x["cc"],
        #     'mail_data': lambda x: x['mail_data'],
        #     "agent_scratchpad": lambda x: format_to_openai_tool_messages(
        #         x["intermediate_steps"]
        #     ),
        # }
            {
                "sender_name": lambda x: x["sender_name"],
                "sender_email": lambda x: x["sender_email"],
                "to": lambda x: x["to"],
                "subject": lambda x: x["subject"],
                "content": lambda x: x["content"],
                "cc": lambda x: x["cc"],
                'tools_meta': lambda x: tools_meta
            }
            | prompt
            | call_model
            # | model
            | model_result_processor
    )

    agent_executor = AgentExecutor(agent=send_mail_agent, tools=tools, verbose=True)

    # test
    agent_executor.invoke(dict(
        sender_name='露露',
        sender_email='shampoo6@163.com',
        to=['454714691@qq.com'],
        subject='测试',
        content='<h1>测试</h1>',
        cc=['luxf_cq@hqyj.com']
    ))
    # agent_executor.invoke({'input': '请问重庆现在的时间是多少？'})

    return


if __name__ == '__main__':
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="default-model",
        base_url='http://127.0.0.1:8000/v1',
        api_key='abc',
        temperature=0.7,  # 控制生成文本的随机性
        top_p=1.  # 核采样策略
    )

    # llm = ChatOpenAI(
    #     model="llama3-3b",
    #     base_url='http://192.168.128.128:9997/v1',
    #     api_key='abc',
    #     temperature=0.7,  # 控制生成文本的随机性
    #     top_p=1.  # 核采样策略
    # )

    create_email_agent(llm)
    # for chunk in llm.stream('你好'):
    #     print(chunk.content, end='')
