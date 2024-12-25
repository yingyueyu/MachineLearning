from langchain_openai import ChatOpenAI


def get_model(temperature=0.7, top_p=0.9):
    return ChatOpenAI(
        model='default',
        base_url='http://127.0.0.1:8000/v1',
        api_key='EMPTY',
        temperature=temperature,
        top_p=top_p
    )


if __name__ == '__main__':
    model = get_model()
    print(model.invoke('你好'))
