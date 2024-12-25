from langchain_openai import ChatOpenAI


def get_model():
    return ChatOpenAI(
        model='default',
        base_url='http://127.0.0.1:8000/v1',
        api_key='Empty',
        temperature=0.7,
        top_p=0.9
    )

if __name__ == '__main__':
    model = get_model()
    print(model.invoke('你好'))
