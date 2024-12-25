import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

model_name = "gpt2"
# 加载分词器
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# 添加填充token和id
tokenizer.pad_token = '[PAD]'
tokenizer.pad_token_id = 0

# 加载预训练模型
model = GPT2LMHeadModel.from_pretrained(model_name)

# 准备数据
data = [
    {"instruction": "Translate the following text to Chinese.", "input": "Hello, how are you?",
     "output": "你好，你好吗？"},
    {"instruction": "Translate the following text to Chinese.", "input": "Good morning!", "output": "早上好！"},
    {"instruction": "Translate the following text to Chinese.", "input": "What is your name?",
     "output": "你叫什么名字？"},
    {"instruction": "Translate the following text to Chinese.", "input": "I love to read books.",
     "output": "我喜欢读书。"},
    {"instruction": "Translate the following text to Chinese.", "input": "Can you help me?", "output": "你能帮我吗？"},
    {"instruction": "Translate the following text to Chinese.", "input": "Where is the nearest hospital?",
     "output": "最近的医院在哪里？"},
    {"instruction": "Translate the following text to Chinese.", "input": "Thank you very much!", "output": "非常感谢！"},
    {"instruction": "Translate the following text to Chinese.", "input": "See you later.", "output": "再见。"},
    {"instruction": "Translate the following text to Chinese.", "input": "Have a nice day!",
     "output": "祝你有美好的一天！"},
    {"instruction": "Translate the following text to Chinese.", "input": "The weather is beautiful today.",
     "output": "今天的天气很好。"},
    {"instruction": "Translate the following text to Chinese.", "input": "I am learning Chinese.",
     "output": "我在学习中文。"},
    {"instruction": "Translate the following text to Chinese.", "input": "This is a wonderful place.",
     "output": "这是一个很棒的地方。"},
    {"instruction": "Translate the following text to Chinese.", "input": "How much does this cost?",
     "output": "这个多少钱？"},
    {"instruction": "Translate the following text to Chinese.", "input": "I need to make a phone call.",
     "output": "我需要打个电话。"},
    {"instruction": "Translate the following text to Chinese.", "input": "What time is it?", "output": "现在几点？"},
    {"instruction": "Translate the following text to Chinese.", "input": "I am very hungry.", "output": "我很饿。"},
    {"instruction": "Translate the following text to Chinese.", "input": "Do you speak English?",
     "output": "你会说英语吗？"},
    {"instruction": "Translate the following text to Chinese.", "input": "Where are you from?", "output": "你从哪里来？"}
]


# 数据预处理
# 使用 tokenizer 做词嵌入和填充掩码
def preprocess_data(_data):
    input_text = [f'{d["instruction"]} {d["input"]}' for d in _data]
    output_text = [d["output"] for d in _data]
    # 处理数据时需要符合模型的输入输出，gpt2的输入输出长度相同
    input_encodings = tokenizer(input_text, max_length=20, padding='max_length', truncation=True)
    output_encodings = tokenizer(output_text, max_length=20, padding='max_length', truncation=True)
    return input_encodings, output_encodings


# 预处理数据
input_encodings, output_encodings = preprocess_data(data)


# 准备数据集
class ConversationDataset(torch.utils.data.Dataset):
    def __init__(self, input_encodings, output_encodings):
        self.input_encodings = input_encodings
        self.output_encodings = output_encodings
        self.inputs = input_encodings['input_ids']
        self.labels = output_encodings['input_ids']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 找到对应索引数据
        result = {key: torch.tensor(value[idx]) for key, value in self.input_encodings.items()}
        result['labels'] = torch.tensor(self.labels[idx])
        return result


ds = ConversationDataset(input_encodings, output_encodings)

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=3,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# 定义训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
)

# 开始训练
trainer.train()
