# 导入 vllm 相关包
from vllm import LLM, SamplingParams

# 创建提示词
prompts = [
    "你好",
    "今天天气怎么样？",
]
# 创建模型参数
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# 加载模型
# trust_remote_code=True 信任远程代码，不加这句话无法加载本地模型
llm = LLM(model="/opt/chatglm3-6b", trust_remote_code=True, quantization_config={"dtype": "int4"})
# llama 创建时的参数
llm = LLM(model="/opt/Llama-3.2-3B-Instruct-Q8_0.gguf", trust_remote_code=True, device='cpu', max_model_len=2048)

# 生成结果
outputs = llm.generate(prompts, sampling_params)

# 输出
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
