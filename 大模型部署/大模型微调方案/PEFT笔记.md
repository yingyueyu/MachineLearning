# PEFT

Parameter-Efficient Fine-Tuning 参数高效微调

[官网](https://huggingface.co/docs/peft/index)

[Transformers 预训练模型](https://huggingface.co/docs/transformers/index)

Transformers 提供 API 和工具来轻松下载和训练最先进的预训练模型。

## LoRA

Low-Rank Adaptation 低秩自适应，[论文](https://arxiv.org/abs/2106.09685)

配置方法:

```py
# 添加 LoRA 微调配置
config = peft.LoraConfig(
    r=8,
    target_modules=["seq.0", "seq.2"],
    modules_to_save=["seq.4"],
)
```

### 重要配置参数解析

重要配置选项如下:

#### init_lora_weights

初始化 lora 权重的方法，默认情况使用 Kaiming-uniform 方法初始化权重，该方法将权重均匀初始化在一个范围内

1. 对于具有 ReLU 激活函数的层：
   1. 如果该层是全连接层（全连接神经网络中的层），范围为：sqrt(6 / fan_in)，其中 fan_in 是输入单元的数量。
   2. 如果该层是卷积层，范围为：sqrt(6 / (fan_in + fan_out))，其中 fan_in 是输入通道数，fan_out 是输出通道数。

2. 对于具有其他激活函数的层，如 Sigmoid 或 Tanh：
   1. 上述范围的数值分布会有所调整，以适应不同的激活函数的性质。

```
# 默认初始化
init_lora_weights=True
# 完全随机初始化，只有测试时会这样做
init_lora_weights=False
# 高斯分布初始化
init_lora_weights="gaussian"
# LoftQ 初始化
init_lora_weights="loftq"
```

##### 关于 LoftQ

LoftQ 是一种新兴的量化技术，旨在进一步优化模型的推理性能和效率。

LoftQ，全称是 Low-Fidelity Quantization，是一种基于低保真度量化的技术。这种方法通过将模型参数量化到更低的精度来减少计算和存储需求，从而提高推理速度和减少内存占用。

LoftQ 采用低比特宽度（如 8 位、4 位甚至更低）来表示模型权重和激活值。这与传统的 32 位浮点数表示相比，可以显著减少模型的大小和计算需求。尽管量化降低了数值的精度，但通过精心设计和优化，LoftQ 可以在保持模型性能的同时，实现显著的加速效果。

官方建议在目标层较多时，使用 LoftQ，例如:

```
LoraConfig(..., target_modules="all-linear") 
```

此外，在使用 4 位量化时，您应该在量化配置中使用 nf4 作为量化类型，即 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")` 。

#### use_rslora

使用秩稳定的 LoRA，例如: `config = LoraConfig(use_rslora=True, ...)`

该选项会引入缩放因子到 LoRA 的权重中，该缩放因子是个标量，取决于 LoRA 内部 r 的大小（r 是 LoRA 内矩阵相乘被消掉的维度）。

#### DoRA

权重分解低阶适应 Weight-Decomposed Low-Rank Adaptation

该技术将权重的更新分解为两个部分：大小和方向。方向由普通 LoRA 处理，而幅度由单独的可学习参数处理。这可以提高 LoRA 的性能，尤其是在 low rank 时。有关 DoRA 的更多信息，请参阅这里的[论文](https://arxiv.org/abs/2402.09353)

代码如下:

```
config = LoraConfig(use_dora=True, ...)
```

注意事项:

- DoRA 目前仅支持线性层。
- DoRA 比纯 LoRA 引入了更大的开销，因此建议合并权重进行推理，请参阅 [LoraModel.merge_and_unload()](https://huggingface.co/docs/peft/v0.10.0/en/package_reference/lora#peft.LoraModel.merge_and_unload)
- DoRA 应使用以位和字节量化的权重（“QDoRA”）。然而，在将 QDoRA 与 DeepSpeed Zero2 结合使用时，已报告出现问题。

#### 复制层

layer_replication 层的副本，也就是复制层。在模型结构中，有时有些层会重复使用，例如:

假设原始模型有 5 层 [0, 1, 2 ,3, 4] ，我们想创建一个具有 7 层排列为 [0, 1, 2, 3, 2, 3, 4] 的模型。

首先原始 5 层的**微调参数**在内存中是独立的，未来每次重复使用这些层，不会新增内存开销，例如上面 7 层模型，重复使用了 2，3 层数据，则 2，3 层数据的**微调参数**是重复的，并不会新增内存开销

然后我们的配置可以这样写 `config = LoraConfig(layer_replication=[[0,4], [2,5]], ...)`

其中 `[0, 4]`，代表取原始模型的 `0 1 2 3` 层

`[2, 5]` 代表取原始模型的 `2 3 4`

所以我们可以通过配置获得一个 7 层的新模型，如: [0, 1, 2, 3, 2, 3, 4]

在这个新模型中，每一次都有一个独立的 LoRA 适配器

### 合并适配器

LoRA 的微调参数，会通过适配器应用到原模型层中。微调后的模型在不合并的情况下使用，则在前向传播过程中，每次都会执行适配器的运算，降低了运算效率。所以我们可以把微调后，满意的适配器结果合并到原始模型中

```py
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
peft_model_id = "alignment-handbook/zephyr-7b-sft-lora"
model = PeftModel.from_pretrained(base_model, peft_model_id)
# 此处融合是不可逆的
model.merge_and_unload()
```

若想要融合后能够撤销，则可使用以下方法

```py
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
peft_model_id = "alignment-handbook/zephyr-7b-sft-lora"
model = PeftModel.from_pretrained(base_model, peft_model_id)
# 融合，但保留原始副本
model.merge_adapter()
# 撤销融合，应用原始副本
model.unmerge_adapter()
```

### 完整配置参数解析

[官方配置文档](https://huggingface.co/docs/peft/v0.10.0/en/package_reference/lora#peft.LoraConfig)

- ==**r (int)**==: 秩，LoRA 的学习维度
- ==**target_modules (Optional[Union[List[str], str]])**== : 代表哪些层需要添加 LoRA 适配器。参数为字符串，则正则匹配；参数是字符串列表，则要么精准匹配，要么匹配结尾。参数为 'all-linear' 则匹配所有线性层和Conv1D层，但不包含输出层。若不指定值，则只有特定架构模型能识别，无法识别模型架构时报错
- **lora_alpha (int)**: 秩稳定的 LoRA 的缩放系数，缩放因子 = lora_alpha / r 得到
- **lora_dropout (float)**: LoRA 层的 dropout 率
- **fan_in_fan_out (bool)**: 权重存储是否已 (fan_in, fan_out) 形式存储，fan_in 输入连接数，fan_out 输出连接数。通常权重参数形式为 (fan_out, fan_in)，若该配置为 True，则权重维度进行转置。（源代码就是这样写的）
- **bias (str)**: 微调哪些偏置，候选项为: 'none', 'all', 'lora_only'。'none' 代表禁用 bias，'all' 和 'lora_only' 会训练指定的偏置
- **use_rslora (bool)**: 是否使用稳定秩lora
- ==**modules_to_save (List[str])**==: 除了适配器外，指定哪些层可训练，并最后保存下来。除此外其他层参数均被冻结，不可训练
- **init_lora_weights (bool | Literal["gaussian", "loftq"])**: 初始化 LoRA 层权重的方法
- **layers_to_transform (Union[List[int], int])**: 指定层的索引，将被变换成 LoRA 适配器，仅当 target_modules 被指定时生效
- **layers_pattern (str)**: 层的模板名，仅在 layers_to_transform 不是 None 时有效
- **rank_pattern (dict)**: 给不同层指定自己的 rank 值，这样的话，不同层将获得不同于 r 的属于自己的 rank 值
- **alpha_pattern (dict)**: 指定不同层自己的 lora_alpha 值
- **megatron_config (Optional[dict])**: Megatron 是 NVIDIA 开发的大规模 Transformer 模型训练框架，此选项用于配置它
- **megatron_core (Optional[str])**: 配置 Megatron 核心
- **loftq_config (Optional[LoftQConfig])**: LoftQ 的配置。如果这不是 None，则 LoftQ 将用于量化主干权重并初始化 Lora 层。不要同时设置 `init_lora_weights='loftq'`
- **use_dora (bool)**: 是否使用 DoRA
- **layer_replication(`List[Tuple[int, int]]`)**: 副本层配置

==**涂黄的部分是重点配置**==

### LoraModel

[官方文档](https://huggingface.co/docs/peft/v0.10.0/en/package_reference/lora#peft.LoraModel)

通过配置我们可以包装原始模型，变成一个 LoraModel，LoraModule 是 torch.nn.Module 的子类

实例化 LoraModel 对象，例如:

```py
from transformers import AutoModelForSeq2SeqLM
from peft import LoraModel, LoraConfig

config = LoraConfig(
    task_type="SEQ_2_SEQ_LM",
    r=8,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.01,
)

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
# 直接调用构造函数即可
lora_model = LoraModel(model, config, "default")
```

### PeftModel

[官方文档](https://huggingface.co/docs/peft/v0.10.0/en/package_reference/peft_model)

这是更为通用的包装类，用于包装原始模型，可以使用 `peft.get_peft_model(module, config)` 获取包装后的 PeftModel，PeftModel 类似 LoraModel

- module: 要包装的模型
- config: peft.LoraConfig 配置对象
- adepter_name: 适配器名称，选填
