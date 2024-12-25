# NLP 评价指标

## 文本分类（Text Classification）

### 任务描述

将文本分配到预定义的类别，例如垃圾邮件检测、情感分析。

### 评估指标

- **准确率（Accuracy）**：正确分类的样本数占总样本数的比例。
- **精确率（Precision）**：正确预测为正类的样本数占所有预测为正类的样本数的比例。
- **召回率（Recall）**：正确预测为正类的样本数占所有实际为正类的样本数的比例。
- **F1得分（F1 Score）**：精确率和召回率的调和平均数。公式 $F1=2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$。F1得分的取值范围是0到1，值越高表示模型的性能越好。它提供了一个**同时考虑精确度和召回率**的平均指标。
- **ROC曲线（ROC Curve）**：真阳率-假阳率曲线，用来衡量二分类问题，模型的判断力。ROC曲线是以假阳性率（FPR）为横轴，真阳性率（TPR）为纵轴绘制的曲线。
  - 真阳率（TPR）又称召回率（Recall）：衡量正例占所有正例的比例
  - 假阳性率（FPR）：衡量误判为正例的样本占所有负例的比例，公式 $\text{FPR} = \frac{FP}{FP + TN}$
- **AUC值（Area Under the Curve）**：是ROC曲线（Receiver Operating Characteristic Curve）下面积的数值表示，是衡量二分类模型性能的重要指标。它反映了模型区分正类和负类样本的能力。AUC值的范围是0到1，值越大表示模型的性能越好。当 AUC 值接近 0.5 时，意味着模型的二元判断几乎是靠猜，没有什么判断力，而越接近 1，判断力越强

## 机器翻译（Machine Translation）

### 任务描述

将一种语言的文本翻译成另一种语言。

### 评估指标

- **BLEU（Bilingual Evaluation Understudy）**：衡量翻译文本和参考文本之间的相似度。
- **METEOR（Metric for Evaluation of Translation with Explicit ORdering）**：考虑了词形变化和同义词的相似度。主要考虑以下几点:
  - 词形变化匹配（Stemming）：允许对词形变化（如动词的不同时态、名词的复数形式等）进行匹配，而不仅仅是直接的词匹配。
  - 同义词匹配（Synonymy）：考虑同义词之间的匹配，这意味着在翻译中使用同义词不会被视为错误。
  - 词序（Word Order）：通过对词序的显式考虑，衡量翻译文本与参考文本在语序上的相似度。
- **TER（Translation Edit Rate）**：衡量翻译需要编辑多少次才能与参考文本一致。编辑方式包括插入、删除、替换、移动。

## 命名实体识别（Named Entity Recognition, NER）

### 任务描述

识别文本中具有特定意义的实体，如人名、地名、组织名等。

### 评估指标

- **准确率（Accuracy）**：预测正确的实体数量占总实体数量的比例。
- **精确率（Precision）**：正确预测的实体数量占所有预测的实体数量的比例。
- **召回率（Recall）**：正确预测的实体数量占所有实际实体数量的比例。
- **F1得分（F1 Score）**：精确率和召回率的调和平均数。

## 情感分析（Sentiment Analysis）

### 任务描述

检测文本中的情感极性，如正面、负面或中性。

### 评估指标

- **准确率（Accuracy）**：正确分类的样本数占总样本数的比例。
- **精确率（Precision）**：正确预测为某情感类别的样本数占所有预测为该情感类别的样本数的比例。
- **召回率（Recall）**：正确预测为某情感类别的样本数占所有实际为该情感类别的样本数的比例。
- **F1得分（F1 Score）**：精确率和召回率的调和平均数。

## 问答系统（Question Answering）

### 任务描述

从文本中自动回答问题。

### 评估指标

- **精确率（Precision）**：预测答案与正确答案之间的匹配程度。
- **召回率（Recall）**：模型能够找到正确答案的比例。
- **F1得分（F1 Score）**：精确率和召回率的调和平均数。
- **EM（Exact Match）**：预测答案与正确答案完全匹配的比例。

## 文本生成（Text Generation）

### 任务描述

自动生成与输入内容相关的文本，如摘要生成、对话生成。

### 评估指标

- **BLEU**：衡量生成文本与参考文本的相似度。
- **ROUGE（Recall-Oriented Understudy for Gisting Evaluation）**：评估生成摘要与参考摘要的重叠程度，常用的子指标有ROUGE-N（N-gram）、ROUGE-L（Longest Common Subsequence）。
- **METEOR**：综合评估生成文本的准确性、流畅性和同义词的使用情况。

## 依存句法分析（Dependency Parsing）

### 任务描述

分析句子中词语之间的依存关系。

依存句法分析（Dependency Parsing）是一种自然语言处理技术，用于分析句子中词语之间的依存关系，即词语如何通过语法结构相互联系。每个词之间的关系通过依存弧（dependency arc）表示，其中包括一个支配词（head）和一个从属词（dependent）。

#### 关键概念

- **支配词（Head）**：在依存关系中起支配作用的词。
- **从属词（Dependent）**：受支配词支配的词。
- **依存弧（Dependency Arc）**：连接支配词和从属词的弧，标注有依存关系的类型（如主谓关系、动宾关系等）。
- **根节点（Root）**：句子中的主要动词或主要成分，通常是句子的核心动词。

#### 依存关系类型

一些常见的依存关系类型包括：

- **nsubj（名词主语）**：主语和动词之间的关系。
- **dobj（直接宾语）**：动词和直接宾语之间的关系。
- **iobj（间接宾语）**：动词和间接宾语之间的关系。
- **amod（形容词修饰语）**：形容词和名词之间的修饰关系。
- **advmod（副词修饰语）**：副词和动词之间的修饰关系。

#### 示例分析

以句子 "The cat sat on the mat." 为例进行依存句法分析：

```
句子: The cat sat on the mat.
```

其依存句法树如下：

```
     sat
    / | \
  /   |   \
The  cat  on
           |
          the
           |
          mat

```

每个词的依存关系可以表示为：

| 从属词 | 依存类型 | 支配词 | 解释
| ------ | -------- | ------ | ---
| The    | det      | cat    | **限定词（Determiner）**：表示“the”是名词“cat”的限定词，限定了名词的范围或数量。
| cat    | nsubj    | sat    | **名词主语（Nominal Subject）**：表示“cat”是动词“sat”的主语，即“cat”是执行“sat”这一动作的主体。
| sat    | root     | root   | **根（Root）**：句子的核心动词，没有依存于其他词，是整个句子的核心。
| on     | prep     | sat    | **介词（Preposition）**：表示“on”是动词“sat”的介词，连接了动词与后续的名词短语。
| the    | det      | mat    | **限定词（Determiner）**：表示“the”是名词“mat”的限定词，限定了名词的范围或数量。
| mat    | pobj     | on     | **介词宾语（Prepositional Object）**：表示“mat”是介词“on”的宾语，说明动作“sat on”作用的对象。

### 评估指标

- **LAS（Labeled Attachment Score）**：正确预测的带标签的依存关系数量占总依存关系数量的比例。
  - 例如:
  - ```
    The → cat (det)
    cat → sat (nsubj)
    sat → on (prep)
    on → mat (pobj)
    the → mat (det)
    ```
  - 后面圆括号中的是标签，前面的是依存关系
- **UAS（Unlabeled Attachment Score）**：正确预测的依存关系数量占总依存关系数量的比例，不考虑标签。