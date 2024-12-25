# pytorch环境搭建

## 什么是 pytorch

AI 学习引擎

深度学习框架

## python 安装

### conda 命令

创建环境

```shell
conda create --name <env-name> python=<python-version>
# 例如:
conda create --name pytorch-env python=3.11
```

删除环境

```shell
# --all 删除环境下所有软件包
conda remove --name <环境名称> --all
# 或
conda env remove --name <环境名称>
```

激活环境

```shell
conda activate <env-name>
# 例如
conda activate pytorch-env
```

查看环境列表

```shell
conda env list
```

#### Q&A

##### SafetyError 解决办法

遇到如下问题

```
SafetyError: The package for pytorch located at C:\pythoncharm\anaconda\pkgs\pytorch-2.2.1-py3.11_cpu_0
appears to be corrupted. The path 'Lib/site-packages/torch/lib/dnnl.lib'
has an incorrect size.
```

运行命令，清空包:

```shell
conda clean --packages --tarballs
```

再重新安装就好了

## pytorch 安装

[官网](https://pytorch.org/get-started/locally/)

根据不同情况选择不同平台

![](md-img/pytorch环境搭建_2023-11-09-09-42-27.png)

这里在 `Compute Platform` 选择 `CPU` 然后复制 `Run this Command` 中的代码，并放到 `cmd` 中运行

## pycharm 安装
