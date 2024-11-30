# 基于LSTM的诗词生成模型

## 环境配置

```bash
pip install -r requirements.txt
```

## 项目结构
```bash
.
├── data
│   ├── 全唐诗 # 原始数据集
│   └── poetry_dataset # 清洗后的数据集
│       ├── data-00000-of-00001.arrow
│       ├── dataset_info.json
│       └── state.json
├── model
│   └── best_poetry_lstm.bin # 模型文件
├── dataset_processing.py # 数据集构建
├── model.py # 模型建模
├── tokenizer.py # 分词器
├── vocab.json # 词表
├── train.py # 训练
├── requirements.txt # 环境依赖
└── README.md 

```

## 数据集构建

```bash
python dataset_processing.py
```

## 模型训练

```bash
python train.py
```

## 推理测试

```bash
python model.py
```

## 致谢
本项目参考了以下项目：
- [chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)：原始数据集
- [tom-ml/LSTM_POET](https://github.com/tom-ml/LSTM_POET)：参考了数据集清洗及构建过程、任务建模过程
- [0809zheng/automatic-poetry-pytorch](https://github.com/0809zheng/automatic-poetry-pytorch)：参考了模型训练过程