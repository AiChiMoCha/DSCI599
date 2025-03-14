import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import pipeline
from datasets import Dataset

# 1. 读取 emails_extracted.csv 数据集，并保留 Label 和 Body 两列
df = pd.read_csv("emails_extracted.csv", encoding="utf-8")
df = df[['Label', 'Body']]
df.columns = ['label', 'text']

print("数据集样本:")
print(df.head())

# 2. 随机抽取 20% 作为测试集
_, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"\n测试集样本数量: {len(test_df)}")

# 3. 将测试集转换为 Hugging Face Dataset
test_dataset = Dataset.from_pandas(test_df)

# 4. 定义候选标签
candidate_labels = ["spam", "ham"]

# 5. 加载 zero-shot 分类 pipeline，使用 DistilBERT 模型（fine-tuned on MNLI），device=0 表示使用 GPU
classifier = pipeline("zero-shot-classification",
                        model="typeform/distilbert-base-uncased-mnli",
                        device=0)

# 6. 定义批量分类函数，对于空文本直接跳过
def classify_batch(batch):
    texts = batch["text"]
    # 初始化结果列表
    predicted_labels = [None] * len(texts)
    
    # 收集非空文本的索引及文本内容
    valid_indices = []
    valid_texts = []
    for i, text in enumerate(texts):
        if text is None or not text.strip():
            predicted_labels[i] = "skipped"
        else:
            valid_indices.append(i)
            valid_texts.append(text)
    
    # 如果存在非空文本，则调用 classifier
    if valid_texts:
        outputs = classifier(valid_texts, candidate_labels, batch_size=32)
        for idx, output in zip(valid_indices, outputs):
            predicted_labels[idx] = output["labels"][0]
    
    return {"predicted_label": predicted_labels}

# 7. 使用 dataset.map 批量处理测试集
test_dataset = test_dataset.map(classify_batch, batched=True, batch_size=32)

# 8. 计算整体的 zero-shot 准确率（跳过空文本）
correct = 0
total = 0
for example in test_dataset:
    pred = example["predicted_label"].lower()
    true = example["label"].lower()
    # 只有当文本不为空时才计入评估
    if pred != "skipped":
        total += 1
        if pred == true:
            correct += 1

if total > 0:
    accuracy = correct / total
    print(f"\n测试集上的 zero-shot 准确率: {accuracy:.2f}")
else:
    print("\n测试集中没有有效的文本数据用于分类。")

# 9. 保存结果到 CSV 文件
result_df = pd.DataFrame(test_dataset)
result_df.to_csv("spamAssassin_predictions.csv", index=False)
print("结果已保存到 spamAssassin_predictions.csv")
