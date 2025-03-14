import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import pipeline
from datasets import Dataset

# 1. 读取数据集 "spam.csv"（假定包含 v1 和 v2 列）
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

print("Dataset sample:")
print(df.head())

# 2. 使用 train_test_split 抽取 20% 作为测试集（保持标签分布）
_, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
print(f"\n测试集样本数量: {len(test_df)}")

# 3. 将测试集转换为 Hugging Face Dataset
test_dataset = Dataset.from_pandas(test_df)

# 4. 定义候选标签
candidate_labels = ["spam", "ham"]

# 5. 加载 zero-shot 分类 pipeline（使用 GPU device=0）
classifier = pipeline("zero-shot-classification",
                        model="typeform/distilbert-base-uncased-mnli",
                        device=0)

# 6. 定义批量分类函数，对于空文本直接跳过
def classify_batch(batch):
    texts = batch["text"]
    predicted_labels = [None] * len(texts)
    valid_indices = []
    valid_texts = []
    for i, text in enumerate(texts):
        if text is None or not text.strip():
            predicted_labels[i] = "skipped"
        else:
            valid_indices.append(i)
            valid_texts.append(text)
    if valid_texts:
        outputs = classifier(valid_texts, candidate_labels, batch_size=32)
        for idx, output in zip(valid_indices, outputs):
            predicted_labels[idx] = output["labels"][0]
    return {"predicted_label": predicted_labels}

# 7. 使用 dataset.map 对测试集进行批量处理
test_dataset = test_dataset.map(classify_batch, batched=True, batch_size=32)

# 8. 计算整体 zero-shot 准确率（仅统计非 "skipped" 的样本）
correct = 0
total = 0
for example in test_dataset:
    pred = example["predicted_label"].lower()
    true = example["label"].lower()
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
result_df.to_csv("sms_spam_predictions.csv", index=False)
print("Results saved to sms_spam_predictions.csv")
