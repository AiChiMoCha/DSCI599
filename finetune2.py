import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, load_metric
from sklearn.model_selection import train_test_split

# 1. 读取 emails_extracted.csv 数据集，并保留 Label 和 Body 两列
df = pd.read_csv("emails_extracted.csv", encoding="utf-8")
df = df[['Label', 'Body']]
df.columns = ['label', 'text']
df["label"] = df["label"].map({"ham": 0, "spam": 1})

print("Dataset sample:")
print(df.head())

# 2. 划分训练集和测试集（20% 用于测试），保持标签分布
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
print(f"Train set: {train_df.shape}, Test set: {test_df.shape}")

# 3. 将 DataFrame 转换为 Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset  = Dataset.from_pandas(test_df)

# 4. 加载 tokenizer 和模型
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 5. 定义预处理函数，对文本进行 tokenization，确保所有文本均为字符串
def preprocess_function(examples):
    texts = [str(t) if t is not None else "" for t in examples["text"]]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# 移除训练中不需要的列（例如原始文本列）
columns_to_remove = set(train_dataset.column_names) - {"input_ids", "attention_mask", "label"}
train_dataset = train_dataset.remove_columns(list(columns_to_remove))
columns_to_remove = set(test_dataset.column_names) - {"input_ids", "attention_mask", "label"}
test_dataset = test_dataset.remove_columns(list(columns_to_remove))

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# 6. 定义训练参数（使用GPU加速）
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
)

# 7. 定义评价指标函数（使用准确率）
metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=-1)
    acc = metric.compute(predictions=predictions, references=labels)
    return acc

# 8. 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 9. Fine-tune 模型
trainer.train()

# 9.5 保存 fine-tuned 模型到本地目录（使用新的保存目录名称）
new_model_dir = "finetuned_spamAssassin_model"
trainer.save_model(new_model_dir)
tokenizer.save_pretrained(new_model_dir)
print(f"Fine-tuned model saved to '{new_model_dir}'.")

# 10. 在测试集上评估并获取预测结果
predictions = trainer.predict(test_dataset)
pred_logits = predictions.predictions
pred_labels = torch.argmax(torch.tensor(pred_logits), axis=-1).numpy()

# 11. 将预测结果保存到 CSV 文件
test_results = pd.DataFrame({
    "text": test_df["text"].values,
    "true_label": test_df["label"].values,
    "predicted_label": pred_labels
})
test_results.to_csv("spamAssassin_test_predictions.csv", index=False)
print("Test predictions saved to spamAssassin_test_predictions.csv")
