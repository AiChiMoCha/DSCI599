import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# 加载零样本预测结果（假定文件包含列：label, text, predicted_label）
df_zeroshot = pd.read_csv("spamAssassin_predictions.csv")
true_labels_zeroshot = df_zeroshot["label"]
predicted_zeroshot = df_zeroshot["predicted_label"]

# 加载 fine-tuned 预测结果（假定文件包含列：true_label, text, predicted_label）
df_finetuned = pd.read_csv("spamAssassin_test_predictions.csv")
true_labels_finetuned = df_finetuned["true_label"]
predicted_finetuned = df_finetuned["predicted_label"]

# 计算零样本预测准确率
accuracy_zeroshot = accuracy_score(true_labels_zeroshot, predicted_zeroshot)
print("Zero-shot Test Accuracy:", accuracy_zeroshot)
print("\nZero-shot Classification Report:")
print(classification_report(true_labels_zeroshot, predicted_zeroshot, zero_division=0))

# 计算 fine-tuned 模型预测准确率
accuracy_finetuned = accuracy_score(true_labels_finetuned, predicted_finetuned)
print("\nFine-tuned Test Accuracy:", accuracy_finetuned)
print("\nFine-tuned Classification Report:")
print(classification_report(true_labels_finetuned, predicted_finetuned, zero_division=0))
