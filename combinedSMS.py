import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------
# 1. 加载 SMS Spam 数据集
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'text']
df["label"] = df["label"].map({"ham": 0, "spam": 1})
df["text"] = df["text"].fillna("")

print("Dataset sample:")
print(df.head())

# ---------------------------
# 2. 划分训练集和测试集（80%训练，20%测试）
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
print(f"Train set: {train_df.shape}, Test set: {test_df.shape}")

# ---------------------------
# 3. 加载 fine-tune 的 Transformer 模型及 tokenizer
sms_model_dir = "finetuned_model"
tokenizer = AutoTokenizer.from_pretrained(sms_model_dir)
transformer_model = AutoModelForSequenceClassification.from_pretrained(sms_model_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer_model.to(device)
transformer_model.eval()

def get_transformer_probs(texts, batch_size=32, max_length=128):
    """获取 Transformer 模型在文本上的预测概率（每个样本对应两个类别的概率）"""
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encodings = tokenizer(batch_texts,
                                  return_tensors="pt",
                                  padding=True,
                                  truncation=True,
                                  max_length=max_length)
            encodings = {k: v.to(device) for k, v in encodings.items()}
            outputs = transformer_model(**encodings)
            logits = outputs.logits  # shape: (batch_size, 2)
            probs = F.softmax(logits, dim=-1)  # 转换为概率
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_probs)

# ---------------------------
# 4. 训练 Naive Bayes 模型（基于 TfidfVectorizer）
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train = vectorizer.fit_transform(train_df["text"])
X_test = vectorizer.transform(test_df["text"])
nb_model = MultinomialNB()
nb_model.fit(X_train, train_df["label"])
# 获取 Naive Bayes 在测试集上的预测概率
nb_probs = nb_model.predict_proba(X_test)  # shape: (num_samples, 2)

# ---------------------------
# 5. 获取 Transformer 模型在测试集上的预测概率
test_texts = test_df["text"].tolist()
transformer_probs = get_transformer_probs(test_texts, batch_size=32, max_length=128)

# ---------------------------
# 6. 融合预测：加权平均（权重各占 0.5）
w_transformer = 0.5
w_nb = 0.5
fused_probs = w_transformer * transformer_probs + w_nb * nb_probs
fused_preds = np.argmax(fused_probs, axis=1)

# ---------------------------
# 7. 评估融合后的预测结果
true_labels = test_df["label"].values
accuracy = accuracy_score(true_labels, fused_preds)
print("融合模型 Test Accuracy:", accuracy)
print("\n融合模型 Classification Report:")
print(classification_report(true_labels, fused_preds, zero_division=0))
