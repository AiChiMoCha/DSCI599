import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------
# 1. 加载并预处理 SpamAssassin 数据集
df = pd.read_csv("emails_extracted.csv", encoding="utf-8")
df = df[['Label', 'Body']]
df.columns = ['label', 'text']
df["label"] = df["label"].map({"ham": 0, "spam": 1})
df["text"] = df["text"].fillna("")

print("Dataset sample:")
print(df.head())

# 划分训练集和测试集（例如 80%训练，20%测试）
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
print(f"Train set: {train_df.shape}, Test set: {test_df.shape}")

# ---------------------------
# 2. 加载 fine-tune 的 transformer 模型及 tokenizer
model_dir = "finetuned_spamAssassin_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
transformer_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer_model.to(device)
transformer_model.eval()

def get_transformer_probs(texts, batch_size=32, max_length=128):
    """获取 transformer 模型的预测概率（返回二维数组，每行两个概率）"""
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encodings = tokenizer(batch_texts, return_tensors="pt", 
                                  padding=True, truncation=True, max_length=max_length)
            encodings = {k: v.to(device) for k, v in encodings.items()}
            outputs = transformer_model(**encodings)
            logits = outputs.logits  # shape: (batch_size, num_labels)
            probs = F.softmax(logits, dim=-1)  # 转换为概率
            all_probs.extend(probs.cpu().numpy())
    return all_probs

# ---------------------------
# 3. 训练 Naive Bayes 模型（基于 TfidfVectorizer）
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train = vectorizer.fit_transform(train_df["text"])
X_test = vectorizer.transform(test_df["text"])
nb_clf = MultinomialNB()
nb_clf.fit(X_train, train_df["label"])

# 对测试集，获取 Naive Bayes 的预测概率
nb_probs = nb_clf.predict_proba(X_test)  # shape: (num_samples, num_labels)

# ---------------------------
# 4. 对测试集使用 transformer 模型获得预测概率
test_texts = test_df["text"].tolist()
transformer_probs = get_transformer_probs(test_texts, batch_size=32, max_length=128)

# ---------------------------
# 5. 融合预测结果：加权平均（此处简单取均值，可根据需要调整权重）
import numpy as np
transformer_probs = np.array(transformer_probs)  # shape: (N, 2)
nb_probs = np.array(nb_probs)                    # shape: (N, 2)

# 设置权重，例如各占 0.5
w_transformer = 0.5
w_nb = 0.5
fused_probs = w_transformer * transformer_probs + w_nb * nb_probs
fused_preds = np.argmax(fused_probs, axis=1)

# ---------------------------
# 6. 评估融合后的预测结果
true_labels = test_df["label"].values
accuracy = accuracy_score(true_labels, fused_preds)
print("融合模型 Test Accuracy:", accuracy)
print("\n融合模型 Classification Report:")
print(classification_report(true_labels, fused_preds, zero_division=0))
