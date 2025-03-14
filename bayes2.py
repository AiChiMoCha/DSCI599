import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. 读取 SpamAssassin 数据集，保留 Label 和 Body 两列
df = pd.read_csv("emails_extracted.csv", encoding="utf-8")
df = df[['Label', 'Body']]
df.columns = ['label', 'text']
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# 填充缺失文本，防止后续处理报错
df["text"] = df["text"].fillna("")

print("Dataset sample:")
print(df.head())

# 2. 划分训练集和测试集（20%作为测试集），保持标签分布
train_df, test_df = train_test_split(df, test_size=0.4, stratify=df["label"], random_state=42)
print(f"Train set: {train_df.shape}, Test set: {test_df.shape}")

# 3. 使用 TfidfVectorizer 进行特征提取（移除英文停用词，最多保留 10000 个特征）
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train = vectorizer.fit_transform(train_df["text"])
X_test = vectorizer.transform(test_df["text"])

y_train = train_df["label"]
y_test = test_df["label"]

# 4. 初始化并训练 Multinomial Naive Bayes 分类器
clf = MultinomialNB()
clf.fit(X_train, y_train)

# 5. 对测试集进行预测
y_pred = clf.predict(X_test)

# 6. 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
print("Naive Bayes Test Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
