import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# ---------------------
# 加载 SMS Spam 数据集 (数据集1)
# 文件：spam.csv，假定包含 v1（标签）和 v2（文本）
sms_df = pd.read_csv("spam.csv", encoding="latin-1")
sms_df = sms_df[['v1', 'v2']]
sms_df.columns = ['label', 'text']
sms_df["label"] = sms_df["label"].map({"ham": 0, "spam": 1})
sms_df["text"] = sms_df["text"].fillna("")

print("SMS Spam 数据集样本:")
print(sms_df.head())

# ---------------------
# 加载 SpamAssassin 数据集 (数据集2)
# 文件：emails_extracted.csv，假定包含 Label 和 Body 两列
emails_df = pd.read_csv("emails_extracted.csv", encoding="utf-8")
emails_df = emails_df[['Label', 'Body']]
emails_df.columns = ['label', 'text']
emails_df["label"] = emails_df["label"].map({"ham": 0, "spam": 1})
emails_df["text"] = emails_df["text"].fillna("")

print("\nSpamAssassin 数据集样本:")
print(emails_df.head())

# ---------------------
# 交叉测试 1：用 SMS Spam 训练，测试 SpamAssassin
print("\n----- 交叉测试 1：SMS Spam 训练，SpamAssassin 测试 -----")

# 训练阶段：对 SMS Spam 数据集全量训练
vectorizer_sms = TfidfVectorizer(stop_words='english', max_features=10000)
X_sms_train = vectorizer_sms.fit_transform(sms_df["text"])
y_sms_train = sms_df["label"]

# 使用 SMS 数据集训练模型
clf_sms = MultinomialNB()
clf_sms.fit(X_sms_train, y_sms_train)

# 测试阶段：将 SpamAssassin 数据集文本用 SMS 训练时的向量器转换
X_emails_test = vectorizer_sms.transform(emails_df["text"])
y_emails_test = emails_df["label"]

y_pred_sms2emails = clf_sms.predict(X_emails_test)

accuracy_sms2emails = accuracy_score(y_emails_test, y_pred_sms2emails)
print("SMS->SpamAssassin Test Accuracy:", accuracy_sms2emails)
print("Classification Report:")
print(classification_report(y_emails_test, y_pred_sms2emails, zero_division=0))

# ---------------------
# 交叉测试 2：用 SpamAssassin 训练，测试 SMS Spam
print("\n----- 交叉测试 2：SpamAssassin 训练，SMS Spam 测试 -----")

# 训练阶段：对 SpamAssassin 数据集全量训练
vectorizer_emails = TfidfVectorizer(stop_words='english', max_features=10000)
X_emails_train = vectorizer_emails.fit_transform(emails_df["text"])
y_emails_train = emails_df["label"]

clf_emails = MultinomialNB()
clf_emails.fit(X_emails_train, y_emails_train)

# 测试阶段：用 SpamAssassin 训练时的向量器转换 SMS Spam 数据集
X_sms_test = vectorizer_emails.transform(sms_df["text"])
y_sms_test = sms_df["label"]

y_pred_emails2sms = clf_emails.predict(X_sms_test)

accuracy_emails2sms = accuracy_score(y_sms_test, y_pred_emails2sms)
print("SpamAssassin->SMS Test Accuracy:", accuracy_emails2sms)
print("Classification Report:")
print(classification_report(y_sms_test, y_pred_emails2sms, zero_division=0))
