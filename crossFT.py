import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batch_predict(model, tokenizer, texts, batch_size=32, max_length=128):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encodings = tokenizer(batch_texts,
                                  return_tensors="pt",
                                  padding=True,
                                  truncation=True,
                                  max_length=max_length)
            encodings = {k: v.to(device) for k, v in encodings.items()}
            outputs = model(**encodings)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
    return predictions

# -------------------------
# 加载 SMS Spam 数据集 (数据集1)
sms_df = pd.read_csv("spam.csv", encoding="latin-1")
sms_df = sms_df[['v1', 'v2']]
sms_df.columns = ['label', 'text']
sms_df["label"] = sms_df["label"].map({"ham": 0, "spam": 1})
sms_df["text"] = sms_df["text"].fillna("")

print("SMS Spam 数据集样本:")
print(sms_df.head())

# -------------------------
# 加载 SpamAssassin 数据集 (数据集2)
emails_df = pd.read_csv("emails_extracted.csv", encoding="utf-8")
emails_df = emails_df[['Label', 'Body']]
emails_df.columns = ['label', 'text']
emails_df["label"] = emails_df["label"].map({"ham": 0, "spam": 1})
emails_df["text"] = emails_df["text"].fillna("")

print("\nSpamAssassin 数据集样本:")
print(emails_df.head())

# -------------------------
# 加载两个 fine-tuned 模型及其 tokenizer
# 第一个模型（SMS 模型）保存在 "finetuned_model"
# 第二个模型（SpamAssassin 模型）保存在 "finetuned_spamAssassin_model"
sms_model_dir = "finetuned_model"
emails_model_dir = "finetuned_spamAssassin_model"

sms_tokenizer = AutoTokenizer.from_pretrained(sms_model_dir)
sms_model = AutoModelForSequenceClassification.from_pretrained(sms_model_dir).to(device)

emails_tokenizer = AutoTokenizer.from_pretrained(emails_model_dir)
emails_model = AutoModelForSequenceClassification.from_pretrained(emails_model_dir).to(device)

# -------------------------
# 交叉测试 1: 用 SMS 模型测试 SpamAssassin 数据集
print("\n----- 交叉测试 1: SMS 模型 -> SpamAssassin 数据集 -----")
texts_emails = emails_df["text"].tolist()
true_labels_emails = emails_df["label"].tolist()

preds_sms_on_emails = batch_predict(sms_model, sms_tokenizer, texts_emails, batch_size=32, max_length=128)
accuracy_sms_emails = accuracy_score(true_labels_emails, preds_sms_on_emails)
print("SMS 模型在 SpamAssassin 数据集上的准确率:", accuracy_sms_emails)
print("分类报告:")
print(classification_report(true_labels_emails, preds_sms_on_emails, zero_division=0))

# -------------------------
# 交叉测试 2: 用 SpamAssassin 模型测试 SMS 数据集
print("\n----- 交叉测试 2: SpamAssassin 模型 -> SMS 数据集 -----")
texts_sms = sms_df["text"].tolist()
true_labels_sms = sms_df["label"].tolist()

preds_emails_on_sms = batch_predict(emails_model, emails_tokenizer, texts_sms, batch_size=32, max_length=128)
accuracy_emails_sms = accuracy_score(true_labels_sms, preds_emails_on_sms)
print("SpamAssassin 模型在 SMS 数据集上的准确率:", accuracy_emails_sms)
print("分类报告:")
print(classification_report(true_labels_sms, preds_emails_on_sms, zero_division=0))
