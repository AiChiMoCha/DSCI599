import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, load_metric
from sklearn.model_selection import train_test_split

# 1. Read the SMS Spam dataset.
# Assumes "spam.csv" has columns: v1 (label) and v2 (text)
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'text']
df["label"] = df["label"].map({"ham": 0, "spam": 1})

print("Dataset sample:")
print(df.head())

# 2. Split the dataset into training and testing sets.
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
print(f"Train set: {train_df.shape}, Test set: {test_df.shape}")

# 3. Convert the DataFrames to Hugging Face Datasets.
train_dataset = Dataset.from_pandas(train_df)
test_dataset  = Dataset.from_pandas(test_df)

# 4. Load the tokenizer and model.
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 5. Define a preprocessing function.
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Remove columns not needed for training.
train_dataset = train_dataset.remove_columns(["__index_level_0__"])
test_dataset = test_dataset.remove_columns(["__index_level_0__"])
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# 6. Define training arguments with GPU acceleration.
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

# 7. Define a metric function.
metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=-1)
    acc = metric.compute(predictions=predictions, references=labels)
    return acc

# 8. Initialize the Trainer.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 9. Fine-tune the model.
trainer.train()

# 9.5 Save the fine-tuned model to a local directory.
trainer.save_model("finetuned_model")
tokenizer.save_pretrained("finetuned_model")
print("Fine-tuned model saved to 'finetuned_model'.")

# 10. Evaluate on test set and get predictions.
predictions = trainer.predict(test_dataset)
pred_logits = predictions.predictions
pred_labels = torch.argmax(torch.tensor(pred_logits), axis=-1).numpy()
true_labels = test_dataset["label"]

# 11. Save predictions to a CSV.
test_results = pd.DataFrame({
    "text": test_df["text"].values,
    "true_label": test_df["label"].values,
    "predicted_label": pred_labels
})
test_results.to_csv("sms_spam_test_predictions.csv", index=False)
print("Test predictions saved to sms_spam_test_predictions.csv")
