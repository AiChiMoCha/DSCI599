import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Load the zero-shot predictions CSV file.
# For zero-shot, the CSV file has columns: label, text, predicted_label.
df_zeroshot = pd.read_csv("sms_spam_predictions.csv")
true_labels_zeroshot = df_zeroshot["label"]  # ground truth in zero-shot file
predicted_zeroshot = df_zeroshot["predicted_label"]

# Load the fine-tuned predictions CSV file.
# For fine-tuned, the CSV file has columns: true_label, text, predicted_label.
df_finetuned = pd.read_csv("sms_spam_test_predictions.csv")
true_labels_finetuned = df_finetuned["true_label"]
predicted_finetuned = df_finetuned["predicted_label"]

# Compute accuracy for zero-shot results.
accuracy_zeroshot = accuracy_score(true_labels_zeroshot, predicted_zeroshot)
print("Zero-shot Test Accuracy:", accuracy_zeroshot)
print("\nZero-shot Classification Report:")
print(classification_report(true_labels_zeroshot, predicted_zeroshot))

# Compute accuracy for fine-tuned results.
accuracy_finetuned = accuracy_score(true_labels_finetuned, predicted_finetuned)
print("\nFine-tuned Test Accuracy:", accuracy_finetuned)
print("\nFine-tuned Classification Report:")
print(classification_report(true_labels_finetuned, predicted_finetuned))
