from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load the tokenizer from the original base model.
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# Load the fine-tuned model from your saved directory.
model = AutoModelForSequenceClassification.from_pretrained("finetuned_model")

# Create a text classification pipeline.
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)

# Test with fabricated examples:
spam_sample = """
I am Ms. Joanna Liu a staff of CITIBANK HONG KONG; I am contacting you concerning a customer and an investment placed under our banks management, I have some business opportunities that I believe will be of great interest to you. You have the chance to be beneficiary, contact for more information.


Warm regards,
Ms. Joanna Liu
"""
ham_sample = "Hey, I'll be a few minutes late for our meeting. Please start without me."

spam_prediction = classifier(spam_sample)
ham_prediction = classifier(ham_sample)

print("Spam Sample:")
print("Text:", spam_sample)
print("Prediction:", spam_prediction)

print("\nHam Sample:")
print("Text:", ham_sample)
print("Prediction:", ham_prediction)
