import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Read the SMS Spam dataset.
# Assumes "spam.csv" has columns: v1 (label) and v2 (text)
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'text']
df["label"] = df["label"].map({"ham": 0, "spam": 1})

print("Dataset sample:")
print(df.head())

# 2. Split the dataset into training and testing sets.
train_df, test_df = train_test_split(df, test_size=0.4, stratify=df["label"], random_state=42)
print(f"Train set: {train_df.shape}, Test set: {test_df.shape}")

# 3. Feature extraction using TfidfVectorizer.
# Here we remove common English stop words and limit to top 10,000 features.
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train = vectorizer.fit_transform(train_df["text"])
X_test = vectorizer.transform(test_df["text"])

y_train = train_df["label"]
y_test = test_df["label"]

# 4. Initialize and train the Multinomial Naive Bayes classifier.
clf = MultinomialNB()
clf.fit(X_train, y_train)

# 5. Predict on the test set.
y_pred = clf.predict(X_test)

# 6. Evaluate the performance.
accuracy = accuracy_score(y_test, y_pred)
print("Naive Bayes Test Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
