import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset (Using a placeholder dataset for spam detection)
data = {
    "text": [
        "Win a $1000 cash prize now!",
        "Meeting scheduled at 10 AM tomorrow",
        "Congratulations, you have won a lottery!",
        "Please send the documents by EOD",
        "Earn money quickly by signing up here",
        "Team lunch is planned for Friday",
        "Claim your free gift card now!",
        "Are you available for the project discussion?",
        "Exclusive offer just for you!",
        "Can we reschedule the meeting to next week?"
    ],
    "label": [
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0  # 1 for spam, 0 for not spam
    ]
}

# Convert the dataset to a DataFrame
df = pd.DataFrame(data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.3, random_state=42
)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_test_vec)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the results
print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", conf_matrix)

# Example predictions
example_texts = [
    "Congratulations, you've been selected to win a free iPhone!",
    "Can we finalize the project details by this week?"
]
example_vec = vectorizer.transform(example_texts)
example_predictions = model.predict(example_vec)
for text, pred in zip(example_texts, example_predictions):
    print(f"Text: '{text}' -> Prediction: {'Spam' if pred == 1 else 'Not Spam'}")
