# %%
import xlrd
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# %%
# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')


# %%
# Load dataset from Excel using xlrd
workbook = xlrd.open_workbook('edit01.xls')
sheet = workbook.sheet_by_index(0)  # Assuming data is in the first sheet

# Extract data from Excel sheet
data = {
    'TITLE': [],
    'SelfText': [],
    'Regret Type': [],
    'Domain related to object': []
}
for row in range(1, sheet.nrows):
    data['TITLE'].append(sheet.cell_value(row, 0))
    data['SelfText'].append(sheet.cell_value(row, 1))
    data['Regret Type'].append(sheet.cell_value(row, 2))
    data['Domain related to object'].append(sheet.cell_value(row, 3))

# Create DataFrame
df = pd.DataFrame(data)

# Fill missing values
df.fillna('', inplace=True)

# %%
# Preprocessing
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    tokens = word_tokenize(str(text).lower())  # Tokenization and lowercasing
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]  # Remove stopwords and non-alphanumeric tokens
    return " ".join(filtered_tokens)

df['processed_text'] = df['TITLE'].apply(preprocess_text) + ' ' + df['SelfText'].apply(preprocess_text)
# Update target variables
y_regret = df['Regret Type']
y_domain = df['Domain related to object']

# Feature Extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['processed_text'])

# Splitting dataset
X_train, X_test, y_regret_train, y_regret_test, y_domain_train, y_domain_test = train_test_split(X, y_regret, y_domain, test_size=0.2, random_state=42)


# %%
# Model Training for Regret Detection
regret_classifier = MultinomialNB()
regret_classifier.fit(X_train, y_regret_train)

# Model Evaluation for Regret Detection
y_regret_pred = regret_classifier.predict(X_test)
regret_accuracy = accuracy_score(y_regret_test, y_regret_pred)
print("Regret Detection Accuracy:", regret_accuracy)

# Example Prediction for Regret Detection
new_text = "I regret not studying harder for my exam."
new_text_vectorized = vectorizer.transform([preprocess_text(new_text)])
regret_prediction = regret_classifier.predict(new_text_vectorized)
print("Predicted regret type:", regret_prediction[0])


# %%
# Model Training for Domain Identification
domain_classifier = MultinomialNB()
domain_classifier.fit(X_train, y_domain_train)

# Model Evaluation for Domain Identification
y_domain_pred = domain_classifier.predict(X_test)
domain_accuracy = accuracy_score(y_domain_test, y_domain_pred)
print("Domain Identification Accuracy:", domain_accuracy)

# Example Prediction for Domain Identification
new_text = "I need advice on my health."
new_text_vectorized = vectorizer.transform([preprocess_text(new_text)])
domain_prediction = domain_classifier.predict(new_text_vectorized)
print("Predicted domain:", domain_prediction[0])


