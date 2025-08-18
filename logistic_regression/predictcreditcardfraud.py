import seaborn
import pandas as pd
import numpy as np
import codecademylib3
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import codecademylib3

# Load the data
transactions = pd.read_csv('transactions_modified.csv')
print(transactions.head())
print(transactions.info())


# How many fraudulent transactions?
print(transactions['isFraud'].sum())


# Summary statistics on amount column
print(transactions['amount'].describe())


# Create isPayment field
transactions['isPayment'] = transactions['type'].apply(
  lambda x: 1 if (x == 'PAYMENT' or x == 'DEBIT') else 0
)


# Create isMovement field
transactions['isMovement'] = transactions['type'].apply(
  lambda x: 1 if (x == 'CASH_OUT' or x == 'TRANSFER') else 0
)


# Create accountDiff field
transactions['accountDiff'] = (transactions['oldbalanceOrg'] - transactions['oldbalanceDest']).abs()


# Create features and label variables
features = transactions[['amount', 'isPayment', 'isMovement', 'accountDiff']]
label = transactions['isFraud']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3, stratify=label)


# Normalize the features variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Fit the model to the training data
model = LogisticRegression()
fitted = model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Score the model on the training data
training_score = model.score(X_train_scaled, y_train)
print(training_score)


# Score the model on the test data
test_score = model.score(X_test_scaled, y_test) 
print(test_score)


# Print the model coefficients
print(model.coef_, model.intercept_)


# New transaction data
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])

# Create a new transaction
your_transaction = np.array([6472.54, 1.0, 0.0, 55901.23])


# Combine new transactions into a single array
sample_transactions = np.stack([transaction1, transaction2, transaction3, your_transaction])

# Normalize the new transactions
sample_transactions = scaler.transform(sample_transactions)


# Predict fraud on the new transactions
print(model.predict(sample_transactions))

# Show probabilities on the new transactions
print(model.predict_proba(sample_transactions))

