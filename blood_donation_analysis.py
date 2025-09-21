# blood_donation_analysis.py
# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Step 2: Load the blood transfusion dataset
transfusion = pd.read_csv('datasets/transfusion.data')
print("First 5 rows of dataset:")
print(transfusion.head())

# Step 3: Inspect dataset structure
print("\nDataset info:")
transfusion.info()

# Step 4: Rename target column for simplicity
transfusion.rename(
    columns={'whether he/she donated blood in March 2007': 'target'},
    inplace=True
)
print("\nFirst 2 rows after renaming target column:")
print(transfusion.head(2))

# Step 5: Check target incidence proportions
print("\nTarget value proportions (normalized):")
print(transfusion.target.value_counts(normalize=True).round(3))

# Step 6: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    transfusion.drop(columns='target'),
    transfusion['target'],
    test_size=0.25,
    random_state=42,
    stratify=transfusion['target']
)
print("\nFirst 2 rows of X_train:")
print(X_train.head(2))

# Step 7: Check variance of features
print("\nVariance of X_train features:")
print(X_train.var().round(3))

# Step 8: Log normalization of 'Monetary (c.c. blood)' column
X_train_normed, X_test_normed = X_train.copy(), X_test.copy()
col_to_normalize = 'Monetary (c.c. blood)'

for df_ in [X_train_normed, X_test_normed]:
    df_['monetary_log'] = np.log(df_[col_to_normalize])
    df_.drop(columns=col_to_normalize, inplace=True)

print("\nVariance of normalized features:")
print(X_train_normed.var().round(3))

# Step 9: Train a logistic regression model
logreg = linear_model.LogisticRegression(solver='liblinear', random_state=42)
logreg.fit(X_train_normed, y_train)

# Step 10: Evaluate model using AUC
logreg_auc_score = roc_auc_score(y_test, logreg.predict_proba(X_test_normed)[:, 1])
print(f"\nAUC score of logistic regression: {logreg_auc_score:.4f}")

# Step 11: Model ranking (only logistic regression here)
print("\nModel ranking based on AUC:")
model_ranking = [('logreg', logreg_auc_score)]
sorted_models = sorted(model_ranking, key=lambda x: x[1], reverse=True)
print(sorted_models)

# Step 12: Plot ROC curve and save as image
y_prob = logreg.predict_proba(X_test_normed)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'LogReg (AUC = {logreg_auc_score:.2f})', color='blue', linewidth=2)
plt.plot([0,1], [0,1], 'k--', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.savefig('roc_curve.png')  # Saves the ROC curve image
plt.show()
print("\nROC curve saved as 'roc_curve.png'")

# Step 13: Show feature importance
feature_importance = pd.DataFrame({
    'Feature': X_train_normed.columns,
    'Coefficient': logreg.coef_[0]
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("\nFeature importance (sorted by absolute value):")
print(feature_importance)
