Tumor Cancer Classification
A machine learning project that classifies tumors as malignant or benign using an ensemble voting model combining multiple classifiers.
What it does
Takes tumor feature data as input and predicts whether the tumor is cancerous (malignant) or non-cancerous (benign). Uses a voting ensemble that combines the predictions of multiple models for improved accuracy.
Models Used

Logistic Regression
Decision Tree
Support Vector Machine (SVM)
Voting Ensemble (combines all three models)

How it works

Preprocessing — Prepares and cleans the tumor dataset
Model Training — Trains Logistic Regression, Decision Tree, and SVM classifiers individually
Voting Module — Combines model predictions using a majority voting strategy (VOTINGMODULE)
Evaluation — Compares individual model performance against the ensemble

Tech Stack

Python
Scikit-learn
Pandas
NumPy
