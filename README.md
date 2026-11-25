Customer Churn Classification

Binary customer churn classification pipeline in Python using pandas and scikit-learn (LogisticRegression), trained and evaluated on customer_churn_basic_dataset.csv (a synthetic practice dataset of 150 rows and 9 columns).
Dataset & Preprocessing

    Target: churn (97 non-churn / 53 churn, 35.33% churn rate), kept untouched.

    Feature exclusion: customer_id is excluded from model predictors.

    Missing values: age (3 missing), monthly_charges (2 missing), and payment_method (3 missing) are handled inside the pipeline after splitting.

    Train / Test split: 80/20 stratified split (random_state=42), producing 120 training rows and 30 test rows (19 non-churn, 11 churn).

    Pipeline: ColumnTransformer applying median imputation and StandardScaler to numeric features, and most-frequent imputation plus OneHotEncoder(handle_unknown="ignore") to categorical features, followed by LogisticRegression(max_iter=1000, random_state=42).

How to Run

    Place churn_classification.py and customer_churn_basic_dataset.csv in the same folder.

    Run:


python churn_classification.py

The script prints the exploratory summary, evaluation metrics, and confusion matrix, and saves confusion_matrix.png.
Test Set Results (30 Samples)

    Accuracy: 0.7000

    Precision: 0.6000

    Recall: 0.5455

    F1-score: 0.5714

    Confusion Matrix ([rows = actual, cols = predicted]): [[15, 4], [5, 6]]

Limitations

    The dataset is a small synthetic practice dataset (150 total rows; 30 test rows with 11 actual churn cases), so metrics on the holdout split are sensitive to sample size and should not be treated as a guarantee on real-world production data.

    Default classification uses a 0.5 probability threshold rather than custom business cost weights.
