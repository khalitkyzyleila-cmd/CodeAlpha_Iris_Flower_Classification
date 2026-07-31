# CodeAlpha — Iris Flower Classification

## Internship Task 1

This project classifies Iris flowers into **Iris-setosa, Iris-versicolor, and Iris-virginica** using four measurements:

- Sepal length
- Sepal width
- Petal length
- Petal width

### Approach

1. Load `data/iris.csv`.
2. Remove the identifier column and missing values.
3. Split the data into 80% training and 20% testing sets using stratification.
4. Standardize the numerical features.
5. Train a Logistic Regression classifier.
6. Evaluate the model with accuracy, precision, recall, F1-score, and a confusion matrix.
7. Create visualizations showing petal measurements and classification performance.

### Result

Using `random_state=42`, the test set contains 30 observations and the model achieved:

**Accuracy: 93.3%**

The model classified all 10 setosa flowers correctly. The main errors were between versicolor and virginica, which is expected because these two species have more similar measurements than setosa.

### Repository structure

```text
CodeAlpha_Iris_Flower_Classification/
├── data/
│   └── iris.csv
├── outputs/
│   ├── classification_report.txt
│   ├── confusion_matrix.csv
│   ├── confusion_matrix.png
│   └── petal_measurements.png
├── src/
│   └── iris_classification.py
├── requirements.txt
└── README.md
```

### Run locally

```bash
pip install -r requirements.txt
python src/iris_classification.py
```

### Dataset

The dataset supplied for the CodeAlpha internship task was used directly. The original filename was `Iris.csv.xls`; it is stored in the project as `iris.csv` because its contents are CSV data.

### Skills demonstrated

Python, pandas, scikit-learn, data preprocessing, train/test splitting, classification, model evaluation, data visualization, and reproducible machine-learning workflows.
