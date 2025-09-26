# AIML-task4-Logistic_Regression
Binary classification using Logistic Regression on the Breast Cancer Wisconsin dataset. Includes preprocessing, model training, evaluation (confusion matrix, precision, recall, ROC-AUC), threshold tuning, and sigmoid function explanation.

## Requirements
- Python 3.x must be installed on your system.
- Install dependencies before running:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## What I Did
- Chose the **Breast Cancer Wisconsin dataset** as a binary classification problem.
- Dropped irrelevant columns (`id, Unnamed: 32`) and preprocessed the dataset.
- Converted `diagnosis` column into numeric labels (M = 1, B = 0).
- Split the dataset into training and testing sets.
- Standardized numerical features using `StandardScaler`.
- Trained a **Logistic Regression model** using scikit-learn.
- Evaluated performance using confusion matrix, precision, recall, F1-score, and ROC-AUC.
- Tuned the classification threshold to find the optimal decision boundary.
- Explained the **sigmoid function** used in Logistic Regression.

## Key Insights
- Logistic Regression is effective for binary classification tasks like cancer detection.
- **Confusion Matrix** shows how well the model distinguishes malignant vs benign cases.
- **Precision** measures how many predicted malignant cases were correct.
- **Recall** measures how many actual malignant cases were detected.
- **ROC-AUC** score summarizes model performance across thresholds (closer to 1 is better).
- **Sigmoid Function**: Logistic Regression uses the sigmoid function to map any real number to a probability between 0 and 1.
    - Values close to 1 - higher chance of being malignant.
    - Values close to 0 - higher chance of being benign.

## Tools & Libraries
- **Python**
- **Pandas, NumPy** - Data handling
- **Matplotlib, Seaborn** - Visualization
- **Scikit-learn** - Logistic Regression, metrics, preprocessing

## How to Run
1. Clone this repository or download the files.
2. Make sure **Python 3.x** is installed.
3. Install dependencies (see Requirements).
4. Download the dataset breast_cancer.csv.
5. Place ```breast_cancer.csv``` in the same folder as task4code.py.
6. Run the script:

```bash
python task4code.py
```

7. The script will display:
- Confusion Matrix heatmap
- Classification report (Precision, Recall, F1-score)
- ROC curve with AUC value
- Optimal classification threshold
