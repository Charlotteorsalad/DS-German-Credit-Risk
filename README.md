# DS-German-Credit-Risk
Data Science Assignment

<h2>Project Overview</h2>
<p>This project uses the <strong>XGBoost</strong> algorithm to perform classification on a dataset. The objective is to demonstrate using XGBoost for classification tasks and evaluate the model’s performance through accuracy and other evaluation metrics. This assignment is performing data science techniques with different algorithms handled by each of the teammates, whereas the preprocessing steps are group work.</p>

<h4>Dataset: [https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data]</h4>

<h4>Me: Kam Bee Foong (XGBoost Classification)</h4>
<h4>Team member 1: Ooi Yi Xuen</h4>
<h4>Team member 2: Ho Jun Min</h4>
<h4>Team member 3: Hee JingXi</h4>

<h2>Table of Contents</h2>
<ol>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#dataset-overview">Dataset Overview</a></li>
    <li><a href="#steps-in-the-notebook">Steps in the Notebook</a>
        <ul>
            <li><a href="#data-loading">1. Data Loading</a></li>
            <li><a href="#data-preprocessing">2. Data Preprocessing</a></li>
            <li><a href="#model-training">3. Model Training</a></li>
            <li><a href="#model-evaluation">4. Model Evaluation</a></li>
        </ul>
    </li>
    <li><a href="#results-and-plots">Results and Plots</a></li>
    <li><a href="#conclusion">Conclusion</a></li>
    <li><a href="#how-to-run">How to Run</a></li>
</ol>

<h2 id="installation">Installation</h2>
<p>To run this notebook, you will need to install the following libraries:</p>
<pre><code>pip install xgboost pandas numpy matplotlib scikit-learn</code></pre>
<p>If you are using Jupyter notebooks, make sure to install <code>notebook</code> as well:</p>
<pre><code>pip install notebook</code></pre>

<h2 id="project-structure">Project Structure</h2>
<pre><code>
.
├── XGBoost_Classification.ipynb   # Main notebook file
└── README.md                      # This file
</code></pre>

<h2 id="dataset-overview">Dataset Overview</h2>
<p>The dataset used for this project consists of several features that contribute to the classification task. Here’s a brief overview of the dataset:</p>
<ul>
**Purpose**: Classify individuals as good or bad credit risks.

**Instances**: 1000

**Attributes**: 20 (7 numerical, 13 categorical)

---

**Attribute Information**:

- **Numerical Attributes**: Age, credit amount, duration, etc.
- **Categorical Attributes**: Checking account status, credit history, purpose, savings, employment status, etc.

---

**Target Variable**:

- **Class**: Good (700 instances) or Bad (300 instances)

---

**Data Characteristics**:

- **Balance**: The dataset is somewhat imbalanced with more good credit cases.
- **Missing Values**: None specified.
- **Cost Matrix**: Provided for misclassification costs.

</ul>

<h2 id="steps-in-the-notebook">Steps in the Notebook</h2>

<h3 id="data-loading">1. Data Loading</h3>
<p>In this section, the dataset is loaded using <code>pandas</code> and basic exploration is performed to understand the structure and summary statistics of the dataset.</p>
<pre><code>import pandas as pd

# Load the dataset
df = pd.read_csv('path_to_dataset.csv')

# Display the first few rows
df.head()
</code></pre>

<h3 id="data-preprocessing">2. Data Preprocessing</h3>
<p>Data preprocessing includes handling missing values, encoding categorical variables, and splitting the dataset into training and testing sets.</p>
<pre><code>from sklearn.model_selection import train_test_split

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
</code></pre>

<h3 id="model-training">3. Model Training</h3>
<p>The <strong>XGBoost</strong> classifier is trained using the training data. We tune hyperparameters and evaluate the model's performance.</p>
<pre><code>from xgboost import XGBClassifier

# Initialize the XGBoost model
model = XGBClassifier()

# Train the model
model.fit(X_train, y_train)
</code></pre>

<h3 id="model-evaluation">4. Model Evaluation</h3>
<p>The model is evaluated using metrics such as accuracy, precision, recall, and F1-score. Additionally, confusion matrices and ROC curves are plotted to visualize the model's performance.</p>
<pre><code>from sklearn.metrics import accuracy_score, classification_report

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
</code></pre>

<h2 id="results-and-plots">Results and Plots</h2>

<h3>1. Confusion Matrix</h3>
<p>A confusion matrix is plotted to evaluate the model's performance.</p>
<img src="path_to_confusion_matrix_image.png" alt="Confusion Matrix" width="500"/>

<h3>2. ROC Curve</h3>
<p>The ROC curve is plotted to assess the model's classification ability across different thresholds.</p>
<img src="path_to_roc_curve_image.png" alt="ROC Curve" width="500"/>

<h3>3. Feature Importance</h3>
<p>The following plot shows the feature importance for the XGBoost model:</p>
<img src="path_to_feature_importance_image.png" alt="Feature Importance" width="500"/>

<h2 id="conclusion">Conclusion</h2>
<p>The XGBoost classifier performed well on the dataset, achieving an accuracy of [accuracy_score]. The model's strength lies in its ability to handle large datasets efficiently and its robustness to overfitting. By tuning the hyperparameters, further improvements in accuracy and generalization may be possible.</p>

<h2 id="how-to-run">How to Run</h2>
<ol>
    <li>Clone this repository to your local machine:</li>
    <pre><code>git clone https://github.com/yourusername/xgboost-classification.git</code></pre>
    <li>Install the required libraries:</li>
    <pre><code>pip install -r requirements.txt</code></pre>
    <li>Run the Jupyter notebook:</li>
    <pre><code>jupyter notebook XGBoost_Classification.ipynb</code></pre>
</ol>

<h2>License</h2>
<p>This project is part of a university assignment and is not intended for commercial use.</p>
