# DS-German-Credit-Risk
Data Science Assignment

<h2>Project Overview</h2>
<p>This project uses the <strong>XGBoost</strong> algorithm to perform classification on a dataset. The objective is to demonstrate using XGBoost for classification tasks and evaluate the model’s performance through accuracy and other evaluation metrics. This assignment is performing data science techniques with different algorithms handled by each of the teammates, whereas the preprocessing steps are group work.</p>

<h4>Dataset: [https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data]</h4>

<h4>Me: Kam Bee Foong (XGBoost Classification)</h4>
<h4>Team member 1: Ooi Yi Xuen</h4>
<h4>Team member 2: Ho Jun Min</h4>
<h4>Team member 3: Hee JingXi</h4>

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
<p>In this section, the dataset is loaded using <code>pandas</code>, and basic exploration is performed to understand the structure and summary statistics of the dataset.</p>


# Load the dataset and perform subsequent data understanding
<img src="https://github.com/user-attachments/assets/65df8b51-3fe7-406e-8d0c-557e622a0eb1" alt="image" width="500"/>

<br>

<img src="https://github.com/user-attachments/assets/3107f338-0938-43b7-a10f-261968e4fe91" alt="image" width="500"/>

<h3 id="data-preprocessing">2. Data Preprocessing</h3>
<p>Data preprocessing includes handling missing values, encoding categorical variables, and splitting the dataset into training and testing sets.</p>

# Handle Categorical Features by LabelEncoder
<img src="https://github.com/user-attachments/assets/933b4969-d96b-4851-b3b3-55a33c857ab7" alt="image" width="500"/>

# Checking for Missing Value
<img src="https://github.com/user-attachments/assets/a8fc2234-73ec-42d5-a8f0-5f1300cbc2d3" alt="image" width="500"/>

# Distribution of Numerical Features
<img src="https://github.com/user-attachments/assets/49247f14-4e9f-4d53-b613-55ce98045335" alt="image" width="500"/>

# Outlier Treatment
<img src="https://github.com/user-attachments/assets/7c59d76e-2352-4d68-8111-05315626a9f8" alt="image" width="500"/>

# Feature Scaling
<img src="https://github.com/user-attachments/assets/8b4e73f5-64d7-492d-9e1c-02701a973506" alt="image" width="500"/>

# Define Target Variable
<img src="https://github.com/user-attachments/assets/f8f8067b-18c0-482b-b0c8-4bcb011f8fc9" alt="image" width="500"/>

# Target Variable Distribution
<img src="https://github.com/user-attachments/assets/64a9a119-4395-4236-96bf-814ce14fbddd" alt="image" width="500"/>

# Feature Selection
<img src="https://github.com/user-attachments/assets/19b22fde-4ed6-40a2-be74-6df70f04ca90" alt="image" width="500"/>

# Class Balancing
<img src="https://github.com/user-attachments/assets/cc907855-244a-449e-b827-97e92e399928" alt="image" width="500"/>

# Train-Test Split
<img src="https://github.com/user-attachments/assets/e31bd470-527e-4b56-9383-b7c5eb7d7575" alt="image" width="500"/>

<h3 id="model-training">3. Model Training</h3>
<p>The <strong>XGBoost</strong> classifier is trained using the training data. We tune hyperparameters and evaluate the model's performance.</p>
<pre><code>from xgboost import XGBClassifier

# Initialize the XGBoost model and Train the model
xgb_baseline_params = {
    'max_depth': 3,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'use_label_encoder': False
}
xgb_baseline_model = xgb.XGBClassifier(**xgb_baseline_params)

xgb_start_time = time.time()
xgb_baseline_model.fit(X_train, y_train)
</code></pre>

<h3 id="model-evaluation">4. Model Evaluation of Best Model Selected: Trial 4- XGBoost Ensemble Model</h3>
<p>The model is evaluated using metrics such as accuracy, precision, recall, and F1-score. Additionally, confusion matrices and ROC curves are plotted to visualize the model's performance.</p>

<img src="https://github.com/user-attachments/assets/acab295c-fd75-49c6-85ed-10b6d4715a27" alt="image" width="500"/>

<br>

<img src="https://github.com/user-attachments/assets/b9b585ca-2c81-4f70-9d7e-6fd926391aee" alt="image" width="500"/>

<br>

<img src="https://github.com/user-attachments/assets/8699a1c9-bb82-4324-92bf-dba057c2d009" alt="image" width="500"/>

<br>

<img src="https://github.com/user-attachments/assets/da5847ab-9fa7-46f0-8339-a9901acc50b7" alt="image" width="500"/>

<h3> Prediction Outcomes after ensemble the best model with teammates' best models </h3>
<img src="https://github.com/user-attachments/assets/ee16a29a-3780-4ffc-b244-6df4d1901d7f" alt="image" width="500"/>

<h2 id="conclusion">Conclusion</h2>
<p>The XGBoost classifier performed well on the dataset, achieving an accuracy of [82.25%]. The model's strength lies in its ability to handle large datasets efficiently and its robustness to overfitting. By tuning the hyperparameters, further improvements in accuracy and generalization may be possible.</p>

<h2 id="how-to-run">How to Run</h2>
<ol>
    <li>Clone this repository to your local machine:</li>
    <pre><code>git clone https://github.com/charlotteorsalad/xgboost-classification.git</code></pre>
    <li>Install the required libraries:</li>
    <pre><code>pip install -r requirements.txt</code></pre>
    <li>Run the Jupyter notebook:</li>
    <pre><code>jupyter notebook XGBoost_Classification.ipynb</code></pre>
</ol>

<h2>License</h2>
<p>This project is part of a university assignment and is not intended for commercial use.</p>
