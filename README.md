<div align="center">
  <h1>🧠 Predictive Modeling for Depression Risk</h1>
  <p><i>An end-to-end machine learning pipeline analyzing psychometric data to classify depression severity.</i></p>
</div>

---

## 📖 Project Overview
This repository contains a comprehensive machine learning analysis aimed at predicting depression risk levels in university students. By evaluating demographic details, academic metrics, and psychological questionnaires, the project highlights the compounding effects of academic pressure on mental health and successfully groups individuals using both supervised and unsupervised algorithms.

## 📊 About the Dataset
The dataset (`Depression Data.csv`) consists of self-reported psychological evaluations and academic metrics. 
* **Total Instances:** 1,977 student records (1,971 after cleaning).
* **Total Features:** 37 attributes.
  * **Quantitative (29):** Ordinal responses to psychological/academic pressure questionnaires (Likert scale) and aggregate scores for Stress, Anxiety, and Depression.
  * **Categorical (8):** Demographic and academic background information (e.g., Age, Gender, Academic Year, CGPA).
* **Target Variable:** `Depression Label` — A multi-class variable categorized into 6 distinct severity levels: *No Depression, Minimal, Mild, Moderate, Moderately Severe,* and *Severe*.
* **Data Characteristics:** The dataset exhibits a significant class imbalance, skewing heavily toward the "Moderately Severe" and "Severe" categories. It also contains a noticeable demographic skew, with male instances representing a much larger portion of the severe categories than female instances.

## 🛠️ Methodology & Tech Stack
The pipeline is built using **Python, Pandas, Seaborn, and Scikit-Learn**, following a structured data science workflow:
* **Exploratory Data Analysis (EDA):** Extracted demographic skews and mapped the strong positive correlation between aggregate stress values and depression severity using heatmaps and boxplots.
* **Data Pre-Processing:** Cleaned missing instances, transformed categorical text via Label Encoding, and normalized scales utilizing Min-Max Scaling.
* **Supervised Learning:** Trained Logistic Regression, Decision Tree, and Multi-Layer Perceptron (Neural Network) models using a stratified 80/20 split to ensure minority classes were accurately represented in the testing phase.
* **Unsupervised Learning:** Deployed K-Means clustering mapped over a 2D space using Principal Component Analysis (PCA) to discover natural algorithmic groupings based purely on survey responses.

## 📈 Key Insights & Results
* **Algorithmic Performance:** The Neural Network established a highly robust 0.9671 accuracy and a 0.9987 AUC score, successfully modeling the complex, non-linear relationships within the 29-dimensional survey data. Logistic Regression provided a strong linear baseline at 0.9038 accuracy.
* **Data Leakage Discovery:** The Decision Tree achieved a perfect 1.0000 accuracy score. Further analysis revealed this was due to deterministic mapping: the algorithm "memorized" the exact mathematical thresholds of the aggregate `Depression Value` column used to assign the target labels. This provided an excellent real-world case study of feature leakage in training datasets.

## 🚀 Future Work
To simulate a true predictive diagnostic tool, future iterations of this project will involve dropping the aggregate "Value" columns (Stress, Anxiety, Depression) before training. This will force the models to predict depression severity relying *exclusively* on the raw, individual survey questionnaire responses, eliminating the data leakage issue.

## 📂 Repository Structure
* `depression_analysis.ipynb` — The primary Jupyter Notebook containing the full code, analysis, and metric evaluations.
* `Depression Data.csv` — The raw dataset used for training and testing.


