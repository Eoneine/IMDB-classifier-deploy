<div align="center">

# 🎬 IMDB Sentiment Analysis Classifier 🍿

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](#)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white)](#)

A beautiful, interactive web application built with **Streamlit** to classify IMDB movie reviews as either **Positive** or **Negative** using Machine Learning.

</div>

---

## 📊 Dataset Description

The dataset driving this project is the **IMDB Movie Reviews Dataset**, which is one of the most popular benchmark datasets for evaluating text-based sentiment analysis models.

* **🧵 Data Characteristics:** Contains raw text comprising movie reviews written by audiences directly on the IMDB platform.
* **⚖️ Class Distribution:** **Perfectly Balanced (50:50)**. This dataset holds an exact equal proportion of positive and negative labeled reviews (Imbalance Ratio = 1.0). This allows the model to learn both classes fairly without needing any resampling techniques.
* **🎯 Target Labels:**
  * **🟢 Positive (1):** Movie reviews expressing satisfaction, praise, or recommendations.
  * **🔴 Negative (0):** Movie reviews expressing disappointment, critique, or dissatisfaction.
* **📚 Vocabulary Size:** The original text boasts an extremely rich vocabulary (reaching over **129,000 unique words** before undergoing preprocessing steps like case folding, stopword removal, and stemming).

---

## 🛠️ Installation & Setup

Want to run this app on your local machine? Just follow these simple steps!

1. **Clone or Download** the repository to your local machine.
2. **Create a virtual environment** (optional but highly recommended):
   ```bash
   conda create -n ds-tools python=3.10
   conda activate ds-tools
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run the App

Once everything is installed, you can start the Streamlit web application by typing the following command in your terminal:

```bash
streamlit run app.py
```

The application will open automatically in your default web browser at `http://localhost:8501`. 

---

## 📁 Repository Structure

* `app.py`: The main Streamlit web application script.
* `lr_best.pkl`: The trained Logistic Regression model (or your best model).
* `vec_terpilih.pkl`: The TF-IDF / Count Vectorizer used for feature extraction.
* `label_encoder.pkl`: Encoder for mapping targets back to their original labels.
* `treshold.txt`: Contains the optimal decision threshold for classification.
* `requirements.txt`: List of Python packages required to run the project.

---
<div align="center">
  <i>Created for PPKD Jakarta Selatan - Data Analyst Program ✨</i>
</div>