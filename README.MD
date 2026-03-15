# 🎬 Hybrid Movie Recommendation System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Movie Recommendation Engine built with Python and Streamlit. This project use the **MovieLens 100k dataset** to provide highly accurate movie recommendations by combining the strengths of Content-Based Filtering and Collaborative Filtering into a power Hybrid Architecture.

---

## 🚀 Key Features and User Flow

The application has a dynamic UI that adapts to the user's authentication status, using Streamlit's session state:

* **🌍 Guest Mode (Cold-Start Handled):** Non-logged-in users are recommended with trending, top-rated movies (filtered for items with >50 ratings to prevent noise).
* **👤 Logged-in User (Personalized Dashboard):** Users who log in with their ID receive a Top 10 movie list based on their historical ratings and the Hybrid Algorithm.
* **🔍 Interactive Search & Discovery:** A responsive search bar that allows users to input any movie title (case-insensitive) and view two distinct recommendation networks:
    * *Content-Based Network:* Movies with similar genres and tags.
    * *Collaborative Network:* "Users who watched this also watched..." based on community behavior.

---

## 🧠 The Algorithm: Weighted Hybrid Approach

To overcome the inherent limitations of standalone recommendation models (such as the Cold-Start problem in Collaborative Filtering and the "Circled Filter" in Content-Based models), this system implements a **Weighted Hybrid Recommender**.

The final predicted score $\hat{y}_{hybrid}$ for a user-item pair is calculated as:

$$\hat{y}_{hybrid} = \alpha \cdot \hat{y}_{cb} + (1 - \alpha) \cdot \hat{y}_{cf}$$

* **Content-Based Filtering ($\hat{y}_{cb}$):** Uses TF-IDF to vectorize movie profiles and Ridge Regression to learn user-specific genre preferences.
* **Collaborative Filtering ($\hat{y}_{cf}$):** Implements an Item-Item Collaborative approach using Cosine Similarity to find patterns among the community.
* **Hyperparameter Tuning ($\alpha$):** The system was evaluated across a grid of $\alpha$ weights to minimize the Root Mean Squared Error (RMSE) on a test set, ensuring optimal performance in production.

---

## 📁 Project Structure

```text
Movie-Recommendation-System/
├── data/
│   ├── raw/                  # Original MovieLens 100k dataset
│   └── processed/            # Cleaned data, TF-IDF matrices, mapping files
├── models/                   # Object models files (.pkl)
├── notebooks/                
│   ├── 1_EDA_&_Data_Processing.ipynb
│   ├── 2_Content_Based_Filtering.ipynb
│   ├── 3_Collaborative_Filtering.ipynb
│   └── 4_Hybrid_Evaluation.ipynb
├── src/                      # Production source code
│   ├── __init__.py
│   ├── recommender.py        # Core algorithm classes
│   └── utils.py              # Some utility functions
├── app.py                    # Streamlit web application
├── requirements.txt          # Dependencies
└── README.MD                 # Project documentation
```

---

## 🛠️ Installation and Local Setup

Follow simple steps to run this project on local machine

### 1. Clone the reponsity

```bash
git clone https://github.com/tuan0306/Recommendation-system.git
cd Recommendation-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Web App
```bash 
streamlit run app.py
```

---

## 📈 Future Improvements
* Use Matrix Factorization (SVD) for advanced Collabrative Filtering
* Use API to fetch real-time movie posters for UI