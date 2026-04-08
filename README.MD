# 🎬 Movie Recommendation System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ed?logo=docker&logoColor=white)
![CI/CD](https://img.shields.io/badge/CI%2FCD-Automated-2088FF?logo=github-actions&logoColor=white)
![Azure](https://img.shields.io/badge/Azure-Cloud%20Hosted-0078d4?logo=microsoft-azure&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Project Overview

Movie Recommendation Engine built with Python and Streamlit. This project use the **MovieLens 100k dataset** to provide highly accurate movie recommendations by combining the strengths of Content-Based Filtering and Collaborative Filtering into a power Hybrid Architecture.

Moving beyond a simple Jupyter Notebook, this project bridges the gap between Data Science and Software Engineering. It is deployed as a **Production-ready Microservice**, fully containerized with **Docker**, and features an automated **CI/CD pipeline** hosted on a **Microsoft Azure Virtual Machine**.

🟢 **Live Web App:** [http://tuan-webs.koreacentral.cloudapp.azure.com:8502/](http://tuan-webs.koreacentral.cloudapp.azure.com:8502/)

🟢 **Live API Docs (Swagger):** [http://tuan-webs.koreacentral.cloudapp.azure.com:8001/docs/](http://tuan-webs.koreacentral.cloudapp.azure.com:8001/docs/)

---

## 🚀 Key Features

* **🌍 Guest Mode:** Handles the "Cold-Start" problem by suggesting trending/top-rated movies.
* **👤 Personalized Dashboard:** Users logging in with their ID receive a Top 10 list based on historical behavior.
* **🔍 Dual-Network Search:** Responsive search bar providing:
    * *Content-Based Network:* Similar genres and metadata.
    * *Collaborative Network:* "Users who watched this also watched..."
* **⚡ Optimized Performance:** Implements server-side caching to ensure sub-second response times.

---

## 📊 Dataset

This project utilizes the classic **MovieLens 100K Dataset** provided by GroupLens Research. It contains 100,000 ratings from 1,000 users on 1,700 movies.

👉 **[Download the MovieLens 100K Dataset here](https://grouplens.org/datasets/movielens/100k/)**

---

## 🧠 The Algorithm: Weighted Hybrid Approach

To overcome the inherent limitations of standalone recommendation models (such as the Cold-Start problem in Collaborative Filtering and the "Circled Filter" in Content-Based models), this system implements a **Weighted Hybrid Recommender**.

The final predicted score $\hat{y}_{hybrid}$ for a user-item pair is calculated as:

$$\hat{y}_{hybrid} = \alpha \cdot \hat{y}_{cb} + (1 - \alpha) \cdot \hat{y}_{cf}$$

* **Content-Based Filtering ($\hat{y}_{cb}$):** Uses TF-IDF to vectorize movie profiles and Ridge Regression to learn user-specific genre preferences.
* **Collaborative Filtering ($\hat{y}_{cf}$):** Implements an Item-Item Collaborative approach using Cosine Similarity to find patterns among the community.
* **Hyperparameter Tuning ($\alpha$):** The system was evaluated across a grid of $\alpha$ weights to minimize the Root Mean Squared Error (RMSE) on a test set, ensuring optimal performance in production.

---

## 🏗️ System Architecture & CI/CD

The system follows a modern DevOps workflow:
1.  **Containerization:** Both Frontend and Backend are isolated using **Docker Compose**.
2.  **Cloud Hosting:** Deployed on an **Ubuntu VM (Azure)**.
3.  **Automated Pipeline:** **GitHub Actions** triggers on every push to `main`:
    * Authenticates via SSH.
    * Synchronizes code with `git reset --hard`.
    * Triggers `docker compose up -d --build` for zero-downtime updates.

---


## 📁 Project Structure

```text
Recommendation-system
├── .github/workflows/         
│   └── deploy.yml           # Automated CI/CD deployment pipeline
├── api/                     # Backend API (FastAPI)
│   ├── __init__.py
│   ├── models/              # Pre-trained ML models (.pkl)
│   ├── src/                 # Backend source code
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI entry point
│   │   ├── recommender.py   # Algorithm logic
│   │   └── utils_backend.py # Backend utility functions
│   ├── Dockerfile           # Backend containerization
│   └── requirements.txt     # Backend dependencies
├── app/                     # Frontend UI (Streamlit)
│   ├── src/                 # Frontend source code
│   │   ├── app.py           # Streamlit main application
│   │   ├── client_api.py    # API client to communicate with Backend
│   │   └── utils_ui.py      # UI helper functions
│   ├── Dockerfile           # Frontend containerization
│   └── requirements.txt     # Frontend dependencies
├── data/                    # Dataset storage
│   ├── raw/                 # MovieLens 100k dataset
│   └── processed/           # Cleaned data, TF-IDF matrices, mapping files
├── notebooks/               # Research & Model development (Jupyter Notebooks)
│   ├── Content_Base.ipynb
│   ├── EDA_Data_Cleaning.ipynb
│   ├── Hybrid.ipynb
│   └── Neighborhood_Based_Collaborative.ipynb
├── .env                     # Environment variables (Secrets)
├── .gitignore               # Files to be ignored by Git
├── docker-compose.yml       # Orchestration for both API and App
├── requirements.txt         # Project dependencies
└── README.MD                # Project documentation
```

---

## 🚀 How to Run

Follow these steps to get the system up and running on your local machine.

### 📋 Prerequisites
* **Docker** & **Docker Compose** (Recommended)
* **Python 3.9+** (For manual installation)
* **Git**

### 📥 Step 1: Clone the Project
First, download the repository to your local machine:
```bash
git clone https://github.com/tuan0306/Recommendation-system.git
cd Recommendation-system
```

### 📦 Step 2: Models & Configuration Setup
1. **Download Pre-trained Models:**
* This project requires model weights (`.pkl` files) to function.
* Download them from: [Models Storage (Google Drive)](https://drive.google.com/drive/folders/1sw44diCClaMo1Uql0gSTJV9RPv3ApIUM?hl=vi)
* Place all `.pkl` files inside the `/api/models/` directory
2. **Environment Configuration:**
This project uses environment variables to manage configurations. Since the `.env` file is not included in the repository for security, you need to create it manually:
* Create file `.env` in the project root
* **Add configurations:** Open the `.env` file with your text editor and add the following line:
  ```plaintext
  TMDB_KEY=<YOUR_KEY_HERE>
  ```

### 🛠️ Step 3: Launching the Application
**Option A: Using Docker (Recommended)**
```bash
docker-compose up -d --build
```
* **Web UI**: [http://localhost:8502](http://localhost:8502)
* **API Docs**: [http://localhost:8001/docs](http://localhost:8001/docs)

**Option B: Manual Installation**
1. **Install requirements:** `pip install -r requirements.txt`
2. **Start Backend:**
   ```bash
   cd api
   uvicorn src.main:app --reload
   ```
3. **Start Frontend:**
   ```bash
   cd app/src
   streamlit run app.py
   ```

---

## 📈 Future Improvements
* Use Matrix Factorization (SVD) for advanced Collabrative Filtering

---

## 👨‍💻 Author
**Nguyễn Đình Tuấn**
* **Github Profile:** [https://github.com/tuan0306](https://github.com/tuan0306) 