# 🏆 Customer Segmentation - Mismatched Clusters Analysis

Deployed link : https://customersegmentation-6s99huyf9ifkafk9p9cuyy.streamlit.app/

This project analyzes customer segmentation using **K-Means** and **Hierarchical Clustering**. The focus is on identifying mismatched customers—those assigned to different clusters by the two models.

## 📌 Key Features

- **Mismatched Customer Analysis**: Identifies customers grouped differently by clustering models.
- **Interactive Dashboard**: Built with **Streamlit** for real-time insights.
- **Data Exploration**: View mismatched customers' spending behavior, income levels, and age distributions.
- **Model Performance Comparison**: Compare **K-Means** vs **Hierarchical Clustering** results.

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/AdityaSareen06/Customer_Segmentation.git
cd Customer_Segmentation

# Set up Virtual Env
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

#Install dependencies 

pip install -r requirements.txt

#Run dashboard
streamlit run src/dashboard.py

📊 Dashboard Overview
The interactive dashboard allows users to:

Explore mismatched clusters visually.
Filter customers based on spending, income, and age.
View key insights & model comparisons.

📈 Insights from Analysis
Most mismatched customers fall into the low spending category.
Age distribution shows significant variation, with a mix of middle-aged and senior customers.
Some outliers in spending scores indicate high-potential customers for premium offers.
Hierarchical clustering tends to group more customers into a single cluster compared to K-Means.

🛠 Technologies Used
Python 🐍
Streamlit 📊
Scikit-learn 🤖
Pandas, NumPy 🏗️
Plotly 📈

📜 License
This project is licensed under the MIT License.

✨ Author
👤 Aditya Sareen
📧 Email: adityasareen.216@gmail.com
🔗 GitHub: AdityaSareen06
