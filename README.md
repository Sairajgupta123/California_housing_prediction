# 🏠 California Housing Price Prediction

A Machine Learning project that predicts California house prices using multiple regression algorithms. This project compares the performance of **Linear Regression**, **Polynomial Regression**, **Ridge Regression**, and **Lasso Regression** on the California Housing Dataset from Scikit-learn.

---

## 📌 Project Overview

This project demonstrates the complete Machine Learning workflow:

- Loading the California Housing Dataset
- Data preprocessing
- Feature scaling
- Model training
- Performance evaluation
- House price prediction
- Model comparison

The trained Polynomial Regression model is saved using **Pickle** and can be used in a Streamlit web application for real-time predictions.

---

## 🚀 Features

- California Housing Dataset
- Data Cleaning & Preprocessing
- StandardScaler for Feature Scaling
- Polynomial Feature Engineering
- Multiple Regression Models
- RMSE and R² Score Evaluation
- Model Comparison
- Save Trained Model with Pickle
- Ready for Streamlit Deployment

---

## 🛠️ Technologies Used

- Python
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Pickle
- Streamlit

---

## 📂 Project Structure

```
California_housing_prediction/
│
├── California_houseing_prediction.py
├── House_price.ipynb
├── train_model.py
├── app.py
├── linear_model.pkl
├── scaler.pkl
├── poly.pkl
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 📊 Machine Learning Models

This project compares the following algorithms:

- Linear Regression
- Polynomial Regression (Degree = 2)
- Ridge Regression
- Lasso Regression

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Sairajgupta123/California_housing_prediction.git
```

Move into the project folder

```bash
cd California_housing_prediction
```

Install the required libraries

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

Train the model

```bash
python train_model.py
```

Run the Streamlit application

```bash
streamlit run app.py
```

The application will open at

```
http://localhost:8501
```

---

## 📈 Evaluation Metrics

The models are evaluated using:

- Root Mean Squared Error (RMSE)
- R² Score

These metrics help compare the prediction accuracy of each regression model.

---

## 📷 Streamlit Application

The Streamlit web application allows users to enter housing details such as:

- Median Income
- House Age
- Average Rooms
- Average Bedrooms
- Population
- Average Occupancy
- Latitude
- Longitude

The trained model predicts the estimated house price instantly.

---

## 📚 Dataset

This project uses the **California Housing Dataset** provided by Scikit-learn.

Dataset Source:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

---

## 🔮 Future Improvements

- Hyperparameter Tuning
- Random Forest Regression
- XGBoost Regressor
- Gradient Boosting
- Feature Importance Visualization
- Deploy using Streamlit Cloud
- Docker Support

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Push the branch
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Sairaj Gupta**

- GitHub: https://github.com/Sairajgupta123
- LinkedIn: www.linkedin.com/in/sairaj-gupta-974a76263

---

⭐ If you found this project helpful, please consider giving it a **Star** on GitHub!
