# **Stock Market Prediction Using LSTM Networks**

## **Overview**
This project implements a Long Short-Term Memory (LSTM) network to predict stock market trends based on historical stock prices. The model processes time-series data, learns stock market patterns, and forecasts future stock prices. The dataset used includes daily stock prices of Apple Inc. (AAPL) and other global stocks.

## **Features**
- Uses **LSTM layers** to learn sequential stock price patterns.
- Implements **MinMax scaling** to normalize stock price data.
- Supports **multi-step forecasting (7-day prediction).**
- Incorporates **an attention mechanism** to enhance predictive accuracy.
- Trained with **Adam optimizer** and **Mean Squared Error (MSE) loss.**
- Data sourced from **Kaggle and financial APIs**.
- Provides visualization of **actual vs. predicted stock prices.**

## **Dataset**
- **Source**: Kaggle (World Stock Prices Dataset)
- **Features**:
  - `Date`: Timestamp of stock data.
  - `Open`: Opening stock price.
  - `High`: Highest stock price of the day.
  - `Low`: Lowest stock price of the day.
  - `Close`: Closing stock price.
  - `Volume`: Number of shares traded.
- **Preprocessing**:
  - Converts `Date` column to datetime format.
  - Sorts data chronologically.
  - Filters stock data for AAPL.
  - Applies **Z-score normalization (StandardScaler).**

## **Installation & Setup**
### **1. Prerequisites**
Ensure you have Python installed along with the following dependencies:
```bash
pip install numpy pandas tensorflow keras scikit-learn matplotlib
```
### **2. Clone Repository**
```bash
git clone https://github.com/your-repo/stock-lstm.git
cd stock-lstm
```
### **3. Run the Model**
```bash
python stock_prediction.py
```

## **Model Architecture**
- **Input Layer**: Accepts stock price sequences.
- **Two LSTM Layers**: Each with 50 units, responsible for learning sequential dependencies.
- **Dropout Layers**: 30% dropout applied to prevent overfitting.
- **Attention Mechanism**: Enhances the model’s ability to focus on relevant price movements.
- **Dense Output Layer**: Predicts the next closing price.
- **Activation Function**: Uses **Softmax (incorrectly, should be Linear)**.
- **Optimizer**: Adam (learning rate = 0.001).
- **Loss Function**: Mean Squared Error (MSE).

## **Usage**
### **1. Training the Model**
The model is trained using:
```python
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
```
### **2. Making Predictions**
To predict stock prices for the next 7 days:
```python
forecast = forecast_next_week_with_attention(model, last_sequence, num_days=7)
```

## **Results & Issues**
- The **LSTM model captures general stock market trends** but fails to predict sharp fluctuations accurately.
- The **7-day forecast is constant at 104.18**, indicating incorrect behavior due to the **Softmax activation function.**
- Validation loss remains high (~1.1910), suggesting **underfitting or poor optimization**.
- Future improvements include **replacing Softmax with Linear activation, fine-tuning hyperparameters, and using hybrid models like CNN-LSTM.**

## **Future Enhancements**
- Replace **Softmax activation with Linear activation**.
- Improve model by **incorporating external financial indicators** (trading volume, sentiment analysis, market indices).
- Experiment with **transformer-based models** for better sequence learning.
- Optimize hyperparameters using **grid search**.

## **Contributors**
- **Rizwan Muhammad Harris**
  - Email: mharrisrizwan@gmail.com
  - Institution: CUAS, Villach
  - Research Interest: Time-Series Forecasting, Machine Learning

## **License**
This project is licensed under the **MIT License**.

## **References**
- Kaggle, "World Stock Prices Dataset". [Online]. Available: https://www.kaggle.com/datasets
- Staudemeyer, R. C., & Morris, E. R. "Understanding LSTM – A Tutorial into Long Short-Term Memory Recurrent Neural Networks." Faculty of Computer Science, Schmalkalden University of Applied Sciences, Germany, 2019.
