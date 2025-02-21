# Stock Price Prediction with Transformers🚀: A Deep Dive into Time2Vec and Data Challenges

## Introduction

When I first thought of using a Transformer model for stock price prediction, my approach was simple - treat time series data like NLP data. After all, in NLP, words are converted into vector embeddings where similar words have high dot products, which helps in calculating attention scores - the heart of a Transformer model. Could I apply a similar principle to financial time series data?

That's where **Time2Vec**⏳ came in - a technique that introduces periodic patterns to linear time series data, much like positional encodings in NLP. My goal was to make stock prices more "Transformer-friendly," enabling the model to effectively capture long-term dependencies in financial trends. However, the journey wasn't straightforward. Several challenges in data preprocessing, feature engineering, model architecture, and loss function design had to be overcome to make the model perform well.

Let's break down the entire process, the obstacles faced, and how I solved them.

---

## Making Time Series Data Transformer-Friendly

### Why Time2Vec?

In NLP, words are transformed into embeddings (like Word2Vec) and combined with positional encodings so that Transformers can understand the sequence of words. Similarly, for stock price prediction, we needed an approach that could capture both linear time variations and periodic market patterns.

🔹**Time2Vec** solves this problem by adding periodic properties to the time series data using sine and cosine functions with different frequencies.   
🔹The peaks of the sine wave correspond to the most influential time steps, allowing the model to capture cyclical trends in stock movements.   
🔹Essentially, Time2Vec acts like a combination of word embeddings and positional encoding for time series data.



## Preparing the Input Data

📌The final input to our Transformer model consists of:

- ✅**Time Features from Time2Vec** (which capture cyclic trends)
- ✅ **Stock Market Features** (high, low, close, open, volume, etc.)

This way, we ensure that the model gets a rich set of features that combine temporal dependencies and financial indicators.

---

## Data Preprocessing: A Make-or-Break Step

Data preprocessing is one of the most crucial parts of stock price prediction. Poor preprocessing can significantly degrade the model's performance. Here's what I learned the hard way:

### 1. The Skewed Distribution Problem📉

Many stocks start with very low prices and gradually increase over time. This creates a left-skewed distribution. Applying a standard scaler on this kind of data forces most values near the mean, which distorts the stock's actual price movements. As a result, the model fails to predict rising stock prices accurately.

Below is a comparison of how **Word2Vec + Positional Encoding** works in NLP versus **Time2Vec + Stock Prices** in time series:

![comparison between word2Vec and stock data distribution](https://github.com/Abhi-sheKkK/Stock_Transformer/raw/main/predictions/other_images/skewed_price_data.png)

🛑**Solution**: Instead of using `StandardScaler` or `MinMaxScaler` directly, I applied a combination of:

- **Log transformation** (to stabilize variance and reduce skewness)
- **Quantile (Gaussian) transformation** (to make the data normally distributed)

This solved the issue and made the data more suitable for Transformers, which prefer normal distributions. After prediction, the inverse transform is done using an exponential function.

---

### Visual Comparison: Standard Scaler vs. Log + Quantile Scaled Predictions

Below is a comparison of predicted prices using **Standard Scaler** versus **Log + Quantile Scaling**:

![Standard Scaler Predicted Prices](https://github.com/Abhi-sheKkK/Stock_Transformer/raw/main/predictions/other_images/standard_scaler_performance.jpg)  
*Figure 3: Predicted Prices with Standard Scaler.*

![Log + Quantile Scaled Predicted Prices](https://github.com/Abhi-sheKkK/Stock_Transformer/raw/main/predictions/other_images/log%2Bquantile_performance.jpg)  
*Figure 4: Predicted Prices with Log + Quantile Scaling.*

---

### 2. Feature Engineering🔍

Stock prices alone don't provide enough information. I manually added additional indicators like:

- 📊**12-day and 24-day MACD** (Moving Average Convergence Divergence)
- 📊**Exponential Moving Averages (EMA)**

These features help capture market momentum and trends, improving prediction accuracy.

---

## Transformer Model Architecture🏗️

The architecture consists of:

- 📌**Two Encoder and Decoder Blocks** (to capture dependencies in stock movements)
- 📌**Feed Forward Layers with Dropout** (to prevent overfitting)
- 📌**Time2Vec Embeddings** (to encode time-based dependencies)

---

## Training Strategy🎯

⚙**Hyperparameters used**:

- Lookback Window: 100 days
- Prediction Horizon: 5 days
- Epochs: 100
- Batch Size: 64
- Model Dimension (`d_model`): 64
- Number of Heads: 4
- Dropout: Added to prevent overfitting

📌The loss function was calculated on the close price feature.

---

## Insights from Model Performance📊

I tested the model on various stocks, and here's what I observed:

### 1. The Importance of Data Distribution

- ✔When stock price data is skewed, the log + quantile transform combination significantly improves performance.
- ✔When stock price data is already near to normal distribution, `MinMaxScaler` sometimes works better.

---

### 2. Data Availability Matters

- Transformers need large datasets. Stocks with fewer historical data points were not predicted as accurately.

---

### 3. Limitations in Predicting Sudden Spikes🚀

- The model underestimates exponential price surges in fast-growing stocks like Nvidia.
- It performs well when stocks have reached a peak and are now declining or stabilizing (e.g., crude oil, Intel, Vodafone Idea).

---

### 4. Model Performance Across Different Stocks📈

I tested it on Tata Motors, Crude Oil, Apple, and Tesla. The performance was accurate for stable and declining stocks but slightly underestimated sharp upward trends.

---

### Visual Comparison: Actual vs. Predicted Prices for 4 Stocks

Below are the actual vs. predicted price plots for four different stocks:

![Tata Motors: Actual vs. Predicted](https://github.com/Abhi-sheKkK/Stock_Transformer/raw/main/predictions/tata_motors/predicted_vs_actual.png)  
*Figure 5: Tata Motors - Actual vs. Predicted Prices.*  
**Test MSE**: 871.0430  
**Test RMSE**: 29.5134  
**Test MAPE**: 3.73%


![Crude Oil: Actual vs. Predicted](https://github.com/Abhi-sheKkK/Stock_Transformer/raw/main/predictions/crude_oil/actual_vs_predicted.jpg)  
*Figure 6: Crude Oil - Actual vs. Predicted Prices.*


![Apple: Actual vs. Predicted](https://github.com/Abhi-sheKkK/Stock_Transformer/raw/main/predictions/Apple/actual_vs_predicted_price.png)  
*Figure 7: Apple - Actual vs. Predicted Prices.*  
**Test MSE**: 367.5187  
**Test RMSE:** 19.1708  
**Test MAPE:** 8.58%

![Tesla: Actual vs. Predicted](https://github.com/Abhi-sheKkK/Stock_Transformer/raw/main/predictions/tesla/predicted_vs_actual.png)  
*Figure 8: Tesla - Actual vs. Predicted Prices.*  
**Test MSE:** 137.8093   
**Test RMSE:** 11.7392  
**Test MAPE:** 3.40%  

---

## Future Enhancements🔮

### 1. Hybrid Model for Better Accuracy

- Combine the Transformer model with other models (e.g., LSTMs or XGBoost) to leverage their strengths in different market conditions.

---

### 2. Predicting Initial Price Surges

- Improve the model's ability to predict the first sudden price increases, which are currently slightly underestimated.

---

### 3. Experimenting with Different Attention Mechanisms

- Modify the self-attention mechanism to assign more weight to recent trends while still capturing long-term dependencies.

---

## Conclusion🏁

This project was an exciting experiment in adapting NLP techniques for time series forecasting. By introducing **Time2Vec** and refining data preprocessing, I was able to significantly improve the Transformer model's performance in stock price prediction.  
**🚀 Key Takeaways:**    
✔ Time2Vec effectively converts time series data into a Transformer-friendly format     
✔ Log + Quantile transformations improve skewed stock data prediction     
✔ Feature engineering is essential for capturing market trends   
✔ Transformer models need a lot of data; small datasets perform poorly    

While the model performs well in many scenarios, predicting sharp upward trends remains a challenge. Moving forward, hybrid models and modified attention mechanisms could further enhance its predictive power.

If you're interested in exploring the code and results, check out my repository!

---

## References

1. ["Attention is All You Need" (2017)](https://arxiv.org/pdf/1706.03762)
2. Time2Vec: [https://arxiv.org/abs/1907.05321](https://arxiv.org/abs/1907.05321)
3. Transformer architecture : [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)

---

## Contact Me

- **GitHub**: [Abhi-sheKkK](https://github.com/Abhi-sheKkk)
- **LinkedIn**: [Abhishek Kotwani](https://in.linkedin.com/in/abhishek-kotwani-906b2828a)
- **Email**: [Abhishek.9.kotwani@gmail.com](mailto:abhishek.9.kotwani@gmail.com)

Would love to hear your thoughts! Feel free to drop a comment or reach out for collaborations! 🚀
