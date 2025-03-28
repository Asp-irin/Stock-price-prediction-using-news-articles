# Stock Price Movement Prediction Using Financial News Sentiment Analysis  

## Introduction  
Stock price prediction is a challenging task influenced by multiple factors, including market trends, economic indicators, and news sentiment. This project utilizes financial news headlines to predict stock price movements using a fine-tuned BERT model for sentiment analysis and machine learning models for prediction.  

## Project Workflow  

1. **Data Collection**  
   - Scraped news headlines from Economic Times archives.  
   - Collected historical stock price data and financial indicators.  

2. **Sentiment Analysis**  
   - Fine-tuned a pre-trained BERT model on financial news sentiment datasets.  
   - Applied the model to assign sentiment scores (positive, neutral, negative) to news headlines.  
   - Used TinyBERT for sentiment probability scoring.  

3. **Stock Price Prediction**  
   - Used sentiment scores and historical stock data as input features.  
   - Implemented machine learning models (Random Forest, LightGBM) to predict price movements.  
   - Evaluated performance using accuracy, F1-score, and confusion matrix.  

## Technologies Used  
- **Python Libraries**: Transformers, Scikit-learn, Pandas, Matplotlib, Seaborn  
- **Scraping Tools**: BeautifulSoup, Selenium  
- **ML Frameworks**: PyTorch, TensorFlow  
- **Visualization**: Plotly, Seaborn  

## Repository Structure  
```bash
├── data/  
│   ├── news_data.csv  
│   ├── stock_data.csv  
├── models/  
│   ├── fine_tuned_bert/  
│   ├── ml_model.pkl  
├── notebooks/  
│   ├── 1_webscraping.ipynb  
│   ├── 2_bert_training.ipynb  
│   ├── 3_bert_inference.ipynb  
│   ├── 4_ml_predictions.ipynb  
│   ├── 5_tinybert_sentiment.ipynb  
