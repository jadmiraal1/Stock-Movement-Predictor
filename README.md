# Stock Movement Predictor

This project aims to build a stock movement prediction system that leverages both historical price data and real-time sentiment analysis from financial news and social media.
By combining transformer-based models (e.g., BERT/FinBERT) for natural language understanding with time-series modeling of stock prices, we create a hybrid machine learning
pipeline that predicts whether a given stock will go up or down over a short-term horizon (1–5 days).

<<<<<<< HEAD
The goal is to develop a model that is:
-More informative than price data alone, by including market sentiment signals
-Fully backtestable, with clear performance metrics
-Deployable for paper or live trading, using APIs like Alpaca

## Installation (pip-based)

python -m venv venv  
source venv/bin/activate  # or venv\Scripts\activate on Windows  
pip install -r requirements.txt  

## Installation (Conda-based)

conda env create -f environment.yml  
<<<<<<< Updated upstream
conda activate stock-predictor-env =
=======
conda activate stock-predictor-env  
=======
>>>>>>> Stashed changes
The goal is to develop a model that is:  
-More informative than price data alone, by including market sentiment signals  
-Fully backtestable, with clear performance metrics  
-Deployable for paper or live trading, using APIs like Alpaca  

## Installation

git clone https://github.com/jadmiraal1/Stock-Movement-Predictor.git  
cd Stock-Movement-Predictor  
  
python -m venv venv  
source venv/bin/activate      # On Windows: venv\Scripts\activate  

pip install --upgrade pip  
pip install -r requirements.txt  
<<<<<<< Updated upstream
>>>>>>> origin/main
=======
>>>>>>> 12d142cf135db447f6bd6b1a4917ac86d6a95e75
>>>>>>> Stashed changes

Future Directions  
How do different stocks affect each other?
