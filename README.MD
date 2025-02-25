# **AQF - Automated Quantitative Forecasting**

AQF is a data pipeline designed to fetch, process, and generate datasets for financial forecasting. It integrates financial data, technical indicators, economic indicators, and news sentiment analysis to build a robust dataset for machine learning or deep learning models.

## **Project Structure**

```
AQF/
├── config/                     # Configuration files
│   ├── gen_dataset_config.json  # Config for dataset generation
│   ├── loader_config.json       # Config for data loader
│   ├── model_config.json        # Config for model training
│   ├── feature_eng_config.json  # Config for feature engineering
├── data/                        # Raw and processed data storage
├── loader/                      # Data loading and transformation
│   ├── loader.py                # Orchestrates the data loading process
│   ├── data_loader.py           # Handles fetching and merging financial data
│   ├── data_transform.py        # Transforms data from the API response
│   ├── api_client.py            # API client to interact with Alpha Vantage
├── gen_dataset/                 # Dataset generation
│   ├── gen_dataset.py           # Main script for dataset generation
│   ├── check_news_dataset.py    # Processes and cleans news sentiment data
│   ├── check_tec_dataset.py     # Processes and validates technical indicators
│   ├── dataset_generator.py     # Merges all datasets into a final structured dataset
│   ├── feature_engineering.py   # Applies advanced feature engineering techniques
├── models/                      # Model training and evaluation
│   ├── model.py                 # Executes model training pipeline
│   ├── model_trainer.py         # Trains and evaluates machine learning models
│   ├── model_preprocessing.py   # Prepares data for training
│   ├── model_utils.py           # Utility functions for model selection and metrics
├── utils/                       # Utility functions and exploratory analysis
│   ├── utils.py                 # General helper functions
│   ├── eda.py                   # Exploratory Data Analysis (EDA) functions
├── main.py                      # Entry point for running the entire pipeline
└── README.md                    # Project documentation
```

## **Installation & Setup**

### **1. Clone the repository:**
```bash
git clone https://github.com/your-repo/AQF.git
cd AQF
```

### **2. Set up a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **3. Install dependencies:**
```bash
pip install -r requirements.txt
```

### **4. Set API Key:**
Export the Alpha Vantage API key:
```bash
export ALPHAVKEY=your_api_key
```
On Windows (PowerShell):
```powershell
$env:ALPHAVKEY="your_api_key"
```

## **Usage**

### **1. Running the Data Pipeline**
Run the `main.py` script to fetch, process, and generate datasets:
```bash
python main.py
```

### **2. How It Works**
#### **Data Loading (`loader/`)**
- `loader.py`: Controls the entire data loading process, coordinating the different modules.
- `data_loader.py`: Fetches financial, macroeconomic, and news data and structures it into raw datasets.
- `data_transform.py`: Cleans and formats raw API responses into structured tables.
- `api_client.py`: Connects to Alpha Vantage API and retrieves data.

#### **Dataset Processing (`gen_dataset/`)**
- `gen_dataset.py`: Executes the entire dataset generation pipeline.
- `check_news_dataset.py`: Cleans and processes news-related data, filtering relevant information and calculating sentiment scores.
- `check_tec_dataset.py`: Cleans and validates technical indicators, ensuring consistency and quality in the dataset.
- `feature_engineering.py`: Applies transformations such as moving averages, lags, volatility calculations, and sentiment interactions.
- `dataset_generator.py`: Merges all the datasets into a structured, ready-to-use format.

#### **Feature Engineering (`feature_engineering.py`)**
- **Lags**: Adds previous values of technical indicators to capture trends.
- **Moving Averages**: Smoothens noisy data to identify longer-term trends.
- **Sentiment Interactions**: Combines sentiment data with technical indicators for improved predictions.
- **Temporal Encoding**: Converts time-based data into cyclical features for better pattern detection.

#### **Model Training (`models/`)**
- `model.py`: Runs the full machine learning pipeline, from preprocessing to evaluation.
- `model_trainer.py`: Trains various classification models and evaluates their performance.
- `model_preprocessing.py`: Splits datasets, balances classes, and scales features before training.
- `model_utils.py`: Includes functions for evaluating models, selecting thresholds, and optimizing classification.

#### **Exploratory Data Analysis (`utils/eda.py`)**
- `eda.py`: Generates visualizations and statistics to understand dataset structure.
- **Target Distribution**: Examines the balance of the dataset’s target variable.
- **Correlation Matrices**: Identifies relationships between features.
- **Sentiment Analysis**: Evaluates how news sentiment affects price movement.

## **Final Dataset (`df_aqf.csv`)**
- Stored in `data/df_aqf.csv`, ready for model training.
- Contains features derived from technical analysis, economic indicators, and sentiment analysis.

## **Planned Enhancements**
✅ **Next steps:**
- Train and evaluate **ML/DL models** using `models/`
- Optimize **dataset aggregation methods**
- Implement a **backtesting framework**

## **Author**
**David Vázquez Doce**

