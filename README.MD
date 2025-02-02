# **AQF - Automated Quantitative Forecasting**

AQF is a data pipeline designed to fetch, process, and generate datasets for financial forecasting. It integrates financial data, technical indicators, economic indicators, and news sentiment analysis to build a robust dataset for machine learning or deep learning models.

## **Project Structure**
```
AQF/
├── config/                   # Configuration files
│   ├── gen_dataset_config.json  # Config for dataset generation
│   ├── loader_config.json       # Config for data loader
├── data/                     # Raw and processed data storage
│   ├── df_unemployment.csv   # Unemployment data
│   ├── df_cpi.csv            # CPI data
│   └── df_aqf.csv            # Final dataset output
├── loader/                   # Data loading and transformation
│   ├── loader.py             # Main script for data loading
│   ├── DataLoader.py         # Handles fetching and merging financial data
│   ├── DataTransformer.py    # Transforms data from the API response
│   ├── ApiClient.py          # API client to interact with Alpha Vantage
├── gen_dataset/              # Dataset generation
│   ├── GenDataset.py         # Runs the dataset generation pipeline
│   ├── CheckNewsDataset.py   # Processes news sentiment data
│   ├── CheckTecDataset.py    # Processes technical indicators and market data
│   ├── DatasetGenerator.py   # Merges and cleans datasets
├── models/                   # Model training and evaluation (to be implemented)
├── utils/                    # Utility functions
│   ├── utils.py              # General helper functions
│   ├── ConfigLogger.py       # Logs configuration history
├── main.py                   # Entry point for running the pipeline
└── README.md                 # Project documentation
```

## **Installation & Setup**
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/AQF.git
   cd AQF
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set API Key**:
   - Export the Alpha Vantage API key:
     ```bash
     export ALPHAVKEY=your_api_key
     ```
   - On Windows (PowerShell):
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
- **Data Loading (`loader.py`)**
  - Fetches **intraday stock price data**, **technical indicators** (SMA, RSI, MACD), and **economic indicators** (CPI, unemployment, nonfarm payroll).
  - Fetches **news sentiment data**, filtering relevant news by ticker and affected topics.
  - Merges all data into a structured format.

- **Dataset Processing (`GenDataset.py`)**
  - Cleans and processes missing data.
  - Generates **news-based features** such as sentiment scores and topic relevance.
  - Applies **technical analysis transformations** like volatility and trend detection.
  - Creates the **target variable** for prediction.

- **Final Dataset (`df_aqf.csv`)**
  - Stored in `data/df_aqf.csv`, ready for model training.

## **Key Components**
### **1. Data Collection**
- **Market Data (`DataLoader.py`)**
  - Retrieves historical stock prices and technical indicators.
  - Merges with macroeconomic indicators.

- **News Sentiment (`CheckNewsDataset.py`)**
  - Filters news by ticker.
  - Aggregates sentiment scores and topic relevance.

### **2. Feature Engineering**
- **Technical Features (`CheckTecDataset.py`)**
  - Computes SMA, RSI, and MACD indicators.
  - Extracts volatility and trend information.
  - Cleans and normalizes missing values.

- **News-Based Features (`CheckNewsDataset.py`)**
  - Computes average sentiment scores per hour.
  - Measures topic relevance and news volume impact.

- **Temporal Features (`DatasetGenerator.py`)**
  - Aggregates past-hour statistics.
  - Assigns target labels for price movement prediction.

## **Planned Enhancements**
✅ **Next steps:**
- Train and evaluate ML/DL models using `models/`
- Optimize dataset aggregation methods
- Implement a backtesting framework

---

### **Author**
David Vázquez Doce

