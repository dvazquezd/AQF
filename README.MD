# Asset Quote Forecast Project with Alpha Vantage

## Description
This project uses historical pricing data, technical and economic indicators, as well as news analysis to predict an asset's price in the next hour. The data is provided by the Alpha Vantage API, and the system is designed to process this information and generate Machine Learning (ML) and Deep Learning (DL) models to achieve the highest possible accuracy.

## Main Features
- **Data Download**: Fetch historical price data, technical indicators (RSI, SMA, MACD), and economic indicators (nonfarm payrolls, CPI, unemployment rate) using the Alpha Vantage API.
- **News Processing**: Analyze financial news, including sentiment scores and relevance.
- **Dataset Generation**: Create combined datasets incorporating historical pricing, technical and economic indicators, and news data.
- **Data Validation**: Validate and transform data for ML and DL models.
- **Centralized Configuration**: Modular configuration system for customizable execution and storage parameters.

## Project Structure

```
.
├── ApiClient.py         # Client to interact with the Alpha Vantage API.
├── ConfigManager.py     # Centralized configuration management.
├── DataLoader.py        # Data loading and storage.
├── DataTransformer.py   # Data transformations and validation.
├── GenDataset.py        # Combined dataset generation.
├── CheckTecDataset.py   # Validation of technical datasets.
├── utils.py             # Utility functions.
├── main.py              # Main project script.
├── loader.py            # Configuration loader system.
├── loader_config.json   # Initial loader configuration.
├── gen_dataset_config.json # Dataset generation configuration.
```

## Installation and Configuration

### Prerequisites
- Python 3.8 or higher
- Alpha Vantage API keys
- Libraries listed in `requirements.txt`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/your_repository.git
   cd your_repository
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration
1. Define your API keys and other parameters in `loader_config.json` and `gen_dataset_config.json`.
2. Adjust settings in `ConfigManager.py` as needed.

## Usage

1. **Run the main script**:
   ```bash
   python main.py
   ```
2. **General workflow**:
   - The script downloads the necessary data using `ApiClient.py`.
   - Transforms and validates it with `DataTransformer.py`.
   - Generates combined datasets using `GenDataset.py`.

## Technical Details

### Key Methods and Functions
#### ApiClient.py
- `get_technical_data()`: Retrieves technical indicators.
- `get_economic_data()`: Downloads economic data.

#### DataTransformer.py
- `validate_data()`: Validates data quality.
- `transform_data()`: Performs required transformations for ML/DL models.

#### GenDataset.py
- `generate_dataset()`: Creates an integrated dataset from multiple data sources.

## Contribution
Contributions are welcome. Please open an issue or a pull request in the repository.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
- **Author**: David Vázquez Doce
- **Email**: [dvazquezd@gmail.com](mailto:dvazquezd@gmail.com)
- **GitHub**: [https://github.com/dvazquezd](https://github.com/dvazquezd)
