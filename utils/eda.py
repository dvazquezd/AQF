import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.utils import get_time_now

def inspect_dataset(df):
    """
    Provides a set of exploratory data analysis (EDA) outputs for a given pandas DataFrame.
    This function is designed to quickly summarize the key aspects of a DataFrame,
    including general structural information, descriptive statistics, and the distribution
    of NaN values across columns.

    Args:
        df (pandas.DataFrame): The dataset to be analyzed.

    Raises:
        AttributeError: If the passed object does not have methods like 'info', 'describe',
        or isnull.

    Returns:
        None
    """
    print(f'\n{get_time_now()} :: EDA: Dataset general info:')
    print(df.info())

    print('\n Descriptive statistics:')
    print(df.describe())

    print('\n NaN per column:')
    print(df.isnull().sum())

def plot_target_distribution(df):
    """
    Plots the distribution of the target variable in a dataset.

    This function generates a bar plot showing the count distribution of
    the target variable within the provided DataFrame. It visually
    distinguishes the categories in the target using color coding and
    provides an indication of the relative frequency of each label.

    Args:
        df (DataFrame): A pandas DataFrame containing the target variable.

    Raises:
        KeyError: If the 'target' column is not found in the provided DataFrame.
    """
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df['target'], hue=df['target'], palette='coolwarm', legend=False)
    plt.title('Target distribution (0 = Down, 1 = Up)')
    plt.xlabel('Target')
    plt.ylabel('Register count')
    plt.show()

def plot_pearson_correlation_matrix(df):
    """
    Plot the Pearson correlation matrix for numeric variables in the given DataFrame.

    This function selects numeric columns from the given DataFrame, calculates the Pearson
    correlation matrix, and visualizes it using a heatmap. If no numeric columns are found
    in the DataFrame, it prints a message and exits without plotting.

    It uses the matplotlib library to configure the figure size and display the plot, and
    the seaborn library to generate the correlation heatmap.

    Parameters:
        df : pandas.DataFrame
            The input DataFrame containing the data to be analyzed.

    Raises:
        None

    Returns:
        None
    """
    numeric_df = df.select_dtypes(include=['number'])

    if numeric_df.empty:
        print('There is not numeric columns to calculate the correlation matrix')
        return

    plt.figure(figsize=(18, 12))
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, annot_kws={'size': 8})

    plt.title('Pearson correlation matrix - numeric variables')
    plt.show()

def plot_price_trend(df):
    """
    Plots the closing price trend over time from a given DataFrame.

    This function visualizes the trend of the closing price as a line chart.
    The x-axis represents dates, and the y-axis represents the closing prices.
    A legend is added for clarity, and the date labels on the x-axis are rotated
    for better readability.

    Parameters:
        df (pandas.DataFrame): A DataFrame containing at least 'datetime' and
            'close' columns. The 'datetime' column represents the timestamps,
            and the 'close' column represents the closing prices.

    Returns:
        None
    """
    plt.figure(figsize=(12, 5))
    df['datetime'] = pd.to_datetime(df['datetime'])
    plt.plot(df['datetime'], df['close'], label='Close price', color='blue')
    plt.title('Close price evolution')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

def plot_technical_indicators(df):
    """
    Plots histograms for selected technical indicators from a given DataFrame if they exist
    in the data. The method visualizes the distribution of specific indicators such as RSI,
    SMA, and MACD to provide an analytical representation of technical metrics.

    Parameters:
    df : pandas.DataFrame
        The input DataFrame containing technical indicators. It must include columns with
        names matching the specified indicators to generate the plots.

    Raises:
    KeyError
        If none of the required indicators are present in the DataFrame columns.

    Notes:
    - The function assumes the passed DataFrame may contain the following technical
      indicators: "rsi_5", "rsi_7", "rsi_9", "sma_5", "sma_10", "sma_12", and "MACD".
    - Only the columns that exist in the DataFrame and match these names will be
      visualized as histograms.
    - The function creates the plots using matplotlib's histogram visualization
      functions and displays them in a single multi-plot figure with edge-colored
      bins for clarity.

    See Also:
    pandas.DataFrame.hist : Method to plot histograms from DataFrame data.
    matplotlib.pyplot.suptitle : Sets a super title for the entire figure in Matplotlib.
    """
    indicators = ['rsi_5', 'rsi_7', 'rsi_9', 'sma_5', 'sma_10', 'sma_12', 'MACD']
    if indicators[0] in df.columns:
        df[indicators].hist(figsize=(12, 8), bins=30, edgecolor='black')
        plt.suptitle('Technical indicators distribution', fontsize=14)
        plt.show()

def plot_economic_indicators(df):
    """
     plot_economic_indicators(df)

     Summary:
     This function generates and displays a boxplot for specific economic indicators
     contained in a given DataFrame. The boxplot will include Consumer Price Index (CPI),
     unemployment rate, and non-farm payroll data. It ensures that these specific indicators
     are present in the input DataFrame columns before creating the visualization. The graph
     provides a clear distribution of the specified economic variables.

     Args:
         df (pandas.DataFrame): A DataFrame containing economic indicator data. It must include
         the columns ['cpi', 'unemployment', 'nonfarm_payroll'].

     Returns:
         None

     Raises:
         None
    """
    economic_vars = ['cpi', 'unemployment', 'nonfarm_payroll']
    if all(col in df.columns for col in economic_vars):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df[economic_vars])
        plt.title('Boxplot Economic indicators')
        plt.xticks(rotation=20)
        plt.show()

def plot_sentiment_vs_target(df):
    """
    Plot the relationship between sentiment scores and target values from a DataFrame.

    This function generates a scatter plot to visualize the correlation between the average
    sentiment score (ticker_ssm) and the target variable (target) that indicates the increase
    (1) or decrease (0) of a value. The scatter plot is displayed if the required column
    'ticker_ssm' is found in the given DataFrame. Otherwise, an error message is printed.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame object containing the relevant data. Must include a 'ticker_ssm' column
        with average sentiment scores and a 'target' column with binary target values.

    Raises
    ------
    ValueError
        If the input DataFrame does not contain the necessary column 'ticker_ssm'.
    """
    if 'ticker_score' in df.columns:
        plt.figure(figsize=(10, 5))
        sns.scatterplot(x=df['ticker_score'], y=df['target'], alpha=0.5)
        plt.title('News target relation (Up/Down)')
        plt.xlabel('Average ticker sentiment')
        plt.ylabel('Target (0 = Down, 1 = Up)')
        plt.show()
    else:
        print('The column ticker_score is not in the dataset.')

def run_eda(df):
    """
    Conducts exploratory data analysis (EDA) on the given dataset, providing insights
    through various visualizations and inspections. The function operates on a
    provided DataFrame and generates plots for correlations, trends, technical,
    economic, and sentiment indicators. This enables a broader understanding of
    underlying dataset patterns and relationships.

    Args:
        df (DataFrame): The dataset to analyze, provided as a pandas DataFrame.
                        It is assumed the DataFrame contains relevant features
                        needed for analysis.

    Returns:
        None: The function generates plots but does not return any value.

    Raises:
        TypeError: If the provided argument is not a pandas DataFrame.
    """
    if df is not None:
        inspect_dataset(df)
        plot_target_distribution(df)
        plot_pearson_correlation_matrix(df)
        plot_price_trend(df)
        plot_technical_indicators(df)
        plot_economic_indicators(df)
        plot_sentiment_vs_target(df)