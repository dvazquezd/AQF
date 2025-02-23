import numpy as np
import utils.utils as ut

class FeatureEngineering:
    def __init__(self, df):
        """
        Class implementing feature engineering utilities for a dataset. It stores
        original and transformed datasets, identifies numeric columns, and loads
        feature engineering configurations from a specified JSON file.

        Attributes:
        df_original (DataFrame): The original unmodified dataframe provided at
            initialization.
        df (DataFrame): The dataframe on which transformation operations are applied.
        config (dict): Configuration dictionary loaded from a JSON file, containing
            settings for feature engineering.
        numeric_columns (list[str]): List of column names from the dataframe that
            contain numeric (float) data types.
        all_columns (list[str]): List of all column names present in the dataframe.

        Parameters:
        df (DataFrame): The input dataframe to be processed. The provided dataframe
            serves as a source for data manipulation and analysis.
        """

        self.df_original = df.copy()  # Save a copy of the original dataset
        self.df = df.copy()  # Dataset on which we apply transformations
        self.config = ut.load_config('feature_eng_config')  # Load the configuration from JSON
        self.numeric_columns = self.df.select_dtypes(include=['float']).columns.tolist()
        self.all_columns = self.df.columns.tolist()

    def add_lags(self):
        """
        Adds lagged features to numeric columns in the dataframe based on the
        configuration provided. This method modifies the dataframe by appending
        new columns for each lag specified in the configuration for numeric
        columns where lagging is enabled.

        Raises
        ------
        KeyError
            Raised if a required key is missing in the configuration dictionary.

        Parameters
        ----------
        self : object
            The instance of the class containing the dataframe (`self.df`), the
            numeric columns (`self.numeric_columns`), and the configuration
            dictionary (`self.config`).

        Returns
        -------
        self : object
            Returns the instance of the class, allowing for method chaining.
        """
        for feature in self.numeric_columns:
            if self.config['apply_lag'].get(feature, False):
                for lag in self.config['lags']:
                    self.df[f'{feature}_lag{lag}'] = self.df[feature].shift(lag)

        return self

    def add_moving_avg(self):
        """
        Adds moving averages to the DataFrame for specified numeric columns using the given windows.

        This function iterates through the numeric columns of the DataFrame and checks the
        configuration to determine which features require the application of moving averages.
        For each applicable feature, it calculates the moving average for different window
        sizes specified in the configuration and appends the calculated averages as new
        columns in the DataFrame.

        Returns:
            This method returns the instance of the class (self) for chaining purposes.

        Raises:
            This implementation does not directly raise errors, but pandas operations used
            such as `rolling` and `mean` may raise errors when operating on invalid data.
        """

        for feature in self.numeric_columns:
            if self.config['apply_moving_avg'].get(feature, False):
                for window in self.config['windows']:
                    self.df[f'{feature}_ma{window}'] = self.df[feature].rolling(window=window).mean()

        return self

    def add_differences(self):
        """
        Adds differences and percentage changes to the DataFrame.

        This method processes the DataFrame columns to calculate differences
        and percentage changes based on the configuration settings. For each
        column, if the option for calculating difference is enabled in the
        configuration, a new column is added to store the computed difference.
        Additionally, for specific numerical columns like 'close', 'volume',
        and 'MACD', percentage changes are also calculated and stored in
        separate columns.

        Parameters
        ----------
        Non parameters are required.

        Returns
        -------
        self : object
            Returns the modified object instance with the updated DataFrame.
        """
        for feature in self.df.columns:
            if self.config['apply_diff'].get(feature, False):  # Check if the difference is enabled
                self.df[f'{feature}_diff'] = self.df[feature].diff()

                # If the column is numeric and represents a price or volume, calculate the % change
                if feature in ['close', 'volume', 'MACD']:
                    self.df[f'{feature}_pct_change'] = self.df[feature].pct_change().round(6)

        print('Aggregated differences and percentage changes')
        return self

    def add_sentiment_interactions(self):
        """
        Adds interaction columns to the DataFrame based on sentiment score and other metrics.

        This method checks for the existence of specific columns in the instance's DataFrame
        and calculates interaction terms based on 'ticker_sentiment_score_mean'. These
        interaction terms are added as new columns to enhance the analysis of sentiment
        impact on metrics such as price trends and volume.

        Returns:
            self: The current instance with updated DataFrame, including the newly added
                  interaction columns.

        Raises:
            None
        """
        if 'ticker_sentiment_score_mean' in self.df.columns and 'price_trend' in self.df.columns:
            self.df['sentiment_price_interaction'] = (
                    self.df['ticker_sentiment_score_mean'] * self.df['price_trend']
            )

        if 'ticker_sentiment_score_mean' in self.df.columns and 'volume' in self.df.columns:
            self.df['sentiment_volume_interaction'] = (
                    self.df['ticker_sentiment_score_mean'] * self.df['volume']
            )

        print('Aggregated interactions between sentiment and price')
        return self

    def encode_temporal_features(self):
        """
        Encodes temporal features in a DataFrame by applying transformations to the 'hour' column to create
        sinusoidal and cosinusoidal representations and, if present, generates one-hot encoded variables for
        the 'day_of_week' column.

        Returns
        -------
        self
            The modified object with the encoded temporal features included in its DataFrame.

        Notes
        -----
        This method assumes the DataFrame (`self.df`) contains features related to time such as 'hour' or
        'day_of_week'. If these columns are not present in the DataFrame, the method will not make changes
        to the respective features.

        Examples
        --------
        This section is intentionally omitted as per documentation rules.
        """
        if 'time' in self.df.columns:
            self.df['hour_sin'] = np.sin(2 * np.pi * self.df['time'] / 24)
            self.df['hour_cos'] = np.cos(2 * np.pi * self.df['time'] / 24)

        return self

    def validate_features(self):
        """
        Validates the presence of specific feature columns in the DataFrame based on the configuration.

        The method iterates through the feature columns specified in the configuration
        under the keys `apply_lag`, `apply_diff`, and `apply_moving_avg`.
        It identifies any columns that are defined in the configuration but do not exist
        in the provided DataFrame. Missing columns, if any, are appended to a list
        and a warning is printed to notify the user.

        Raises:
            None
        """
        missing_columns = []
        all_columns = list(self.config['apply_lag'].keys()) + \
                      list(self.config['apply_diff'].keys()) + \
                      list(self.config['apply_moving_avg'].keys())

        for col in set(all_columns):
            if col not in self.df.columns:
                missing_columns.append(col)

        if missing_columns:
            print(f'Warning: The following columns defined in the configuration do NOT exist in the dataset and '
                  f'will be ignored: {missing_columns}')

    def delete_no_necessary_col(self):
        """
        Removes unnecessary columns from the dataframe and retains only the specified columns.

        This method filters the columns of the dataframe based on the specified list of
        columns to keep. If no such list is provided, the dataframe remains unchanged.

        Attributes:
            df (DataFrame): The dataframe from which columns will be filtered.
            config (dict): A dictionary containing configurations, including the
                'columns_to_keep' key.

        Returns:
            The updated object with the filtered dataframe.

        Raises:
            KeyError: If one or more specified columns do not exist in the dataframe.
        """
        columns_to_keep = self.config.get('columns_to_keep', [])
        self.df = self.df[columns_to_keep] if columns_to_keep else self.df
        return self

    def add_advanced_features(self):
        """
        Adds advanced features, cycle analysis, and aggregated perspective indicators to the dataframe based on the
        configuration provided. The method checks for specific advanced indicators, analyzes specified cycles, and
        computes aggregated perspectives to enhance the dataset.

        Returns
        -------
        DataFrame
            A dataframe with the added advanced indicators and analysis as per the configuration.

        Raises
        ------
        None

        """
        # Advanced metrics
        if self.config['advanced_indicators'].get('intraday_volatility', False):
            self.add_intraday_volatility()

        if self.config['advanced_indicators'].get('volume_ratio', False):
            self.add_volume_ratio()

        if self.config['advanced_indicators'].get('price_trend', False):
            self.add_price_trend()

        if self.config['advanced_indicators'].get("previous_hours_target", False):
            self.add_previous_targets()

        # Cycle analysis
        if self.config['cycle_analysis'].get('monthly_cycle', False):
            self.add_monthly_cycle()

        if self.config['cycle_analysis'].get('yearly_cycle', False):
            self.add_yearly_cycle()

        # Aggregated perspective
        if self.config['aggregated_perspective'].get('closing_moving_avg', False):
            self.add_closing_moving_avg()

        if self.config['aggregated_perspective'].get('cumulative_change_in_volume', False):
            self.add_cumulative_change_in_volume()

        return self.df

    def add_intraday_volatility(self):
        """
        Adds a new column to the DataFrame containing the calculated intraday volatility.

        The intraday volatility is computed as the difference between the 'high' and
        'low' values in each row of the DataFrame. The result is rounded to four
        decimal places and stored in a new column named 'intraday_volatility'.

        Args:
            This method operates directly on the object's DataFrame attribute.

        Returns:
            DataFrame: The updated DataFrame with the new 'intraday_volatility' column added.

        Raises:
            KeyError: If the 'high' or 'low' columns are missing in the DataFrame.
        """

        self.df['intraday_volatility'] = (self.df['high'] - self.df['low']).round(4)
        return self.df

    def add_volume_ratio(self):
        """
        Computes a new column 'volume_ratio' by dividing the 'volume' column by the rolling mean of
        the 'volume' column, with a window of 5 and a minimum of 1 period. The result is rounded
        to four decimal places and added as a new column to the data frame.

        Returns
        -------
        DataFrame
            The updated DataFrame with the new 'volume_ratio' column added.
        """
        self.df['volume_ratio'] = (self.df['volume'] / self.df['volume'].rolling(window=5, min_periods=1).mean()).round(4)
        return self.df

    def add_price_trend(self):
        """
        Adds a price trend calculation to the DataFrame.

        This method computes the percentage change in the 'close' column
        of the DataFrame, rounds the result to four decimal places, and
        assigns it to a new column named 'price_trend'. It modifies the
        original DataFrame and returns it with the added column.

        Args:
            Non arguments are required.

        Returns:
            pandas.DataFrame: Modified DataFrame with a new 'price_trend' column.
        """
        self.df['price_trend'] = self.df['close'].pct_change().round(4)
        return self.df

    def add_monthly_cycle(self):
        """
        Adds a monthly cycle field to the dataframe.

        This method calculates the monthly cycle days from the 'day' column in
        the dataframe and appends it as a new column named 'month_cycle'. The
        original dataframe is then returned with the added 'month_cycle' column.

        Returns:
            pandas.DataFrame: The modified dataframe with an additional
            'month_cycle' column.

        """
        self.df['month_cycle'] = self.df['day']
        return self.df

    def add_yearly_cycle(self):
        """
        Adds a yearly cycle column to the dataframe by extracting the quarter
        of the year from the datetime column.

        Parameters
        ----------
        self : object
            Instance of the class containing the dataframe as an attribute.

        Returns
        -------
        DataFrame
            Modified DataFrame with a newly added column 'yearly_cycle' that
            represents the quarter of the year.
        """
        self.df['yearly_cycle'] = self.df['datetime'].dt.quarter
        return self.df

    def add_closing_moving_avg(self):
        """
        Calculates the moving average of the 'close' column of a DataFrame and rounds it
        to 4 decimal places. Adds a new column 'closing_moving_avg' containing these
        calculated values to the DataFrame and returns the modified DataFrame.

        Returns:
            pandas.DataFrame: The DataFrame with an added 'closing_moving_avg' column.
        """
        self.df['closing_moving_avg'] = self.df['close'].rolling(window=5, min_periods=1).mean().round(4)
        return self.df

    def add_cumulative_change_in_volume(self):
        """
        Adds a cumulative change in volume column to the DataFrame by calculating the
        cumulative sum of the 'volume' column and rounding it to four decimal places.

        Calculates the cumulative sum of the 'volume' column in the DataFrame. The newly
        generated cumulative sum is assigned to a new column named
        'cumulative_change_in_volume'. The method modifies the original DataFrame with
        this additional column and returns it.

        Returns:
            DataFrame: The updated DataFrame containing the newly added
            'cumulative_change_in_volume' column.
        """
        self.df['cumulative_change_in_volume'] = self.df['volume'].cumsum().round(4)
        return self.df

    def add_previous_targets(self):
        """
        Adds columns for previous target values to the DataFrame if enabled in the configuration.

        This method checks the configuration dictionary for the 'previous_hours_target'
        option under 'advanced_indicators'. If the option is set to True, it will add
        new columns to the DataFrame containing the shifted close values for the last
        five hours.

        Args:
            Non arguments are required.

        Raises:
            None

        Returns:
            pandas.DataFrame: DataFrame containing the new columns for previous target
            values, if the feature is enabled in the configuration. Otherwise, the
            original DataFrame is returned unchanged.
        """
        if self.config['advanced_indicators'].get('previous_hours_target', False):
            for i in range(1, 6):
                self.df[f"target-{i}"] = self.df["close"].shift(i)

        return self.df
