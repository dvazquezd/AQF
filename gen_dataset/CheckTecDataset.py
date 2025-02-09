import pandas as pd
import utils.utils as ut

class CheckTecDataset:
    def __init__(self, df):
        """
        A class responsible for initializing and preparing a data frame for further processing.

        Attributes:
        config : dict
            Configuration settings loaded from the 'gen_dataset_config' file.
        original_df : pandas.DataFrame
            A copy of the original data frame provided during initialization.
        df : pandas.DataFrame
            A working copy of the data frame provided during initialization, used for
            modifications and processing.
        target_ticker : str
            The target ticker symbol determined by the get_target_ticker method.

        Parameters:
        df : pandas.DataFrame
            The data frame to be initialized, processed, and utilized within the class.
        """
        self.config = ut.load_config('gen_dataset_config')
        self.original_df = df.copy()
        self.df = df.copy()
        self.target_ticker = self.get_target_ticker()

    def apply_corrections(self):
        """
        Applies a series of data correction methods to the DataFrame based on the configuration
        settings provided. The methods include forward fill, backward fill, moving average,
        and marking incomplete days. Each method is applied only if enabled in the configuration.

        Returns
        -------
        DataFrame
            The corrected DataFrame with applied methods based on the configuration.

        Raises
        ------
        KeyError
            If required configuration keys are missing.
        """
        if self.config['tec_correction_methods'].get('forward_fill', False):
            self.df = self.forward_fill()
        if self.config['tec_correction_methods'].get('backward_fill', False):
            self.df = self.backward_fill()
        if self.config['tec_correction_methods'].get('moving_average', False):
            self.df = self.moving_average()
        if self.config['tec_correction_methods'].get('mark_incomplete_days', False):
            self.df = self.mark_incomplete_days()

        return self.df

    def apply_economic_indicators(self):
        """
        apply_economic_indicators(self)

        Apply economic indicators to the DataFrame based on the configuration. This method ensures that
        economic indicators configured as disabled are removed from the DataFrame while leaving enabled
        ones unaltered.

        Raises:
            ValueError: If the DataFrame (`self.df`) is None or empty.
        """
        # Verify that self.df is a valid DataFrame
        if self.df is None or self.df.empty:
            raise ValueError('The technical dataset is empty or has not been initialized correctly')

        # Retrieve economic indicators from the configuration
        for indicator, enabled in self.config['tec_economic_indicators'].items():
            if not enabled and indicator in self.df.columns:
                self.df.drop(columns=[indicator], inplace=True)

    def calculate_missing_indicators(self):
        """
        Calculates missing technical indicators for the dataset based on configuration.

        This method computes selective technical indicators like Simple Moving Average (SMA),
        Relative Strength Index (RSI), and Moving Average Convergence/Divergence (MACD)
        for the dataset provided in the object. It evaluates the configuration settings
        to decide which indicators to calculate and applies the respective calculation
        methods for specific periods.

        Returns:
            pandas.DataFrame: A DataFrame with the computed technical indicators
            appended or updated.
        """
        if self.config['tec_calculate_indicators'].get('sma', False):
            for period in [5, 10, 12]:
                self.df = self.calculate_sma_partial(period)

        if self.config['tec_calculate_indicators'].get('rsi', False):
            for period in [5, 7, 9]:
                self.df = self.calculate_rsi_partial(period)

        if self.config['tec_calculate_indicators'].get('macd', False):
            self.df = self.calculate_macd_partial()

        return self.df

    def apply_date_time_actions(self):
        """
        apply_date_time_actions(self)

        This method applies a series of date-time related transformations to the dataframe
        held within the object. The operations to be applied are determined by the
        configuration specified in the 'config' attribute of the object. These transformations
        include splitting dates, filling missing temporal values, and adding temporal
        features. Each operation is conditionally executed based on the configuration
        settings.

        Returns:
            pandas.DataFrame: The modified dataframe after applying the specified
            date-time transformations.
        """
        if self.config['global_date_time_actions'].get('date_split', False):
            self.df = self.split_date()

        if self.config['global_date_time_actions'].get('fill_missing_days', False):
            self.df = self.fill_missing_days()

        if self.config['global_date_time_actions'].get('fill_missing_hours', False):
            self.df = self.fill_missing_hours()

        if self.config['global_date_time_actions'].get('add_temporal_features', False):
            self.df = self.add_temporal_features()

        return self.df

    def forward_fill(self):
        """
        This method performs forward filling of missing data in a DataFrame. It replaces NaN
        values in the DataFrame by propagating the value from previous rows downwards, ensuring
        that gaps in the data are filled with the most recent non-NaN value.

        Returns:
            pandas.DataFrame: A DataFrame with forward-filled missing values.
        """
        return self.df.fillna(method='ffill')

    def backward_fill(self):
        """
        Performs backward fill on the DataFrame.

        This method replaces NaN values in the DataFrame using the backward fill
        method, which propagates the next valid value to fill gaps.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with NaN values replaced using backward filling.
        """
        return self.df.fillna(method='bfill')

    def moving_average(self):
        """
        Calculates the moving average for numeric columns in a DataFrame.

        This method operates on the DataFrame stored in the 'df' attribute. It
        applies a rolling window calculation with a specified window size to
        compute the moving average for each numeric column. If a column's data
        type is not numeric, it remains unchanged. Missing values in the numeric
        columns are also handled using the rolling average.

        Returns
        -------
        DataFrame
            A new DataFrame where each numeric column contains the computed
            moving average values, and non-numeric columns are unaffected.
        """
        return self.df.apply(
            lambda col: col.fillna(col.rolling(window=5, min_periods=1).mean())
            if col.dtype in [float, int] else col
        )

    def mark_incomplete_days(self):
        """
        Marks incomplete records in the dataset and removes them.

        This function identifies rows with incomplete data by checking for
        any null values across all columns. It then flags those rows as
        "incomplete" in a new column called "is_incomplete". Following this,
        rows marked as incomplete are removed from the dataset.

        Returns
        -------
        DataFrame
            The dataset after marking and removing incomplete records.
        """
        self.df['is_incomplete'] = self.df.isnull().any(axis=1)
        self.df = self.remove_incomplete_records()

        return self.df

    def calculate_sma_partial(self, period, column='close'):
        """
        Calculate the Simple Moving Average (SMA) for a specific period and column in the dataframe
        and update the dataframe with the calculated SMA.

        Updates the dataframe to include a new column for SMA if it doesn't already exist. The SMA
        is calculated using a rolling window over the specified `period`. The method ensures that
        SMA values are only calculated for rows where the SMA column is not already populated. The
        calculated SMA values are rounded to four decimal places.

        Args:
            period (int): The number of data points to consider while calculating the rolling SMA.
            column (str, optional): The name of the column in the dataframe to use for SMA
                calculation. Defaults to "close".

        Returns:
            pandas.DataFrame: The dataframe updated with the SMA values in a new column named
            based on the format `sma_<period>`.
        """
        sma_column = f'sma_{period}'
        if sma_column not in self.df.columns:
            self.df[sma_column] = pd.NA

        sma_series = self.df[column].rolling(window=period, min_periods=1).mean()
        self.df.loc[self.df[sma_column].isnull(), sma_column] = sma_series[self.df[sma_column].isnull()]
        self.df[sma_column] = self.df[sma_column].round(4)

        return self.df

    def calculate_rsi_partial(self, period, column='close'):
        """
        Calculates the Relative Strength Index (RSI) for a given period and updates the corresponding column
        in the dataframe. RSI is a momentum oscillator that measures the speed and change of price movements.

        RSI is applied to a specific column of the dataframe (default is 'close') and calculates the RSI
        values for the defined period. The function partially updates rows in the dataframe only
        where RSI values are not already present. RSI values are rounded to 4 decimal places before
        being updated.

        Parameters:
            period (int): The lookback period for calculating RSI.
            column (str): The column name in the dataframe to compute RSI on. Defaults to "close".

        Returns:
            pandas.DataFrame: The dataframe with an additional or updated RSI column.

        Raises:
            None
        """
        rsi_column = f'rsi_{period}'
        if rsi_column not in self.df.columns:
            self.df[rsi_column] = pd.NA

        delta = self.df[column].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi_series = 100 - (100 / (1 + rs))
        self.df.loc[self.df[rsi_column].isnull(), rsi_column] = rsi_series[self.df[rsi_column].isnull()]
        self.df[rsi_column] = self.df[rsi_column].round(4)

        return self.df

    def calculate_macd_partial(self, short_window=12, long_window=26, signal_window=9, column='close'):
        """
        Calculates the MACD (Moving Average Convergence Divergence) indicator and its related components
        for a financial time series. The method computes the MACD line, Signal line, and Histogram
        using the specified short, long, and signal window periods. If MACD, Signal, or Histogram values
        already exist in the DataFrame as null, they will be replaced with newly calculated values.

        Parameters:
            short_window (int): The period for the short-term exponential moving average.
            long_window (int): The period for the long-term exponential moving average.
            signal_window (int): The period for the signal line's exponential moving average.
            column (str): The column name in the DataFrame from which the MACD is calculated.

        Returns:
            pandas.DataFrame: DataFrame with updated columns ("MACD", "MACD_Signal", and "MACD_Hist") containing
            the calculated MACD, Signal line, and Histogram values.
        """
        if 'MACD' not in self.df.columns:
            self.df['MACD'] = pd.NA
        if 'MACD_Signal' not in self.df.columns:
            self.df['MACD_Signal'] = pd.NA
        if 'MACD_Hist' not in self.df.columns:
            self.df['MACD_Hist'] = pd.NA

        short_ema = self.df[column].ewm(span=short_window, adjust=False).mean()
        long_ema = self.df[column].ewm(span=long_window, adjust=False).mean()
        macd_series = short_ema - long_ema
        macd_signal_series = macd_series.ewm(span=signal_window, adjust=False).mean()
        macd_hist_series = macd_series - macd_signal_series

        # Fill only the null values
        self.df.loc[self.df['MACD'].isnull(), 'MACD'] = macd_series[self.df['MACD'].isnull()].round(4)
        self.df.loc[self.df['MACD_Signal'].isnull(), 'MACD_Signal'] = macd_signal_series[self.df['MACD_Signal'].isnull()].round(4)
        self.df.loc[self.df['MACD_Hist'].isnull(), 'MACD_Hist'] = macd_hist_series[self.df['MACD_Hist'].isnull()].round(4)

        return self.df

    def remove_incomplete_records(self):
        """
        Removes incomplete records from the dataset.

        This method identifies and removes rows marked as incomplete in the
        'dataset' based on the presence of an 'is_incomplete' column. The
        number of removed records is printed for logging purposes. If the
        'is_incomplete' column does not exist, the dataset remains unchanged
        and a message indicating no rows were removed is printed.

        Returns:
            pandas.DataFrame: The modified dataset after removing incomplete
            rows. If no rows were removed or the 'is_incomplete' column does
            not exist, the original dataset is returned.
        """
        if 'is_incomplete' in self.df.columns:
            original_size = len(self.df)
            self.df = self.df[~self.df['is_incomplete']].reset_index(drop=True)
            removed = original_size - len(self.df)
            print(f'{ut.get_time_now()} :: Dataset generation: Deleted rows: {removed}')

            # Deleting 'is_incomplete' colum
            self.df = self.df.drop(columns=['is_incomplete'])

            return self.df
        else:
            print(f'{ut.get_time_now()} ::  Dataset Generation: The is_incomplete row does not exists in dataset. Rows have not been removed.')

    def split_date(self):
        """
        Splits a 'datetime' column in the dataset into separate derived columns for day, month, year, and hour, while
        removing unnecessary columns 'date' and 'year_month'. Ensures that the 'datetime' column exists and is correctly
        formatted before proceeding with processing.

        Raises
        ------
        ValueError
            Raised if the 'datetime' column does not exist in the dataset.

        Parameters
        ----------
        None

        Returns
        -------
        pd.DataFrame
            Modified DataFrame with additional derived columns: 'day', 'month', 'year', 'time'. Removes any columns named
            'date' and 'year_month' if they exist.
        """
        # Verifying dataset existence
        if 'datetime' not in self.df.columns:
            raise ValueError(f'{ut.get_time_now()} ::  Dataset generation: The dataset has not datetime column. Cannot split date.')

        # Convert datetime if it is not in the correct format.
        self.df['datetime'] = pd.to_datetime(self.df['datetime'], errors='coerce')

        # Remove unnecessary fields
        self.df = self.df.drop(columns=['date', 'year_month'], errors='ignore')

        # Create new fields derived from datetime
        self.df['day'] = self.df['datetime'].dt.day
        self.df['month'] = self.df['datetime'].dt.month
        self.df['year'] = self.df['datetime'].dt.year
        self.df['time'] = self.df['datetime'].dt.hour

        return self.df

    def fill_missing_hours(self):
        """
        Fills in missing hourly data for each day in the dataset, ensuring the time series is complete.

        Summary:
        This method processes the given dataframe to identify missing hourly entries for each unique day in
        the dataset, and fills the missing hours with data. When filling in missing hours, the values are
        extrapolated either from the first hour of the day (for missing hours before the available data) or
        from the last hour of the day (for missing hours after the available data). After filling in the gaps,
        the dataset is updated, sorted by date and time, and auxiliary columns used during processing
        are removed.

        Parameters:
        None

        Raises:
        ValueError: If the dataset does not contain the required columns 'year', 'month', 'day', and 'time'.

        Returns:
        pandas.DataFrame
            Updated dataframe with missing hourly data filled in and sorted by date and time.
        """
        # Verify that the necessary fields exist
        required_columns = {'year', 'month', 'day', 'time'}
        if not required_columns.issubset(self.df.columns):
            raise ValueError(f'The dataset must contain the columns: {required_columns}')

        # Generate the 24 hours for each day
        unique_days = self.df[['year', 'month', 'day']].drop_duplicates()

        filled_rows = []
        for year, month, day in unique_days.itertuples(index=False, name=None):
            # Filter the data for the current day
            day_data = self.df[(self.df['year'] == year) & (self.df['month'] == month) & (self.df['day'] == day)]

            # Create a list of complete hours
            all_hours = set(range(24))
            existing_hours = set(day_data['time'].unique())

            missing_hours = sorted(all_hours - existing_hours)

            if not day_data.empty:
                # First and last hour of the day
                first_hour_data = day_data[day_data['time'] == day_data['time'].min()].iloc[0]
                last_hour_data = day_data[day_data['time'] == day_data['time'].max()].iloc[0]

                for hour in missing_hours:
                    row = None
                    if hour < day_data['time'].min():  # Before the first hour
                        row = first_hour_data.copy()
                    elif hour > day_data['time'].max():  # After the last hour
                        row = last_hour_data.copy()

                    if row is not None:
                        row['time'] = hour
                        filled_rows.append(row)

        # Add the filled rows to the original dataset
        if filled_rows:
            self.df = pd.concat([self.df, pd.DataFrame(filled_rows)], ignore_index=True)

        # Ensure that 'time' is interpreted as an hour
        self.df['hour'] = self.df['time'].astype(int)

        # Reconstruct the 'datetime' field
        self.df['datetime'] = pd.to_datetime(self.df[['year', 'month', 'day', 'hour']])

        # Sort by date and time
        self.df.sort_values(by=['year', 'month', 'day', 'hour']).reset_index(drop=True)

        # Remove the auxiliary column 'hour'
        if 'hour' in self.df.columns:
            self.df = self.df.drop(columns=['hour'])

        return self.df

    def fill_missing_days(self):
        """
        Fills in missing days in the dataset by calculating and appending rows for absent dates, based
        on the last known data prior to the missing day. Preserves the structure of the dataset and
        ensures the date-time continuity by generating hour-wise data for the imputed days.

        Raises:
            ValueError: If the required columns 'year', 'month', 'day', or 'time' are missing in the dataset.

        Returns:
            The dataset (pandas.DataFrame) with missing days filled, sorted chronologically by 'datetime' column.
        """
        # Verify that the required fields exist.
        required_columns = {'year', 'month', 'day', 'time'}
        if not required_columns.issubset(self.df.columns):
            raise ValueError(f'The dataset must contain the required columns: {required_columns}')

        # Convert to datetime to facilitate date handling
        self.df['datetime'] = pd.to_datetime(self.df[['year', 'month', 'day']]) + pd.to_timedelta(self.df['time'],
                                                                                                  unit='h')

        # Generate a full date range excluding the last day
        full_date_range = pd.date_range(
            start=self.df['datetime'].min().normalize(),
            end=self.df['datetime'].max().normalize() - pd.Timedelta(days=1),
            freq='d'
        )

        # Identify existing days
        existing_dates = self.df['datetime'].dt.date.unique()

        # Detect missing days
        missing_dates = sorted(set(full_date_range.date) - set(existing_dates))

        # Generate rows for missing days
        filled_rows = []
        for missing_date in missing_dates:
            # Last available day before the missing date
            previous_day_data = self.df[self.df['datetime'].dt.date < missing_date]
            if previous_day_data.empty:
                continue

            last_day_data = previous_day_data[previous_day_data['datetime'] == previous_day_data['datetime'].max()]
            for hour in range(24):  # Generate all 24 hours for the missing day
                row = last_day_data.iloc[0].copy()
                row['year'], row['month'], row['day'], row[
                    'time'] = missing_date.year, missing_date.month, missing_date.day, hour
                row['datetime'] = pd.Timestamp(year=row['year'], month=row['month'], day=row['day'],
                                               hour=row['time'])  # Fix datetime column
                filled_rows.append(row)

        # Add generated rows to the original dataset
        if filled_rows:
            self.df = pd.concat([self.df, pd.DataFrame(filled_rows)], ignore_index=True)

        # Sort and finalize datetime column
        self.df = self.df.sort_values(by=['datetime']).reset_index(drop=True)

        return self.df

    def add_temporal_features(self):
        """
        Adds temporal features to the dataset for better analysis and categorization of time-dependent data.

        This method extracts several temporal features from a dataset containing a `datetime` column, along with other
        time-based classifications. It validates if the required column exists and then calculates the following features:
        day of the week, whether the date falls on a weekend, and market session indicators (pre-market, market,
        post-market) based on time intervals. Market session indicators are further adjusted by zeroing values outside
        of trading hours or during weekends.

        Raises:
            ValueError: Raised if the required column 'datetime' is missing from the dataset.

        Returns:
            DataFrame: The updated DataFrame with the added temporal features.
        """
        if 'datetime' not in self.df.columns:
            raise ValueError(
                'The dataset does not contain the required datetime column to calculate temporal features')

        self.df['day_of_week'] = self.df['datetime'].dt.dayofweek

        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)

        self.df['is_premarket'] = self.df['time'].between(4, 9, inclusive='both').astype(int)
        self.df['is_market'] = self.df['time'].between(10, 16, inclusive='both').astype(int)
        self.df['is_postmarket'] = self.df['time'].between(17, 20, inclusive='both').astype(int)

        # Set indicators to zero outside working hours and on weekends
        self.df.loc[self.df['is_weekend'] == 1, ['is_premarket', 'is_market', 'is_postmarket']] = 0
        self.df.loc[~self.df['time'].between(4, 20, inclusive='both'), ['is_premarket', 'is_market', 'is_postmarket']] = 0

        return self.df

    def get_target_ticker(self):
        """
        Determines and retrieves the single, unique ticker symbol from the technical dataset.

        This method checks whether the dataset is loaded and contains a single unique ticker symbol. If these conditions are
        not met, it raises an appropriate error. The method is primarily intended to validate that the dataset contains
        exactly one ticker before proceeding with other analyses.

        Raises:
            ValueError: Raised if the dataset (`df`) is not loaded, if the `ticker` column is missing, or if there is not
            exactly one unique ticker in the dataset.

        Returns:
            str: The single unique ticker symbol found in the dataset.
        """
        if self.df is None or 'ticker' not in self.df.columns:
            raise ValueError('The technical dataset is not loaded or does not contain the ticker column')

        tickers = self.df['ticker'].unique()
        if len(tickers) != 1:
            raise ValueError('A single ticker was expected in the technical dataset, but multiple or none were found')

        return tickers[0]

    def delete_no_news_dates(self):
        """
        Deletes rows from the dataframe that are dated before a specified cutoff date.

        This method checks a configuration flag to determine whether rows with no news
        prior to a given cutoff date should be deleted. If the condition is met, it filters
        the dataframe to retain only rows dated on or after the cutoff date, resets the
        index of the dataframe, and calculates the number of removed rows, which is then
        logged via a printed message.

        Parameters
        ----------
        self :
            The calling object context that holds `config` and `df` attributes.

        Raises
        ------
        None

        """
        if self.config['tec_delete_no_news_dates']:
            cutoff_date = pd.Timestamp('2022-03-01')  # Deadline to retain records
            initial_size = len(self.df)

            # Filter records, keeping only those from the deadline onwards
            self.df = self.df[self.df['datetime'] >= cutoff_date].reset_index(drop=True)

            removed_rows = initial_size - len(self.df)
            print(
                f'{ut.get_time_now()} :: Dataset generation: Removed {removed_rows} row on dates previous to {cutoff_date}.')