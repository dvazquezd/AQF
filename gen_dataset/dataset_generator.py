import pandas as pd
import utils.utils as ut


class DatasetGenerator:
    def __init__(self, df_news, df_tec):
        """
        Class to manage and process datasets for news and technology-related data.

        Attributes:
        df_news (pd.DataFrame): Copy of the provided news dataset.
        df_tec (pd.DataFrame): Copy of the provided technology dataset.
        df (pd.DataFrame): An empty DataFrame initialized for subsequent operations.
        config (dict): Configuration settings loaded from the 'gen_dataset_config'
        file using the utility function.

        Parameters:
        df_news (pd.DataFrame): The input DataFrame containing news data.
        df_tec (pd.DataFrame): The input DataFrame containing technology data.
        """
        self.df_news = df_news.copy()
        self.df_tec = df_tec.copy()
        self.df = pd.DataFrame()
        self.config = ut.load_config('gen_dataset_config')

    def complete_missing_times(self):
        """
        Completes missing timestamps in the news dataset by aligning it with the technical dataset.

        This method ensures that all timestamps present in the technical dataset (`df_tec`)
        are also present in the news dataset (`df_news`). If timestamps are missing in `df_news`,
        they are added with default values.

        Returns:
            None: Updates `self.df` with the aligned dataset.
        """
        self.df_news['datetime'] = pd.to_datetime(self.df_news['datetime'])
        self.df_tec['datetime'] = pd.to_datetime(self.df_tec['datetime'])

        # Create a full date range based on the technical dataset
        full_range = pd.DataFrame({'datetime': self.df_tec['datetime'].unique()})

        # Merge ensuring all `df_tec` timestamps exist in `df_news`
        self.df = full_range.merge(self.df_news, on='datetime', how='left')

        # Fill missing values with 0
        self.df.fillna(0, inplace=True)

    def aggregate_previous_hours(self):
        """
        Aggregates news sentiment data over a specified number of previous hours.

        If enabled in the configuration, this method applies a rolling window over
        numerical news data, calculating the mean over a specified number of hours.

        Returns:
            DataFrame: The updated dataset with aggregated numerical columns.
        """
        if self.config['news_aggregate_hours'].get('aggregate_news_execute', False):
            hours = self.config['news_aggregate_hours']['aggregate_news_horus']
            # Identify numeric columns excluding datetime
            numeric_columns = self.df.select_dtypes(include=['number']).columns.tolist()

            # Ensure data is sorted by 'datetime'
            self.df = self.df.sort_values(by='datetime')

            if not numeric_columns:
                print('No numeric columns to aggregate.')
                return self.df

            # Apply a rolling window to compute the mean over the last 'hours' hours
            self.df.set_index('datetime', inplace=True)
            self.df[numeric_columns] = self.df[numeric_columns].rolling(f'{hours}h', min_periods=1).mean()
            self.df.reset_index(inplace=True)

            return self.df

    def merge_datasets(self):
        """
        Merges the technical and news datasets, and calculates target labels.

        This method merges the processed news dataset (`self.df`) with the technical dataset (`df_tec`)
        based on timestamps. It also calculates the target variable for predicting future price changes.

        Returns:
            DataFrame: The final merged dataset with calculated target values.
        """
        self.df = pd.merge(self.df_tec, self.df, on='datetime', how='left')
        if 'close' in self.df.columns:  # Ensure the 'close' column exists
            # 1 if price goes up, 0 if it goes down
            self.df = self.df.sort_values(by='datetime')
            self.df['target'] = (self.df['close'].shift(-1) > self.df['close']).astype(int)

            self.df['close_pct_change'] = (
                (self.df['close'] - self.df['close'].shift(1)) / self.df['close'].shift(1)).round(6)

        self.df = self.df.sort_values(by='datetime').drop_duplicates(subset=['datetime'], keep='last')

        self.df.to_csv('data/gen_data.csv',encoding='utf-8',index=False)

        return self.df
