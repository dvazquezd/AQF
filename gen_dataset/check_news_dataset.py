import pandas as pd
import utils.utils as ut

class CheckNewsDataset:
    def __init__(self, df, target_ticker):
        """
        This class is responsible for managing and processing a financial dataset. It provides
        functionality to filter the dataset based on a specific target stock ticker and store
        the filtered data along with other related configurations and properties.

        Attributes:
            config (dict): A dictionary containing configuration settings loaded at initialization.
            target_ticker (str): The stock ticker used to filter the dataset.
            original_df (pandas.DataFrame): A copy of the original input dataframe before filtering.
            df (pandas.DataFrame): The filtered dataframe containing only the rows related to the
            target stock ticker.

        Args:
            df (pandas.DataFrame): The input dataframe containing financial data.
            target_ticker (str): The stock ticker to filter the dataset by.
        """
        self.config = ut.load_config('gen_dataset_config')
        self.target_ticker = target_ticker
        self.original_df = df.copy()
        self.filtered_df = self.filter_by_ticker()
        self.df = pd.DataFrame()

    def filter_by_ticker(self):
        """
        Filters news data for a specific ticker and creates a copy of the filtered dataframe.

        This method isolates rows in a dataframe where the 'ticker' column matches the
        specified target ticker value. It then creates and returns a copy of the filtered
        dataframe to prevent modifications to the original dataset.

        Returns:
            DataFrame: A new dataframe containing only the rows where the 'ticker' column
            matches the target ticker value.
        """
        # Filter news for the target ticker and create a copy
        self.filtered_df = (self.original_df[self.original_df['ticker'] == self.target_ticker]).copy()

        return self.filtered_df

    def generate_ticker_features(self):
        """
        This method generates ticker features based on the configuration provided. Depending on the flags set
        in the 'generate_ticker_features' section of the configuration, it applies different operations to
        adjust the dataframe, such as weighting ticker metrics or averaging specific ticker values.

        Arguments:
            None

        Returns:
            None
        """
        if self.config['generate_ticker_features'].get('weight_ticker_value',False):
            self.df = self.weight_ticker_metrics()
        if self.config['generate_ticker_features'].get('average_ticker_value',False):
            self.df = self.average_ticker_value()

    def weight_ticker_metrics(self):
        """
        Calculates and aggregates weighted ticker metrics based on sentiment and relevance scores,
        grouping by time intervals. This method computes weighted metrics, aggregates them, and adds
        additional statistics such as the count of unique news articles.

        Parameters
        ----------
        This method does not take any parameters. It operates on the `self.filtered_df` attribute
        and updates the `self.df` attribute.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the aggregated and weighted ticker metrics along with the count
            of unique news articles, grouped by `datetime`.

        Raises
        ------
        This method does not explicitly raise exceptions but may propagate errors from operations
        such as data type conversion or aggregation.

        Examples
        --------
        Examples are not provided in this documentation.

        Notes
        -----
        - The method assumes that `self.filtered_df` contains columns necessary for calculating
          weighted metrics: 'overall_sentiment_score', 'ticker_sentiment_score',
          'affected_topic_relevance_score', and 'relevance_score'.
        - Computes numerical conversions for specific columns to ensure proper aggregation and
          processing.
        - Weighted metrics are aggregated by finding the average per time interval, rounded to
          six decimal places.
        """
        # Ensure that the columns required for numerical calculations are of the appropriate type
        df = self.filtered_df.copy()
        numeric_columns = ['overall_sentiment_score', 'relevance_score', 'ticker_sentiment_score',
                           'affected_topic_relevance_score']
        for col in numeric_columns:
            if col in self.filtered_df.columns:
                self.filtered_df[col] = pd.to_numeric(self.filtered_df[col], errors='coerce')

        # Weight metrics by `relevance_score`
        df['w_ticker_ossm'] = self.filtered_df['overall_sentiment_score'] * self.filtered_df['relevance_score']
        df['w_ticker_ssm'] = self.filtered_df['ticker_sentiment_score'] * self.filtered_df['relevance_score']
        df['w_ticker_atrsm'] = self.filtered_df['affected_topic_relevance_score'] * self.filtered_df[
            'relevance_score']

        df = df.groupby('datetime').agg({
            'w_ticker_ossm': lambda x: round(x.sum() / max(x.count(), 1), 6),
            'w_ticker_ssm': lambda x: round(x.sum() / max(x.count(), 1), 6),
            'w_ticker_atrsm': lambda x: round(x.sum() / max(x.count(), 1), 6),
            'title': 'nunique'  # Number of unique news articles per hour
        }).reset_index()
        df = df.rename(columns={'title': 'w_ticker_nc'})
        df = df.sort_values(by='datetime')
        self.df = self.intermediate_dataset(df)

        return self.df

    def average_ticker_value(self):
        """
        """
        df = self.filtered_df.copy()
        numeric_columns = ['overall_sentiment_score', 'relevance_score', 'ticker_sentiment_score',
                           'affected_topic_relevance_score']
        for col in numeric_columns:
            if col in self.filtered_df.columns:
                self.filtered_df[col] = pd.to_numeric(self.filtered_df[col], errors='coerce')

        df = df.groupby('datetime').agg({
            'overall_sentiment_score': lambda x: round(x.mean(), 6),
            'relevance_score': lambda x: round(x.mean(), 6),
            'ticker_sentiment_score': lambda x: round(x.mean(), 6),
            'affected_topic_relevance_score': lambda x: round(x.mean(), 6),
            'title': 'nunique'
        }).reset_index()

        # Rename the count column for better clarity.
        df = df.rename(columns={'overall_sentiment_score': 'avg_ticker_ossm'})
        df = df.rename(columns={'relevance_score': 'avg_ticker_rsm'})
        df = df.rename(columns={'ticker_sentiment_score': 'avg_ticker_ssm'})
        df = df.rename(columns={'affected_topic_relevance_score': 'avg_ticker_atrsm'})
        df = df.rename(columns={'title': 'avg_ticker_nc'})
        df = df.sort_values(by='datetime')

        self.df = self.intermediate_dataset(df)

        return self.df

    def normalize_topic_names(self):
        """
        Normalizes topic names within a DataFrame by replacing certain predefined topic
        names with their lowercase and/or underscore-separated equivalents. This allows
        for consistent topic naming conventions across the DataFrame.

        Returns:
            pandas.DataFrame: Modified DataFrame with normalized topic names in the
            'affected_topic' column.
        """
        self.original_df['affected_topic'] = self.original_df['affected_topic'].replace({
            'Blockchain': 'blockchain',
            'Earnings': 'earnings',
            'IPO': 'ipo',
            'Mergers & Acquisitions': 'mergers_and_acquisitions',
            'Financial Markets': 'financial_markets',
            'Economy - Macro': 'economy_macro',
            'Economy - Monetary': 'economy_monetary',
            'Economy - Fiscal': 'economy_fiscal',
            'Energy & Transportation': 'energy_transportation',
            'Finance': 'finance',
            'Life Sciences': 'life_sciences',
            'Manufacturing': 'manufacturing',
            'Real Estate & Construction': 'real_estate',
            'Retail & Wholesale': 'retail_wholesale',
            'Technology': 'technology'
        })

        return self.original_df

    def generate_topic_features(self):
        """
        Generates topic-specific features for the dataset based on the configuration and updates the
        main dataset with the computed metrics. Normalizes topic names and iteratively processes topics
        enabled in the configuration. Fills missing values with 0 in the final dataset.

        Yields:
            No return is expected as the function modifies class instance attributes
        """
        self.original_df = self.normalize_topic_names()

        for topic, enabled in self.config.get('news_topic_features', {}).items():
            if enabled:
                df = self._calculate_topic_metrics(topic)
                self.df = self.intermediate_dataset(df)

        return self.df.fillna(0)

    def _calculate_topic_metrics(self, topic):
        """
        Calculates metrics for a given topic based on non-related news data.

        This function determines and processes data related to a specified topic by
        excluding any news articles associated with the target ticker and ensuring
        only valid topic-related entries are considered. It computes several metrics
        such as mean overall sentiment score, mean affected topic relevance score,
        and the average count of news articles per datetime for the provided topic
        from the filtered data. The computed metrics are returned in a structured
        DataFrame format.

        Parameters:
            topic (str): The topic for which metrics are to be calculated.

        Returns:
            pd.DataFrame: A DataFrame containing calculated metrics by datetime
            with the following columns:
                - '<topic>_ossm': Mean overall sentiment score rounded to 6 decimals.
                - '<topic>_atrsm': Mean affected topic relevance score rounded to 6
                  decimals.
                - '<topic>_nc': Average count of news articles per datetime.
        """
        # Identify titles associated with the target_ticker.
        titles_with_target_ticker = self.original_df[self.original_df['ticker'] == self.target_ticker]['title'].unique()

        # Exclude all news whose titles are related to the target_ticker.
        non_related_news = self.original_df[~self.original_df['title'].isin(titles_with_target_ticker)].copy()

        # Exclude rows that do not have a valid topic in the affected_topic field.
        non_related_news = non_related_news[non_related_news['affected_topic'].notnull()]

        # Filter by topic.
        topic_data = non_related_news[non_related_news['affected_topic'] == topic]

        # Select relevant columns and remove duplicates by datetime.
        topic_data = topic_data[
            ['datetime', 'title', 'overall_sentiment_score', 'affected_topic_relevance_score']].drop_duplicates()

        numeric_columns = ['overall_sentiment_score', 'affected_topic_relevance_score']
        topic_data[numeric_columns] = topic_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # Count the number of news articles per datetime.
        topic_data['news_count'] = topic_data.groupby('datetime')['datetime'].transform('count')

        # Group by datetime and calculate metrics similar to those of the ticker.
        topic_metrics = topic_data.groupby('datetime').agg({
            'overall_sentiment_score': lambda x: round(x.mean(), 6),
            'affected_topic_relevance_score': lambda x: round(x.mean(), 6),
            'news_count': lambda x: x.mean()
        }).rename(columns={
            'overall_sentiment_score': f'{topic}_ossm',
            'affected_topic_relevance_score': f'{topic}_atrsm',
            'news_count': f'{topic}_nc'
        }).reset_index()

        return topic_metrics

    def generate_news_global_metrics(self):
        """
        Generate global metrics based on news data.

        This method processes and aggregates certain metrics from news data related to ticker
        sentiment and relevance. It computes new metrics such as `ticker_score` and combines
        them with global metrics like the overall sentiment score and relevance score across
        the provided data. The resulting dataset is sorted, formatted, and saved as part of
        the object's attributes.

        Returns:
            pd.DataFrame: A processed dataframe containing aggregated news-related
            metrics, including `global_score` and `ticker_score`, indexed by datetime.

        Raises:
            None
        """
        df_ticker = self.filter_by_ticker().copy()
        df_ticker = \
        df_ticker.drop_duplicates(subset=['title', 'datetime', 'relevance_score', 'ticker_sentiment_score'])[
            ['title', 'datetime', 'relevance_score', 'ticker_sentiment_score']
        ]

        df_ticker['relevance_score'] = pd.to_numeric(df_ticker['relevance_score'], errors='coerce')
        df_ticker['ticker_sentiment_score'] = pd.to_numeric(df_ticker['ticker_sentiment_score'], errors='coerce')

        df_ticker['ticker_score'] = (df_ticker['relevance_score'] * df_ticker['ticker_sentiment_score']) * 5
        df_ticker = df_ticker.groupby('datetime', as_index=False).agg({'ticker_score': 'sum'})
        df_ticker = df_ticker.sort_values(by='datetime', ascending=True)

        news_data = self.original_df[['datetime', 'title', 'overall_sentiment_score', 'relevance_score']].drop_duplicates()
        news_data[['overall_sentiment_score', 'relevance_score']] = news_data[
            ['overall_sentiment_score', 'relevance_score']].apply(pd.to_numeric, errors='coerce')
        global_metrics = news_data.groupby('datetime').agg({
            'overall_sentiment_score': lambda x: round(x.mean(), 6),
            'relevance_score': lambda x: round(x.mean(), 6)
        }).rename(columns={
            'overall_sentiment_score': 'all_news_ossm',
            'relevance_score': 'all_news_rsm'
        }).reset_index()

        global_metrics['global_score'] = global_metrics['all_news_ossm'] * global_metrics['all_news_rsm']

        df_fn = pd.merge(global_metrics, df_ticker, on='datetime', how='outer')
        df_fn['ticker_score'] = df_fn['ticker_score'].fillna(0)
        df_fn['global_score'] = df_fn['global_score'].fillna(0)

        df_fn = df_fn[['datetime', 'global_score', 'ticker_score']]
        self.df = self.intermediate_dataset(df_fn)

        return self.df

    def intermediate_dataset(self,df):
        """
        Merges two datasets based on the 'datetime' column.

        This function takes an input dataframe and performs an outer join with
        the instance's dataframe based on the 'datetime' column. It is useful
        for combining datasets where you want to retain all rows from both
        dataframes, aligning on the 'datetime' column.

        Args:
            df (pd.DataFrame): The dataframe to be merged with the current
            instance's dataframe.

        Returns:
            pd.DataFrame: A merged dataframe containing all rows from both
            dataframes, aligned on the 'datetime' column.
        """
        if self.df is None or self.df.empty:
            return df.copy()
        return pd.merge(self.df, df, on='datetime', how='outer')