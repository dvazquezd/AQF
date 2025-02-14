import pandas as pd
import utils.utils as ut

class CheckNewsDataset:
    def __init__(self, df, target_ticker):
        """
        Initializes the DatasetGenerator class with a DataFrame and a target ticker.

        Attributes:
            config (dict): Configuration dictionary loaded from the
                'gen_dataset_config' file.
            target_ticker (str): The ticker symbol that is the focus of the dataset.
            original_df (DataFrame): A copy of the input pandas DataFrame provided
                during initialization.
            df (DataFrame): A filtered pandas DataFrame containing only rows that
                correspond to the specified target ticker.

        Parameters:
            df (DataFrame): The input pandas DataFrame to be processed.
            target_ticker (str): The ticker symbol to filter the input DataFrame by.
        """
        self.config = ut.load_config('gen_dataset_config')
        self.target_ticker = target_ticker
        self.original_df = df.copy()
        self.df = self.filter_by_ticker()

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
        self.df = self.original_df[self.original_df['ticker'] == self.target_ticker].copy()

        return self.df

    def generate_ticker_features(self):
        """
        Processes and transforms the DataFrame associated with an instance to generate ticker-level
        features by performing numeric conversions, aggregations, and renaming columns for clarity.

        Raises:
            KeyError: If any required column is missing in the DataFrame.

        Returns:
            pd.DataFrame: A processed DataFrame with aggregated and renamed columns, sorted by datetime.
        """
        # Ensure that the columns required for numerical calculations are of the appropriate type
        numeric_columns = ['overall_sentiment_score', 'relevance_score', 'ticker_sentiment_score',
                           'affected_topic_relevance_score']
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Group by hour and calculate aggregates.
        self.df = self.df.groupby('datetime').agg({
            'overall_sentiment_score': lambda x: round(x.mean(), 6),
            'relevance_score': lambda x: round(x.mean(), 6),
            'ticker_sentiment_score': lambda x: round(x.mean(), 6),
            'affected_topic_relevance_score': lambda x: round(x.mean(), 6),
            'title': 'nunique'
        }).reset_index()

        # Rename the count column for better clarity.
        self.df = self.df.rename(columns={'overall_sentiment_score': 'ticker_ossm'})
        self.df = self.df.rename(columns={'relevance_score': 'ticker_rsm'})
        self.df = self.df.rename(columns={'ticker_sentiment_score': 'ticker_ssm'})
        self.df = self.df.rename(columns={'affected_topic_relevance_score': 'ticker_atrsm'})
        self.df = self.df.rename(columns={'title': 'ticker_nc'})
        self.df = self.df.sort_values(by='datetime')

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

        self.df = self.df.fillna(0)

    def intermediate_dataset(self,df):
        if self.df is None:
            return df
        return pd.merge(self.df, df, on='datetime', how='outer')

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
        Generate aggregated global metrics for news data.

        This function processes the original dataframe to compute global metrics such
        as the overall sentiment score mean and relevance score mean for each datetime,
        aggregated across all news items. The processed metrics are then transformed
        into an intermediate dataset for further use.

        Args:
            None

        Returns:
            pd.DataFrame: The dataframe containing aggregated global metrics for news data
            with columns for datetime, all_news_ossm (overall sentiment score mean), and
            all_news_rsm (relevance score mean).
        """
        news_data = self.original_df[
            ['datetime', 'title', 'overall_sentiment_score', 'relevance_score']].drop_duplicates()
        news_data[['overall_sentiment_score', 'relevance_score']] = news_data[
            ['overall_sentiment_score', 'relevance_score']].apply(pd.to_numeric, errors='coerce')

        global_metrics = news_data.groupby('datetime').agg({
            'overall_sentiment_score': lambda x: round(x.mean(), 6),
            'relevance_score': lambda x: round(x.mean(), 6)
        }).rename(columns={
            'overall_sentiment_score': 'all_news_ossm',
            'relevance_score': 'all_news_rsm'
        }).reset_index()

        self.df = self.intermediate_dataset(global_metrics)

        return self.df