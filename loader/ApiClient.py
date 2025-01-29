import requests
import os
import time
from datetime import datetime

class ApiClient:

    def __init__(self):
        """
        """

        self.apiKey = os.getenv('ALPHAVKEY')
        self.baseUrl = 'https://www.alphavantage.co/query'
        self.requests_made = []

    def _control_rate_limit(self):
        """
        """
        current_time = time.time()
        self.requests_made = [t for t in self.requests_made if current_time - t < 60]  # Mantiene solo las solicitudes del último minuto
        
        if len(self.requests_made) >= 75:  # Si hemos hecho 75 peticiones en el último minuto
            time_stamp = datetime.now()
            sleep_time = 60 - (current_time - self.requests_made[0])  # Calcular el tiempo que falta para completar el minuto
            print(f"{time_stamp.strftime("%Y-%m-%d %H:%M:%S")} :: Límite de 75 peticiones por minuto alcanzado. Tiempo de espera {sleep_time:.2f} segundos.")
            time.sleep(sleep_time)  # Pausar hasta que podamos hacer más peticiones

        self.requests_made.append(current_time)  # Añadir el timestamp de la nueva petición

    def _get(self, params):
        """
        Executes an HTTP GET request to an API endpoint, managing rate limits, appending
        the API key, and handling errors.

        This method interacts with an external API by performing a GET request with
        the specified query parameters. It includes rate-limiting controls, handles
        potential network or API errors, and parses the response as JSON to extract
        data. If an error is detected in the API response, or the response lacks
        data, the method logs the issue and returns a None value.

        Parameters:
            params (dict): A dictionary of query parameters to be sent with the API
            request.

        Returns:
            dict | None: A dictionary containing the JSON-parsed data of the response
            if successful, or None in case of an error or invalid response.
        """
        self._control_rate_limit()  # Controlar el rate limit antes de hacer la solicitud
        params['apikey'] = self.apiKey
        try:
            response = requests.get(self.baseUrl, params=params)
            response.raise_for_status()
            data = response.json()
            if not data or "Error" in data:
                time_stamp = datetime.now()
                print(f"{time_stamp.strftime("%Y-%m-%d %H:%M:%S")} :: Error in API response: {data}")
                return None
            return data
        except requests.exceptions.RequestException as e:
            time_stamp = datetime.now()
            print(f"{time_stamp.strftime("%Y-%m-%d %H:%M:%S")} :: Request failed: {e}")
            return None

    def get_data(self, function, symbol, **kwargs):
        """
        Fetches and processes data from a remote API using specified function and symbol parameters.

        This method constructs a parameterized request to retrieve data based on a user-provided function
        and symbol. It supports additional customization through optional keyword arguments. It determines
        the appropriate response analysis key depending on the function type and returns the relevant data
        if available. In the case of a missing or invalid response, it logs an appropriate error message
        and returns None.

        Parameters:
            function: str
                The specific operation or data type to request from the API, such as a technical analysis
                function or a data category like 'NEWS_SENTIMENT'.
            symbol: str
                The stock or asset identifier for which data is being requested.
            **kwargs: dict
                Additional optional key-value pairs to customize the API request parameters.

        Returns:
            dict or None:
                If the data is retrieved successfully and the relevant analysis key exists in the response,
                returns the corresponding value as a dictionary. Otherwise, returns None.
        """
        params = {
            'function': function,
            'symbol': symbol
            }
        
        params.update(kwargs)
        time_stamp = datetime.now()
        print(f'{time_stamp.strftime("%Y-%m-%d %H:%M:%S")} - Getting data: {params} ')
        data = self._get(params)

        # Definir la clave correspondiente según el tipo de función solicitada
        if function == 'NEWS_SENTIMENT':  # Supongamos que 'NEWS' es el nombre para solicitar datos de noticias
            analysis_key = 'feed'
        elif function == 'TIME_SERIES_INTRADAY':
            analysis_key = 'Time Series (60min)'
        else:
            analysis_key = f'Technical Analysis: {function}'
    
        # Validar y devolver los datos correspondientes
        if data and analysis_key in data:
            return data[analysis_key]
        else:
            time_stamp = datetime.now()
            print(f'{time_stamp.strftime("%Y-%m-%d %H:%M:%S")} :: Error o datos no encontrados en la respuesta')
            return None
 
    def get_economic_indicator(self, function):
        """
        Fetches economic indicator data based on the specified function.

        This method utilizes the 'function' parameter to construct the API
        request and fetch the corresponding economic indicator data. The request
        is sent using the internal `_get` method with the required parameters.

        Parameters:
          function (str): The name of the economic indicator function to be requested.

        Returns:
          dict: The API response containing economic indicator data.
        """
        params = {
            'function': function
        }
        return self._get(params)
    
    # Métodos para funciones específicas
    def get_intraday_data(self, symbol, month=None, interval='60min', outputsize='full', entitlement ='delayed', extended_hours='true'):
        """
        Fetches intraday time series data for a specific stock symbol.

        This method retrieves intraday stock market data based on the provided
        symbol and optional parameters like month, interval, and additional
        entitlements. It supports fetching data for various time intervals
        and extended market hours. The details of the data are regulated
        by the 'TIME_SERIES_INTRADAY' API functionality.

        Args:
            symbol (str): The stock symbol for the desired data.
            month (Optional[int]): The specific month's data to retrieve
                (e.g., 1 for January), if applicable. Defaults to None.
            interval (str): The time interval between data points. Common
                values include '1min', '5min', '30min', and '60min'.
                Defaults to '60min.'
            outputsize (str): The amount of data to retrieve. Options
                typically include 'compact' (latest few entries) and 'full'
                (complete dataset). Defaults to 'full.'
            entitlement (str): The data entitlement level. Usually
                'delayed' or 'real-time.' Defaults to 'delayed.'
            extended_hours (str): Whether to include extended market hours'
                data. Defaults to 'true.'

        Returns:
            dict: A dictionary containing the intraday time series data
                organized per the provided parameters.

        Raises:
            N/A
        """
        return self.get_data('TIME_SERIES_INTRADAY', symbol, interval=interval, outputsize=outputsize, entitlement=entitlement, extended_hours=extended_hours, month=month)

    def get_sma(self, symbol, month=None, time_period=200, interval='60min', series_type='close', entitlement ='delayed', extended_hours='true'):
        """
        Fetches Simple Moving Average (SMA) data for a given symbol using specified parameters. The method
        uses additional options such as time periods, intervals, series type, entitlement level, and extended
        hours trading adjustments.

        Args:
            symbol (str): The stock symbol for which SMA data will be retrieved.
            month (Optional[int]): The month for which the data is needed. Default is None.
            time_period (int): The number of data points to use for calculating the SMA. The default is 200.
            interval (str): The time interval between data points. Default is '60min'.
            series_type (str): The data series type (e.g., 'close', 'open'). Default is 'close'.
            entitlement (str): The level of entitlement for the data ('delayed', 'real-time').
                               Default is 'delayed'.
            extended_hours (str): Whether to include data from extended trading hours. Default is 'true'.

        Returns:
            The SMA data as retrieved by the function. The format of the returned data depends on the
            underlying API response from `get_data`.

        Raises:
            Any error(s) raised are based on the implementation of the `get_data` method invoked. Errors
            could include invalid parameters, connectivity issues, or API-level exceptions.
        """
        return self.get_data('SMA', symbol, time_period=time_period, interval=interval, series_type=series_type, entitlement=entitlement, extended_hours=extended_hours, month=month)

    def get_macd(self, symbol, month=None, interval='60min', series_type='close', entitlement ='delayed', extended_hours='true'):
        """
        Fetches the Moving Average Convergence Divergence (MACD) indicator for a specified
        symbol and returns its data. The method uses the given parameters to customize
        the request, allowing the user to specify time intervals, series type, data
        entitlement, and other options. MACD is a trend-following momentum indicator
        that shows the relationship between two moving averages of a security’s price.

        Parameters:
            symbol (str): The financial instrument to fetch data for.
            month (Optional[int]): An optional filter for a specific month. Defaults to None.
            interval (str): The time interval between data points, e.g., '60min'. Defaults to '60min'.
            series_type (str): The price series to use, e.g., 'close'. Defaults to 'close'.
            entitlement (str): The data entitlement level, e.g., 'delayed'. Defaults to 'delayed'.
            extended_hours (str): Whether to include extended hours data, e.g., 'true'. Defaults to 'true'.

        Returns:
            Any: The MACD data for the specified symbol and configuration.
        """
        return self.get_data('MACD', symbol, interval=interval, series_type=series_type, entitlement=entitlement, extended_hours=extended_hours, month=month)

    def get_rsi(self, symbol, month=None, time_period=200, interval='60min', series_type='close',  entitlement ='delayed', extended_hours='true'):
        """
        Calculate the Relative Strength Index (RSI) for a given financial instrument. The RSI is a momentum oscillator
        that measures the speed and change of price movements. It is often used to identify overbought or oversold
        conditions in a traded stock or other asset.

        Parameters
        ----------
        symbol : str
            The symbol of the financial instrument for which the RSI is to be calculated.
        month : int, optional
            The month for which the data should be retrieved. If not provided, the data for the latest available month
            will be used.
        time_period : int
            The number of periods to use in the RSI calculation. Defaults to 200.
        interval : str
            The time interval between data points for the RSI calculation. Defaults to '60min'.
        series_type : str
            The type of price data to use for the RSI calculation. Defaults to 'close'.
        entitlement : str
            The data entitlement type, e.g., delayed or real-time. Defaults to 'delayed'.
        extended_hours : str
            Indicates whether to include data from extended trading hours. Accepts 'true' or 'false'. Defaults to 'true'.

        Returns
        -------
        dict
            A dictionary containing the RSI data for the given symbol and specific parameters.

        Raises
        ------
        ValueError
            If any of the input parameters are invalid or do not meet the requirements of the data provider API.
        """
        return self.get_data('RSI', symbol, time_period=time_period, interval=interval, series_type=series_type, entitlement=entitlement, extended_hours=extended_hours, month=month)

    def get_news_sentiment(self, topics='economy_macro', time_from='20240410T0130', time_to='20240415T0130', limit=1000, sort='RELEVANCE'):
        """
        Retrieve sentiment analysis data for news articles based on specified criteria.

        This method fetches sentiment-related data for news articles, offering a
        wide range of filtering options such as topics, time frame, results limit,
        and sorting criteria. It is designed to filter news sentiment effectively
        and return the requested information as structured data.

        Parameters:
            topics (str): The topic or category for the news articles. Defaults to
                'economy_macro'.
            time_from (str): The start of the time range for fetching news sentiment
                data, specified in the format 'YYYYMMDDTHHMM'.
            time_to (str): The end of the time range for fetching news sentiment
                data, specified in the format 'YYYYMMDDTHHMM'.
            limit (int): The maximum number of news sentiment data entries to be
                returned. Defaults to 1000.
            sort (str): The criteria by which the results should be sorted. Defaults
                to 'RELEVANCE'.

        Returns:
            Any: The fetched data containing news sentiment analysis information,
            formatted according to the specified parameters.
        """
        return self.get_data('NEWS_SENTIMENT', None, topics=topics, time_from=time_from, time_to=time_to, limit=limit, sort=sort)
