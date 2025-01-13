import requests
import os
import time

class ApiClient:

    def __init__(self):
        self.apiKey = os.getenv('ALPHAVKEY')
        self.baseUrl = 'https://www.alphavantage.co/query'
        self.requests_made = []


    def _control_rate_limit(self):
        """
        Controla el número de peticiones por minuto para evitar sobrepasar el límite.
        """
        current_time = time.time()
        self.requests_made = [t for t in self.requests_made if current_time - t < 60]  # Mantiene solo las solicitudes del último minuto
        
        if len(self.requests_made) >= 75:  # Si hemos hecho 75 peticiones en el último minuto
            sleep_time = 60 - (current_time - self.requests_made[0])  # Calcular el tiempo que falta para completar el minuto
            print(f"Límite de 75 peticiones por minuto alcanzado. Tiempo de espera {sleep_time:.2f} segundos.")
            time.sleep(sleep_time)  # Pausar hasta que podamos hacer más peticiones

        self.requests_made.append(current_time)  # Añadir el timestamp de la nueva petición


    def _get(self, params):
        '''
        '''
        self._control_rate_limit()  # Controlar el rate limit antes de hacer la solicitud
        params['apikey'] = self.apiKey
        try:
            response = requests.get(self.baseUrl, params=params)
            response.raise_for_status()
            data = response.json()
            if not data or "Error" in data:
                print(f"Error in API response: {data}")
                return None
            return data
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None


    def get_data(self, function, symbol, **kwargs):
        '''
        Función genérica para obtener cualquier tipo de dato desde Alpha Vantage.
        '''
        params = {
            'function': function,
            'symbol': symbol
            }
        
        params.update(kwargs)
        print(f'Getting data: {params} ')
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
            print('Error o datos no encontrados en la respuesta')
            return None
        

    def get_economic_indicator(self, function):
        """
        Obtiene un indicador económico como unemployment, CPI, inflation, etc.
        """
        params = {
            'function': function
        }
        return self._get(params)
    

    # Métodos para funciones específicas
    def get_intraday_data(self, symbol, month=None, interval='60min', outputsize='full', entitlement ='delayed', extended_hours='true'):
        '''
        '''
        return self.get_data('TIME_SERIES_INTRADAY', symbol, interval=interval, outputsize=outputsize, entitlement=entitlement, extended_hours=extended_hours, month=month)


    def get_sma(self, symbol, month=None, time_period=200, interval='60min', series_type='close', entitlement ='delayed', extended_hours='true'):
        '''
        '''
        return self.get_data('SMA', symbol, time_period=time_period, interval=interval, series_type=series_type, entitlement=entitlement, extended_hours=extended_hours, month=month)


    def get_macd(self, symbol, month=None, interval='60min', series_type='close', entitlement ='delayed', extended_hours='true'):
        '''
        '''
        return self.get_data('MACD', symbol, interval=interval, series_type=series_type, entitlement=entitlement, extended_hours=extended_hours, month=month)


    def get_rsi(self, symbol, month=None, time_period=200, interval='60min', series_type='close',  entitlement ='delayed', extended_hours='true'):
        '''
        '''
        return self.get_data('RSI', symbol, time_period=time_period, interval=interval, series_type=series_type, entitlement=entitlement, extended_hours=extended_hours, month=month)


    def get_news_sentiment(self, topics='economy_macro', time_from='20240410T0130', time_to='20240415T0130', limit=1000, sort='RELEVANCE'):
        '''
        '''
        return self.get_data('NEWS_SENTIMENT', None, topics=topics, time_from=time_from, time_to=time_to, limit=limit, sort=sort)
