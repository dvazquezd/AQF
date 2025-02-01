import os
import json
import pandas as pd
from datetime import datetime
from utils.utils import load_config, get_time_now


class ConfigLogger:
    def __init__(self, config_file='gen_dataset_config', output_dir='data'):
        self.config_file = config_file
        self.output_file = os.path.join(output_dir, 'config_history.csv')
        self.config = load_config(self.config_file)

        # Asegurar que el directorio de salida existe
        os.makedirs(output_dir, exist_ok=True)

    def log_config(self):
        """
        Guarda la configuración de la ejecución en un CSV dentro de 'data/config_history.csv'.
        Cada nueva ejecución se añade como una fila nueva.
        """
        # Obtener fecha y hora actual
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Convertir la configuración en un solo diccionario plano
        config_flat = self.flatten_config(self.config)

        # Agregar la fecha y hora de ejecución
        config_flat['execution_time'] = timestamp

        # Crear DataFrame con la configuración
        df = pd.DataFrame([config_flat])

        # Si el archivo no existe, se crea; si existe, se añade la nueva ejecución
        if os.path.exists(self.output_file):
            df_existing = pd.read_csv(self.output_file)
            df = pd.concat([df_existing, df], ignore_index=True)

        # Guardar en CSV
        df.to_csv(self.output_file, index=False, encoding='utf-8')
        print(f"{get_time_now()} :: Saving configuration: Configuration saved in {self.output_file}")

    def flatten_config(self, config, parent_key=''):
        """
        Convierte la configuración JSON en un diccionario plano.
        """
        items = {}
        for k, v in config.items():
            new_key = f"{parent_key}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self.flatten_config(v, new_key + '_'))
            else:
                items[new_key] = v
        return items