import json
import os

class ConfigManager:
    def __init__(self, config_name):
        """
        Inicializa ConfigManager con un archivo de configuración.
        """
        base_dir = os.path.join(os.path.dirname(__file__), '../config')  # Directorio base de configuración
        config_path = os.path.join(base_dir, f'{config_name}.json')

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file {config_path} not found.")

        with open(config_path, 'r') as file:
            self.config = json.load(file)

    def is_feature_active(self, feature_name):
        """
        Comprueba si una característica está activa en la configuración.
        """
        return self.config.get("features", {}).get(feature_name, False)

    def get_parameter(self, param_name, default=None):
        """
        Obtiene un parámetro específico de la configuración.
        """
        return self.config.get("parameters", {}).get(param_name, default)

    def get_section(self, section_name):
        """
        Obtiene una sección completa de la configuración.
        """
        return self.config.get(section_name, {})

    def __getitem__(self, key):
        """
        Hace que ConfigManager sea subscriptable.
        """
        return self.config[key]