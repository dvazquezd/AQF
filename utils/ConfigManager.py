import json
import os

class ConfigManager:
    def __init__(self, config_name):
        """
            Initializes a configuration loader and loads configuration data from a
            JSON file. The file is expected to be located under a specified directory
            and named after the provided configuration name. If the file does not
            exist, an exception is raised.

            Attributes:
                config: dict
                    Dictionary containing the loaded configuration data from the
                    specified JSON file.

            Parameters:
                config_name: str
                    Name of the configuration file (without extension) to be loaded.
            Raises:
                FileNotFoundError:
                    Raised when the specified configuration file cannot be found at
                    the expected path.
        """
        base_dir = os.path.join(os.path.dirname(__file__), '../config')  # Directorio base de configuración
        config_path = os.path.join(base_dir, f'{config_name}.json')

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file {config_path} not found.")

        with open(config_path, 'r') as file:
            self.config = json.load(file)

    def is_feature_active(self, feature_name):
        """
            Checks whether a specific feature is active by looking up its status in
            the configuration settings.

            Args:
                feature_name (str): The name of the feature to check.

            Returns:
                bool: True if the feature is active, otherwise False.
        """
        return self.config.get("features", {}).get(feature_name, False)

    def get_parameter(self, param_name, default=None):
        """
            Retrieve a parameter value from a configuration dictionary.

            This method searches for a specified parameter name within a
            nested "parameters" dictionary inside the `config` attribute. If the
            parameter is not found, it returns a provided default value.

            Args:
                param_name (str): The name of the parameter to retrieve.
                default (Any, optional): The default value to return if the
                    parameter is not found. Defaults to None.

            Returns:
                Any: The value of the specified parameter, or the default
                    value if the parameter is not found.
        """
        return self.config.get("parameters", {}).get(param_name, default)

    def get_section(self, section_name):
        """
            Fetches and returns the configuration values of the specified section.

            The method retrieves configuration details from the internal 'config'
            dictionary. If the specified section does not exist, it returns an empty
            dictionary by default, ensuring graceful handling of missing configuration
            sections.

            Parameters:
            section_name: str
                The name of the section whose configuration data is to be retrieved.

            Returns:
            dict
                A dictionary containing the configuration values for the specified
                section. If the section does not exist, an empty dictionary is
                returned.
        """
        return self.config.get(section_name, {})

    def __getitem__(self, key):
        """
            Retrieves an item from the configuration based on the provided key.

            This method allows access to elements in the configuration dictionary by
            their respective key. The key must exist within the configuration; otherwise,
            an exception will be raised.

            Parameters:
                key: The key corresponding to the item to retrieve from the configuration.

            Returns:
                The value associated with the provided key in the configuration.

            Raises:
                KeyError: If the specified key does not exist in the configuration.
        """
        return self.config[key]