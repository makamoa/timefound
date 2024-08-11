import os
import yaml
from box import Box

class Config(Box):
    """
    A class to load and access configuration settings from a YAML file using Box.
    """

    def __init__(self, config_path='.config/settings.yaml', frozen_box=False):
        """
        Initializes the Config object by loading the YAML file and setting attributes using Box.

        Args:
            config_path (str): The path to the YAML configuration file.
        """
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        super().__init__(config_data, frozen_box=frozen_box)  # frozen_box=True to make it immutable if needed

if __name__ == '__main__':
    config = Config()
    print(config)