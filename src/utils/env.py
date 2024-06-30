"""
Environment
"""
from typing import Dict

import yaml


def load_config(filepath="../../.secret.config.yaml") -> Dict:
    """

    :param filepath:
    :return:
    """
    try:
        with open(filepath) as yf:
            return yaml.safe_load(yf)
    except Exception as e:
        print(">>> e", e)
        return {}


config = load_config()

# configs
QWEN_API_KEY = config["qwen_api_key"]
CALENDAR_API_KEY = config["calendar_api_key"]
WEATHER_API_KEY = config["weather_api_key"]
COHERE_API_KEY = config["cohere_api_key"]


if __name__ == '__main__':
    load_config()
