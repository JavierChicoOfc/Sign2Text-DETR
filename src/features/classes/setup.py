import json
from typing import List, Union


def get_classes() -> Union[List[str], str]:
    """
    Load the list of classes from the config file.

    Returns:
        List[str]: List of class names if successful.
        str: Error message if loading fails.
    """
    try:
        with open("src/config.json") as f:
            config = json.load(f)
        classes = config["classes"]
        assert isinstance(classes, list) and len(classes) > 0, (
            'You need to specify classes inside of config.json e.g. {"classes":["hello", "iloveyou", "hola"]}'
        )
        return classes
    except Exception as e:
        return f"Something went wrong loading your config file: {e}"


def get_colors() -> Union[List[List[int]], str]:
    """
    Load the list of colors from the config file, ensuring one color per class.

    Returns:
        List[List[int]]: List of RGB color lists if successful.
        str: Error message if loading fails.
    """
    try:
        with open("src/config.json") as f:
            config = json.load(f)
        classes = config["classes"]
        colors = config["colors"]
        assert isinstance(colors, list) and len(colors) > 0, (
            'You need to specify colors in RGB inside of config.json e.g. {"colors":[[131, 193, 103], [240, 172, 95]]}'
        )
        assert len(classes) == len(colors), (
            f"Please specify one color per class. You have {len(colors)} colours and {len(classes)} classes"
        )
        return colors
    except Exception as e:
        return f"Something went wrong loading your config file: {e}"
