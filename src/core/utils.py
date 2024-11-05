from box import ConfigBox
import yaml


def load_params(params_file: str) -> ConfigBox:
    """
    Load parameters from a YAML file and return them as a ConfigBox object.

    This function reads a YAML file, loads its contents into a dictionary, and
    wraps it into a ConfigBox object for easier access to the configuration
    parameters.

    Args:
        params_file (str): The path to the YAML file containing configuration
            parameters.

    Returns:
        ConfigBox: A ConfigBox object containing the parameters loaded from the
        YAML file.

    Raises:
        FileNotFoundError: If the specified `params_file` does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.

    Example:
        >>> config = load_params("config.yaml")
        This will load the parameters from the 'config.yaml' file and return a
        ConfigBox object.
    """
    with open(params_file, "r") as f:
        params = yaml.safe_load(f)
    return ConfigBox(params)
