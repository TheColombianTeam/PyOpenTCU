import configparser, os


def config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.config')
    general_config = configparser.ConfigParser()
    general_config.read(config_path)
    return general_config
