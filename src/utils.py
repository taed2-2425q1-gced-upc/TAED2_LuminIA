import configparser


def load_config(path):
    config = configparser.ConfigParser(allow_no_value=True, delimiters=('=', ':'), strict=False)
    config.optionxform = str  # Mantener la capitalización exacta de las opciones
    config.read(path, encoding='utf-8')
    return config