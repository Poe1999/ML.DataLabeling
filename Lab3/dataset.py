#import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Способ 1: Через переменные окружения
#os.environ['KAGGLE_USERNAME'] = 'poelll'
#os.environ['KAGGLE_KEY'] = '2d60cb02edbada844ff02d5a05abf8ce'

# Инициализация API
api = KaggleApi()
api.authenticate()

# Скачивание датасета
try:
    api.dataset_download_files(
        'irkaal/english-premier-league-results',
        path='./data',
        unzip=True
    )
    print("Датасет успешно скачан!")
except Exception as e:
    print(f"Ошибка: {e}")