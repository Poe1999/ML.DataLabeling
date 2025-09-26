import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import sqlite3

api = KaggleApi()
api.authenticate()

os.makedirs("./data", exist_ok=True)

dataset_name = "hugomathien/soccer"
api.dataset_download_files(dataset_name, path="./data", unzip=True)

sqlite_files = [f for f in os.listdir('./data') if f.endswith('.sqlite')]
if not sqlite_files:

    sqlite_files = [f for f in os.listdir('./data') if f.endswith('.db')]

if sqlite_files:
    sqlite_path = os.path.join('./data', sqlite_files[0])

    conn = sqlite3.connect(sqlite_path)

    tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql_query(tables_query, conn)

    print("Таблицы в базе данных:")
    print(tables)

    table_name = tables['name'].iloc[0]
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)


    print(f"Размер датасета: {df.shape}")
    print(f"Колонки: {df.columns.tolist()}")
    print(df.head())

matches = pd.read_sql_query("SELECT * FROM Match", conn)

matches['target'] = matches['home_team_goal'] - matches['away_team_goal']

print(f"Данные: {matches.shape}")
print(f"Цель: разница голов (положительная = победа дома)")
print(matches[['home_team_goal', 'away_team_goal', 'target']].head())
conn.close()

