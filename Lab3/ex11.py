import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class PipeLine:
    def __init__(self, data):
        self.data = data.copy()
        self.report = []

    def visualize(self):
        print(f"Размер данных: {self.data.shape}")
        print("\nТипы данных:")
        for dtype, count in self.data.dtypes.value_counts().items():
            print(f"  {dtype}: {count} колонок")

        plt.figure(figsize=(15, 10))

        plt.subplot(2, 3, 1)
        missing = self.data.isnull().sum()
        missing_cols = missing[missing > 0]
        if len(missing_cols) > 0:
            plt.bar(missing_cols.index, missing_cols.values)
            plt.title('попущенные значения)')
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, 'Нет попущенных значений',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('попущенные значения')

        plt.subplot(2, 3, 2)
        numeric_cols = self.data.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            first_numeric = self.data[numeric_cols[0]]
            plt.hist(first_numeric.dropna(), bins=20, alpha=0.7, color = 'darkred')
            plt.title(f'распределение {numeric_cols[0]}')
        else:
            plt.text(0.5, 0.5, 'Нет числовых данных',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Распределение числовых данных')

        plt.subplot(2, 3, 3)
        categorical_cols = self.data.select_dtypes(include='object').columns
        if len(categorical_cols) > 0:
            unique_counts = [self.data[col].nunique() for col in categorical_cols]
            plt.bar(categorical_cols, unique_counts, color = 'darkred')
            plt.title('категориальные значения')
            plt.xticks(rotation=25)
        else:
            plt.text(0.5, 0.5, 'Нет категориальных данных',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('категориальные данные')


        plt.subplot(2, 3, 4)
        if len(numeric_cols) > 0:
            cols_to_plot = numeric_cols[:min(5, len(numeric_cols))]
            data_to_plot = [self.data[col].dropna() for col in cols_to_plot]
            plt.boxplot(data_to_plot, tick_labels=cols_to_plot)
            plt.title('выбросы')
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, 'Нет числовых данных',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('выбросы')

        plt.subplot(2, 3, 5)
        if 'Season' in self.data.columns or 'season' in self.data.columns:
            season_col = 'Season' if 'Season' in self.data.columns else 'season'
            season_counts = self.data[season_col].value_counts()
            plt.bar(season_counts.index, season_counts.values, color = '#BC8F8F')
            plt.title('матчи по сезонам')
            plt.xticks(rotation=90)
        else:
            plt.bar(['Всего'], [len(self.data)])
            plt.title(f'Всего записей: {len(self.data)}')

        plt.subplot(2, 3, 6)
        result_col = None
        for col in ['FTR', 'result', 'Result']:
            if col in self.data.columns:
                result_col = col
                break

        if result_col:
            result_counts = self.data[result_col].value_counts()
            colors= ['#8B0000','#FF4500','#DAA520']
            plt.pie(result_counts.values, labels=result_counts.index, autopct='%1.1f%%', colors = colors)
            plt.title('соотношение результатов')
        else:
            plt.text(0.5, 0.5, 'нет данных о результатах',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('результаты матчей')

        plt.tight_layout()
        plt.show()

    def to_numeric(self):
        print("\nпреобразование данных к числовым типам")

        categorical_cols = self.data.select_dtypes(include='object').columns

        if len(categorical_cols) == 0:
            print("ннет категориальных данных для преобразования")
            return

        for col in categorical_cols:

            converted = pd.to_numeric(self.data[col], errors='coerce')

            if converted.isna().sum() < len(self.data) * 0.5:
                self.data[col] = converted
                print(f"{col} числовой тип")
            else:

                self.data[col] = self.data[col].astype('category').cat.codes
                print(f"{col} категориальные коды")

        self.report.append('Преобразование в числовые форматы завершено')

    def missing_values(self):
        print("\nобработка пропусков")

        missing_before = self.data.isnull().sum().sum()

        if missing_before == 0:
            print("Пропущенные значения отсутствуют")
            return

        print(f"Найдено пропущенных значений: {missing_before}")

        for col in self.data.columns:
            missing_count = self.data[col].isnull().sum()
            if missing_count > 0:
                if self.data[col].dtype in ['float64', 'int64']:
                    self.data[col].fillna(self.data[col].median(), inplace=True)
                    print(f"{col}: {missing_count} пропусков заполнены медианой")
                else:
                    self.data[col].fillna(self.data[col].mode()[0], inplace=True)
                    print(f"✓ {col}: {missing_count} пропусков заполнены модой")

        missing_after = self.data.isnull().sum().sum()
        print(f"Пропущенных значений после обработки: {missing_after}")

        self.report.append('Обработка пропущенных значений завершена')

    def normalize(self):
        print("\nyjhvfkbpfwbz")

        numeric_cols = self.data.select_dtypes(include=np.number).columns

        if len(numeric_cols) == 0:
            print("Нет числовых данных для нормализации")
            return

        scaler = StandardScaler()
        self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])

        print(f"yормализовано {len(numeric_cols)} числовых колонок")

        self.report.append('Нормализация данных завершена')

    def run(self):
        self.visualize()
        self.to_numeric()
        self.missing_values()
        self.normalize()

        return self.data


if __name__ == "__main__":
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    try:
        api.dataset_download_files(
            'irkaal/english-premier-league-results',
            path='./data',
            unzip=True
        )
    except Exception as e:
        print(f"Ошибка: {e}")

    fdf = pd.read_csv('results.csv', encoding='windows-1252', na_values=['NA'])
    print(fdf.head())
    print(fdf.info())


    processor = PipeLine(fdf)
    processed_data = processor.run()


    print("\n первые 5 строк")
    print(processed_data.head())