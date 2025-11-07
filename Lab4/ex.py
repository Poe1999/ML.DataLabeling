import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import TomekLinks
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

try:
    api.dataset_download_files(
        'irkaal/english-premier-league-results',
        path='./data',
        unzip=True
    )
    print("Датасет успешно скачан")
except Exception as e:
    print(f"Ошибка: {e}")

df = pd.read_csv('results.csv', encoding='windows-1252', na_values=['NA'])



print("Размер датасета:", df.shape)
print("\nПервые 5 строк:")
print(df.head())

#целевая переменная для регрессии
df['TotalGoals'] = df['FTHG'] + df['FTAG']



def goals_to_class(total_goals):
    if total_goals <= 2:
        return 0
    elif total_goals <= 4:
        return 1
    else:
        return 2


df['GoalClass'] = df['TotalGoals'].apply(goals_to_class)

print("\nРаспределение классов:")
print(df['GoalClass'].value_counts(normalize=True))

numeric_features = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR',
                    'AR']

medians = df[numeric_features].median()
for col in numeric_features:
    if df[col].isnull().any():
        print(f"{col}: {medians[col]:.2f}")
df[numeric_features] = df[numeric_features].fillna(medians)
print(df[numeric_features].isnull().sum().sum(), "пропусков осталось")

X = df[numeric_features]
y = df['GoalClass']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nРазмер тренировочной выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")

#базовое обучение
base_model = DecisionTreeClassifier(random_state=42)
base_model.fit(X_train, y_train)
y_pred_base = base_model.predict(X_test)
accuracy_base = accuracy_score(y_test, y_pred_base)

print(f"Accuracy: {accuracy_base:.4f}")
print("\nРаспределение классов в тренировочной выборке:")
train_class_dist = Counter(y_train)
for class_label, count in train_class_dist.items():
    print(f"Класс {class_label}: {count} samples ({count / len(y_train) * 100:.1f}%)")

#искусственнфй дисбалан
majority_class = max(train_class_dist, key=train_class_dist.get)
majority_count = train_class_dist[majority_class]

classes_to_reduce = [cls for cls in train_class_dist.keys() if cls != majority_class]
class_to_reduce = classes_to_reduce[0] if classes_to_reduce else list(train_class_dist.keys())[1]

reduce_to_count = int(majority_count * 0.1)

X_train_imbalanced = X_train.copy()
y_train_imbalanced = y_train.copy()

reduce_class_indices = y_train_imbalanced[y_train_imbalanced == class_to_reduce].index
reduce_class_keep = np.random.choice(reduce_class_indices, size=reduce_to_count, replace=False)

keep_indices = list(y_train_imbalanced[y_train_imbalanced != class_to_reduce].index) + list(reduce_class_keep)
X_train_imbalanced = X_train_imbalanced.loc[keep_indices]
y_train_imbalanced = y_train_imbalanced.loc[keep_indices]

print(f"Уменьшаем класс {class_to_reduce} до {reduce_to_count} samples (10% от класса {majority_class})")
print("Распределение после создания дисбаланса:")
imbalanced_dist = Counter(y_train_imbalanced)
for class_label, count in imbalanced_dist.items():
    print(f"Класс {class_label}: {count} samples ({count / len(y_train_imbalanced) * 100:.1f}%)")

#обучение модели на дисбалансированных данных
imbalanced_model = DecisionTreeClassifier(random_state=42)
imbalanced_model.fit(X_train_imbalanced, y_train_imbalanced)
y_pred_imbalanced = imbalanced_model.predict(X_test)
accuracy_imbalanced = accuracy_score(y_test, y_pred_imbalanced)

print(f"Accuracy на дисбалансированных данных: {accuracy_imbalanced:.4f}")

methods = {
    'Random Oversampling': RandomOverSampler(random_state=42),
    'SMOTE': SMOTE(random_state=42),
    'ADASYN': ADASYN(random_state=42),
    'Tomek Links': TomekLinks()
}

results = {}

for method_name, sampler in methods.items():
    print(f"\n{method_name}")

    if method_name == 'Tomek Links':
        X_balanced, y_balanced = sampler.fit_resample(X_train, y_train)
    else:
        X_balanced, y_balanced = sampler.fit_resample(X_train, y_train)

    print(f"распределение после балансировки: {Counter(y_balanced)}")

    #обучаем модель
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_balanced, y_balanced)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    results[method_name] = accuracy
    print(f"Accuracy: {accuracy:.4f}")


print("\nрезультатоs:")
print(f"{'Метод':<20} {'Accuracy':<10}")
print(f"{'Базовая модель':<20} {accuracy_base:.4f}")
print(f"{'С дисбалансом':<20} {accuracy_imbalanced:.4f}")

for method, acc in results.items():
    print(f"{method:<20} {acc:.4f}")

#эффективност методов
comparison_df = pd.DataFrame({
    'Method': ['Base Model', 'Imbalanced'] + list(results.keys()),
    'Accuracy': [accuracy_base, accuracy_imbalanced] + list(results.values())
})

print(f"\nЛучший метод: {comparison_df.iloc[0]['Method']} (Accuracy: {comparison_df.iloc[0]['Accuracy']:.4f})")

best_method_name = comparison_df.iloc[0]['Method']
if best_method_name == 'Base Model':
    best_model = base_model
    y_pred_best = y_pred_base
elif best_method_name == 'Imbalanced':
    best_model = imbalanced_model
    y_pred_best = y_pred_imbalanced
else:
    sampler = methods[best_method_name]
    X_best, y_best = sampler.fit_resample(X_train, y_train)
    best_model = DecisionTreeClassifier(random_state=42)
    best_model.fit(X_best, y_best)
    y_pred_best = best_model.predict(X_test)

print(f"\nотчет классификации для {best_method_name}:")
print(classification_report(y_test, y_pred_best))

print("\nВажные признаки в модели:")
feature_importance = pd.DataFrame({
    'feature': numeric_features,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))


#Искусственный дисбаланс значительно снижает accuracy модели
#Методы балансировки могут улучшить или незначительно снизить accuracy по сравнению с базовой моделью
#Oversampling методы (SMOTE, ADASYN) обычно показывают лучшие результаты на несбалансированных данных
#Undersampling методы (Tomek Links) могут быть менее эффективны при сильном дисбалансе
#Выбор метода зависит от конкретной задачи и метрик качества (не только accuracy)