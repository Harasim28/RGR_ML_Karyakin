"""
Скрипт обучения и сериализации 6 моделей ML для РГР
Датасет: smoke_detector_task_main.csv (классификация)
"""

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier

os.makedirs('models', exist_ok=True)

# 1. Загрузка данных
print("Загрузка данных...")
df = pd.read_csv('data/smoke_detector_task_main.csv')
print(f"Размер датасета: {df.shape}")

# 2. Предобработка
# Удаляем ненужные колонки
df = df.drop(['Unnamed: 0', 'UTC', 'CNT'], axis=1, errors='ignore')
# Преобразуем целевую переменную
df['Fire Alarm'] = df['Fire Alarm'].map({'No': 0, 'Yes': 1})
# Удаляем пропуски
df = df.dropna()

X = df.drop('Fire Alarm', axis=1)
y = df['Fire Alarm']
feature_names = list(X.columns)

print(f"Признаки: {feature_names}")
print(f"Распределение классов:\n{y.value_counts(normalize=True)}")

# 3. Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# 4. Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Сохраняем scaler и feature_names
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(feature_names, 'models/feature_names.joblib')
print("Scaler сериализован.")

# 5. Функция для обучения и оценки модели
def train_and_save(model, name, filepath, X_tr, y_tr, X_te, y_te):
    print(f"\nОбучение {name}...")
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec = recall_score(y_te, y_pred, zero_division=0)
    f1 = f1_score(y_te, y_pred, zero_division=0)
    
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    
    joblib.dump(model, filepath)
    print(f"  Сохранено: {filepath}")
    return {'name': name, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

# 6. Обучение моделей
results = []

# ML1: LogisticRegression (классическая модель)
ml1 = LogisticRegression(C=10, class_weight='balanced', max_iter=2000, random_state=42)
results.append(train_and_save(ml1, 'ML1: LogisticRegression', 'models/ml1_logreg.joblib',
                              X_train_scaled, y_train, X_test_scaled, y_test))

# ML2: GradientBoosting (бустинг)
ml2 = GradientBoostingClassifier(n_estimators=100, random_state=42)
results.append(train_and_save(ml2, 'ML2: GradientBoosting', 'models/ml2_boosting.joblib',
                              X_train_scaled, y_train, X_test_scaled, y_test))

# ML3: CatBoost (продвинутый градиентный бустинг)
ml3 = CatBoostClassifier(iterations=200, verbose=0, random_state=42)
results.append(train_and_save(ml3, 'ML3: CatBoost', 'models/ml3_catboost.joblib',
                              X_train_scaled, y_train, X_test_scaled, y_test))

# ML4: RandomForest (бэггинг)
ml4 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
results.append(train_and_save(ml4, 'ML4: RandomForest', 'models/ml4_rf.joblib',
                              X_train_scaled, y_train, X_test_scaled, y_test))

# ML5: Stacking (стэкинг)
ml5 = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42))
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    passthrough=False
)
results.append(train_and_save(ml5, 'ML5: Stacking', 'models/ml5_stack.joblib',
                              X_train_scaled, y_train, X_test_scaled, y_test))

# ML6: MLP (глубокая полносвязная нейронная сеть)
ml6 = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=500, random_state=42)
results.append(train_and_save(ml6, 'ML6: MLP (FCNN)', 'models/ml6_mlp.joblib',
                              X_train_scaled, y_train, X_test_scaled, y_test))

# 7. Сводная таблица
print("\n" + "="*60)
print("СВОДНАЯ ТАБЛИЦА МОДЕЛЕЙ")
print("="*60)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Сохраняем таблицу
results_df.to_csv('models/metrics_summary.csv', index=False)
print("\nМетрики сохранены в models/metrics_summary.csv")
print("Все модели сериализованы!")
