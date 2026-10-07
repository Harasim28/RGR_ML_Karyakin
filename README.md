# ML Inference Dashboard (РГР)

## Описание

Web-приложение (дашборд) для инференса (вывода) моделей машинного обучения и анализа данных, разработанное с использованием фреймворка **Streamlit**.

**Тема РГР:** «Разработка Web-приложения (дашборда) для инференса моделей ML и анализа данных»

**Датасет:** Smoke Detection (классификация наличия пожара по показаниям датчиков)

**Задача:** бинарная классификация — определить, сработала ли пожарная сигнализация (`Fire Alarm: Yes/No`) на основе показаний датчиков температуры, влажности, концентрации газов и твердых частиц.

---

## Структура проекта

```
RGR/
├── venv/                       # Виртуальное окружение Python 3.10
├── models/                     # Сериализованные модели ML
│   ├── ml1_logreg.joblib       # ML1: LogisticRegression
│   ├── ml2_boosting.joblib     # ML2: GradientBoosting
│   ├── ml3_catboost.joblib     # ML3: CatBoost
│   ├── ml4_rf.joblib           # ML4: RandomForest
│   ├── ml5_stack.joblib        # ML5: Stacking
│   ├── ml6_mlp.joblib          # ML6: MLP (FCNN)
│   ├── scaler.joblib           # StandardScaler
│   ├── feature_names.joblib    # Список признаков
│   └── metrics_summary.csv     # Сводная таблица метрик
├── data/
│   └── smoke_detector_task_main.csv
├── assets/
│   └── photo.jpg               # Фото разработчика (добавить)
├── app.py                       # Главный файл Streamlit
├── train_models.py              # Скрипт обучения и сериализации
├── README.md                    # Этот файл
└── requirements.txt             # Зависимости Python
```

---

## Модели ML

| Модель | Тип | F1-score | Accuracy |
|--------|-----|----------|----------|
| ML1: LogisticRegression | Классическая | 0.9283 | 0.9015 |
| ML2: GradientBoosting | Бустинг | 0.9997 | 0.9996 |
| ML3: CatBoost | Продвинутый градиентный бустинг | 0.9999 | 0.9999 |
| ML4: RandomForest | Бэггинг | 1.0000 | 1.0000 |
| ML5: Stacking | Стэкинг | 0.9998 | 0.9997 |
| ML6: MLP (FCNN) | Глубокая нейронная сеть | 0.9999 | 0.9999 |

---

## Установка и запуск

### 1. Клонирование репозитория

```bash
git clone https://github.com/Harasim28/RGR_ML_Karyakin.git
cd RGR
```

### 2. Создание виртуального окружения

```bash
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Обучение моделей (опционально — модели уже сериализованы)

```bash
python train_models.py
```

### 5. Запуск дашборда

```bash
streamlit run app.py
```

Приложение откроется в браузере по адресу `http://localhost:8501`.

---

## Страницы дашборда

1. **🏠 О разработчике** — информация о студенте, тема РГР, описание проекта
2. **📋 О датасете** — описание предметной области, признаки, статистика, EDA
3. **📈 Визуализации** — 4 вида графиков (распределение, boxplot, корреляция, scatter)
4. **🔮 Инференс** — загрузка CSV или ручной ввод данных, выбор модели, предсказание с вероятностью

---

## Сериализация моделей

- **Scikit-learn модели** — `joblib.dump()`
- **CatBoost** — `CatBoostClassifier.save_model()` (внутренний формат, загружается через `joblib` в текущей реализации)

---

## Технологический стек

- **Python 3.10**
- **Streamlit** — фреймворк для Web-приложений
- **scikit-learn** — классические ML модели
- **CatBoost** — градиентный бустинг
- **pandas, numpy** — обработка данных
- **matplotlib, seaborn** — визуализация
- **joblib** — сериализация

---

**Автор**

**ФИО:** Карякин Алексей Петрович  
**Группа:** МО-241  
**Университет:** ОмГТУ  
**Дисциплина:** Машинное обучение и большие данные

---

## Ссылки

- **GitHub:** https://github.com/Harasim28/RGR_ML_Karyakin
- **Streamlit Cloud:** https://rgr-ml-karyakin.streamlit.app

---

## Лицензия

Проект выполнен в рамках учебной расчетно-графической работы.
