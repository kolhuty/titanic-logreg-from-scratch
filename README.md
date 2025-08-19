# Titanic Survival Prediction - Logistic Regression from Scratch

Решение задачи классификации выживших на Titanic Dataset с Kaggle.  
Результат на Kaggle: **0.78**  
Реализация логистической регрессии **без ML-библиотек** (только `numpy`, `pandas`).

---

## Структура проекта
- `model.py` — реализация бинарного классификатора (логистическая регрессия + градиентный спуск)  
- `prepare_data.py` — функции подготовки train/test данных
- `main.py` — запуск обучения и формирование submission для Kaggle
- `data\` — тренировочные и тестовые данные