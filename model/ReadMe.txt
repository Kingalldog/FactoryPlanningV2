# Простое обучение
python3 heuristic_recommender.py --mode train --train_data task_results/dataset_table_20_20.csv

# Обучение с сохранением модели в другой файл
python3 heuristic_recommender.py --mode train --train_data task_results/dataset_table_20_20.csv --save_model my_model.pkl


# Использовать модель по умолчанию (heuristic_model.pkl)
python3 heuristic_recommender.py --mode recommend --problem path/to/problem.txt

# С детальной информацией
python3 heuristic_recommender.py --mode recommend --problem path/to/problem.txt --detailed

# С указанием другой модели
python3 heuristic_recommender.py --mode recommend --model my_model.pkl --problem path/to/problem.txt

# Создайте файл tasks.txt с путями к задачам (по одной на строку)
python3 heuristic_recommender.py --mode batch --problem_list tasks.txt --output results.csv

# интерактивный режим
python3 interactive_recommender.py


# Обучение
python3 dynamic_recommender.py --mode train --train_data results.csv --save_model dynamic_model.pkl

# Рекомендация для одной задачи
python3 dynamic_recommender.py --mode recommend --model dynamic_model.pkl --problem ../../factory/dataset_directory_big/20_20/2349008_20_20_0.300000_0.300000_0.300000_1000_100_.txt --detailed

# Пакетная обработка
python3 dynamic_recommender.py --mode batch --model dynamic_model.pkl --problem_list problems.txt --output recommendations.csv


🤖 Система рекомендации эвристик для задач планирования
📋 Описание
Данная система предназначена для автоматического выбора наиболее эффективной эвристики сортировки операций при решении задач планирования производства. На основе характеристик задачи (количество операций, структура графа зависимостей, временные параметры и т.д.) модель предсказывает, какая эвристика даст наименьший штраф.

🎯 Целевые эвристики
Система выбирает из 6 эвристик:

Эвристика	Описание
Dummy	Базовая эвристика (порядок по умолчанию)
Directive	Приоритет по директивным срокам
Stoppable	Учет возможности прерывания операций
Robin	Циклический перебор (Round Robin)
FineSorter	Сортировка по штрафам
Dependent	Учет зависимостей между операциями
🔍 Как это работает
1. Извлечение признаков
Из файла задачи извлекаются следующие группы признаков:

📊 Размерностные признаки
n_tools - количество инструментов

n_operations - количество операций

n_works - количество работ

stoppable_ratio - доля прерываемых операций

⏱️ Временные характеристики
avg_time - среднее время выполнения операции

std_time - стандартное отклонение времени

min_time / max_time - минимальное и максимальное время

time_range - разброс времен

time_cv - коэффициент вариации

matrix_density - плотность матрицы "операция-инструмент"

📈 Характеристики работ
avg_work_duration - средняя длительность работы

std_work_duration - разброс длительностей

avg_fine_coef / std_fine_coef - средний штраф и его вариативность

avg_ops_per_work - среднее количество операций в работе

max_ops_per_work - максимальный размер работы

avg_slack / min_slack - средний и минимальный запас времени

slack_range - разброс запаса времени

🔗 Структура графа
n_edges - количество зависимостей

graph_density - плотность графа

avg_in_degree / max_in_degree - входящие связи

avg_out_degree / max_out_degree - исходящие связи

🛠️ Загрузка инструментов
avg_tool_busy - среднее время занятости инструмента

std_tool_busy - разброс времени занятости

total_tool_busy - общая занятость всех инструментов

2. Обучение модели
Модель использует Random Forest Classifier со следующими параметрами:

200 деревьев

максимальная глубина: 10

min_samples_split: 5

class_weight: 'balanced' (для работы с несбалансированными классами)

3. Принятие решения
Для новой задачи система:

Извлекает все признаки из файла

Передает их в обученную модель

Получает вероятности для каждой эвристики

Рекомендует эвристику с наибольшей вероятностью

📊 Текущая производительность
На тестовой выборке модель показывает точность 38.89%, что в 2.3 раза лучше случайного угадывания (16.7%).

Распределение лучших эвристик в обучающей выборке:
text
FineSorter: 33.7% (182 задачи)
Directive:  30.0% (162 задачи)
Robin:      15.7% (85 задач)
Dependent:  10.7% (58 задач)
Dummy:      7.4%  (40 задач)
Stoppable:  2.4%  (13 задач)
Качество предсказания по эвристикам:
Эвристика	Precision	Recall	F1-score
FineSorter	0.65	0.67	0.66
Dummy	0.43	0.38	0.40
Directive	0.32	0.31	0.32
Dependent	0.27	0.33	0.30
Robin	0.06	0.06	0.06
Stoppable	0.00	0.00	0.00
🔑 Ключевые признаки
Наиболее важные признаки для выбора эвристики:

Ранг	Признак	Важность	Влияние
1	n_operations	6.5%	Количество операций - самый важный фактор
2	n_edges	5.7%	Количество зависимостей между операциями
3	graph_density	5.1%	Плотность графа зависимостей
4	std_fine_coef	4.8%	Разброс коэффициентов штрафа
5	std_ops_per_work	4.7%	Вариативность размера работ
6	min_slack	4.4%	Минимальный запас времени
7	slack_range	4.2%	Разброс запаса времени
8	stoppable_ratio	4.0%	Доля прерываемых операций
9	std_work_duration	3.8%	Разброс длительностей работ
10	matrix_density	3.8%	Плотность матрицы операция-инструмент
Интерпретация важных признаков:
Структурные признаки (n_operations, n_edges, graph_density) - определяют сложность графа задач

Признаки неопределенности (std_fine_coef, std_ops_per_work) - показывают разнородность задач

Временные ограничения (min_slack, slack_range) - критичность сроков

Ресурсные признаки (matrix_density) - доступность инструментов