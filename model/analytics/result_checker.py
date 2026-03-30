import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# ===== НАСТРОЙКИ =====
PREDICTIONS_FILE = "mk_predictions_results_2.csv"  # файл с предсказаниями
# =====================

# Загружаем данные
with open('new_test_data_25_03_2026.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()

headers = lines[0].strip().split(';')
if headers[0] == '':
    headers[0] = 'file_path'

data = []
for line in lines[1:]:
    parts = line.strip().split(';')
    file_path = parts[0]
    values = []
    for val in parts[1:]:
        try:
            clean_val = val.strip().replace(',', '.')
            if clean_val == '':
                values.append(0.0)  # Пустые = 0 штрафа
            else:
                values.append(float(clean_val))
        except ValueError:
            values.append(0.0)
    data.append([file_path] + values)

df = pd.DataFrame(data, columns=headers)
print("Данные загружены!")
print(f"Размер: {df.shape}")

# Извлекаем имя файла из пути
df['filename'] = df['file_path'].apply(lambda x: os.path.basename(x) if isinstance(x, str) else '')

algo_columns = df.columns[1:-1]  # все кроме file_path и filename

# Статистика штрафов
print("\n=== СТАТИСТИКА ШТРАФОВ ===")
print("(0 = идеально, больше = хуже)\n")

stats_df = df[algo_columns].describe()
print(stats_df)

# Процент идеальных решений (штраф = 0)
zero_percent = (df[algo_columns] == 0).sum() / len(df) * 100
print("\n=== ПРОЦЕНТ ИДЕАЛЬНЫХ РЕШЕНИЙ (штраф = 0) ===")
print(zero_percent.sort_values(ascending=False).round(2))

# Средний штраф (когда он не нулевой)
print("\n=== СРЕДНИЙ ШТРАФ (только когда есть ошибки) ===")
for col in algo_columns:
    non_zero = df[df[col] > 0][col]
    if len(non_zero) > 0:
        print(f"{col}: средний штраф {non_zero.mean():.2f} (в {len(non_zero)} случаях)")
    else:
        print(f"{col}: никогда не ошибается!")

# 1. Тепловая карта штрафов (первые 20 примеров)
plt.figure(figsize=(15, 8))
sample_df = df[algo_columns].head(20)
sns.heatmap(sample_df.T, annot=True, fmt='.1f', cmap='YlOrRd_r', 
            cbar_kws={'label': 'Штраф (меньше = лучше)'})
plt.title('Штрафы алгоритмов на первых 20 примерах')
plt.xlabel('Номер примера')
plt.ylabel('Алгоритм')
plt.tight_layout()
plt.show()

# 2. Box plot штрафов
plt.figure(figsize=(14, 6))
df[algo_columns].boxplot()
plt.xticks(rotation=45)
plt.title('Распределение штрафов по алгоритмам')
plt.ylabel('Штраф')
plt.yscale('symlog')  # симметричная логарифмическая шкала для обработки нулей
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3. Доля идеальных решений
ideal_rates = (df[algo_columns] == 0).mean() * 100
plt.figure(figsize=(12, 6))
ideal_rates.sort_values().plot(kind='barh')
plt.xlabel('Процент идеальных решений (%)')
plt.title('Доля случаев с нулевым штрафом')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ml_models = ['MlSorter', 'MlHardSorter']
ml_models = ['MlSorter']
classic_algos = ['Directive', 'Stoppable', 'Robin', 'FineSorter', 'Dependent']
dummy = 'Dummy'

print("\n=== АНАЛИЗ ML МОДЕЛЕЙ (штрафы) ===\n")

# 1. Идеальные решения
for model in ml_models + [dummy] + classic_algos[:1]:
    ideal = (df[model] == 0).sum()
    total = len(df)
    print(f"{model}: идеальных решений {ideal}/{total} ({ideal/total*100:.1f}%)")

# 2. Средний штраф (чем меньше, тем лучше)
print("\n=== СРЕДНИЙ ШТРАФ (меньше = лучше) ===")
mean_penalties = df[algo_columns].mean().sort_values()
print(mean_penalties)

# 3. Медианный штраф (устойчив к выбросам)
print("\n=== МЕДИАННЫЙ ШТРАФ ===")
median_penalties = df[algo_columns].median().sort_values()
print(median_penalties)

# 4. Максимальный штраф (худший случай)
print("\n=== МАКСИМАЛЬНЫЙ ШТРАФ ===")
max_penalties = df[algo_columns].max().sort_values()
print(max_penalties)

# ИСПРАВЛЕННАЯ ФУНКЦИЯ: Подсчет сколько раз каждый алгоритм был лучшим и в топ-2
def count_top_positions(df, algo_columns):
    """
    Подсчитывает для каждого алгоритма:
    - сколько раз он был лучшим (минимальный штраф)
    - сколько раз был в топ-2 (первое или второе место)
    - сколько раз был в топ-3 (первое, второе или третье место)
    Исправлено: корректно учитывает случаи с несколькими алгоритмами на первом месте
    """
    
    top1_counts = {col: 0 for col in algo_columns}
    top2_counts = {col: 0 for col in algo_columns}
    top3_counts = {col: 0 for col in algo_columns}
    
    for idx, row in df[algo_columns].iterrows():
        # Получаем значения штрафов для этой строки
        values = row.values
        
        # Находим уникальные значения и сортируем
        unique_values = sorted(set(values))
        
        # ТОП-1: алгоритмы с минимальным значением
        min_val = unique_values[0]
        min_algos = [algo_columns[i] for i, val in enumerate(values) if val == min_val]
        
        # Каждый из алгоритмов с минимальным значением считается лучшим
        for algo in min_algos:
            top1_counts[algo] += 1
        
        # ТОП-2: алгоритмы с минимальным или вторым минимальным значением
        if len(unique_values) >= 2:
            second_min = unique_values[1]
            # Все алгоритмы со значением <= second_min (т.е. минимальные и вторые)
            for i, val in enumerate(values):
                algo = algo_columns[i]
                if val <= second_min:
                    top2_counts[algo] += 1
        else:
            # Если все значения одинаковые, то все в топ-2
            for i, val in enumerate(values):
                algo = algo_columns[i]
                top2_counts[algo] += 1
        
        # ТОП-3: алгоритмы с минимальным, вторым или третьим минимальным значением
        if len(unique_values) >= 3:
            third_min = unique_values[2]
            for i, val in enumerate(values):
                algo = algo_columns[i]
                if val <= third_min:
                    top3_counts[algo] += 1
        else:
            # Если меньше 3 уникальных значений, то все алгоритмы с val <= max(unique_values)
            # (фактически все, у кого значение не больше максимального из топ-3)
            max_top3 = unique_values[-1] if len(unique_values) > 0 else 0
            for i, val in enumerate(values):
                algo = algo_columns[i]
                if val <= max_top3:
                    top3_counts[algo] += 1
    
    return top1_counts, top2_counts, top3_counts

# Подсчитываем топ-позиции
top1_counts, top2_counts, top3_counts = count_top_positions(df, algo_columns)

print("\n" + "="*60)
print("=== СТАТИСТИКА ТОП-ПОЗИЦИЙ ===")
print("="*60)

# Сортируем по убыванию
top1_sorted = sorted(top1_counts.items(), key=lambda x: x[1], reverse=True)
top2_sorted = sorted(top2_counts.items(), key=lambda x: x[1], reverse=True)
top3_sorted = sorted(top3_counts.items(), key=lambda x: x[1], reverse=True)

print("\nТОП-1 (минимальный штраф):")
print("-" * 50)
for algo, count in top1_sorted:
    percentage = count / len(df) * 100
    bar = "█" * int(percentage / 2)
    print(f"{algo:15} : {count:4} раз ({percentage:5.1f}%) {bar}")

print("\nТОП-2 (1-е или 2-е место):")
print("-" * 50)
for algo, count in top2_sorted:
    percentage = count / len(df) * 100
    bar = "█" * int(percentage / 2)
    print(f"{algo:15} : {count:4} раз ({percentage:5.1f}%) {bar}")

print("\nТОП-3 (1-е, 2-е или 3-е место):")
print("-" * 50)
for algo, count in top3_sorted:
    percentage = count / len(df) * 100
    bar = "█" * int(percentage / 2)
    print(f"{algo:15} : {count:4} раз ({percentage:5.1f}%) {bar}")

# Проверка на вашем примере
print("\n" + "="*60)
print("=== ПРОВЕРКА НА ПРИМЕРЕ ===")
print("="*60)
example_file = '2349008_35_70_0.300000_0.300000_0.450000_1000_500_.txt'
if example_file in df['filename'].values:
    print(f"Файл: {example_file}")
    print("Значения:")
    for algo in algo_columns:
        val = df[df['filename'] == example_file][algo].values[0]
        print(f"  {algo}: {val}")
    print("\nРезультат подсчета для этого примера:")
    print(f"  MlSorter в топ-1: {'✅' if top1_counts['MlSorter'] > 0 else '❌'}")
    print(f"  MlSorter в топ-3: {'✅' if top3_counts['MlSorter'] > 0 else '❌'}")
else:
    print(f"Файл {example_file} не найден в данных")

# Создаем DataFrame для визуализации
top_df = pd.DataFrame({
    'Алгоритм': algo_columns,
    'Топ-1': [top1_counts[algo] for algo in algo_columns],
    'Топ-2': [top2_counts[algo] for algo in algo_columns],
    'Топ-3': [top3_counts[algo] for algo in algo_columns]
})
top_df = top_df.sort_values('Топ-1', ascending=False)

# Визуализация топ-позиций
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# График для топ-1
ax1 = axes[0]
bars1 = ax1.barh(top_df['Алгоритм'], top_df['Топ-1'], color='gold')
ax1.set_xlabel('Количество раз')
ax1.set_title('ТОП-1: Количество раз алгоритм был лучшим')
ax1.grid(axis='x', alpha=0.3)
for i, (idx, row) in enumerate(top_df.iterrows()):
    percentage = row['Топ-1'] / len(df) * 100
    ax1.text(row['Топ-1'] + 0.5, i, f'{percentage:.1f}%', va='center')

# График для топ-2
ax2 = axes[1]
bars2 = ax2.barh(top_df['Алгоритм'], top_df['Топ-2'], color='silver')
ax2.set_xlabel('Количество раз')
ax2.set_title('ТОП-2: Количество раз алгоритм был в топ-2')
ax2.grid(axis='x', alpha=0.3)
for i, (idx, row) in enumerate(top_df.iterrows()):
    percentage = row['Топ-2'] / len(df) * 100
    ax2.text(row['Топ-2'] + 0.5, i, f'{percentage:.1f}%', va='center')

# График для топ-3
ax3 = axes[2]
bars3 = ax3.barh(top_df['Алгоритм'], top_df['Топ-3'], color='#cd7f32')  # бронза
ax3.set_xlabel('Количество раз')
ax3.set_title('ТОП-3: Количество раз алгоритм был в топ-3')
ax3.grid(axis='x', alpha=0.3)
for i, (idx, row) in enumerate(top_df.iterrows()):
    percentage = row['Топ-3'] / len(df) * 100
    ax3.text(row['Топ-3'] + 0.5, i, f'{percentage:.1f}%', va='center')

plt.tight_layout()
plt.show()

# НОВАЯ ФУНКЦИЯ: Анализ предсказаний из файла
def analyze_predictions(predictions_file, df, ml_model_name='MlSorter', top3_counts=None):
    """
    Анализирует файл с предсказаниями и считает статистику:
    - сколько раз первое предсказание было лучшим
    - сколько раз лучший алгоритм был в топ-2 предсказаний
    - сколько раз лучший алгоритм был в топ-3 предсказаний
    """
    print("\n" + "="*60)
    print("=== АНАЛИЗ КАЧЕСТВА ПРЕДСКАЗАНИЙ ===")
    print("="*60)
    
    try:
        # Загружаем предсказания
        pred_df = pd.read_csv(predictions_file, sep=';', encoding='utf-8')
        print(f"Загружено предсказаний: {len(pred_df)}")
        
        # Показываем колонки для отладки
        print("\n📋 КОЛОНКИ В ФАЙЛЕ:")
        for i, col in enumerate(pred_df.columns):
            print(f"  {i}: '{col}'")
        
        # Имена колонок из вашего файла
        filename_col = 'имя файла'
        top1_col = '1_лучший алгоритм'
        top2_col = '2'
        top3_col = '3'
        
        print(f"\n✅ Используем колонки:")
        print(f"  - имя файла: '{filename_col}'")
        print(f"  - топ-1: '{top1_col}'")
        print(f"  - топ-2: '{top2_col}'")
        print(f"  - топ-3: '{top3_col}'")
        
        # Проверяем, что все колонки существуют
        missing_cols = []
        for col in [filename_col, top1_col, top2_col, top3_col]:
            if col not in pred_df.columns:
                missing_cols.append(col)
        
        if missing_cols:
            print(f"❌ Отсутствуют колонки: {missing_cols}")
            print("Доступные колонки:", pred_df.columns.tolist())
            return None, None
        
        # Создаем словарь для быстрого доступа к предсказаниям по имени файла
        predictions_dict = {}
        success_count = 0
        
        for idx, row in pred_df.iterrows():
            try:
                # Получаем имя файла
                filename = str(row[filename_col]).strip()
                # Убираем возможные пути
                filename = os.path.basename(filename)
                
                # Получаем предсказания
                top1 = str(row[top1_col]).strip() if pd.notna(row[top1_col]) else None
                top2 = str(row[top2_col]).strip() if pd.notna(row[top2_col]) else None
                top3 = str(row[top3_col]).strip() if pd.notna(row[top3_col]) else None
                
                # Очищаем от лишних символов
                if top1:
                    top1 = top1.replace(';', '').strip()
                if top2:
                    top2 = top2.replace(';', '').strip()
                if top3:
                    top3 = top3.replace(';', '').strip()
                
                predictions_dict[filename] = {
                    'top1': top1,
                    'top2': top2,
                    'top3': top3
                }
                success_count += 1
                
            except Exception as e:
                if success_count < 5:
                    print(f"  ⚠️ Ошибка в строке {idx}: {e}")
        
        print(f"\n📊 Успешно загружено предсказаний: {success_count}")
        
        # Покажем несколько примеров для проверки
        print("\n🔍 ПРИМЕРЫ ПРЕДСКАЗАНИЙ (первые 5):")
        for i, (filename, pred) in enumerate(list(predictions_dict.items())[:5]):
            actual_row = df[df['filename'] == filename]
            if not actual_row.empty:
                actual_best = get_best_algorithm(actual_row.iloc[0])
                print(f"  {i+1}. {filename}:")
                print(f"     Предсказано: 1='{pred['top1']}', 2='{pred['top2']}', 3='{pred['top3']}'")
                print(f"     Реально лучший: {actual_best}")
        
        # Статистика
        stats = {
            'total': 0,
            'top1_correct': 0,        # первое предсказание = лучший алгоритм
            'top2_contains_best': 0,   # лучший алгоритм в первых двух
            'top3_contains_best': 0,    # лучший алгоритм в первых трех
            'detailed': []
        }
        
        matched_files = 0
        
        for idx, row in df.iterrows():
            filename = row['filename']
            if not filename or filename not in predictions_dict:
                continue
                
            matched_files += 1
            pred = predictions_dict[filename]
            
            # Пропускаем ошибки
            if not pred['top1'] or pred['top1'] in ['ERROR', 'EXCEPTION', 'None', 'none', '']:
                continue
                
            stats['total'] += 1
            
            # Определяем реально лучший алгоритм
            algo_values = {}
            for algo in algo_columns:
                if algo in row:
                    algo_values[algo] = row[algo]
            
            if not algo_values:
                continue
                
            min_val = min(algo_values.values())
            best_algos = [a for a, v in algo_values.items() if v == min_val]
            
            # Проверяем каждую позицию
            top1 = pred['top1']
            top2 = pred['top2']
            top3 = pred['top3']
            
            # 1. Первое предсказание лучше всех?
            if top1 and top1 in best_algos:
                stats['top1_correct'] += 1
            
            # 2. Лучший алгоритм в топ-2 предсказаний?
            if (top1 and top1 in best_algos) or (top2 and top2 in best_algos):
                stats['top2_contains_best'] += 1
            
            # 3. Лучший алгоритм в топ-3 предсказаний?
            if (top1 and top1 in best_algos) or (top2 and top2 in best_algos) or (top3 and top3 in best_algos):
                stats['top3_contains_best'] += 1
            
            # Детальная информация
            stats['detailed'].append({
                'filename': filename,
                'top1_pred': top1,
                'top2_pred': top2,
                'top3_pred': top3,
                'actual_best': best_algos,
                'top1_correct': top1 in best_algos if top1 else False,
                'top2_correct': (top1 in best_algos or top2 in best_algos) if top1 or top2 else False,
                'top3_correct': (top1 in best_algos or top2 in best_algos or top3 in best_algos) if top1 or top2 or top3 else False
            })
        
        print(f"\n📊 Найдено совпадений по именам файлов: {matched_files}")
        
        # Вывод статистики
        print(f"\n" + "="*60)
        print("📊 СТАТИСТИКА КАЧЕСТВА ПРЕДСКАЗАНИЙ")
        print("="*60)
        
        if stats['total'] == 0:
            print("❌ Нет данных для анализа!")
            return None, None
        
        print(f"\nВсего проанализировано задач: {stats['total']}")
        print(f"\n🎯 ТОЧНОСТЬ ПРЕДСКАЗАНИЙ:")
        
        # Топ-1 точность
        top1_pct = stats['top1_correct'] / stats['total'] * 100
        print(f"  🥇 Первое предсказание лучший алгоритм: {stats['top1_correct']}/{stats['total']} = {top1_pct:.1f}%")
        
        # Топ-2 точность
        top2_pct = stats['top2_contains_best'] / stats['total'] * 100
        print(f"  🥈 Лучший алгоритм в топ-2: {stats['top2_contains_best']}/{stats['total']} = {top2_pct:.1f}%")
        
        # Топ-3 точность
        top3_pct = stats['top3_contains_best'] / stats['total'] * 100
        print(f"  🥉 Лучший алгоритм в топ-3: {stats['top3_contains_best']}/{stats['total']} = {top3_pct:.1f}%")
        
        print(f"\n📈 УЛУЧШЕНИЕ:")
        print(f"  Топ-2 лучше топ-1 на {top2_pct - top1_pct:.1f}%")
        print(f"  Топ-3 лучше топ-2 на {top3_pct - top2_pct:.1f}%")
        print(f"  Топ-3 лучше топ-1 на {top3_pct - top1_pct:.1f}%")
        
        # Анализ ошибок
        print(f"\n🔍 АНАЛИЗ ОШИБОК:")
        
        # Случаи, когда топ-1 не угадал
        top1_wrong = stats['total'] - stats['top1_correct']
        print(f"  Топ-1 ошибся в {top1_wrong} случаях ({top1_wrong/stats['total']*100:.1f}%)")
        
        if top1_wrong > 0:
            # Из них сколько исправил топ-2
            fixed_by_top2 = stats['top2_contains_best'] - stats['top1_correct']
            print(f"    Из них исправлено топ-2: {fixed_by_top2} ({fixed_by_top2/top1_wrong*100:.1f}% ошибок топ-1)")
            
            # Из них сколько исправил топ-3
            fixed_by_top3 = stats['top3_contains_best'] - stats['top2_contains_best']
            print(f"    Из них исправлено топ-3: {fixed_by_top3} ({fixed_by_top3/top1_wrong*100:.1f}% ошибок топ-1)")
        
        # Визуализация
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Столбчатая диаграмма точности
        ax1 = axes[0]
        categories = ['Топ-1', 'Топ-2', 'Топ-3']
        values = [top1_pct, top2_pct, top3_pct]
        colors = ['gold', 'silver', '#cd7f32']
        bars = ax1.bar(categories, values, color=colors)
        ax1.set_ylim([0, 100])
        ax1.set_ylabel('Точность (%)')
        ax1.set_title('Точность предсказаний')
        ax1.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{val:.1f}%', ha='center', va='bottom')
        
        # 2. Круговая диаграмма для топ-1
        ax2 = axes[1]
        labels = ['Угадал топ-1', 'Не угадал топ-1']
        sizes = [stats['top1_correct'], stats['total'] - stats['top1_correct']]
        colors = ['#4CAF50', '#f44336']
        ax2.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Топ-1 предсказание')
        
        # 3. Накопительная диаграмма
        ax3 = axes[2]
        x = ['Топ-1', 'Топ-2', 'Топ-3']
        y_correct = [stats['top1_correct'], stats['top2_contains_best'], stats['top3_contains_best']]
        y_wrong = [stats['total'] - c for c in y_correct]
        
        ax3.bar(x, y_correct, label='Угадал', color='#4CAF50')
        ax3.bar(x, y_wrong, bottom=y_correct, label='Не угадал', color='#f44336')
        ax3.set_ylabel('Количество задач')
        ax3.set_title('Накопительная точность')
        ax3.legend()
        
        # Добавляем проценты на столбцы
        for i, (correct, wrong) in enumerate(zip(y_correct, y_wrong)):
            total = correct + wrong
            ax3.text(i, correct/2, f'{correct}', ha='center', va='center', color='white', fontweight='bold')
            ax3.text(i, correct + wrong/2, f'{wrong}', ha='center', va='center', color='white', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Покажем примеры ошибок
        print(f"\n📋 ПРИМЕРЫ ОШИБОК (первые 5):")
        errors = [d for d in stats['detailed'] if not d['top3_correct']][:5]
        if errors:
            for e in errors:
                print(f"  Файл: {e['filename']}")
                print(f"    Предсказано: 1={e['top1_pred']}, 2={e['top2_pred']}, 3={e['top3_pred']}")
                print(f"    Реально лучшие: {e['actual_best']}")
        else:
            print("  Нет ошибок! Все предсказания попали в топ-3")
        
        return stats
        
    except FileNotFoundError:
        print(f"❌ Файл {predictions_file} не найден!")
        return None
    except Exception as e:
        print(f"❌ Ошибка при анализе предсказаний: {e}")
        import traceback
        traceback.print_exc()
        return None

# Вспомогательная функция для получения лучшего алгоритма
def get_best_algorithm(row):
    algo_values = {}
    for algo in algo_columns:
        if algo in row:
            algo_values[algo] = row[algo]
    if algo_values:
        min_val = min(algo_values.values())
        return [a for a, v in algo_values.items() if v == min_val]
    return None

def evaluate_penalty_model(df, model_name, all_algos, top1_counts, top2_counts, top3_counts):
    """
    Оценивает модель на основе штрафов
    (меньше штраф = лучше)
    """
    
    print(f"\n=== ОЦЕНКА {model_name} ===")
    
    # Метрика 1: Процент идеальных решений (штраф = 0)
    ideal_pct = (df[model_name] == 0).mean() * 100
    rank_ideal = (df[all_algos].eq(0).mean() * 100).rank(ascending=False)[model_name]
    print(f"Идеальных решений: {ideal_pct:.1f}% (место {rank_ideal:.0f} из {len(all_algos)})")
    
    # Метрика 2: Средний штраф
    mean_penalty = df[model_name].mean()
    rank_mean = df[all_algos].mean().rank()[model_name]
    print(f"Средний штраф: {mean_penalty:.3f} (место {rank_mean:.0f} из {len(all_algos)})")
    
    # Метрика 3: Медианный штраф
    median_penalty = df[model_name].median()
    rank_median = df[all_algos].median().rank()[model_name]
    print(f"Медианный штраф: {median_penalty:.3f} (место {rank_median:.0f} из {len(all_algos)})")
    
    # Метрика 4: 95-й перцентиль (худшие случаи)
    p95 = df[model_name].quantile(0.95)
    rank_p95 = df[all_algos].quantile(0.95).rank()[model_name]
    print(f"95-й перцентиль: {p95:.3f} (место {rank_p95:.0f} из {len(all_algos)})")
    
    # Метрика 5: Интегральный показатель (сумма штрафов)
    total_penalty = df[model_name].sum()
    rank_total = df[all_algos].sum().rank()[model_name]
    print(f"Суммарный штраф: {total_penalty:.3f} (место {rank_total:.0f} из {len(all_algos)})")
    
    # Позиция в топ-рейтинге
    top1_count = top1_counts[model_name]
    top1_rank = sorted(top1_counts.values(), reverse=True).index(top1_count) + 1
    print(f"Был лучшим: {top1_count} раз (место {top1_rank} из {len(all_algos)})")
    
    top2_count = top2_counts[model_name]
    top2_rank = sorted(top2_counts.values(), reverse=True).index(top2_count) + 1
    print(f"Был в топ-2: {top2_count} раз (место {top2_rank} из {len(all_algos)})")
    
    top3_count = top3_counts[model_name]
    top3_rank = sorted(top3_counts.values(), reverse=True).index(top3_count) + 1
    print(f"Был в топ-3: {top3_count} раз (место {top3_rank} из {len(all_algos)})")
    
    # Итоговая оценка
    avg_rank = (rank_ideal + rank_mean + rank_median + rank_p95 + rank_total + top1_rank + top2_rank + top3_rank) / 8
    
    print(f"\nСредний рейтинг: {avg_rank:.1f} место из {len(all_algos)}")
    
    if avg_rank <= 3:
        print("✅ ОТЛИЧНО - модель в топ-3 по всем метрикам")
    elif avg_rank <= 5:
        print("👍 ХОРОШО - модель конкурентоспособна")
    elif avg_rank <= len(all_algos) / 2:
        print("⚠️ СРЕДНЕ - модель работает, но есть лучше")
    else:
        print("❌ ПЛОХО - модель в аутсайдерах")
    
    return avg_rank

# Оцениваем все алгоритмы
all_algos = algo_columns.tolist()
print("\n" + "="*60)
for model in ml_models + [dummy] + classic_algos:
    evaluate_penalty_model(df, model, all_algos, top1_counts, top2_counts, top3_counts)
    print("="*60)

# Сравним ML модели с лучшими классическими
print("\n=== ДЕТАЛЬНОЕ СРАВНЕНИЕ ===\n")

# Найдем лучший классический алгоритм (по сумме штрафов)
classic_performance = {}
for algo in classic_algos:
    classic_performance[algo] = df[algo].sum()

best_classic = min(classic_performance, key=classic_performance.get)
best_classic_sum = classic_performance[best_classic]
print(f"Лучший классический алгоритм: {best_classic}")
print(f"  Суммарный штраф: {best_classic_sum:.3f}")

for model in ml_models:
    model_sum = df[model].sum()
    print(f"\n{model}:")
    print(f"  Суммарный штраф: {model_sum:.3f}")
    
    if model_sum < best_classic_sum:
        improvement = (best_classic_sum - model_sum) / best_classic_sum * 100
        print(f"  ✅ Лучше лучшего классического на {improvement:.1f}%")
    else:
        degradation = (model_sum - best_classic_sum) / best_classic_sum * 100
        print(f"  ❌ Хуже лучшего классического на {degradation:.1f}%")
    
    # Статистическая значимость
    t_stat, p_value = stats.mannwhitneyu(df[model], df[best_classic], alternative='less')
    print(f"  p-value (лучше ли классики): {p_value:.4f}")
    if p_value < 0.05:
        print(f"  ✅ Статистически значимо лучше")

def final_verdict(df, model_name, top1_counts, top2_counts, top3_counts):
    """Выносит окончательный вердикт по модели"""
    
    metrics = {
        'ideal_rate': (df[model_name] == 0).mean() * 100,
        'mean_penalty': df[model_name].mean(),
        'median_penalty': df[model_name].median(),
        'max_penalty': df[model_name].max(),
        'total_penalty': df[model_name].sum()
    }
    
    # Сравниваем со средним по всем алгоритмам
    avg_ideal = df[algo_columns].eq(0).mean().mean() * 100
    avg_mean = df[algo_columns].mean().mean()
    avg_total = df[algo_columns].sum().mean()
    
    print(f"\n=== ИТОГОВЫЙ ВЕРДИКТ ДЛЯ {model_name} ===")
    print(f"Идеальные решения: {metrics['ideal_rate']:.1f}% (среднее по всем: {avg_ideal:.1f}%)")
    print(f"Средний штраф: {metrics['mean_penalty']:.3f} (среднее по всем: {avg_mean:.3f})")
    print(f"Суммарный штраф: {metrics['total_penalty']:.3f} (среднее по всем: {avg_total:.3f})")
    
    # Добавляем информацию о топ-позициях в финальный вердикт
    print(f"Был лучшим: {top1_counts[model_name]} раз ({top1_counts[model_name]/len(df)*100:.1f}%)")
    print(f"Был в топ-2: {top2_counts[model_name]} раз ({top2_counts[model_name]/len(df)*100:.1f}%)")
    print(f"Был в топ-3: {top3_counts[model_name]} раз ({top3_counts[model_name]/len(df)*100:.1f}%)")
    
    # Подсчет баллов
    score = 0
    max_score = 7
    
    if metrics['ideal_rate'] > avg_ideal:
        score += 1
        print("✅ Выше среднего по идеальным решениям")
    
    if metrics['mean_penalty'] < avg_mean:
        score += 1
        print("✅ Ниже среднего штрафа")
    
    if metrics['total_penalty'] < avg_total:
        score += 1
        print("✅ Ниже суммарного штрафа")
    
    # Сравнение с лучшим классическим
    best_classic_ideal = df[classic_algos].eq(0).mean().max() * 100
    if metrics['ideal_rate'] >= best_classic_ideal * 0.9:  # не хуже 90% от лучшего
        score += 1
        print(f"✅ Конкурентоспособна с лучшим классическим ({best_classic_ideal:.1f}%)")
    
    # Бонус за частые победы
    if top1_counts[model_name] / len(df) > 0.2:  # побеждает в >20% случаев
        score += 1
        print("✅ Часто побеждает (>20% случаев)")
    
    if top2_counts[model_name] / len(df) > 0.5:  # в топ-2 в >50% случаев
        score += 1
        print("✅ Часто в топ-2 (>50% случаев)")
    
    if top3_counts[model_name] / len(df) > 0.8:  # в топ-3 в >80% случаев
        score += 1
        print("✅ Почти всегда в топ-3 (>80% случаев)")
    
    print(f"\nИтоговый счет: {score}/{max_score}")
    
    if score >= 6:
        print("🏆 ПРЕВОСХОДНО - модель лучше всех!")
    elif score >= 5:
        print("👍 ОЧЕНЬ ХОРОШО - можно использовать")
    elif score >= 4:
        print("⚠️ ХОРОШО - но есть куда расти")
    elif score >= 3:
        print("📊 ПРИЕМЛЕМО - средний результат")
    elif score >= 2:
        print("❌ ПЛОХО - модель не справляется")
    else:
        print("💀 УЖАСНО - модель бесполезна")
    
    return score

# Итоговая оценка для ML моделей
print("\n" + "="*60)
print("ИТОГОВОЕ ЗАКЛЮЧЕНИЕ")
print("="*60)

for model in ml_models:
    final_verdict(df, model, top1_counts, top2_counts, top3_counts)
    print("-"*40)

# Анализ предсказаний из файла
# Анализ предсказаний из файла - ИСПРАВЛЕНО
stats_predictions = analyze_predictions(PREDICTIONS_FILE, df, ml_models[0], top3_counts)

# Если есть статистика предсказаний, выводим дополнительное заключение
if stats_predictions:
    print("\n" + "="*60)
    print("ЗАКЛЮЧЕНИЕ ПО ПРЕДСКАЗАНИЯМ")
    print("="*60)
    
    model = ml_models[0]
    total = stats_predictions['total']
    
    print(f"\nМодель {model}:")
    
    if total > 0:
        # Используем правильные ключи из новой статистики
        top1_correct = stats_predictions['top1_correct']
        top2_contains = stats_predictions['top2_contains_best']
        top3_contains = stats_predictions['top3_contains_best']
        
        print(f"  📊 Всего проанализировано задач: {total}")
        print(f"\n🎯 ТОЧНОСТЬ ПРЕДСКАЗАНИЙ:")
        print(f"  🥇 Первое предсказание лучший алгоритм: {top1_correct}/{total} = {top1_correct/total*100:.1f}%")
        print(f"  🥈 Лучший алгоритм в топ-2: {top2_contains}/{total} = {top2_contains/total*100:.1f}%")
        print(f"  🥉 Лучший алгоритм в топ-3: {top3_contains}/{total} = {top3_contains/total*100:.1f}%")
        
        # Дополнительная информация
        print(f"\n📈 УЛУЧШЕНИЕ:")
        print(f"  Топ-2 лучше топ-1 на {(top2_contains - top1_correct)/total*100:.1f}%")
        print(f"  Топ-3 лучше топ-2 на {(top3_contains - top2_contains)/total*100:.1f}%")
        
        # Сравнение с реальностью (если есть top3_counts)
        if top3_counts:
            ml_in_actual_top3 = top3_counts[model]
            print(f"\n📊 СРАВНЕНИЕ С РЕАЛЬНОСТЬЮ:")
            print(f"  Реально в топ-3: {ml_in_actual_top3} раз ({ml_in_actual_top3/len(df)*100:.1f}%)")
            print(f"  Предсказания в топ-3: {top3_contains} раз ({top3_contains/total*100:.1f}%)")
    else:
        print("  ❌ Нет данных для анализа")