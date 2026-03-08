import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Загружаем данные
with open('test_data.csv', 'r', encoding='utf-8') as f:
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

algo_columns = df.columns[1:]

# Статистика штрафов
print("=== СТАТИСТИКА ШТРАФОВ ===")
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
plt.show()

# 2. Box plot штрафов
plt.figure(figsize=(14, 6))
df[algo_columns].boxplot()
plt.xticks(rotation=45)
plt.title('Распределение штрафов по алгоритмам')
plt.ylabel('Штраф')
plt.yscale('symlog')  # симметричная логарифмическая шкала для обработки нулей
plt.grid(True, alpha=0.3)
plt.show()

# 3. Доля идеальных решений
ideal_rates = (df[algo_columns] == 0).mean() * 100
plt.figure(figsize=(12, 6))
ideal_rates.sort_values().plot(kind='barh')
plt.xlabel('Процент идеальных решений (%)')
plt.title('Доля случаев с нулевым штрафом')
plt.grid(True, alpha=0.3)
plt.show()

ml_models = ['MlSorter', 'MlHardSorter']
classic_algos = ['Directive', 'Stoppable', 'Robin', 'FineSorter', 'Dependent']
dummy = 'Dummy'

print("=== АНАЛИЗ ML МОДЕЛЕЙ (штрафы) ===\n")

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

def evaluate_penalty_model(df, model_name, all_algos):
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
    
    # Итоговая оценка
    avg_rank = (rank_ideal + rank_mean + rank_median + rank_p95 + rank_total) / 5
    
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
print("="*60)
for model in ml_models + [dummy] + classic_algos[:1]:
    evaluate_penalty_model(df, model, all_algos)
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

def final_verdict(df, model_name):
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
    
    # Подсчет баллов
    score = 0
    max_score = 4
    
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
    
    print(f"\nИтоговый счет: {score}/{max_score}")
    
    if score == 4:
        print("🏆 ПРЕВОСХОДНО - модель лучше всех!")
    elif score == 3:
        print("👍 ОЧЕНЬ ХОРОШО - можно использовать")
    elif score == 2:
        print("⚠️ ПРИЕМЛЕМО - но есть куда расти")
    elif score == 1:
        print("❌ ПЛОХО - модель не справляется")
    else:
        print("💀 УЖАСНО - модель бесполезна")
    
    return score

# Итоговая оценка для ML моделей
print("\n" + "="*60)
print("ИТОГОВОЕ ЗАКЛЮЧЕНИЕ")
print("="*60)

for model in ml_models:
    final_verdict(df, model)
    print("-"*40)