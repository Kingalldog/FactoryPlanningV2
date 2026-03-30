import random

# Просто перечислите все нужные файлы в этом списке
file_paths = [
    "/home/vsamohin/FactoryPlanningV2/model/task_results/dataset_table_20_20.csv",
    "/home/vsamohin/FactoryPlanningV2/model/task_results/dataset_table_20_25.csv", 
    "/home/vsamohin/FactoryPlanningV2/model/task_results/dataset_table_20_40.csv",
    "/home/vsamohin/FactoryPlanningV2/model/task_results/dataset_table_20_70.csv",
    "/home/vsamohin/FactoryPlanningV2/model/task_results/dataset_table_35_20.csv",
    "/home/vsamohin/FactoryPlanningV2/model/task_results/dataset_table_35_25.csv",
    "/home/vsamohin/FactoryPlanningV2/model/task_results/dataset_table_35_40.csv",
    "/home/vsamohin/FactoryPlanningV2/model/task_results/dataset_table_45_20.csv",
    "/home/vsamohin/FactoryPlanningV2/model/task_results/dataset_table_45_25.csv",
    "/home/vsamohin/FactoryPlanningV2/model/task_results/dataset_table_35_40.csv",
    "/home/vsamohin/FactoryPlanningV2/model/task_results/dataset_table_45_40.csv",
    "/home/vsamohin/FactoryPlanningV2/model/task_results/dataset_table_45_70.csv",
    "/home/vsamohin/FactoryPlanningV2/model/task_results/dataset_table_50_20.csv",
    "/home/vsamohin/FactoryPlanningV2/model/task_results/dataset_table_50_25.csv",
    "/home/vsamohin/FactoryPlanningV2/model/task_results/dataset_table_50_40.csv",
    "/home/vsamohin/FactoryPlanningV2/model/task_results/dataset_table_50_70.csv",
]

lines_per_file = 1000
output_file = "/home/vsamohin/FactoryPlanningV2/model/task_results/randomResults.csv"

all_selected_lines = []

for file_path in file_paths:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Пропускаем заголовок если нужно
        data_lines = [line for line in lines[1:] if line.strip()]
        
        if len(data_lines) < lines_per_file:
            print(f"⚠️ В файле {file_path} всего {len(data_lines)} строк. Берём все.")
            selected = data_lines
        else:
            selected = random.sample(data_lines, lines_per_file)
        
        all_selected_lines.extend(selected)
        print(f"✅ Из файла {file_path} взято {len(selected)} строк")
        
    except FileNotFoundError:
        print(f"❌ Файл {file_path} не найден")

# Перемешиваем и сохраняем
random.shuffle(all_selected_lines)
with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(all_selected_lines)

print(f"\n✅ Готово! Всего {len(all_selected_lines)} строк в {output_file}")