import os
import subprocess

FOLDER_PATH = "../../../factory/dataset_directory_big/new_test_data/"
OUTPUT_CSV = "mk_predictions_results_2.csv"
PYTHON_SCRIPT = "../heuristic_recommender.py"
MODEL_PATH = "../heuristic_model.pkl"
TOP_N = 3

# Собираем файлы
files = [os.path.join(root, f) for root, _, files in os.walk(FOLDER_PATH) for f in files]
print(f"Найдено файлов: {len(files)}")

with open(OUTPUT_CSV, 'w', encoding='utf-8') as f:
    f.write("имя файла;1_лучший алгоритм;2;3;\n")
    
    for i, problem_file in enumerate(files, 1):
        filename = os.path.basename(problem_file)
        print(f"[{i}/{len(files)}] {filename}")
        
        cmd = f"python3 {PYTHON_SCRIPT} --mode names_only --model \"{MODEL_PATH}\" --problem \"{problem_file}\" --top_n {TOP_N}"
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                algs = [a.strip() for a in result.stdout.strip().split(',') if a.strip()]
                f.write(f"{filename};{algs[0] if algs else ''};{algs[1] if len(algs)>1 else ''};{algs[2] if len(algs)>2 else ''};\n")
                print(f"  ✅ {', '.join(algs[:3])}")
            else:
                f.write(f"{filename};ERROR;;;\n")
                print(f"  ❌ Ошибка")
        except:
            f.write(f"{filename};EXCEPTION;;;\n")
            print(f"  ❌ Исключение")

print(f"\n✅ Готово! Результаты в {OUTPUT_CSV}")