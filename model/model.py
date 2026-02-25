import os

from lib.model_trainer import ModelTrainer


if __name__ == "__main__":
    # Путь к вашему CSV файлу
    results_csv = "task_results/dataset_table_20_20.csv"
    
    # Создаем и обучаем модель
    trainer = ModelTrainer()
    
    print("Начинаем обработку данных...")
    X, y, problem_paths = trainer.prepare_dataset(results_csv)
    
    if len(X) > 0:
        # Обучаем модель
        model, feature_importance = trainer.train(X, y)
        
        # Пример предсказания для первой задачи из датасета
        if len(problem_paths) > 0:
            print(f"\n=== Пример предсказания ===")
            result = trainer.predict_best_heuristic(problem_paths[0])
            print(f"Файл: {problem_paths[0]}")
            print(f"Предсказанная лучшая эвристика: {result['best']}")
            print(f"Уверенность: {result['confidence']:.2%}")
            print("Топ-3 эвристики:")
            for h, prob in result['probabilities'][:3]:
                print(f"  {h}: {prob:.2%}")
    else:
        print("Не удалось обработать ни одного файла. Проверьте пути в CSV.")