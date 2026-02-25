import os
import sys
import numpy as np
from heuristic_recommender import HeuristicRecommenderSystem

class InteractiveHeuristicRecommender:
    """
    Интерактивная система рекомендации эвристик
    """
    
    def __init__(self, model_path='heuristic_model.pkl'):
        self.model_path = model_path
        self.recommender = None
        
        if os.path.exists(model_path):
            self.recommender = HeuristicRecommenderSystem(model_path)
            print(f"✅ Модель загружена из {model_path}")
        else:
            print(f"⚠️ Модель {model_path} не найдена")
            print("   Сначала обучите модель командой: python heuristic_recommender.py --mode train --train_data <ваш_csv>")
    
    def interactive_mode(self):
        """Интерактивный режим"""
        print("\n" + "="*60)
        print("🤖 ИНТЕРАКТИВНЫЙ ПОМОЩНИК ВЫБОРА ЭВРИСТИКИ")
        print("="*60)
        
        if self.recommender is None:
            print("\n❌ Модель не загружена. Завершение.")
            return
        
        while True:
            print("\n" + "-"*40)
            print("Что хотите сделать?")
            print("1. Рекомендовать эвристику для задачи")
            print("2. Пакетная обработка нескольких задач")
            print("3. Информация о модели")
            print("0. Выход")
            
            choice = input("\nВаш выбор (0-3): ").strip()
            
            if choice == '1':
                self._recommend_single()
            elif choice == '2':
                self._batch_recommend()
            elif choice == '3':
                self._show_model_info()
            elif choice == '0':
                print("До свидания!")
                break
            else:
                print("Неверный выбор, попробуйте снова")
    
    def _recommend_single(self):
        """Рекомендация для одной задачи"""
        print("\n--- Рекомендация для одной задачи ---")
        filepath = input("Введите путь к файлу задачи: ").strip()
        
        if not os.path.exists(filepath):
            print(f"❌ Файл {filepath} не найден")
            return
        
        try:
            result = self.recommender.recommend(filepath, detailed=True)
            
            print("\n💡 ИТОГОВАЯ РЕКОМЕНДАЦИЯ:")
            print(f"   Используйте эвристику: {result['best']}")
            print(f"   (уверенность: {result['confidence']:.1%})")
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
    
    def _batch_recommend(self):
        """Пакетная обработка"""
        print("\n--- Пакетная обработка задач ---")
        
        # Спрашиваем как ввести список
        print("Как хотите ввести список задач?")
        print("1. Текстовый файл (по одной задаче на строку)")
        print("2. Ввести вручную (по одной, пустая строка для завершения)")
        
        choice = input("Ваш выбор (1-2): ").strip()
        
        problem_files = []
        
        if choice == '1':
            list_file = input("Путь к файлу со списком задач: ").strip()
            if not os.path.exists(list_file):
                print(f"❌ Файл {list_file} не найден")
                return
            
            with open(list_file, 'r') as f:
                problem_files = [line.strip() for line in f if line.strip()]
            
            print(f"Загружено {len(problem_files)} задач")
            
        elif choice == '2':
            print("Введите пути к файлам задач (пустая строка для завершения):")
            while True:
                path = input("> ").strip()
                if not path:
                    break
                if os.path.exists(path):
                    problem_files.append(path)
                else:
                    print(f"⚠️ Файл {path} не найден, пропускаем")
        
        if not problem_files:
            print("Нет задач для обработки")
            return
        
        # Спрашиваем куда сохранить
        output = input("Куда сохранить результаты (Enter для пропуска): ").strip()
        if not output:
            output = None
        
        # Запускаем обработку
        results = self.recommender.batch_recommend(problem_files, output)
        
        # Показываем сводку
        print("\n📊 Сводка по результатам:")
        heuristics_count = {}
        for r in results:
            h = r['best_heuristic']
            heuristics_count[h] = heuristics_count.get(h, 0) + 1
        
        for h, count in sorted(heuristics_count.items(), key=lambda x: x[1], reverse=True):
            print(f"   {h}: {count} задач ({count/len(results)*100:.1f}%)")
    
    def _show_model_info(self):
        """Информация о модели"""
        print("\n--- Информация о модели ---")
        
        if hasattr(self.recommender.trainer, 'feature_names') and self.recommender.trainer.feature_names:
            print(f"Количество признаков: {len(self.recommender.trainer.feature_names)}")
            
            if hasattr(self.recommender.trainer.model, 'feature_importances_'):
                importances = self.recommender.trainer.model.feature_importances_
                top_indices = np.argsort(importances)[-5:][::-1]
                
                print("\nТоп-5 важных признаков:")
                for i, idx in enumerate(top_indices, 1):
                    print(f"   {i}. {self.recommender.trainer.feature_names[idx]}: {importances[idx]:.4f}")
        
        if hasattr(self.recommender.trainer.model, 'classes_'):
            print(f"\nПоддерживаемые эвристики: {', '.join(self.recommender.trainer.model.classes_)}")


if __name__ == "__main__":
    # Можно указать путь к модели как аргумент
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'heuristic_model.pkl'
    
    recommender = InteractiveHeuristicRecommender(model_path)
    recommender.interactive_mode()