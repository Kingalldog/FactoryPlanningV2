#!/usr/bin/env python3
# dynamic_recommender.py

import os
import argparse
import pickle
import pandas as pd
import numpy as np
from lib.dynamic_trainer import DynamicTrainer

class DynamicRecommender:
    """
    Динамическая система рекомендации эвристик для задач планирования
    """
    
    def __init__(self, model_path=None):
        self.trainer = DynamicTrainer()
        self.model = None
        self.feature_names = None
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            # print(f"✅ Модель загружена из {model_path}")
        else:
            print(f"ℹ️ Модель не найдена: {model_path}")
    
    def train(self, results_csv, model_save_path=None):
        """
        Обучение модели
        """
        print("=" * 60)
        print("🚀 ДИНАМИЧЕСКОЕ ОБУЧЕНИЕ МОДЕЛИ")
        print("=" * 60)
        
        # Подготовка данных
        print("\n[1/3] Загрузка и обработка данных...")
        X, y, problem_paths, dynamic_names = self.trainer.prepare_dataset(results_csv)
        
        if len(X) == 0:
            print("❌ Ошибка: нет данных для обучения")
            return None
        
        print(f"\n[2/3] Обучение модели на {len(X)} примерах...")
        model, feature_importance = self.trainer.train(X, y)
        self.model = model
        self.feature_names = X.columns.tolist()
        
        # Сохранение модели
        if model_save_path:
            self.save_model(model_save_path)
            print(f"\n[3/3] Модель сохранена в {model_save_path}")
        
        return model
    
    def recommend(self, problem_filepath, top_n=3, detailed=False):
        """
        Рекомендация для одной задачи
        """
        if self.model is None:
            raise ValueError("❌ Модель не загружена!")
        
        # print("=" * 60)
        # print("🎯 ДИНАМИЧЕСКАЯ РЕКОМЕНДАЦИЯ ЭВРИСТИКИ")
        # print("=" * 60)
        
        result = self.trainer.predict(problem_filepath)
        
        if 'error' in result:
            print(f"❌ Ошибка: {result['error']}")
            return result
        
        dynamic_name = result.get('dynamic_filename', os.path.basename(problem_filepath))
        # print(f"\n📄 Файл задачи: {dynamic_name}")
        # print(f"\n🏆 Рекомендуемая эвристика: {result['best']}")
        # print(f"   Уверенность: {result['confidence']:.1%}")
        
        # if result['confidence'] < 0.5:
        #     print("   ⚠️ Низкая уверенность - проверьте альтернативы")
        
        # print(f"\n📊 Топ-{top_n} рекомендаций:")
        for i, (h, prob) in enumerate(result['probabilities'][:top_n], 1):
            marker = "👉" if i == 1 else "  "
            print(f"{h}", end=" ")
        
        if detailed:
            self._explain_recommendation(result)
        
        return result
    
    def batch_recommend(self, problem_list_file, output_csv=None):
        """
        Пакетная рекомендация
        """
        if self.model is None:
            raise ValueError("❌ Модель не загружена!")
        
        # Чтение списка файлов
        with open(problem_list_file, 'r') as f:
            problem_files = [line.strip() for line in f if line.strip()]
        
        print("=" * 60)
        print("📦 ПАКЕТНАЯ РЕКОМЕНДАЦИЯ")
        print("=" * 60)
        print(f"Всего задач: {len(problem_files)}")
        
        results = self.trainer.batch_predict(problem_files)
        
        # Сохранение результатов
        if output_csv:
            df = pd.DataFrame([{
                'file': r.get('dynamic_filename', ''),
                'best_heuristic': r['best'],
                'confidence': r['confidence'],
                **{f'prob_{h}': r.get('all_probs', {}).get(h, 0) 
                   for h in self.trainer.heuristic_names}
            } for r in results])
            df.to_csv(output_csv, index=False)
            print(f"\n💾 Результаты сохранены в {output_csv}")
        
        return results
    
    def _explain_recommendation(self, result):
        """Объяснение рекомендации"""
        print("\n🔍 Детальный анализ:")
        
        confidence = result['confidence']
        if confidence >= 0.8:
            print("   ✅ Очень высокая уверенность")
        elif confidence >= 0.6:
            print("   📈 Хорошая уверенность")
        elif confidence >= 0.4:
            print("   📊 Средняя уверенность")
        else:
            print("   ⚠️ Низкая уверенность")
        
        if len(result['probabilities']) >= 2:
            top1, prob1 = result['probabilities'][0]
            top2, prob2 = result['probabilities'][1]
            ratio = prob1 / prob2 if prob2 > 0 else float('inf')
            
            if ratio > 2:
                print(f"   📊 {top1} значительно лучше {top2} (в {ratio:.1f} раз)")
    
    def save_model(self, path):
        """Сохранение модели"""
        self.trainer.save_model(path)
    
    def load_model(self, path):
        """Загрузка модели"""
        self.trainer.load_model(path)
        self.model = self.trainer.model
        self.feature_names = self.trainer.feature_names


def main():
    parser = argparse.ArgumentParser(description='Динамическая система рекомендации эвристик')
    parser.add_argument('--mode', type=str, choices=['train', 'recommend', 'batch'], 
                       required=True, help='Режим работы')
    
    # Аргументы для обучения
    parser.add_argument('--train_data', type=str, help='CSV с обучающими данными')
    parser.add_argument('--save_model', type=str, default='dynamic_model.pkl', 
                       help='Куда сохранить модель')
    
    # Аргументы для рекомендации
    parser.add_argument('--model', type=str, default='dynamic_model.pkl', 
                       help='Путь к модели')
    parser.add_argument('--problem', type=str, help='Файл задачи')
    parser.add_argument('--problem_list', type=str, help='Список задач')
    parser.add_argument('--output', type=str, help='Выходной CSV')
    parser.add_argument('--top_n', type=int, default=3, help='Топ-N рекомендаций')
    parser.add_argument('--detailed', action='store_true', help='Детальная информация')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        if not args.train_data:
            print("Ошибка: нужно --train_data")
            return
        
        recommender = DynamicRecommender()
        recommender.train(args.train_data, args.save_model)
    
    elif args.mode == 'recommend':
        if not args.problem:
            print("Ошибка: нужно --problem")
            return
        
        if not os.path.exists(args.model):
            print(f"Ошибка: модель {args.model} не найдена")
            return
        
        recommender = DynamicRecommender(args.model)
        recommender.recommend(args.problem, args.top_n, args.detailed)
    
    elif args.mode == 'batch':
        if not args.problem_list:
            print("Ошибка: нужно --problem_list")
            return
        
        recommender = DynamicRecommender(args.model)
        recommender.batch_recommend(args.problem_list, args.output)


if __name__ == "__main__":
    main()