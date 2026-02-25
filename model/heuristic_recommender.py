import os
import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from lib.model_trainer import ModelTrainer

class HeuristicRecommenderSystem:
    """
    Система рекомендации эвристик для задач планирования
    """
    
    def __init__(self, model_path=None):
        """
        Инициализация системы
        model_path: путь к сохраненной модели (если None, то режим только обучения)
        """
        self.trainer = ModelTrainer()
        self.model = None
        self.feature_names = None
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            # print(f"Модель загружена из {model_path}")
        else:
            print(f"Модель не загружена из {model_path}")
    
    def train(self, results_csv, model_save_path=None, test_size=0.2):
        """
        Режим 1: Обучение модели
        
        results_csv: путь к CSV с результатами
        model_save_path: куда сохранить модель (если None, модель не сохраняется)
        test_size: размер тестовой выборки
        """
        print("=" * 60)
        print("РЕЖИМ 1: ОБУЧЕНИЕ МОДЕЛИ")
        print("=" * 60)
        
        # Подготовка данных
        print("\n[1/4] Загрузка и обработка данных...")
        X, y, problem_paths = self.trainer.prepare_dataset(results_csv)
        
        if len(X) == 0:
            print("Ошибка: нет данных для обучения")
            return None
        
        print(f"\n[2/4] Обучение модели на {len(X)} примерах...")
        # Обучаем модель
        model, feature_importance = self.trainer.train(X, y)
        self.model = model
        self.feature_names = X.columns.tolist()
        
        # Сохраняем модель
        if model_save_path:
            self.save_model(model_save_path, feature_importance)
            print(f"\n[3/4] Модель сохранена в {model_save_path}")
        
        # Анализ результатов
        print("\n[4/4] Анализ обученной модели:")
        self._analyze_model_performance(X, y, test_size)
        
        # Анализ сильных сторон эвристик
        self._analyze_heuristic_strengths(X, y, feature_importance)
        
        return model
    
    def _analyze_model_performance(self, X, y, test_size=0.2):
        """Анализ производительности модели"""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Обучаем временную модель для оценки
        temp_model = RandomForestClassifier(n_estimators=200, random_state=42)
        temp_model.fit(X_train, y_train)
        y_pred = temp_model.predict(X_test)
        
        print("\n--- Оценка качества ---")
        print(f"Точность на тестовой выборке ({len(y_test)} примеров): {np.mean(y_pred == y_test):.2%}")
        
        # Матрица ошибок в процентах
        cm = confusion_matrix(y_test, y_pred, labels=temp_model.classes_)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        print("\n--- Матрица ошибок (в % от каждого класса) ---")
        print("      ", "  ".join([f"{c[:4]:>4}" for c in temp_model.classes_]))
        for i, row in enumerate(cm_percent):
            print(f"{temp_model.classes_[i][:6]:>6}", end=" ")
            for val in row:
                print(f"{val:5.1f}", end=" ")
            print()
    
    def _analyze_heuristic_strengths(self, X, y, feature_importance):
        """Анализ для каких задач какие эвристики лучше"""
        print("\n--- Сильные стороны эвристик ---")
        
        # Топ-5 самых важных признаков
        top_features = feature_importance.head(5)['feature'].tolist()
        
        for heuristic in set(y):
            indices = [i for i, val in enumerate(y) if val == heuristic]
            if len(indices) < 3:
                continue
                
            print(f"\n{heuristic} ({len(indices)} задач):")
            
            # Средние значения топ-признаков для этой эвристики
            X_heuristic = X.iloc[indices]
            X_mean = X.mean()
            
            for feat in top_features:
                if feat in X_heuristic.columns:
                    heur_mean = X_heuristic[feat].mean()
                    global_mean = X_mean[feat]
                    diff = (heur_mean - global_mean) / global_mean if global_mean != 0 else 0
                    
                    direction = "выше" if diff > 0.1 else "ниже" if diff < -0.1 else "≈"
                    if abs(diff) > 0.1:
                        print(f"  - {feat}: {direction} среднего на {abs(diff):.1%}")
    
    def recommend(self, problem_filepath, top_n=3, detailed=False):
        """
        Режим 2: Рекомендация эвристики для конкретной задачи
        
        problem_filepath: путь к файлу задачи
        top_n: количество рекомендуемых эвристик
        detailed: показывать детальную информацию
        """
        if self.model is None:
            raise ValueError("Модель не загружена! Сначала обучите или загрузите модель.")
        
        print("=" * 60)
        print("РЕЖИМ 2: РЕКОМЕНДАЦИЯ ЭВРИСТИКИ")
        print("=" * 60)
        
        # Получаем предсказание
        result = self.trainer.predict_best_heuristic(problem_filepath)
        
        print(f"\nФайл задачи: {os.path.basename(problem_filepath)}")
        print(f"\n🎯 Рекомендуемая эвристика: {result['best']}")
        print(f"   Уверенность модели: {result['confidence']:.1%}")
        
        if result['confidence'] < 0.5:
            print("   ⚠️ Низкая уверенность - рекомендуется проверить несколько вариантов")
        
        print(f"\n📊 Топ-{top_n} рекомендуемых эвристик:")
        for i, (h, prob) in enumerate(result['probabilities'][:top_n], 1):
            marker = "👉" if i == 1 else "  "
            print(f"   {marker} {i}. {h}: {prob:.1%}")
        
        if detailed:
            self._explain_recommendation(problem_filepath, result)
        
        return result

    def only_recommend(self, problem_filepath, top_n=3, detailed=False):
        if self.model is None:
            raise ValueError("Модель не загружена! Сначала обучите или загрузите модель.")
        # Получаем предсказание
        result = self.trainer.predict_best_heuristic(problem_filepath)
        
        if detailed:
            self._explain_recommendation(problem_filepath, result)
        
        return result
    
    def batch_recommend(self, problem_files, output_csv=None):
        """
        Пакетная рекомендация для нескольких задач
        
        problem_files: список путей к файлам задач
        output_csv: если указан, сохранить результаты в CSV
        """
        results = []
        
        print("=" * 60)
        print("ПАКЕТНАЯ РЕКОМЕНДАЦИЯ")
        print("=" * 60)
        
        for i, filepath in enumerate(problem_files, 1):
            print(f"\n[{i}/{len(problem_files)}] Обработка: {os.path.basename(filepath)}")
            
            try:
                result = self.trainer.predict_best_heuristic(filepath)
                results.append({
                    'file': filepath,
                    'best_heuristic': result['best'],
                    'confidence': result['confidence'],
                    'probabilities': dict(result['probabilities'])
                })
                print(f"   ✅ Рекомендация: {result['best']} (уверенность: {result['confidence']:.1%})")
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
                results.append({
                    'file': filepath,
                    'best_heuristic': 'ERROR',
                    'confidence': 0,
                    'error': str(e)
                })
        
        # Сохраняем в CSV если нужно
        if output_csv:
            df = pd.DataFrame(results)
            # Распаковываем вероятности в отдельные колонки
            for h in ['Dummy', 'Directive', 'Stoppable', 'Robin', 'FineSorter', 'Dependent']:
                df[f'prob_{h}'] = df['probabilities'].apply(lambda x: x.get(h, 0) if isinstance(x, dict) else 0)
            df = df.drop('probabilities', axis=1)
            df.to_csv(output_csv, index=False)
            print(f"\n💾 Результаты сохранены в {output_csv}")
        
        return results
    
    def _explain_recommendation(self, problem_filepath, result):
        """Объяснение почему выбрана именно эта эвристика"""
        print("\n🔍 Детальный анализ:")
        
        # Анализ уверенности
        if result['confidence'] >= 0.8:
            print("   - Очень высокая уверенность, можно смело использовать")
        elif result['confidence'] >= 0.6:
            print("   - Хорошая уверенность, рекомендуется к использованию")
        elif result['confidence'] >= 0.4:
            print("   - Средняя уверенность, стоит проверить альтернативы")
        else:
            print("   - Низкая уверенность, рекомендуется протестировать топ-3")
        
        # Сравнение топ-2
        if len(result['probabilities']) >= 2:
            top1, prob1 = result['probabilities'][0]
            top2, prob2 = result['probabilities'][1]
            ratio = prob1 / prob2 if prob2 > 0 else float('inf')
            
            if ratio > 2:
                print(f"   - {top1} значительно лучше {top2} (в {ratio:.1f} раз)")
            elif ratio > 1.5:
                print(f"   - {top1} moderately better than {top2}")
    
    def save_model(self, path, feature_importance=None):
        """Сохранение модели и метаданных"""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_importance': feature_importance.to_dict() if feature_importance is not None else None
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, path):
        """Загрузка модели"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        
        # Восстанавливаем модель в trainer
        self.trainer.model = self.model
        self.trainer.feature_names = self.feature_names


def main():
    parser = argparse.ArgumentParser(description='Система рекомендации эвристик для задач планирования')
    parser.add_argument('--mode', type=str, choices=['train', 'recommend', 'names_only', 'batch'], 
                       required=True, help='Режим работы: train (обучение) или recommend (рекомендация)')
    
    # Аргументы для обучения
    parser.add_argument('--train_data', type=str, help='Путь к CSV с обучающими данными')
    parser.add_argument('--save_model', type=str, default='heuristic_model.pkl', 
                       help='Куда сохранить модель (по умолчанию: heuristic_model.pkl)')
    
    # Аргументы для рекомендации
    parser.add_argument('--model', type=str, default='heuristic_model.pkl', 
                       help='Путь к сохраненной модели')
    parser.add_argument('--problem', type=str, help='Путь к файлу задачи для рекомендации')
    parser.add_argument('--problem_list', type=str, help='Путь к текстовому файлу со списком задач (по одной на строку)')
    parser.add_argument('--output', type=str, help='Куда сохранить результаты пакетной обработки')
    parser.add_argument('--top_n', type=int, default=3, help='Количество рекомендуемых эвристик')
    parser.add_argument('--detailed', action='store_true', help='Показывать детальную информацию')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Режим обучения
        if not args.train_data:
            print("Ошибка: для обучения необходимо указать --train_data")
            return
        
        recommender = HeuristicRecommenderSystem()
        recommender.train(args.train_data, args.save_model)
        
    elif args.mode == 'recommend':
        # Режим рекомендации для одной задачи
        if not args.problem:
            print("Ошибка: для рекомендации необходимо указать --problem")
            return
        
        if not os.path.exists(args.model):
            print(f"Ошибка: модель {args.model} не найдена")
            return
        
        recommender = HeuristicRecommenderSystem(args.model)
        
        if not os.path.exists(args.problem):
            print(f"Ошибка: файл задачи {args.problem} не найден")
            return
        
        recommender.recommend(args.problem, args.top_n, args.detailed)
    elif args.mode == 'names_only':
        # Режим вывода только имен алгоритмов
        if not args.problem:
            print("Error: need --problem")
            return
        
        recommender = HeuristicRecommenderSystem(args.model)
        result = recommender.only_recommend(args.problem, args.top_n, False)
        
        # Выводим только имена через запятую
        names = [h for h, prob in result['probabilities'][:args.top_n]]
        print(' '.join(names))
    elif args.mode == 'batch':
        # Режим пакетной обработки
        if not args.problem_list:
            print("Ошибка: для пакетной обработки необходимо указать --problem_list")
            return
        
        if not os.path.exists(args.model):
            print(f"Ошибка: модель {args.model} не найдена")
            return
        
        # Читаем список задач
        with open(args.problem_list, 'r') as f:
            problem_files = [line.strip() for line in f if line.strip()]
        
        recommender = HeuristicRecommenderSystem(args.model)
        recommender.batch_recommend(problem_files, args.output)


if __name__ == "__main__":
    main()