import os
import argparse
import pickle
import pandas as pd
import numpy as np
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
        else:
            if model_path:
                print(f"Модель не найдена по пути: {model_path}")
    
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
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn.preprocessing import LabelEncoder
        import numpy as np
        import pandas as pd
        
        print("\n" + "="*60)
        print("АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ МОДЕЛИ")
        print("="*60)
        
        if self.model is None:
            print("Ошибка: модель не обучена")
            return
        
        heuristic_names = self.trainer.heuristic_names if hasattr(self.trainer, 'heuristic_names') else \
                        ['Dummy', 'Directive', 'Stoppable', 'Robin', 'FineSorter', 'Dependent']
        
        if not y or not isinstance(y[0], str):
            print("Ошибка: ожидаются жесткие метки (hard labels)")
            return
        
        temp_encoder = LabelEncoder()
        y_encoded = temp_encoder.fit_transform(y)
        
        # Разделяем данные
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42,
            stratify=y_encoded if len(np.unique(y_encoded)) > 1 and min(np.bincount(y_encoded)) >= 2 else None
        )
        
        print(f"Размер тестовой выборки: {len(X_test)}")
        
        # Предсказания через predict_from_features
        y_pred = []
        for i in range(len(X_test)):
            test_row = X_test.iloc[i] if hasattr(X_test, 'iloc') else X_test[i]
            test_df = pd.DataFrame([test_row], columns=self.feature_names)
            
            result = self.model.predict_from_features(test_df)
            y_pred.append(result['best'])
        
        y_pred_encoded = temp_encoder.transform(y_pred)
        accuracy = accuracy_score(y_test, y_pred_encoded)
        
        print(f"\nТочность на тестовой выборке: {accuracy:.2%}")
        
        y_test_decoded = temp_encoder.inverse_transform(y_test)
        y_pred_decoded = temp_encoder.inverse_transform(y_pred_encoded)
        
        print("\nДетальный отчет по классам:")
        print(classification_report(y_test_decoded, y_pred_decoded))
        
        print("\n--- Матрица ошибок (в % от каждого класса) ---")
        cm = confusion_matrix(y_test, y_pred_encoded, normalize='true')
        print("       ", "  ".join([f"{c[:4]:>4}" for c in heuristic_names[:6]]))
        for i, h in enumerate(heuristic_names):
            if i < len(cm):
                row = " ".join([f"{val*100:5.1f}" for val in cm[i]])
                print(f"{h[:5]:<6} {row}")

    def _analyze_heuristic_strengths(self, X, y, feature_importance):
        """
        Анализ для каких задач какие эвристики лучше
        Поддерживает как hard, так и soft labels
        """
        print("\n" + "="*60)
        print("АНАЛИЗ СИЛЬНЫХ СТОРОН ЭВРИСТИК")
        print("="*60)
        
        # Получаем данные из trainer
        heuristic_names = self.trainer.heuristic_names
        
        # Определяем тип меток
        is_soft_labels = y and isinstance(y[0], dict)
        
        if is_soft_labels:
            # Для soft labels берем основную эвристику
            y_hard = []
            for soft in y:
                best_heuristic = max(soft.items(), key=lambda x: x[1])[0]
                y_hard.append(best_heuristic)
            y_series = pd.Series(y_hard)
        else:
            y_series = pd.Series(y)
        
        # Топ-10 самых важных признаков
        top_features = feature_importance.head(10)['feature'].tolist()
        
        # Общие средние значения
        overall_means = X.mean()
        
        for heuristic in heuristic_names:
            # Находим задачи, где эта эвристика была лучшей
            mask = y_series == heuristic
            count = mask.sum()
            
            if count < 5:
                print(f"\n{heuristic} ({count} задач): недостаточно данных для анализа")
                continue
            
            print(f"\n{heuristic} ({count} задач):")
            
            # Выбираем задачи с этой эвристикой
            heuristic_data = X[mask]
            
            # Анализируем топ-признаки
            significant_deviations = []
            for feat in top_features:
                if feat in heuristic_data.columns and feat in overall_means.index:
                    heur_mean = heuristic_data[feat].mean()
                    global_mean = overall_means[feat]
                    
                    if global_mean != 0:
                        diff_pct = (heur_mean - global_mean) / abs(global_mean) * 100
                        
                        # Показываем только значимые отклонения (>10%)
                        if abs(diff_pct) > 10:
                            direction = "выше" if diff_pct > 0 else "ниже"
                            significant_deviations.append((feat, direction, abs(diff_pct)))
            
            if significant_deviations:
                # Сортируем по величине отклонения
                significant_deviations.sort(key=lambda x: x[2], reverse=True)
                for feat, direction, pct in significant_deviations[:7]:
                    print(f"  - {feat}: {direction} среднего на {pct:.1f}%")
            else:
                print("  - Нет значимых отклонений (все признаки близки к средним)")

    def _analyze_model_performance_simple(self, X, y, test_size=0.2):
        """
        Упрощенный анализ производительности (для быстрой проверки)
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        
        print("\n" + "="*60)
        print("УПРОЩЕННЫЙ АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ")
        print("="*60)
        
        # Получаем данные из trainer
        label_encoder = self.trainer.label_encoder
        heuristic_names = self.trainer.heuristic_names
        scaler = self.trainer.scaler
        
        # Определяем тип меток
        is_soft_labels = y and isinstance(y[0], dict)
        
        if is_soft_labels:
            # Преобразуем в hard для простоты
            y_hard = []
            for soft in y:
                best_heuristic = max(soft.items(), key=lambda x: x[1])[0]
                y_hard.append(best_heuristic)
            y_encoded = label_encoder.transform(y_hard)
        else:
            if isinstance(y[0], str):
                y_encoded = label_encoder.transform(y)
            else:
                y_encoded = y
        
        # Масштабируем признаки
        X_scaled = scaler.transform(X)
        
        # Разделяем данные
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=test_size, random_state=42,
            stratify=y_encoded
        )
        
        print(f"Размер тестовой выборки: {len(y_test)}")
        
        # Делаем предсказания
        y_pred = self.model.predict(X_test)
        
        # Оценка
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nТочность на тестовой выборке: {accuracy:.2%}")
        
        # Детальный отчет
        y_test_decoded = label_encoder.inverse_transform(y_test)
        y_pred_decoded = label_encoder.inverse_transform(y_pred)
        
        print("\nДетальный отчет по классам:")
        print(classification_report(y_test_decoded, y_pred_decoded))
        
        # Матрица ошибок
        print("\n--- Матрица ошибок (в % от каждого класса) ---")
        cm = confusion_matrix(y_test, y_pred, normalize='true')
        print("       ", "  ".join([f"{c[:4]:>4}" for c in heuristic_names[:6]]))
        for i, h in enumerate(heuristic_names):
            if i < len(cm):
                row = " ".join([f"{val*100:5.1f}" for val in cm[i]])
                print(f"{h[:5]:<6} {row}")
    
    def recommend(self, problem_filepath, top_n=3, detailed=False):
        """
        Режим 2: Рекомендация эвристики для конкретной задачи
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
        """Только рекомендация без вывода"""
        if self.model is None:
            raise ValueError("Модель не загружена!")
        result = self.trainer.predict_best_heuristic(problem_filepath)
        if detailed:
            self._explain_recommendation(problem_filepath, result)
        return result
    
    def batch_recommend(self, problem_files, output_csv=None):
        """Пакетная рекомендация"""
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
        
        if output_csv:
            df = pd.DataFrame(results)
            heuristic_names = self.trainer.heuristic_names
            for h in heuristic_names:
                df[f'prob_{h}'] = df['probabilities'].apply(lambda x: x.get(h, 0) if isinstance(x, dict) else 0)
            df = df.drop('probabilities', axis=1)
            df.to_csv(output_csv, index=False)
            print(f"\n💾 Результаты сохранены в {output_csv}")
        
        return results
    
    def _explain_recommendation(self, problem_filepath, result):
        """Объяснение рекомендации"""
        print("\n🔍 Детальный анализ:")
        
        if result['confidence'] >= 0.8:
            print("   - Очень высокая уверенность, можно смело использовать")
        elif result['confidence'] >= 0.6:
            print("   - Хорошая уверенность, рекомендуется к использованию")
        elif result['confidence'] >= 0.4:
            print("   - Средняя уверенность, стоит проверить альтернативы")
        else:
            print("   - Низкая уверенность, рекомендуется протестировать топ-3")
        
        if len(result['probabilities']) >= 2:
            top1, prob1 = result['probabilities'][0]
            top2, prob2 = result['probabilities'][1]
            ratio = prob1 / prob2 if prob2 > 0 else float('inf')
            
            if ratio > 2:
                print(f"   - {top1} значительно лучше {top2} (в {ratio:.1f} раз)")
            elif ratio > 1.5:
                print(f"   - {top1} moderately better than {top2}")
    
    def save_model(self, path, feature_importance=None):
        """Сохранение модели"""
        self.trainer.save_model(path)
    
    def load_model(self, path):
        """Загрузка модели"""
        self.trainer.load_model(path)
        self.model = self.trainer.model
        self.feature_names = self.trainer.feature_names


def main():
    parser = argparse.ArgumentParser(description='Система рекомендации эвристик для задач планирования')
    parser.add_argument('--mode', type=str, choices=['train', 'recommend', 'names_only', 'batch'], 
                       required=True, help='Режим работы')
    
    # Аргументы для обучения
    parser.add_argument('--train_data', type=str, help='Путь к CSV с обучающими данными')
    parser.add_argument('--save_model', type=str, default='heuristic_model.pkl', 
                       help='Куда сохранить модель')
    
    # Аргументы для рекомендации
    parser.add_argument('--model', type=str, default='heuristic_model.pkl', 
                       help='Путь к сохраненной модели')
    parser.add_argument('--problem', type=str, help='Путь к файлу задачи')
    parser.add_argument('--problem_list', type=str, help='Путь к файлу со списком задач')
    parser.add_argument('--output', type=str, help='Куда сохранить результаты')
    parser.add_argument('--top_n', type=int, default=3, help='Количество рекомендуемых эвристик')
    parser.add_argument('--detailed', action='store_true', help='Показывать детальную информацию')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        if not args.train_data:
            print("Ошибка: для обучения необходимо указать --train_data")
            return
        
        recommender = HeuristicRecommenderSystem()
        recommender.train(args.train_data, args.save_model)
        
    elif args.mode == 'recommend':
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
        if not args.problem:
            print("Error: need --problem")
            return
        
        recommender = HeuristicRecommenderSystem(args.model)
        result = recommender.only_recommend(args.problem, args.top_n, False)
        names = [h for h, prob in result['probabilities'][:args.top_n]]
        print(' '.join(names))
        
    elif args.mode == 'batch':
        if not args.problem_list:
            print("Ошибка: для пакетной обработки необходимо указать --problem_list")
            return
        
        if not os.path.exists(args.model):
            print(f"Ошибка: модель {args.model} не найдена")
            return
        
        with open(args.problem_list, 'r') as f:
            problem_files = [line.strip() for line in f if line.strip()]
        
        recommender = HeuristicRecommenderSystem(args.model)
        recommender.batch_recommend(problem_files, args.output)


if __name__ == "__main__":
    main()