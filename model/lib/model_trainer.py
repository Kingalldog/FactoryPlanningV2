import pandas as pd
import os
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

from lib.feature_parser import ProblemFeatureExtractor

class ModelTrainer:
    """Обучает модель предсказания лучшей эвристики"""
    
    def __init__(self):
        self.extractor = ProblemFeatureExtractor()
        self.model = None
        self.feature_names = None
        self.heuristic_names = ['Dummy', 'Directive', 'Stoppable', 'Robin', 'FineSorter', 'Dependent']
        
    def prepare_dataset(self, results_csv):
        """
        results_csv: путь к CSV с результатами (где первая колонка - путь к файлу задачи)
        """
        # Загружаем результаты
        results_df = pd.read_csv(results_csv, sep=';')
        
        X_list = []
        y_list = []
        problem_paths = []
        skipped_files = []
        
        print(f"Всего записей в CSV: {len(results_df)}")
        
        for idx, row in results_df.iterrows():
            problem_path = row.iloc[0]  # первая колонка - путь к файлу
            
            # Извлекаем признаки из файла задачи
            features = self.extractor.extract_from_file(problem_path)
            
            if features is not None:
                # Определяем лучшую эвристику по результатам
                scores = {}
                for h in self.heuristic_names:
                    if h in row.index:
                        scores[h] = row[h]
                
                if scores:
                    best_heuristic = min(scores, key=scores.get)
                    
                    X_list.append(features)
                    y_list.append(best_heuristic)
                    problem_paths.append(problem_path)
                else:
                    skipped_files.append(problem_path)
            else:
                skipped_files.append(problem_path)
            
            if (idx + 1) % 50 == 0:
                print(f"Обработано {idx + 1} задач...")
        
        X = pd.DataFrame(X_list)
        self.feature_names = X.columns.tolist()
        
        print(f"\nУспешно обработано: {len(X)} задач")
        print(f"Пропущено: {len(skipped_files)} задач")
        print(f"Количество признаков: {len(self.feature_names)}")
        
        if len(skipped_files) > 0:
            print("\nПримеры пропущенных файлов:")
            for f in skipped_files[:5]:
                print(f"  {f}")
        
        # Анализ распределения
        y_series = pd.Series(y_list)
        print("\nРаспределение лучших эвристик:")
        for h in self.heuristic_names:
            count = (y_series == h).sum()
            if count > 0:
                print(f"  {h}: {count} ({count/len(y_series)*100:.1f}%)")
        
        return X, y_list, problem_paths
    
    def train(self, X, y):
        """Обучает модель"""
        # Проверяем, что у нас достаточно данных для каждого класса
        y_series = pd.Series(y)
        min_class_size = y_series.value_counts().min()
        
        if min_class_size < 2:
            print("\nПредупреждение: некоторые классы имеют слишком мало примеров!")
            print("Объединяем редкие классы...")
            # Можно объединить редкие классы или использовать стратификацию
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=y if min_class_size >= 2 else None
        )
        
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        # Оценка на тесте
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n=== Результаты обучения ===")
        print(f"Точность на тестовой выборке: {accuracy:.2%}")
        
        # Детальный отчет
        print("\nДетальный отчет по классам:")
        print(classification_report(y_test, y_pred))
        
        # Кросс-валидация
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        print(f"\nКросс-валидация (5-fold): {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")
        
        # Важность признаков
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nТоп-10 важных признаков:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return self.model, feature_importance
    
    def predict_best_heuristic(self, problem_filepath):
        """Предсказывает лучшую эвристику для новой задачи"""
        if self.model is None:
            # raise ValueError("Модель не обучена! Сначала вызовите train().")
            return {
                'best': 'Dummy',
                'probabilities': [('Dummy', 1.0)],
                'confidence': 1.0,
                'error': 'model not trained'
            }
        
        features = self.extractor.extract_from_file(problem_filepath)
        if features is None:
            # raise ValueError(f"Не удалось извлечь признаки из файла: {problem_filepath}")
            return {
                'best': 'Dummy',
                'probabilities': [('Dummy', 1.0)],
                'confidence': 1.0,
                'error': 'Feature extraction failed'
            }
        
        features_df = pd.DataFrame([features])
        features_df = features_df.reindex(columns=self.feature_names, fill_value=0)
        
        prediction = self.model.predict(features_df)[0]
        proba = self.model.predict_proba(features_df)[0]
        
        proba_dict = dict(zip(self.model.classes_, proba))
        sorted_heuristics = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'best': prediction,
            'probabilities': sorted_heuristics,
            'confidence': max(proba)
        }