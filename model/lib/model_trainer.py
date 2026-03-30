import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

from lib.feature_parser import ProblemFeatureExtractor


class EnsembleHeuristicSelector:
    """
    Ансамбль моделей для выбора лучшей эвристики
    Robin теперь в простых задачах, тривиальные задачи исключаются
    """
    
    def __init__(self):
        self.extractor = ProblemFeatureExtractor()
        self.heuristic_names = ['Dummy', 'Directive', 'Stoppable', 'Robin', 'FineSorter', 'Dependent']
        
        # Модели (теперь только 2 категории: simple и complex)
        self.level1_model = None
        self.simple_model = None   # для Dummy, Directive, Robin
        self.complex_model = None  # для FineSorter, Dependent, Stoppable
        
        # Кодировщики
        self.label_encoder_simple = LabelEncoder()
        self.label_encoder_complex = LabelEncoder()
        self.scaler = StandardScaler()
        
        self.feature_names = None
        self._is_fitted = False
    
    # ==================== ФУНКЦИИ ПОДГОТОВКИ ДАННЫХ ====================
    
    def _categorize_complexity(self, y):
        """Категоризирует задачу по сложности - Robin теперь в простых"""
        categories = []
        for heuristic in y:
            if heuristic in ['Dummy', 'Directive', 'Robin']:
                categories.append('simple')
            else:  # FineSorter, Dependent, Stoppable
                categories.append('complex')
        return categories
    
    def _filter_trivial_tasks(self, results_df, time_diff_threshold=0.01):
        """
        Исключает задачи, где все эвристики показали почти одинаковое время
        """
        filtered_indices = []
        for idx, row in results_df.iterrows():
            times = []
            for h in self.heuristic_names:
                if h in row.index:
                    try:
                        times.append(float(row[h]))
                    except:
                        continue
            if len(times) > 0:
                min_time = min(times)
                max_time = max(times)
                # Если разница меньше threshold, исключаем
                if max_time - min_time > time_diff_threshold * min_time:
                    filtered_indices.append(idx)
        
        filtered_df = results_df.iloc[filtered_indices]
        print(f"Исключено тривиальных задач: {len(results_df) - len(filtered_df)}")
        return filtered_df
    
    def _extract_features_from_row(self, row):
        """Извлекает признаки из одной строки CSV"""
        scores = {}
        for h in self.heuristic_names:
            if h in row.index:
                try:
                    scores[h] = float(row[h])
                except (ValueError, TypeError):
                    continue
        return scores if scores else None
    
    def _get_best_heuristic(self, scores):
        """Возвращает лучшую эвристику по минимальному времени"""
        return min(scores, key=scores.get)
    
    def _process_single_task(self, problem_path):
        """Обрабатывает одну задачу: извлекает признаки"""
        return self.extractor.extract_from_file(problem_path)
    
    def prepare_dataset(self, results_csv, filter_trivial=True, time_diff_threshold=0.01):
        """
        Подготовка данных для ансамбля
        Возвращает: X, y, problem_paths (3 значения для интерфейса)
        """
        results_df = pd.read_csv(results_csv, sep=';')
        
        # Исключаем тривиальные задачи
        if filter_trivial:
            results_df = self._filter_trivial_tasks(results_df, time_diff_threshold)
        
        X_list = []
        y_list = []
        problem_paths = []
        skipped_files = []
        
        print("Загрузка и обработка данных...")
        
        for idx, row in results_df.iterrows():
            problem_path = row.iloc[0]
            features = self._process_single_task(problem_path)
            
            if features is not None:
                scores = self._extract_features_from_row(row)
                
                if scores:
                    best_heuristic = self._get_best_heuristic(scores)
                    X_list.append(features)
                    y_list.append(best_heuristic)
                    problem_paths.append(problem_path)
                else:
                    skipped_files.append(problem_path)
            else:
                skipped_files.append(problem_path)
            
            if (idx + 1) % 50 == 0:
                print(f"  Обработано {idx + 1} задач...")
        
        if not X_list:
            print("Ошибка: не удалось извлечь признаки ни из одного файла")
            return pd.DataFrame(), [], []
        
        X = pd.DataFrame(X_list)
        self.feature_names = X.columns.tolist()
        
        self._print_preparation_stats(X, skipped_files, y_list)
        
        return X, y_list, problem_paths
    
    def _print_preparation_stats(self, X, skipped_files, y_list):
        """Выводит статистику подготовки данных"""
        print(f"\nУспешно обработано: {len(X)} задач")
        print(f"Пропущено: {len(skipped_files)} задач")
        
        y_series = pd.Series(y_list)
        print("\nРаспределение лучших эвристик:")
        for h in self.heuristic_names:
            count = (y_series == h).sum()
            if count > 0:
                print(f"  {h}: {count} ({count/len(y_series)*100:.1f}%)")
    
    # ==================== ФУНКЦИИ ОБУЧЕНИЯ ====================
    
    def _split_data(self, X, y, stratify=True):
        """Разделяет данные на train/test"""
        if stratify and len(np.unique(y)) > 1:
            unique, counts = np.unique(y, return_counts=True)
            if np.min(counts) >= 2:
                stratify_param = y
            else:
                stratify_param = None
        else:
            stratify_param = None
        
        return train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=stratify_param
        )
    
    def _evaluate_model(self, model, X_test, y_test):
        """Оценивает модель на тестовых данных"""
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)
    
    def _train_level1_classifier(self, X_scaled, y_category_encoded):
        """Обучает классификатор для определения типа задачи (бинарный)"""
        print("\n[Уровень 1] Обучение классификатора типа задачи...")
        
        X_train, X_test, y_train, y_test = self._split_data(
            X_scaled, y_category_encoded, stratify=True
        )
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=1.0,
            reg_lambda=1.5,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )
        
        model.fit(X_train, y_train)
        
        accuracy = self._evaluate_model(model, X_test, y_test)
        print(f"  Точность классификации типа задачи: {accuracy:.2%}")
        
        return model
    
    def _train_specialized_model(self, X, y, model_name, n_estimators=400, max_depth=8):
        """
        Обучает специализированную модель
        
        Args:
            X: признаки
            y: метки
            model_name: 'simple' или 'complex'
            n_estimators: количество деревьев
            max_depth: максимальная глубина
        """
        encoders = {
            'simple': self.label_encoder_simple,
            'complex': self.label_encoder_complex
        }
        encoder = encoders.get(model_name)
        
        if encoder is None:
            raise ValueError(f"Unknown model_name: {model_name}. Use 'simple' or 'complex'")
        
        y_encoded = encoder.fit_transform(y)
        
        print(f"\n  Обучение модели для {model_name}...")
        print(f"    Всего примеров: {len(X)}")
        unique, counts = np.unique(y_encoded, return_counts=True)
        print(f"    Распределение классов: {dict(zip(encoder.classes_, counts))}")
        
        # Разделяем на train/val для early stopping
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42,
            stratify=y_encoded if len(np.unique(y_encoded)) > 1 else None
        )
        
        # Создаем модель с регуляризацией и early stopping
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=1.0,
            reg_lambda=1.5,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss',
            verbosity=0,
            early_stopping_rounds=20
        )
        
        # Обучаем с валидацией
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Получаем лучшее количество деревьев
        if hasattr(model, 'best_iteration'):
            best_n_estimators = model.best_iteration + 1
            print(f"    Лучшее количество деревьев: {best_n_estimators}")
        
        # Оценка на валидации
        accuracy = self._evaluate_model(model, X_val, y_val)
        print(f"    Точность на валидации: {accuracy:.2%}")
        
        return model
    
    def _get_data_by_category(self, X_scaled, y, y_categories, category):
        """Возвращает данные для указанной категории"""
        indices = [i for i, cat in enumerate(y_categories) if cat == category]
        if indices:
            return X_scaled.iloc[indices], [y[i] for i in indices]
        return None, None
    
    def _print_training_statistics(self, y_categories, y):
        """Выводит статистику обучения"""
        simple_count = sum(1 for cat in y_categories if cat == 'simple')
        complex_count = sum(1 for cat in y_categories if cat == 'complex')
        total = len(y)
        
        print("\n" + "="*60)
        print("СТАТИСТИКА РАСПРЕДЕЛЕНИЯ")
        print("="*60)
        print(f"Простые задачи (Dummy, Directive, Robin): {simple_count} ({simple_count/total*100:.1f}%)")
        print(f"Сложные задачи (FineSorter, Dependent, Stoppable): {complex_count} ({complex_count/total*100:.1f}%)")
        
        print("\nДетальное распределение:")
        for h in self.heuristic_names:
            count = sum(1 for heur in y if heur == h)
            if count > 0:
                print(f"  {h}: {count} ({count/total*100:.1f}%)")
    
    def train(self, X, y):
        """
        Обучает ансамблевую модель
        Возвращает: model, feature_importance
        """
        print("\n" + "="*60)
        print("ОБУЧЕНИЕ АНСАМБЛЕВОЙ МОДЕЛИ")
        print("="*60)
        
        # Категоризация (теперь 2 категории)
        y_categories = self._categorize_complexity(y)
        
        # Масштабирование
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        # Уровень 1 (бинарная классификация)
        category_map = {'simple': 0, 'complex': 1}
        y_category_encoded = np.array([category_map[cat] for cat in y_categories])
        self.level1_model = self._train_level1_classifier(X_scaled, y_category_encoded)
        
        # Уровень 2
        print("\n[Уровень 2] Обучение специализированных моделей...")
        
        X_simple, y_simple = self._get_data_by_category(X_scaled, y, y_categories, 'simple')
        X_complex, y_complex = self._get_data_by_category(X_scaled, y, y_categories, 'complex')
        
        if X_simple is not None and len(X_simple) > 0:
            self.simple_model = self._train_specialized_model(
                X_simple, y_simple, 'simple',
                n_estimators=400, max_depth=6
            )
        
        if X_complex is not None and len(X_complex) > 0:
            self.complex_model = self._train_specialized_model(
                X_complex, y_complex, 'complex',
                n_estimators=400, max_depth=6
            )
        
        self._is_fitted = True
        self._print_training_statistics(y_categories, y)
        
        feature_importance = self._get_feature_importance()
        
        return self, feature_importance
    
    def _get_feature_importance(self):
        """Собирает важность признаков из всех моделей"""
        importances = {}
        
        models = [
            ('level1', self.level1_model),
            ('simple', self.simple_model),
            ('complex', self.complex_model)
        ]
        
        for name, model in models:
            if model is not None and hasattr(model, 'feature_importances_'):
                for i, feat in enumerate(self.feature_names):
                    importances[feat] = importances.get(feat, 0) + model.feature_importances_[i]
        
        total = sum(importances.values())
        if total > 0:
            for feat in importances:
                importances[feat] /= total
        
        df = pd.DataFrame(list(importances.items()), columns=['feature', 'importance'])
        return df.sort_values('importance', ascending=False)
    
    # ==================== ФУНКЦИИ ПРЕДСКАЗАНИЯ ====================
    
    def _prepare_features_for_prediction(self, problem_filepath):
        """Подготавливает признаки для предсказания"""
        features = self.extractor.extract_from_file(problem_filepath)
        if features is None:
            return None
        
        features_df = pd.DataFrame([features])
        
        if self.feature_names:
            for col in self.feature_names:
                if col not in features_df.columns:
                    features_df[col] = 0
            features_df = features_df[self.feature_names]
        
        return self.scaler.transform(features_df)
    
    def _predict_category(self, features_scaled):
        """Предсказывает категорию задачи"""
        return self.level1_model.predict(features_scaled)[0]
    
    def _predict_simple(self, features_scaled):
        """Предсказывает эвристику для простой задачи (Dummy, Directive, Robin)"""
        if not self.simple_model:
            return 'Directive', [('Directive', 0.5), ('Dummy', 0.5), ('Robin', 0.0)], 0.5
        
        pred_encoded = self.simple_model.predict(features_scaled)[0]
        proba = self.simple_model.predict_proba(features_scaled)[0]
        prediction = self.label_encoder_simple.inverse_transform([pred_encoded])[0]
        confidence = max(proba)
        
        full_proba = {h: 0.0 for h in self.heuristic_names}
        for i, h in enumerate(self.label_encoder_simple.classes_):
            full_proba[h] = proba[i]
        sorted_probs = sorted(full_proba.items(), key=lambda x: x[1], reverse=True)
        
        return prediction, sorted_probs, confidence
    
    def _predict_complex(self, features_scaled):
        """Предсказывает эвристику для сложной задачи"""
        if not self.complex_model:
            return 'FineSorter', [('FineSorter', 0.6), ('Dependent', 0.3), ('Stoppable', 0.1)], 0.6
        
        pred_encoded = self.complex_model.predict(features_scaled)[0]
        proba = self.complex_model.predict_proba(features_scaled)[0]
        prediction = self.label_encoder_complex.inverse_transform([pred_encoded])[0]
        confidence = max(proba)
        
        full_proba = {h: 0.0 for h in self.heuristic_names}
        for i, h in enumerate(self.label_encoder_complex.classes_):
            full_proba[h] = proba[i]
        sorted_probs = sorted(full_proba.items(), key=lambda x: x[1], reverse=True)
        
        return prediction, sorted_probs, confidence
    
    def predict_from_features(self, features_df):
        """
        Предсказывает эвристику на основе DataFrame с признаками
        
        Args:
            features_df: DataFrame с признаками (1 строка)
        
        Returns:
            dict: результат предсказания
        """
        if hasattr(self, 'scaler') and self.scaler is not None:
            features_scaled = self.scaler.transform(features_df)
        else:
            features_scaled = features_df.values
        
        category = self._predict_category(features_scaled)
        
        if category == 0:  # simple
            prediction, sorted_probs, confidence = self._predict_simple(features_scaled)
        else:  # complex
            prediction, sorted_probs, confidence = self._predict_complex(features_scaled)
        
        return {
            'best': prediction,
            'probabilities': sorted_probs,
            'confidence': confidence
        }
    
    def _get_error_response(self, error_msg):
        """Возвращает ответ при ошибке"""
        return {
            'best': 'Dummy',
            'probabilities': [('Dummy', 1.0)],
            'confidence': 1.0,
            'error': error_msg
        }
    
    def predict_best_heuristic(self, problem_filepath):
        """Предсказывает лучшую эвристику"""
        if not self._is_fitted:
            return self._get_error_response('model not trained')
        
        try:
            features_scaled = self._prepare_features_for_prediction(problem_filepath)
            if features_scaled is None:
                return self._get_error_response('Feature extraction failed')
            
            category = self._predict_category(features_scaled)
            
            if category == 0:
                prediction, sorted_probs, confidence = self._predict_simple(features_scaled)
            else:
                prediction, sorted_probs, confidence = self._predict_complex(features_scaled)
            
            return {
                'best': prediction,
                'probabilities': sorted_probs,
                'confidence': confidence
            }
            
        except Exception as e:
            print(f"Ошибка при предсказании: {e}")
            import traceback
            traceback.print_exc()
            return self._get_error_response(str(e))
    
    # ==================== ФУНКЦИИ СОХРАНЕНИЯ/ЗАГРУЗКИ ====================
    
    def save_model(self, filepath):
        """Сохраняет ансамблевую модель"""
        model_data = {
            'level1_model': self.level1_model,
            'simple_model': self.simple_model,
            'complex_model': self.complex_model,
            'label_encoder_simple': self.label_encoder_simple,
            'label_encoder_complex': self.label_encoder_complex,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'heuristic_names': self.heuristic_names,
            '_is_fitted': self._is_fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Ансамблевая модель сохранена в {filepath}")
    
    def load_model(self, filepath):
        """Загружает ансамблевую модель"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.level1_model = model_data['level1_model']
        self.simple_model = model_data['simple_model']
        self.complex_model = model_data['complex_model']
        self.label_encoder_simple = model_data['label_encoder_simple']
        self.label_encoder_complex = model_data['label_encoder_complex']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.heuristic_names = model_data['heuristic_names']
        self._is_fitted = model_data['_is_fitted']


# ============================================================
# ОСНОВНОЙ КЛАСС (для интерфейса HeuristicRecommenderSystem)
# ============================================================

class ModelTrainer:
    """
    Класс для обучения ансамблевой модели
    Совместим с интерфейсом HeuristicRecommenderSystem
    """
    
    def __init__(self, filter_trivial=True, time_diff_threshold=0.01):
        """
        Args:
            filter_trivial: исключать ли тривиальные задачи
            time_diff_threshold: порог для определения тривиальных задач (в процентах)
        """
        self.ensemble = EnsembleHeuristicSelector()
        self.model = None
        self.feature_names = None
        self.label_encoder = LabelEncoder()
        self.heuristic_names = ['Dummy', 'Directive', 'Stoppable', 'Robin', 'FineSorter', 'Dependent']
        self.scaler = StandardScaler()
        self._is_fitted = False
        self.filter_trivial = filter_trivial
        self.time_diff_threshold = time_diff_threshold
    
    # ==================== ФУНКЦИИ ПОДГОТОВКИ ====================
    
    def prepare_dataset(self, results_csv):
        """
        Подготовка данных
        Возвращает: X, y, problem_paths (3 значения)
        """
        return self.ensemble.prepare_dataset(
            results_csv, 
            filter_trivial=self.filter_trivial,
            time_diff_threshold=self.time_diff_threshold
        )
    
    # ==================== ФУНКЦИИ ОБУЧЕНИЯ ====================
    
    def train(self, X, y):
        """
        Обучение ансамблевой модели
        Возвращает: model, feature_importance (2 значения)
        """
        model, feature_importance = self.ensemble.train(X, y)
        self.model = model
        self.feature_names = self.ensemble.feature_names
        self._is_fitted = True
        
        self.label_encoder.classes_ = self.ensemble.heuristic_names
        
        return model, feature_importance
    
    # ==================== ФУНКЦИИ ПРЕДСКАЗАНИЯ ====================
    
    def predict_best_heuristic(self, problem_filepath):
        """Предсказание лучшей эвристики"""
        return self.ensemble.predict_best_heuristic(problem_filepath)
    
    # ==================== ФУНКЦИИ СОХРАНЕНИЯ/ЗАГРУЗКИ ====================
    
    def save_model(self, filepath):
        """Сохранение модели"""
        self.ensemble.save_model(filepath)
    
    def load_model(self, filepath):
        """Загрузка модели"""
        self.ensemble.load_model(filepath)
        self.model = self.ensemble
        self.feature_names = self.ensemble.feature_names
        self._is_fitted = self.ensemble._is_fitted
    
    # ==================== ФУНКЦИИ АНАЛИЗА ====================
    
    def analyze_ties_by_class(self, results_csv):
        """Анализ ничьих по классам"""
        return self.ensemble.analyze_ties_by_class(results_csv)
    
    def analyze_ties_summary(self, results_csv):
        """Краткий анализ ничьих"""
        return self.ensemble.analyze_ties_summary(results_csv)