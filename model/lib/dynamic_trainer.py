# lib/dynamic_trainer.py

import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import pickle
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from lib.dynamic_parser import DynamicParser

class DynamicTrainer:
    """Динамический тренер для модели предсказания лучшей эвристики"""
    
    def __init__(self):
        self.parser = DynamicParser()
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.heuristic_names = ['Dummy', 'Directive', 'Stoppable', 'Robin', 'FineSorter', 'Dependent']
        self._is_fitted = False
        self.scaler = StandardScaler()
        self.training_history = []
        
    def prepare_dataset(self, results_csv):
        """
        Подготовка датасета из CSV файла с результатами
        """
        results_df = pd.read_csv(results_csv, sep=';')
        
        X_list = []
        y_list = []
        problem_paths = []
        skipped_files = []
        dynamic_names = []
        
        print(f"Всего записей в CSV: {len(results_df)}")
        print(f"Колонки: {results_df.columns.tolist()}")
        
        for idx, row in results_df.iterrows():
            problem_path = row.iloc[0]
            
            # Извлекаем признаки
            features = self.parser.extract_features(problem_path)
            
            if features is not None and len(features) > 2:  # Больше 2 признаков (dynamic_id и filename)
                scores = {}
                for h in self.heuristic_names:
                    if h in row.index:
                        try:
                            scores[h] = float(row[h])
                        except (ValueError, TypeError):
                            continue
                
                if scores:
                    best_heuristic = min(scores, key=scores.get)
                    
                    # Сохраняем dynamic имя для истории
                    if 'dynamic_filename' in features:
                        dynamic_names.append(features['dynamic_filename'])
                    
                    # Удаляем строковые признаки для обучения
                    features_for_training = {k: v for k, v in features.items() 
                                           if k not in ['dynamic_filename', 'dynamic_id']}
                    
                    X_list.append(features_for_training)
                    y_list.append(best_heuristic)
                    problem_paths.append(problem_path)
                else:
                    skipped_files.append(problem_path)
            else:
                skipped_files.append(problem_path)
            
            if (idx + 1) % 50 == 0:
                print(f"Обработано {idx + 1} задач...")
        
        if not X_list:
            print("Ошибка: нет данных для обучения")
            return pd.DataFrame(), [], []
        
        X = pd.DataFrame(X_list)
        self.feature_names = X.columns.tolist()
        
        print(f"\n✅ Успешно обработано: {len(X)} задач")
        print(f"❌ Пропущено: {len(skipped_files)} задач")
        print(f"📊 Количество признаков: {len(self.feature_names)}")
        
        if len(skipped_files) > 0:
            print("\nПримеры пропущенных файлов:")
            for f in skipped_files[:5]:
                print(f"  {f}")
        
        # Анализ распределения
        y_series = pd.Series(y_list)
        print("\n📈 Распределение лучших эвристик:")
        total = len(y_series)
        for h in self.heuristic_names:
            count = (y_series == h).sum()
            if count > 0:
                print(f"  {h}: {count} ({count/total*100:.1f}%)")
        
        return X, y_list, problem_paths, dynamic_names
    
    def train(self, X, y, validation_split=0.2):
        """
        Обучение динамической модели
        """
        unique_classes = np.unique(y)
        print(f"\n🎯 Уникальные классы: {unique_classes}")
        print(f"📊 Количество классов: {len(unique_classes)}")
        
        if len(unique_classes) < 2:
            raise ValueError("В данных только один класс! Невозможно обучить модель.")
        
        # Кодирование целевой переменной
        y_encoded = self.label_encoder.fit_transform(y)
        self._is_fitted = True
        
        print("\n📊 Распределение после кодирования:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            count = np.sum(y_encoded == i)
            print(f"  {class_name} (код {i}): {count} примеров")
        
        # Масштабирование признаков
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Разделение на train/test
        y_series = pd.Series(y)
        min_class_size = y_series.value_counts().min()
        
        stratify_param = y_encoded if min_class_size >= 2 and len(unique_classes) > 1 else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=validation_split, 
            random_state=42, stratify=stratify_param
        )
        
        print(f"\n📚 Обучающая выборка: {len(X_train)} примеров")
        print(f"🧪 Тестовая выборка: {len(X_test)} примеров")
        
        # Веса классов
        class_weights = len(y) / (len(unique_classes) * np.bincount(y_encoded))
        sample_weights = class_weights[y_train]
        
        # Создание XGBoost модели
        self.model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=2,
            gamma=0.05,
            reg_alpha=0.5,
            reg_lambda=1.5,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss',
            verbosity=0
        )
        
        # Обучение
        start_time = datetime.now()
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
            sample_weight=sample_weights
        )
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Оценка
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{'='*50}")
        print(f"✅ РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
        print(f"{'='*50}")
        print(f"⏱️ Время обучения: {training_time:.2f} сек")
        print(f"🎯 Точность на тесте: {accuracy:.2%}")
        
        unique_pred = np.unique(y_pred)
        print(f"📊 Уникальные предсказания: {unique_pred}")
        
        # Детальный отчет
        y_test_decoded = self.label_encoder.inverse_transform(y_test)
        y_pred_decoded = self.label_encoder.inverse_transform(y_pred)
        
        print("\n📋 Детальный отчет по классам:")
        print(classification_report(y_test_decoded, y_pred_decoded))
        
        # Матрица ошибок
        cm = confusion_matrix(y_test_decoded, y_pred_decoded, labels=self.label_encoder.classes_)
        print("\n📊 Матрица ошибок:")
        print("      ", "  ".join([f"{c[:4]:>4}" for c in self.label_encoder.classes_]))
        for i, row in enumerate(cm):
            print(f"{self.label_encoder.classes_[i][:6]:>6}", end=" ")
            for val in row:
                print(f"{val:4d}", end=" ")
            print()
        
        # Важность признаков
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔥 Топ-10 важных признаков:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
        
        # Сохраняем в историю
        self.training_history.append({
            'timestamp': datetime.now(),
            'accuracy': accuracy,
            'n_samples': len(X),
            'feature_importance': feature_importance.to_dict()
        })
        
        return self.model, feature_importance
    
    def predict(self, problem_filepath):
        """
        Предсказание для одного файла
        """
        if self.model is None:
            return {
                'best': 'Dummy',
                'probabilities': [('Dummy', 1.0)],
                'confidence': 1.0,
                'error': 'model not trained'
            }
        
        if not self._is_fitted:
            if hasattr(self.model, 'classes_'):
                self.label_encoder.classes_ = self.model.classes_
                self._is_fitted = True
            else:
                return {
                    'best': 'Dummy',
                    'probabilities': [('Dummy', 1.0)],
                    'confidence': 1.0,
                    'error': 'label encoder not fitted'
                }
        
        try:
            features = self.parser.extract_features(problem_filepath)
            
            if features is None or len(features) <= 2:
                return {
                    'best': 'Dummy',
                    'probabilities': [('Dummy', 1.0)],
                    'confidence': 1.0,
                    'error': 'Feature extraction failed'
                }
            
            dynamic_name = features.get('dynamic_filename', os.path.basename(problem_filepath))
            
            # Убираем строковые признаки
            features_for_pred = {k: v for k, v in features.items() 
                               if k not in ['dynamic_filename', 'dynamic_id']}
            
            features_df = pd.DataFrame([features_for_pred])
            
            # Добавляем недостающие признаки
            if self.feature_names:
                for col in self.feature_names:
                    if col not in features_df.columns:
                        features_df[col] = 0
                features_df = features_df[self.feature_names]
            
            # Масштабирование
            features_scaled = self.scaler.transform(features_df)
            features_scaled = pd.DataFrame(features_scaled, columns=self.feature_names)
            
            # Предсказание
            prediction_encoded = self.model.predict(features_scaled)[0]
            prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
            
            proba = self.model.predict_proba(features_scaled)[0]
            
            proba_dict = {}
            for i, class_name in enumerate(self.label_encoder.classes_):
                proba_dict[class_name] = float(proba[i])
            
            sorted_heuristics = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'dynamic_filename': dynamic_name,
                'best': prediction,
                'probabilities': sorted_heuristics,
                'confidence': float(max(proba)),
                'all_probs': proba_dict
            }
            
        except Exception as e:
            print(f"Ошибка при предсказании: {e}")
            import traceback
            traceback.print_exc()
            return {
                'best': 'Dummy',
                'probabilities': [('Dummy', 1.0)],
                'confidence': 1.0,
                'error': str(e)
            }
    
    def batch_predict(self, file_list):
        """
        Пакетное предсказание для списка файлов
        """
        results = []
        for i, filepath in enumerate(file_list):
            print(f"Обработка {i+1}/{len(file_list)}: {os.path.basename(filepath)}")
            result = self.predict(filepath)
            results.append(result)
        return results
    
    def save_model(self, filepath):
        """
        Сохранение модели
        """
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'label_encoder': self.label_encoder,
            'heuristic_names': self.heuristic_names,
            '_is_fitted': self._is_fitted,
            'scaler': self.scaler,
            'training_history': self.training_history
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"💾 Модель сохранена в {filepath}")
    
    def load_model(self, filepath):
        """
        Загрузка модели
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.heuristic_names = model_data['heuristic_names']
        self._is_fitted = model_data['_is_fitted']
        self.label_encoder = model_data['label_encoder']
        self.scaler = model_data['scaler']
        self.training_history = model_data.get('training_history', [])
        
        # print(f"📂 Модель загружена из {filepath}")
        # print(f"📊 Классы: {self.label_encoder.classes_}")
        # print(f"✅ Модель обучена: {self._is_fitted}")