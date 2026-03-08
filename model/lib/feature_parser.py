import numpy as np
import re
from datetime import datetime
import networkx as nx

class ProblemFeatureExtractor:
    """Извлекает признаки из файла задачи"""
    
    def parse_time(self, time_str):
        """Парсит время из формата '24.02.26_11:45:56'"""
        time_part = time_str.split('_')[1]
        h, m, s = map(int, time_part.split(':'))
        return h * 3600 + m * 60 + s
    
    def extract_from_file(self, filepath):
        """Извлекает все признаки из файла задачи"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
        except FileNotFoundError:
            print(f"Файл не найден: {filepath}")
            return None
        
        try:
            lines = content.strip().split('\n')
            i = 0
            data = {}
            
            # Парсим TOOLS
            while i < len(lines) and 'TOOLS' not in lines[i]:
                i += 1
            if i >= len(lines):
                return None
            i += 1
            n_tools = int(lines[i])
            i += 1
            
            # Собираем информацию о занятости инструментов
            tool_intervals = []
            for tool_idx in range(n_tools):
                line = lines[i].strip()
                intervals = []
                if line.startswith('i'):
                    parts = line.split('(')[1:]
                    for part in parts:
                        if ')' in part:
                            times = part.split(')')[0].split()
                            if len(times) == 2:
                                start = self.parse_time(times[0])
                                end = self.parse_time(times[1])
                                intervals.append((start, end))
                tool_intervals.append(intervals)
                i += 1
            
            data['n_tools'] = n_tools
            data['tool_intervals'] = tool_intervals
            
            # Парсим OPERATIONS
            while i < len(lines) and 'OPERATIONS' not in lines[i]:
                i += 1
            if i >= len(lines):
                return None
            i += 1
            n_operations = int(lines[i])
            i += 1
            
            stoppable = list(map(int, lines[i].split()))
            data['n_operations'] = n_operations
            data['stoppable_ratio'] = sum(stoppable) / n_operations if n_operations > 0 else 0
            i += 1
            
            # Парсим TIMES
            while i < len(lines) and 'TIMES' not in lines[i]:
                i += 1
            if i >= len(lines):
                return None
            i += 1
            
            times_matrix = []
            for op_idx in range(n_operations):
                # if i >= len(lines):
                #     row = [0] * n_machines  # или другое значение по умолчанию
                #     times_matrix.append(row)
                #     continue
                row = []
                times_str = lines[i].strip().split()
                for t in times_str:
                    # Парсим "0_00:05:38"
                    days_str, time_str = t.split('_')
                    h, m, s = map(int, time_str.split(':'))
                    total_seconds = int(days_str) * 86400 + h * 3600 + m * 60 + s
                    row.append(total_seconds)
                times_matrix.append(row)
                i += 1
            
            data['times_matrix'] = times_matrix
            
            # Парсим works
            works_data = []
            while i < len(lines):
                if lines[i].strip() == 'work':
                    i += 1
                    if i >= len(lines):
                        break
                    work_header = lines[i].strip().split()
                    if len(work_header) < 6:
                        break
                    start_time = self.parse_time(work_header[0])
                    directive = self.parse_time(work_header[1])
                    fine_coef = float(work_header[2])
                    work_id = int(work_header[3])
                    cnt_edges = int(work_header[4])
                    size = int(work_header[5])
                    i += 1
                    
                    if i >= len(lines):
                        break
                    op_ids = list(map(int, lines[i].strip().split()))
                    i += 1
                    
                    dependencies = []
                    for _ in range(cnt_edges):
                        if i >= len(lines):
                            break
                        dep = tuple(map(int, lines[i].strip().split()))
                        dependencies.append(dep)
                        i += 1
                    
                    works_data.append({
                        'id': work_id,
                        'start': start_time,
                        'directive': directive,
                        'fine_coef': fine_coef,
                        'op_ids': op_ids,
                        'dependencies': dependencies,
                        'duration': directive - start_time
                    })
                else:
                    i += 1
            
            data['works'] = works_data
            data['n_works'] = len(works_data)
            
            # Вычисляем признаки
            features = self.compute_features(data)
            return features
        except Exception as e:
            return {}
    
    def compute_features(self, data):
        """Вычисляет признаки на основе распарсенных данных"""
        features = {}
        
        # 1. Размерностные признаки
        features['n_tools'] = data['n_tools']
        features['n_operations'] = data['n_operations']
        features['n_works'] = data['n_works']
        features['stoppable_ratio'] = data['stoppable_ratio']
        
        # 2. Признаки временной матрицы
        times = np.array(data['times_matrix'])
        non_zero = times[times > 0]
        
        if len(non_zero) > 0:
            features['avg_time'] = np.mean(non_zero)
            features['std_time'] = np.std(non_zero)
            features['min_time'] = np.min(non_zero)
            features['max_time'] = np.max(non_zero)
            features['time_range'] = features['max_time'] - features['min_time']
            features['time_cv'] = features['std_time'] / features['avg_time'] if features['avg_time'] > 0 else 0
        else:
            features['avg_time'] = 0
            features['std_time'] = 0
            features['min_time'] = 0
            features['max_time'] = 0
            features['time_range'] = 0
            features['time_cv'] = 0
        
        # Плотность матрицы
        total_cells = data['n_operations'] * data['n_tools']
        features['matrix_density'] = np.sum(times > 0) / total_cells if total_cells > 0 else 0
        
        # 3. Признаки работ
        if data['works']:
            durations = [w['duration'] for w in data['works']]
            fine_coefs = [w['fine_coef'] for w in data['works']]
            work_sizes = [len(w['op_ids']) for w in data['works']]
            
            features['avg_work_duration'] = np.mean(durations)
            features['std_work_duration'] = np.std(durations)
            features['avg_fine_coef'] = np.mean(fine_coefs)
            features['std_fine_coef'] = np.std(fine_coefs)
            features['avg_ops_per_work'] = np.mean(work_sizes)
            features['max_ops_per_work'] = np.max(work_sizes)
            features['std_ops_per_work'] = np.std(work_sizes)
            
            # Временное давление
            now = min(w['start'] for w in data['works'])
            slacks = [w['directive'] - now for w in data['works']]
            features['avg_slack'] = np.mean(slacks)
            features['min_slack'] = np.min(slacks)
            features['slack_range'] = np.max(slacks) - np.min(slacks)
        else:
            features['avg_work_duration'] = 0
            features['std_work_duration'] = 0
            features['avg_fine_coef'] = 0
            features['std_fine_coef'] = 0
            features['avg_ops_per_work'] = 0
            features['max_ops_per_work'] = 0
            features['std_ops_per_work'] = 0
            features['avg_slack'] = 0
            features['min_slack'] = 0
            features['slack_range'] = 0
        
        # 4. Признаки графа зависимостей
        G = nx.DiGraph()
        G.add_nodes_from(range(data['n_operations']))
        
        for work in data['works']:
            for dep in work['dependencies']:
                G.add_edge(dep[0], dep[1])
        
        features['n_edges'] = G.number_of_edges()
        if data['n_operations'] > 1:
            features['graph_density'] = features['n_edges'] / (data['n_operations'] * (data['n_operations'] - 1))
        else:
            features['graph_density'] = 0
        
        # Степени вершин
        in_degrees = [d for n, d in G.in_degree()]
        out_degrees = [d for n, d in G.out_degree()]
        
        if in_degrees:
            features['avg_in_degree'] = np.mean(in_degrees)
            features['max_in_degree'] = np.max(in_degrees)
        else:
            features['avg_in_degree'] = 0
            features['max_in_degree'] = 0
            
        if out_degrees:
            features['avg_out_degree'] = np.mean(out_degrees)
            features['max_out_degree'] = np.max(out_degrees)
        else:
            features['avg_out_degree'] = 0
            features['max_out_degree'] = 0
        
        # 5. Признаки загрузки инструментов
        all_intervals = []
        for tool_intervals in data['tool_intervals']:
            for start, end in tool_intervals:
                all_intervals.append(end - start)
        
        if all_intervals:
            features['avg_tool_busy'] = np.mean(all_intervals)
            features['std_tool_busy'] = np.std(all_intervals)
            features['total_tool_busy'] = sum(all_intervals)
        else:
            features['avg_tool_busy'] = 0
            features['std_tool_busy'] = 0
            features['total_tool_busy'] = 0
        
        return features