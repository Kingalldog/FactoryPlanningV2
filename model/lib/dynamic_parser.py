import re
from datetime import datetime
import numpy as np
import os

class DynamicParser:
    """Парсер для динамических файлов задач планирования"""
    
    def __init__(self):
        self.data = {}
    
    def parse_file(self, filepath):
        """Основной метод парсинга файла"""
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Добавляем dynamic к имени файла для идентификации
        filename = os.path.basename(filepath)
        name_without_ext = os.path.splitext(filename)[0]
        dynamic_name = f"dynamic_{name_without_ext}.txt"
        
        self.data = {
            'filename': filepath,
            'dynamic_filename': dynamic_name,
            'prefront': self._parse_prefront(content),
            'tools': self._parse_tools(content),
            'operations': self._parse_operations(content),
            'times': self._parse_times(content),
            'work_blocks': self._parse_work_blocks(content),
            'graph_metrics': self._calculate_graph_metrics(content)
        }
        
        return self.data
    
    def _parse_prefront(self, content):
        """Парсинг PREFRONT секции"""
        match = re.search(r'PREFRONT\n(\d+)\n([\d\s]+)', content)
        if match:
            count = int(match.group(1))
            numbers = list(map(int, match.group(2).split()))
            return {'count': count, 'numbers': numbers}
        return {'count': 0, 'numbers': []}
    
    def _parse_tools(self, content):
        """Парсинг TOOLS секции"""
        match = re.search(r'TOOLS\n(\d+)\n(.*?)(?=\n\w+|\Z)', content, re.DOTALL)
        if match:
            count = int(match.group(1))
            tools_text = match.group(2).strip()
            
            # Парсим интервалы для каждого инструмента
            tool_intervals = []
            tool_blocks = re.findall(r'i(\([^)]+\)\s*\([^)]+\)[^i]*)', tools_text)
            
            if not tool_blocks:
                tool_blocks = re.findall(r'\(([^)]+)\s+([^)]+)\)', tools_text)
                if tool_blocks:
                    chunk_size = 40
                    tool_intervals = [tool_blocks[i:i+chunk_size] for i in range(0, len(tool_blocks), chunk_size)]
            else:
                for block in tool_blocks:
                    intervals = re.findall(r'\(([^)]+)\s+([^)]+)\)', block)
                    tool_intervals.append(intervals)
            
            return {
                'count': count,
                'tools': tool_intervals
            }
        return {'count': 0, 'tools': []}
    
    def _parse_operations(self, content):
        """Парсинг OPERATIONS секции"""
        match = re.search(r'OPERATIONS\n(\d+)\n([\d\s]+)', content)
        if match:
            count = int(match.group(1))
            operations = list(map(int, match.group(2).split()))
            return {'count': count, 'operations': operations}
        return {'count': 0, 'operations': []}
    
    def _parse_times(self, content):
        """Парсинг TIMES секции"""
        match = re.search(r'TIMES\n(.*?)(?=\n\w+|\Z)', content, re.DOTALL)
        if match:
            times_text = match.group(1).strip()
            lines = times_text.split('\n')
            times_matrix = []
            for line in lines:
                times = [self._parse_time_str(t.strip()) for t in line.split() if t.strip()]
                if times:
                    times_matrix.append(times)
            return times_matrix
        return []
    
    def _parse_time_str(self, time_str):
        """Парсинг строки времени вида 0_00:05:38"""
        if time_str == '0_00:00:00':
            return 0
        match = re.match(r'(\d+)_(\d+):(\d+):(\d+)', time_str)
        if match:
            days = int(match.group(1))
            hours = int(match.group(2))
            minutes = int(match.group(3))
            seconds = int(match.group(4))
            return days * 24 * 3600 + hours * 3600 + minutes * 60 + seconds
        return 0
    
    def _parse_work_blocks(self, content):
        """Парсинг WORK блоков"""
        work_blocks = []
        work_matches = re.finditer(r'work\n(.*?)(?=\nwork|\Z)', content, re.DOTALL)
        
        for match in work_matches:
            block_text = match.group(1).strip()
            lines = block_text.split('\n')
            
            if len(lines) >= 5:
                parts = lines[0].split()
                if len(parts) >= 3:
                    block = {
                        'start_time': parts[0],
                        'end_time': parts[1],
                        'value': float(parts[2]),
                        'param1': int(parts[3]) if len(parts) > 3 else 0,
                        'param2': int(parts[4]) if len(parts) > 4 else 0,
                        'count1': int(parts[5]) if len(parts) > 5 else 0,
                        'nodes': [],
                        'edges': []
                    }
                    
                    if len(lines) > 1:
                        block['nodes'] = list(map(int, lines[1].split()))
                    
                    for line in lines[2:]:
                        if line.strip() and not line.startswith('work'):
                            edge_parts = line.split()
                            if len(edge_parts) == 2:
                                block['edges'].append((int(edge_parts[0]), int(edge_parts[1])))
                    
                    work_blocks.append(block)
        
        return work_blocks
    
    def _calculate_graph_metrics(self, content):
        """Вычисление метрик графа из WORK блоков"""
        work_blocks = self._parse_work_blocks(content)
        
        total_nodes = 0
        total_edges = 0
        total_value = 0
        max_nodes = 0
        max_edges = 0
        
        for block in work_blocks:
            nodes = len(block['nodes'])
            edges = len(block['edges'])
            total_nodes += nodes
            total_edges += edges
            total_value += block['value']
            max_nodes = max(max_nodes, nodes)
            max_edges = max(max_edges, edges)
        
        return {
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'total_value': total_value,
            'max_nodes': max_nodes,
            'max_edges': max_edges,
            'avg_nodes': total_nodes / len(work_blocks) if work_blocks else 0,
            'avg_edges': total_edges / len(work_blocks) if work_blocks else 0,
            'density': total_edges / (total_nodes * (total_nodes - 1)) if total_nodes > 1 else 0
        }
    
    def extract_features(self, filepath):
        """Извлечение признаков для ML"""
        data = self.parse_file(filepath)
        features = {}
        
        # Dynamic идентификатор
        features['dynamic_id'] = hash(data['dynamic_filename']) % 10000
        features['dynamic_filename'] = data['dynamic_filename']
        
        # 1. Базовые характеристики
        features['n_prefront'] = data['prefront']['count']
        features['n_tools'] = data['tools']['count']
        features['n_operations'] = data['operations']['count']
        features['n_work_blocks'] = len(data['work_blocks'])
        
        # 2. Характеристики операций
        if data['operations']['operations']:
            ops = data['operations']['operations']
            features['ops_mean'] = np.mean(ops)
            features['ops_std'] = np.std(ops)
            features['ops_sum'] = np.sum(ops)
            features['ops_min'] = np.min(ops)
            features['ops_max'] = np.max(ops)
            features['ops_ratio'] = np.sum(ops) / len(ops) if len(ops) > 0 else 0
        else:
            features['ops_mean'] = 0
            features['ops_std'] = 0
            features['ops_sum'] = 0
            features['ops_min'] = 0
            features['ops_max'] = 0
            features['ops_ratio'] = 0
        
        # 3. Характеристики времен
        all_times = []
        for row in data['times']:
            all_times.extend(row)
        if all_times:
            non_zero_times = [t for t in all_times if t > 0]
            features['times_mean'] = np.mean(all_times)
            features['times_std'] = np.std(all_times)
            features['times_max'] = np.max(all_times)
            features['times_min'] = np.min(all_times)
            features['times_min_nonzero'] = np.min(non_zero_times) if non_zero_times else 0
            features['times_zero_ratio'] = sum(1 for t in all_times if t == 0) / len(all_times)
            features['times_total'] = np.sum(all_times)
            features['times_nonzero_count'] = len(non_zero_times)
        else:
            features['times_mean'] = 0
            features['times_std'] = 0
            features['times_max'] = 0
            features['times_min'] = 0
            features['times_min_nonzero'] = 0
            features['times_zero_ratio'] = 1
            features['times_total'] = 0
            features['times_nonzero_count'] = 0
        
        # 4. Графовые метрики
        features['graph_total_nodes'] = data['graph_metrics']['total_nodes']
        features['graph_total_edges'] = data['graph_metrics']['total_edges']
        features['graph_total_value'] = data['graph_metrics']['total_value']
        features['graph_max_nodes'] = data['graph_metrics']['max_nodes']
        features['graph_max_edges'] = data['graph_metrics']['max_edges']
        features['graph_avg_nodes'] = data['graph_metrics']['avg_nodes']
        features['graph_avg_edges'] = data['graph_metrics']['avg_edges']
        features['graph_density'] = data['graph_metrics']['density']
        
        # 5. Характеристики интервалов инструментов
        total_intervals = 0
        total_interval_duration = 0
        for tool in data['tools']['tools']:
            total_intervals += len(tool)
            for interval in tool:
                if isinstance(interval, tuple) and len(interval) == 2:
                    start = self._parse_datetime(interval[0])
                    end = self._parse_datetime(interval[1])
                    if start and end:
                        duration = (end - start).total_seconds()
                        total_interval_duration += duration
        
        features['tools_total_intervals'] = total_intervals
        features['tools_avg_duration'] = total_interval_duration / total_intervals if total_intervals > 0 else 0
        
        # 6. Соотношения
        features['ratio_ops_to_tools'] = features['n_operations'] / features['n_tools'] if features['n_tools'] > 0 else 0
        features['ratio_nodes_to_ops'] = features['graph_total_nodes'] / features['n_operations'] if features['n_operations'] > 0 else 0
        features['ratio_edges_to_nodes'] = features['graph_total_edges'] / features['graph_total_nodes'] if features['graph_total_nodes'] > 0 else 0
        
        return features
    
    def _parse_datetime(self, dt_str):
        """Парсинг строки даты-времени вида 01.02.26_14:57:43"""
        try:
            return datetime.strptime(dt_str, '%d.%m.%y_%H:%M:%S')
        except:
            return None