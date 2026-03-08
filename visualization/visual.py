import json
import plotly.express as px
import pandas as pd
from datetime import datetime

def parse_datetime(date_str):
    return datetime.strptime(date_str, '%d.%m.%y_%H:%M:%S')

def create_gantt_express(json_file_path):
    """Создание графика Ганта с помощью Plotly Express"""
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    gantt_data = []
    
    operation_to_work = {}
    for work in data['works']:
        for operation_id in work['operations']:
            operation_to_work[operation_id] = work['id']

    for tool_group in data['tools']:
        tool = tool_group
        tool_id = tool['id']
            
        for work_process in tool['work_process']:
            operation_id = work_process['operation']
            work_id = operation_to_work.get(operation_id, -1)
                
            start_time = parse_datetime(work_process['start'])
            end_time = parse_datetime(work_process['end'])
                
            operation_info = next((op for op in data['operations'] if op['id'] == operation_id), None)
                
            if operation_info:
                gantt_data.append({
                    'Tool': f'{tool_id}',
                    'Operation': f'{operation_id}',
                    'Work': f'{work_id}',
                    'Start': start_time,
                    'Finish': end_time,
                    'Duration (s)': (end_time - start_time).total_seconds(),
                    'Appointed': 'True' if operation_info['appointed'] else 'False'
                })
    
    df = pd.DataFrame(gantt_data)
    
    if df.empty:
        print("Нет данных для построения графика")
        return

    fig = px.timeline(
        df, 
        x_start="Start", 
        x_end="Finish", 
        y="Tool",
        color="Work",
        hover_data=["Operation", "Appointed", "Duration (s)"],
        title="График Ганта: Выполнение операций"
    )
    
    fig.update_layout(
        height=600,
        xaxis_title="Время",
        yaxis_title="Инструменты",
        showlegend=True
    )
    
    fig.show()

create_gantt_express("../build/samples/solution_robin.json")