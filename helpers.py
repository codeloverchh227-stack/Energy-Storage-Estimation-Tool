# -*- coding: utf-8 -*-
"""
辅助函数
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


def ensure_directory(path):
    """
    确保目录存在

    Args:
        path: 目录路径

    Returns:
        str: 确保存在的目录路径
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def save_json(data, filepath):
    """
    保存数据到JSON文件

    Args:
        data: 要保存的数据
        filepath: 文件路径
    """
    ensure_directory(os.path.dirname(filepath))

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def load_json(filepath):
    """
    从JSON文件加载数据

    Args:
        filepath: 文件路径

    Returns:
        dict: 加载的数据
    """
    if not os.path.exists(filepath):
        return {}

    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_currency(value):
    """
    格式化货币显示

    Args:
        value: 数值

    Returns:
        str: 格式化后的字符串
    """
    if abs(value) >= 1e8:  # 大于1亿
        return f"{value / 1e8:.2f}亿元"
    elif abs(value) >= 1e4:  # 大于1万
        return f"{value / 1e4:.2f}万元"
    else:
        return f"{value:,.2f}元"


def format_power(value):
    """
    格式化功率显示

    Args:
        value: 功率值 (kW)

    Returns:
        str: 格式化后的字符串
    """
    if abs(value) >= 1000:  # 大于1MW
        return f"{value / 1000:.2f}MW"
    else:
        return f"{value:.2f}kW"


def format_energy(value):
    """
    格式化能量显示

    Args:
        value: 能量值 (kWh)

    Returns:
        str: 格式化后的字符串
    """
    if abs(value) >= 1e6:  # 大于1MWh
        return f"{value / 1e6:.2f}MWh"
    elif abs(value) >= 1000:  # 大于1MWh
        return f"{value / 1000:.2f}MWh"
    else:
        return f"{value:.2f}kWh"


def get_time_period_label(hour):
    """
    根据小时获取时段标签

    Args:
        hour: 小时 (0-23)

    Returns:
        str: 时段标签
    """
    if 10 <= hour < 15 or 18 <= hour < 21:
        return '峰'
    elif 15 <= hour < 18:
        return '尖峰'
    elif 8 <= hour < 10 or 21 <= hour < 23:
        return '平'
    else:
        return '谷'


def create_sample_data(output_path="data/sample_data.xlsx"):
    """
    创建示例数据文件

    Args:
        output_path: 输出文件路径
    """
    # 确保目录存在
    ensure_directory(os.path.dirname(output_path))

    # 创建储能信息
    storage_data = {
        '参数': ['容量_kWh', '功率_kW', '充电效率', '放电效率',
                 '最小SOC', '最大SOC', '初始SOC', '退化成本_元/kWh', '循环寿命'],
        '储能场景_1': [1000, 500, 0.95, 0.95, 0.1, 0.9, 0.5, 0.1, 5000],
        '储能场景_2': [2000, 1000, 0.96, 0.96, 0.15, 0.85, 0.6, 0.08, 6000],
        '储能场景_3': [500, 250, 0.94, 0.94, 0.2, 0.8, 0.4, 0.12, 4000]
    }

    # 创建电价和负荷数据
    n_periods = 24 * 4 * 30  # 30天，15分钟间隔

    # 生成时间序列
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    time_series = [start_time + timedelta(minutes=15 * i) for i in range(n_periods)]

    # 生成用户侧电价（分时电价）
    user_prices = []
    for t in time_series:
        hour = t.hour
        if 10 <= hour < 15 or 18 <= hour < 21:  # 峰时段
            price = 0.8 + np.random.random() * 0.2
        elif 15 <= hour < 18:  # 尖峰时段
            price = 1.2 + np.random.random() * 0.3
        elif 8 <= hour < 10 or 21 <= hour < 23:  # 平时段
            price = 0.5 + np.random.random() * 0.2
        else:  # 谷时段
            price = 0.2 + np.random.random() * 0.1
        user_prices.append(price)

    # 生成批发侧电价
    wholesale_prices = [p * 0.7 + np.random.random() * 0.1 for p in user_prices]

    # 生成负荷数据
    base_load = 1000  # 基础负荷
    load_data = []
    for t in time_series:
        hour = t.hour
        # 日负荷曲线
        if 8 <= hour < 12:  # 上午高峰
            load = base_load * (1.5 + np.random.random() * 0.3)
        elif 18 <= hour < 22:  # 晚间高峰
            load = base_load * (1.8 + np.random.random() * 0.4)
        elif 0 <= hour < 6:  # 夜间低谷
            load = base_load * (0.5 + np.random.random() * 0.2)
        else:  # 其他时间
            load = base_load * (0.9 + np.random.random() * 0.2)

        # 添加随机波动
        load *= (1 + np.random.random() * 0.1 - 0.05)
        load_data.append(load)

    # 生成需量数据（每月最大负荷）
    demand_data = []
    for i in range(n_periods):
        if i % (24 * 4) == 0:  # 每天重置
            daily_max = max(load_data[i:i + 24 * 4]) if i + 24 * 4 <= n_periods else load_data[i]
        demand_data.append(daily_max)

    # 创建DataFrame
    price_load_df = pd.DataFrame({
        '时间点': time_series,
        '用户侧电价': user_prices,
        '批发侧电价': wholesale_prices,
        '负荷': load_data,
        '需量': demand_data
    })

    storage_df = pd.DataFrame(storage_data)

    # 写入Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        storage_df.to_excel(writer, sheet_name='储能信息', index=False)
        price_load_df.to_excel(writer, sheet_name='电价负荷数据', index=False)

    print(f"示例数据已创建: {output_path}")
    return output_path


def validate_data(data):
    """
    验证数据完整性

    Args:
        data: 要验证的数据

    Returns:
        tuple: (是否有效, 错误消息)
    """
    if data is None or len(data) == 0:
        return False, "数据为空"

    # 检查必要的列
    required_columns = ['时间点', '用户侧电价']

    for col in required_columns:
        if col not in data.columns:
            return False, f"缺少必要列: {col}"

    # 检查数据有效性
    if data['用户侧电价'].isna().any():
        return False, "电价数据包含空值"

    if (data['用户侧电价'] < 0).any():
        return False, "电价数据包含负值"

    return True, "数据验证通过"