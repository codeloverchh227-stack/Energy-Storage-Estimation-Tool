# -*- coding: utf-8 -*-
"""
数据加载模块
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    def __init__(self):
        pass

    def load_storage_info(self, excel_path):
        """
        加载储能信息

        Args:
            excel_path: Excel文件路径

        Returns:
            dict: 储能参数字典
        """
        try:
            # 读取第一页
            df = pd.read_excel(excel_path, sheet_name=0, header=0)

            # 确保列名正确
            if df.shape[1] < 2:
                raise ValueError("Excel文件格式错误：第一页应至少有两列")

            # 提取储能参数
            storage_data = {}

            # 第一列是参数名，后面的列是不同场景
            param_names = df.iloc[:, 0].tolist()

            for col_idx in range(1, df.shape[1]):
                scenario_name = f"储能场景_{col_idx}" if df.columns[col_idx] == 'Unnamed: 0' else df.columns[col_idx]
                scenario_params = {}

                for i, param_name in enumerate(param_names):
                    value = df.iloc[i, col_idx]

                    # 处理参数名
                    param_key = self._parse_param_name(param_name)

                    # 转换数据类型
                    if isinstance(value, (int, float)):
                        scenario_params[param_key] = value
                    elif isinstance(value, str):
                        try:
                            scenario_params[param_key] = float(value)
                        except:
                            scenario_params[param_key] = value
                    else:
                        scenario_params[param_key] = value

                storage_data[scenario_name] = scenario_params

            return storage_data

        except Exception as e:
            raise Exception(f"加载储能信息失败: {str(e)}")

    def load_price_load_data(self, excel_path):
        """
        加载电价和负荷数据

        Args:
            excel_path: Excel文件路径

        Returns:
            DataFrame: 包含时间、电价、负荷、需量的数据
        """
        try:
            # 读取第二页
            df = pd.read_excel(excel_path, sheet_name=1, header=0)

            # 确保有时间列
            if '时间点' not in df.columns and '时间' not in df.columns:
                raise ValueError("Excel文件格式错误：第二页应包含时间列")

            # 标准化列名
            time_col = '时间点' if '时间点' in df.columns else '时间'
            df = df.rename(columns={time_col: '时间'})

            # 转换时间格式
            df['时间'] = pd.to_datetime(df['时间'])

            # 检查必要的列
            required_columns = ['时间']
            optional_columns = ['用户侧电价', '批发侧电价', '负荷', '需量']

            # 重命名列，统一命名
            column_mapping = {
                '用户电价': '用户侧电价',
                '批发电价': '批发侧电价',
                '用电负荷': '负荷',
                '电力需量': '需量'
            }

            df = df.rename(columns=column_mapping)

            # 检查是否有价格数据
            has_price_data = False
            for col in ['用户侧电价', '批发侧电价']:
                if col in df.columns:
                    has_price_data = True

                    # 清理价格数据
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

            if not has_price_data:
                raise ValueError("未找到电价数据，请确保包含用户侧电价或批发侧电价")

            # 清理负荷数据
            if '负荷' in df.columns:
                df['负荷'] = pd.to_numeric(df['负荷'], errors='coerce')
                df['负荷'] = df['负荷'].fillna(method='ffill').fillna(method='bfill')

            # 清理需量数据
            if '需量' in df.columns:
                df['需量'] = pd.to_numeric(df['需量'], errors='coerce')
                df['需量'] = df['需量'].fillna(method='ffill').fillna(method='bfill')

            # 按时间排序
            df = df.sort_values('时间')

            # 重置索引
            df = df.reset_index(drop=True)

            return df

        except Exception as e:
            raise Exception(f"加载电价负荷数据失败: {str(e)}")

    def calculate_time_step(self, price_data):
        """
        计算时间步长

        Args:
            price_data: 包含时间列的数据

        Returns:
            int: 时间步长（分钟）
        """
        if len(price_data) < 2:
            return 60  # 默认1小时

        time_diff = price_data['时间'].iloc[1] - price_data['时间'].iloc[0]
        step_minutes = time_diff.total_seconds() / 60

        # 取整到常见的步长
        common_steps = [5, 15, 30, 60]
        closest_step = min(common_steps, key=lambda x: abs(x - step_minutes))

        return closest_step

    def _parse_param_name(self, param_name):
        """
        解析参数名

        Args:
            param_name: 原始参数名

        Returns:
            str: 标准化参数名
        """
        # 参数名映射
        param_mapping = {
            '容量': 'capacity_kwh',
            '功率': 'power_kw',
            '充电效率': 'efficiency_charge',
            '放电效率': 'efficiency_discharge',
            '最小SOC': 'soc_min',
            '最大SOC': 'soc_max',
            '初始SOC': 'soc_initial',
            '退化成本': 'degradation_cost',
            '循环寿命': 'cycle_life',
            '运维成本': 'opex_per_kwh',
            '额定容量': 'capacity_kwh',
            '额定功率': 'power_kw'
        }

        # 尝试查找映射
        for key, value in param_mapping.items():
            if key in param_name:
                return value

        # 返回标准化后的名称
        return param_name.lower().replace(' ', '_').replace('（', '(').replace('）', ')')