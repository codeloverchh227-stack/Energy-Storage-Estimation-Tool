# -*- coding: utf-8 -*-
"""
配置文件
"""


class Config:
    # 储能默认参数
    DEFAULT_STORAGE_PARAMS = {
        'capacity_kwh': 1000,  # 容量 (kWh)
        'power_kw': 500,  # 功率 (kW)
        'efficiency_charge': 0.95,  # 充电效率
        'efficiency_discharge': 0.95,  # 放电效率
        'soc_min': 0.1,  # 最小SOC
        'soc_max': 0.9,  # 最大SOC
        'soc_initial': 0.5,  # 初始SOC
        'degradation_cost': 0.1,  # 退化成本 (元/kWh)
        'cycle_life': 5000,  # 循环寿命
        'opex_per_kwh': 0.01,  # 运维成本 (元/kWh)
    }

    # 时间参数
    MINUTES_PER_HOUR = 60
    HOURS_PER_DAY = 24
    DAYS_PER_MONTH = 30

    # 输出设置
    OUTPUT_FORMATS = ['excel', 'csv']
    CHART_FORMATS = ['png', 'svg']

    # 电价时段划分（示例）
    PRICE_PERIODS = {
        '尖峰': (18, 21),
        '峰': (8, 12),
        '平': (12, 18),
        '谷': (0, 8)
    }

    # 颜色配置
    CHART_COLORS = {
        '充电': '#43AA8B',
        '放电': '#277DA1',
        'SOC': '#444444',
        '电价': '#FF6B6B',
        '负荷': '#4ECDC4',
        '用户侧电价': '#FFA600',
        '批发侧电价': '#FF5733',
        '综合电价': '#C44536'
    }

    # 优化参数
    OPTIMIZATION_PARAMS = {
        'solver': 'highs',
        'time_limit': 60,
        'mip_gap': 0.01
    }