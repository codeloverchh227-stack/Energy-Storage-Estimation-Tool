# -*- coding: utf-8 -*-
"""
计算工具函数
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def calculate_equivalent_cycles(charge_energy, discharge_energy, capacity):
    """
    计算等效循环次数

    Args:
        charge_energy: 总充电量 (kWh)
        discharge_energy: 总放电量 (kWh)
        capacity: 储能容量 (kWh)

    Returns:
        float: 等效循环次数
    """
    if capacity <= 0:
        return 0

    # 等效循环次数 = (充电量 + 放电量) / (2 * 容量)
    total_throughput = charge_energy + discharge_energy
    equivalent_cycles = total_throughput / (2 * capacity)

    return equivalent_cycles


def calculate_degradation_cost(charge_energy, discharge_energy, cost_per_kwh):
    """
    计算退化成本

    Args:
        charge_energy: 总充电量 (kWh)
        discharge_energy: 总放电量 (kWh)
        cost_per_kwh: 退化成本 (元/kWh)

    Returns:
        float: 总退化成本
    """
    total_throughput = charge_energy + discharge_energy
    degradation_cost = total_throughput * cost_per_kwh

    return degradation_cost


def calculate_daily_revenue(discharge_power, charge_power, prices, time_step):
    """
    计算每日收益

    Args:
        discharge_power: 放电功率数组 (kW)
        charge_power: 充电功率数组 (kW)
        prices: 电价数组 (元/kWh)
        time_step: 时间步长 (分钟)

    Returns:
        dict: 包含各项收益的字典
    """
    dt = time_step / 60  # 转换为小时

    # 计算放电收入
    discharge_revenue = np.sum(discharge_power * dt * prices)

    # 计算充电成本
    charge_cost = np.sum(charge_power * dt * prices)

    # 计算净收益
    net_revenue = discharge_revenue - charge_cost

    return {
        'discharge_revenue': discharge_revenue,
        'charge_cost': charge_cost,
        'net_revenue': net_revenue
    }


def identify_price_periods(prices, time_stamps):
    """
    识别电价时段

    Args:
        prices: 电价数组
        time_stamps: 时间戳数组

    Returns:
        list: 时段标签列表
    """
    periods = []

    # 简单的时段划分（可根据实际需求调整）
    for price in prices:
        if price >= 1.0:
            periods.append('尖峰')
        elif price >= 0.7:
            periods.append('峰')
        elif price >= 0.4:
            periods.append('平')
        else:
            periods.append('谷')

    return periods


def calculate_load_factor(load_data):
    """
    计算负荷率

    Args:
        load_data: 负荷数据数组

    Returns:
        float: 负荷率
    """
    if len(load_data) == 0:
        return 0

    max_load = np.max(load_data)
    avg_load = np.mean(load_data)

    if max_load > 0:
        load_factor = avg_load / max_load * 100
    else:
        load_factor = 0

    return load_factor


def estimate_demand_charge_reduction(original_load, storage_power, demand_rate=30):
    """
    估算需量电费节省

    Args:
        original_load: 原始负荷 (kW)
        storage_power: 储能功率 (kW)
        demand_rate: 需量电价 (元/kW/月)

    Returns:
        float: 需量电费节省 (元/月)
    """
    if len(original_load) == 0:
        return 0

    # 计算原始最大需量
    original_demand = np.max(original_load)

    # 计算调整后负荷
    adjusted_load = original_load - storage_power

    # 确保负荷非负
    adjusted_load = np.maximum(adjusted_load, 0)

    # 计算新最大需量
    new_demand = np.max(adjusted_load)

    # 计算需量降低
    demand_reduction = max(0, original_demand - new_demand)

    # 计算费用节省
    cost_saving = demand_reduction * demand_rate

    return cost_saving


def calculate_self_consumption_rate(pv_generation, load_data):
    """
    计算自发自用率（如果存在光伏数据）

    Args:
        pv_generation: 光伏发电量 (kWh)
        load_data: 负荷数据 (kWh)

    Returns:
        float: 自发自用率 (%)
    """
    if len(pv_generation) == 0 or len(load_data) == 0:
        return 0

    total_pv = np.sum(pv_generation)
    total_load = np.sum(load_data)

    if total_pv > 0:
        # 简化：自发自用量 = min(光伏发电, 负荷)
        self_consumed = np.minimum(pv_generation, load_data)
        self_consumption_rate = np.sum(self_consumed) / total_pv * 100
    else:
        self_consumption_rate = 0

    return self_consumption_rate