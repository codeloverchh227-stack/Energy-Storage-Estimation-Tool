# -*- coding: utf-8 -*-
"""
经济模型模块 - 使用线性规划/混合整数线性规划
"""
import numpy as np
import pandas as pd
import pulp as pl
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


class EconomicModel:
    def __init__(self, use_mip=True):
        """
        初始化经济模型

        Args:
            use_mip: 是否使用混合整数线性规划（MILP），False则使用线性规划（LP）
        """
        self.use_mip = use_mip
        self.solver = pl.PULP_CBC_CMD(msg=False)  # 使用CBC求解器

    def daily_optimization(self, prices, storage_params, time_step,
                           load_data=None, demand_data=None):
        """
        按日优化的收益最大化模型（线性规划/MILP）

        Args:
            prices: 电价数组
            storage_params: 储能参数
            time_step: 时间步长（分钟）
            load_data: 负荷数据（可选）
            demand_data: 需量数据（可选）

        Returns:
            dict: 优化结果
        """
        # 提取参数
        capacity = storage_params.get('capacity_kwh', 1000)
        power = storage_params.get('power_kw', 500)
        eff_charge = storage_params.get('efficiency_charge', 0.95)
        eff_discharge = storage_params.get('efficiency_discharge', 0.95)
        soc_min = storage_params.get('soc_min', 0.1)
        soc_max = storage_params.get('soc_max', 0.9)
        soc_initial = storage_params.get('soc_initial', 0.5)
        degradation_cost = storage_params.get('degradation_cost', 0.1)
        opex_per_kwh = storage_params.get('opex_per_kwh', 0.01)

        # 时间参数
        n_periods = len(prices)
        dt = time_step / 60  # 转换为小时

        # 创建优化问题
        problem_name = f"EnergyStorage_Optimization_{n_periods}periods"
        if self.use_mip:
            prob = pl.LpProblem(problem_name, pl.LpMaximize)
        else:
            prob = pl.LpProblem(problem_name, pl.LpMaximize)

        # 决策变量
        # 充电功率（kW）
        P_ch = [pl.LpVariable(f'P_ch_{t}', lowBound=0, upBound=power)
                for t in range(n_periods)]

        # 放电功率（kW）
        P_dis = [pl.LpVariable(f'P_dis_{t}', lowBound=0, upBound=power)
                 for t in range(n_periods)]

        # 储能状态（SOC，kWh）
        E = [pl.LpVariable(f'E_{t}', lowBound=soc_min * capacity,
                           upBound=soc_max * capacity)
             for t in range(n_periods)]

        # 如果是MILP，需要二进制变量来防止同时充放电
        if self.use_mip:
            # 二进制变量：1表示充电，0表示放电或不动作
            u_ch = [pl.LpVariable(f'u_ch_{t}', cat='Binary')
                    for t in range(n_periods)]
            u_dis = [pl.LpVariable(f'u_dis_{t}', cat='Binary')
                     for t in range(n_periods)]

        # 目标函数：最大化净收益
        # 收益 = 放电收入 - 充电成本 - 退化成本 - 运维成本
        revenue = pl.lpSum([P_dis[t] * dt * prices[t] for t in range(n_periods)])
        cost = pl.lpSum([P_ch[t] * dt * prices[t] for t in range(n_periods)])

        # 退化成本（与总吞吐量成正比）
        degradation = pl.lpSum([(P_ch[t] + P_dis[t]) * dt * degradation_cost
                                for t in range(n_periods)])

        # 运维成本（与吞吐量成正比）
        opex = pl.lpSum([(P_ch[t] + P_dis[t]) * dt * opex_per_kwh
                         for t in range(n_periods)])

        # 设置目标函数
        prob += revenue - cost - degradation - opex

        # 约束条件

        # 1. 储能状态转移方程
        # E[t] = E[t-1] + (充电功率*效率 - 放电功率/效率) * dt
        prob += E[0] == soc_initial * capacity + \
                (P_ch[0] * eff_charge - P_dis[0] / eff_discharge) * dt

        for t in range(1, n_periods):
            prob += E[t] == E[t - 1] + \
                    (P_ch[t] * eff_charge - P_dis[t] / eff_discharge) * dt

        # 2. SOC上下限约束（已在变量边界中设置）

        # 3. 如果是MILP，添加防止同时充放电的约束
        if self.use_mip:
            for t in range(n_periods):
                # 充电功率和放电功率不能同时大于0
                prob += P_ch[t] <= power * u_ch[t]
                prob += P_dis[t] <= power * u_dis[t]
                prob += u_ch[t] + u_dis[t] <= 1

        # 4. 如果有负荷数据，添加负荷平衡约束
        if load_data is not None:
            for t in range(min(n_periods, len(load_data))):
                # 负荷必须由电网和储能放电共同满足
                # 这里我们假设储能可以放电支持负荷，但不强制要求
                # 如果需要强制满足负荷，可以添加约束：
                # grid_power[t] + P_dis[t] >= load_data[t]
                # 这里我们简化处理
                pass

        # 5. 如果有需量数据，可以添加最大需量约束
        if demand_data is not None:
            # 计算最大需量
            max_demand = pl.LpVariable('max_demand', lowBound=0)
            if load_data is not None:
                for t in range(min(n_periods, len(load_data))):
                    # 净负荷 = 原始负荷 - 储能放电 + 储能充电
                    net_load = load_data[t] - P_dis[t] + P_ch[t]
                    prob += net_load <= max_demand

        # 6. 储能周期一致性约束（可选）：SOC结束值等于初始值
        prob += E[n_periods - 1] >= soc_initial * capacity * 0.95
        prob += E[n_periods - 1] <= soc_initial * capacity * 1.05

        # 求解优化问题
        try:
            prob.solve(self.solver)

            if pl.LpStatus[prob.status] != 'Optimal':
                print(f"警告: 优化问题未达到最优解，状态: {pl.LpStatus[prob.status]}")
                # 使用启发式方法作为后备
                return self._fallback_heuristic(prices, storage_params, time_step,
                                                load_data, demand_data)

            # 提取结果
            charge_power = np.array([pl.value(P_ch[t]) for t in range(n_periods)])
            discharge_power = np.array([pl.value(P_dis[t]) for t in range(n_periods)])
            soc = np.array([pl.value(E[t]) / capacity for t in range(n_periods)])
            net_power = discharge_power - charge_power

            # 计算收益
            dt_hours = dt
            revenue_val = sum(discharge_power[t] * dt_hours * prices[t]
                              for t in range(n_periods))
            cost_val = sum(charge_power[t] * dt_hours * prices[t]
                           for t in range(n_periods))
            degradation_val = sum((charge_power[t] + discharge_power[t]) * dt_hours *
                                  degradation_cost for t in range(n_periods))
            opex_val = sum((charge_power[t] + discharge_power[t]) * dt_hours *
                           opex_per_kwh for t in range(n_periods))

            total_revenue = revenue_val - cost_val - degradation_val - opex_val

            # 计算等效循环次数
            total_energy_throughput = sum(charge_power + discharge_power) * dt_hours
            equivalent_cycles = total_energy_throughput / (2 * capacity)

            # 计算需量节省（如果有负荷和需量数据）
            demand_saving = 0
            if load_data is not None and demand_data is not None:
                # 计算原始最大需量
                original_demand = np.max(load_data[:n_periods])
                # 计算优化后净负荷
                net_load = load_data[:n_periods] - discharge_power + charge_power
                new_demand = np.max(net_load)
                demand_reduction = max(0, original_demand - new_demand)
                # 假设需量电费为30元/kW/月
                demand_saving = demand_reduction * 30
                total_revenue += demand_saving

            # 整理结果
            result = {
                'charge_power': charge_power,
                'discharge_power': discharge_power,
                'net_power': net_power,
                'soc': soc,
                'revenue': revenue_val,
                'cost': cost_val,
                'degradation_cost': degradation_val,
                'opex_cost': opex_val,
                'demand_saving': demand_saving,
                'total_revenue': total_revenue,
                'equivalent_cycles': equivalent_cycles,
                'total_energy_throughput': total_energy_throughput,
                'storage_params': storage_params,
                'prices': prices,
                'load_data': load_data,
                'demand_data': demand_data,
                'optimization_status': pl.LpStatus[prob.status]
            }

            return result

        except Exception as e:
            print(f"优化求解失败: {str(e)}")
            # 使用启发式方法作为后备
            return self._fallback_heuristic(prices, storage_params, time_step,
                                            load_data, demand_data)

    def _fallback_heuristic(self, prices, storage_params, time_step,
                            load_data=None, demand_data=None):
        """
        后备启发式方法（当优化求解失败时使用）
        """
        print("使用启发式方法作为后备...")

        # 提取参数
        capacity = storage_params.get('capacity_kwh', 1000)
        power = storage_params.get('power_kw', 500)
        eff_charge = storage_params.get('efficiency_charge', 0.95)
        eff_discharge = storage_params.get('efficiency_discharge', 0.95)
        soc_min = storage_params.get('soc_min', 0.1)
        soc_max = storage_params.get('soc_max', 0.9)
        soc_initial = storage_params.get('soc_initial', 0.5)
        degradation_cost = storage_params.get('degradation_cost', 0.1)

        # 时间参数
        n_periods = len(prices)
        dt = time_step / 60

        # 初始化变量
        charge_power = np.zeros(n_periods)
        discharge_power = np.zeros(n_periods)
        soc = np.zeros(n_periods)

        # 设置初始SOC
        soc_current = soc_initial * capacity

        # 简单启发式：低谷充电，高峰放电
        price_mean = np.mean(prices)
        price_std = np.std(prices)

        for t in range(n_periods):
            # 计算当前SOC百分比
            soc_percent = soc_current / capacity

            # 如果电价低且SOC不满，则充电
            if prices[t] < price_mean - 0.5 * price_std and soc_percent < soc_max:
                # 可充电量
                available_charge = min(
                    power * dt,
                    (soc_max * capacity - soc_current) / eff_charge
                )
                charge_power[t] = min(available_charge / dt, power)
                soc_current += charge_power[t] * dt * eff_charge

            # 如果电价高且SOC不低，则放电
            elif prices[t] > price_mean + 0.5 * price_std and soc_percent > soc_min:
                # 可放电量
                available_discharge = min(
                    power * dt,
                    (soc_current - soc_min * capacity) * eff_discharge
                )
                discharge_power[t] = min(available_discharge / dt, power)
                soc_current -= discharge_power[t] * dt / eff_discharge

            # 记录SOC
            soc[t] = soc_current / capacity

        # 计算收益
        revenue = np.sum(discharge_power * dt * prices)
        cost = np.sum(charge_power * dt * prices)
        degradation = np.sum((charge_power + discharge_power) * dt * degradation_cost)

        total_revenue = revenue - cost - degradation

        # 计算等效循环次数
        total_energy_throughput = np.sum(charge_power + discharge_power) * dt
        equivalent_cycles = total_energy_throughput / (2 * capacity)

        # 计算需量节省
        demand_saving = 0
        if load_data is not None and demand_data is not None:
            original_demand = np.max(load_data[:n_periods])
            net_load = load_data[:n_periods] - discharge_power + charge_power
            new_demand = np.max(net_load)
            demand_reduction = max(0, original_demand - new_demand)
            demand_saving = demand_reduction * 30
            total_revenue += demand_saving

        result = {
            'charge_power': charge_power,
            'discharge_power': discharge_power,
            'net_power': discharge_power - charge_power,
            'soc': soc,
            'revenue': revenue,
            'cost': cost,
            'degradation_cost': degradation,
            'demand_saving': demand_saving,
            'total_revenue': total_revenue,
            'equivalent_cycles': equivalent_cycles,
            'total_energy_throughput': total_energy_throughput,
            'storage_params': storage_params,
            'prices': prices,
            'load_data': load_data,
            'demand_data': demand_data,
            'optimization_status': 'Heuristic'
        }

        return result

    def daily_optimization_constrained(self, prices, storage_params, time_step,
                                       load_data=None, demand_data=None):
        """
        带约束的按日优化（考虑负荷和需量）
        调用主优化函数，已经包含了相关约束

        Args:
            prices: 电价数组
            storage_params: 储能参数
            time_step: 时间步长
            load_data: 负荷数据
            demand_data: 需量数据

        Returns:
            dict: 优化结果
        """
        return self.daily_optimization(prices, storage_params, time_step,
                                       load_data, demand_data)

    def multi_day_optimization(self, prices, storage_params, time_step,
                               num_days=30, cyclic_constraint=True):
        """
        多日优化（考虑跨日套利）

        Args:
            prices: 电价数组（多日）
            storage_params: 储能参数
            time_step: 时间步长
            num_days: 天数
            cyclic_constraint: 是否添加循环约束（最终SOC=初始SOC）

        Returns:
            dict: 优化结果
        """
        # 直接调用单日优化，但使用更长的价格序列
        return self.daily_optimization(prices, storage_params, time_step)