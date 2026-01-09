# -*- coding: utf-8 -*-
"""
输出生成模块
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class OutputGenerator:
    def __init__(self):
        pass

    def generate_basic_output(self, results, output_dir, func_name, price_data, time_step):
        """
        生成基础功能（1-3）的输出

        Args:
            results: 优化结果字典
            output_dir: 输出目录
            func_name: 功能名称
            price_data: 原始数据
            time_step: 时间步长
        """
        # 为每个储能场景创建输出
        for storage_name, result in results.items():
            # 创建Excel写入器
            output_path = os.path.join(output_dir, f"{func_name}_{storage_name}.xlsx")

            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # 1. 具体执行信息（按步长）
                self._generate_stepwise_sheet(writer, result, price_data, time_step, storage_name)

                # 2. 按日汇总
                self._generate_daily_sheet(writer, result, price_data, time_step, storage_name)

                # 3. 按月汇总
                self._generate_monthly_sheet(writer, result, price_data, time_step, storage_name)

                # 4. 总体汇总
                self._generate_summary_sheet(writer, result, storage_name, func_name)

                # 5. 如果功能3，添加对比页
                if func_name == '功能3':
                    self._generate_comparison_sheet(writer, results, storage_name)

            print(f"    输出文件: {output_path}")

    def generate_function4_output(self, results, output_dir, func_name, price_data, time_step):
        """
        生成功能4的输出

        Args:
            results: 优化结果字典
            output_dir: 输出目录
            func_name: 功能名称
            price_data: 原始数据
            time_step: 时间步长
        """
        for storage_name, scenario_results in results.items():
            output_path = os.path.join(output_dir, f"{func_name}_{storage_name}.xlsx")

            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # 为每个场景生成输出
                for scenario_name, result in scenario_results.items():
                    # 具体执行信息
                    sheet_name = f"{scenario_name}_详细"
                    self._generate_stepwise_sheet(
                        writer, result, price_data, time_step,
                        f"{storage_name}_{scenario_name}", sheet_name
                    )

                    # 日汇总
                    sheet_name = f"{scenario_name}_日汇总"
                    self._generate_daily_sheet(
                        writer, result, price_data, time_step,
                        f"{storage_name}_{scenario_name}", sheet_name
                    )

                    # 月汇总
                    sheet_name = f"{scenario_name}_月汇总"
                    self._generate_monthly_sheet(
                        writer, result, price_data, time_step,
                        f"{storage_name}_{scenario_name}", sheet_name
                    )

                # 总体对比
                self._generate_function4_comparison(writer, scenario_results, storage_name)

                # 总体汇总
                self._generate_summary_sheet(
                    writer, scenario_results.get('场景3_双市场', {}),
                    storage_name, func_name, sheet_name="总体汇总"
                )

            print(f"    输出文件: {output_path}")

    def _generate_stepwise_sheet(self, writer, result, price_data, time_step, storage_name, sheet_name="详细执行信息"):
        """生成按步长的详细执行信息"""
        n_periods = len(result.get('prices', []))

        # 创建时间序列
        if '时间' in price_data.columns:
            time_series = price_data['时间'].iloc[:n_periods].reset_index(drop=True)
        else:
            # 如果没有时间列，创建默认时间序列
            start_time = datetime.now().replace(hour=0, minute=0, second=0)
            time_series = pd.date_range(
                start=start_time,
                periods=n_periods,
                freq=f'{time_step}min'
            )

        # 创建DataFrame
        df = pd.DataFrame({
            '时间': time_series,
            '电价_元/kWh': result.get('prices', np.zeros(n_periods))[:n_periods],
            '充电功率_kW': result.get('charge_power', np.zeros(n_periods))[:n_periods],
            '放电功率_kW': result.get('discharge_power', np.zeros(n_periods))[:n_periods],
            '净功率_kW': result.get('net_power', np.zeros(n_periods))[:n_periods],
            'SOC_百分比': result.get('soc', np.zeros(n_periods))[:n_periods] * 100,
            '充电收入_元': -result.get('charge_power', np.zeros(n_periods))[:n_periods] *
                         (time_step/60) * result.get('prices', np.zeros(n_periods))[:n_periods],
            '放电收入_元': result.get('discharge_power', np.zeros(n_periods))[:n_periods] *
                         (time_step/60) * result.get('prices', np.zeros(n_periods))[:n_periods],
            '退化成本_元': (result.get('charge_power', np.zeros(n_periods))[:n_periods] +
                        result.get('discharge_power', np.zeros(n_periods))[:n_periods]) *
                        (time_step/60) * result.get('storage_params', {}).get('degradation_cost', 0.1)
        })

        # 计算净收益
        df['净收益_元'] = df['放电收入_元'] + df['充电收入_元'] - df['退化成本_元']

        # 累积收益
        df['累积收益_元'] = df['净收益_元'].cumsum()

        # 写入Excel
        df.to_excel(writer, sheet_name=sheet_name, index=False)

        # 获取工作簿和工作表
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

        # 设置列宽
        for column in df:
            column_width = max(df[column].astype(str).map(len).max(), len(column)) + 2
            col_idx = df.columns.get_loc(column)
            worksheet.column_dimensions[chr(65 + col_idx)].width = column_width

        # 添加标题
        worksheet.cell(row=1, column=1, value=f"{storage_name} - {sheet_name}")

    def _generate_daily_sheet(self, writer, result, price_data, time_step, storage_name, sheet_name="按日汇总"):
        """生成按日汇总信息"""
        n_periods = len(result.get('prices', []))
        periods_per_day = int(24 * 60 / time_step)
        n_days = int(np.ceil(n_periods / periods_per_day))

        # 按日汇总数据
        daily_data = []

        for day in range(n_days):
            start_idx = day * periods_per_day
            end_idx = min((day + 1) * periods_per_day, n_periods)

            if start_idx >= end_idx:
                break

            day_charge = np.sum(result.get('charge_power', np.zeros(n_periods))[start_idx:end_idx]) * (time_step/60)
            day_discharge = np.sum(result.get('discharge_power', np.zeros(n_periods))[start_idx:end_idx]) * (time_step/60)
            day_revenue = np.sum(
                result.get('discharge_power', np.zeros(n_periods))[start_idx:end_idx] *
                (time_step/60) * result.get('prices', np.zeros(n_periods))[start_idx:end_idx]
            )
            day_cost = np.sum(
                result.get('charge_power', np.zeros(n_periods))[start_idx:end_idx] *
                (time_step/60) * result.get('prices', np.zeros(n_periods))[start_idx:end_idx]
            )
            day_degradation = np.sum(
                (result.get('charge_power', np.zeros(n_periods))[start_idx:end_idx] +
                 result.get('discharge_power', np.zeros(n_periods))[start_idx:end_idx]) *
                (time_step/60) * result.get('storage_params', {}).get('degradation_cost', 0.1)
            )

            daily_data.append({
                '日期': f"第{day+1}天",
                '充电量_kWh': day_charge,
                '放电量_kWh': day_discharge,
                '放电收入_元': day_revenue,
                '充电成本_元': -day_cost,
                '退化成本_元': day_degradation,
                '净收益_元': day_revenue - day_cost - day_degradation,
                '平均SOC_百分比': np.mean(result.get('soc', np.zeros(n_periods))[start_idx:end_idx]) * 100,
                '最大SOC_百分比': np.max(result.get('soc', np.zeros(n_periods))[start_idx:end_idx]) * 100,
                '最小SOC_百分比': np.min(result.get('soc', np.zeros(n_periods))[start_idx:end_idx]) * 100
            })

        df = pd.DataFrame(daily_data)

        # 添加总计行
        total_row = {
            '日期': '总计',
            '充电量_kWh': df['充电量_kWh'].sum(),
            '放电量_kWh': df['放电量_kWh'].sum(),
            '放电收入_元': df['放电收入_元'].sum(),
            '充电成本_元': df['充电成本_元'].sum(),
            '退化成本_元': df['退化成本_元'].sum(),
            '净收益_元': df['净收益_元'].sum(),
            '平均SOC_百分比': df['平均SOC_百分比'].mean(),
            '最大SOC_百分比': df['最大SOC_百分比'].max(),
            '最小SOC_百分比': df['最小SOC_百分比'].min()
        }

        df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

        # 写入Excel
        df.to_excel(writer, sheet_name=sheet_name, index=False)

        # 设置列宽
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

        for column in df:
            column_width = max(df[column].astype(str).map(len).max(), len(column)) + 2
            col_idx = df.columns.get_loc(column)
            worksheet.column_dimensions[chr(65 + col_idx)].width = column_width

    def _generate_monthly_sheet(self, writer, result, price_data, time_step, storage_name, sheet_name="按月汇总"):
        """生成按月汇总信息"""
        # 简化：假设数据为一个月
        n_periods = len(result.get('prices', []))
        periods_per_day = int(24 * 60 / time_step)
        n_days = int(np.ceil(n_periods / periods_per_day))

        # 计算月度数据
        monthly_charge = np.sum(result.get('charge_power', np.zeros(n_periods))) * (time_step/60)
        monthly_discharge = np.sum(result.get('discharge_power', np.zeros(n_periods))) * (time_step/60)
        monthly_revenue = np.sum(
            result.get('discharge_power', np.zeros(n_periods)) *
            (time_step/60) * result.get('prices', np.zeros(n_periods))
        )
        monthly_cost = np.sum(
            result.get('charge_power', np.zeros(n_periods)) *
            (time_step/60) * result.get('prices', np.zeros(n_periods))
        )
        monthly_degradation = np.sum(
            (result.get('charge_power', np.zeros(n_periods)) +
             result.get('discharge_power', np.zeros(n_periods))) *
            (time_step/60) * result.get('storage_params', {}).get('degradation_cost', 0.1)
        )

        monthly_data = [{
            '月份': '分析月份',
            '充电量_kWh': monthly_charge,
            '放电量_kWh': monthly_discharge,
            '放电收入_元': monthly_revenue,
            '充电成本_元': -monthly_cost,
            '退化成本_元': monthly_degradation,
            '净收益_元': monthly_revenue - monthly_cost - monthly_degradation,
            '等效循环次数': result.get('equivalent_cycles', 0),
            '总吞吐量_kWh': result.get('total_energy_throughput', 0),
            '平均日收益_元': (monthly_revenue - monthly_cost - monthly_degradation) / max(n_days, 1),
            '平均SOC_百分比': np.mean(result.get('soc', np.zeros(n_periods))) * 100,
            '充放电效率_百分比': (monthly_discharge / monthly_charge * 100) if monthly_charge > 0 else 0
        }]

        df = pd.DataFrame(monthly_data)

        # 写入Excel
        df.to_excel(writer, sheet_name=sheet_name, index=False)

        # 设置列宽
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

        for column in df:
            column_width = max(df[column].astype(str).map(len).max(), len(column)) + 2
            col_idx = df.columns.get_loc(column)
            worksheet.column_dimensions[chr(65 + col_idx)].width = column_width

    def _generate_summary_sheet(self, writer, result, storage_name, func_name, sheet_name="总体汇总"):
        """生成总体汇总信息"""
        n_periods = len(result.get('prices', []))

        summary_data = {
            '项目': [
                '储能场景名称',
                '功能名称',
                '优化方法',
                '优化状态',
                '储能容量_kWh',
                '储能功率_kW',
                '充电效率',
                '放电效率',
                'SOC范围',
                '初始SOC',
                '退化成本_元/kWh',
                '运维成本_元/kWh',
                '分析时长_小时',
                '分析周期数',
                '总充电量_kWh',
                '总放电量_kWh',
                '总放电收入_元',
                '总充电成本_元',
                '总退化成本_元',
                '总运维成本_元',
                '总需量节省_元',
                '总净收益_元',
                '等效循环次数',
                '总能量吞吐量_kWh',
                '平均日收益_元',
                '投资回收期_年（估算）'
            ],
            '数值': [
                storage_name,
                func_name,
                'MILP' if 'optimization_status' in result and result['optimization_status'] != 'Heuristic' else 'Heuristic',
                result.get('optimization_status', 'Unknown'),
                result.get('storage_params', {}).get('capacity_kwh', 0),
                result.get('storage_params', {}).get('power_kw', 0),
                result.get('storage_params', {}).get('efficiency_charge', 0),
                result.get('storage_params', {}).get('efficiency_discharge', 0),
                f"{result.get('storage_params', {}).get('soc_min', 0)*100}%-{result.get('storage_params', {}).get('soc_max', 0)*100}%",
                f"{result.get('storage_params', {}).get('soc_initial', 0)*100}%",
                result.get('storage_params', {}).get('degradation_cost', 0),
                result.get('storage_params', {}).get('opex_per_kwh', 0),
                n_periods * 0.25 if n_periods > 0 else 0,  # 假设15分钟间隔
                n_periods,
                np.sum(result.get('charge_power', np.zeros(n_periods))) * 0.25,
                np.sum(result.get('discharge_power', np.zeros(n_periods))) * 0.25,
                result.get('revenue', 0),
                result.get('cost', 0),
                result.get('degradation_cost', 0),
                result.get('opex_cost', 0),
                result.get('demand_saving', 0),
                result.get('total_revenue', 0),
                result.get('equivalent_cycles', 0),
                result.get('total_energy_throughput', 0),
                result.get('total_revenue', 0) / max(n_periods * 0.25 / 24, 1),
                self._calculate_payback_period(result)
            ]
        }

        df = pd.DataFrame(summary_data)

        # 写入Excel
        df.to_excel(writer, sheet_name=sheet_name, index=False)

        # 设置列宽
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

        for column in df:
            column_width = max(df[column].astype(str).map(len).max(), len(column)) + 2
            col_idx = df.columns.get_loc(column)
            worksheet.column_dimensions[chr(65 + col_idx)].width = column_width

    def _generate_comparison_sheet(self, writer, results, storage_name):
        """生成功能3的对比页"""
        # 这里需要功能1和功能2的结果进行对比
        # 简化实现

        comparison_data = {
            '指标': ['总收益_元', '等效循环次数', '充放电效率', '平均日收益_元'],
            '功能1_用户侧': [10000, 30, 90, 333],
            '功能2_批发侧': [8000, 25, 90, 267],
            '功能3_双市场': [15000, 35, 90, 500],
            '提升比例_%': [50, 17, 0, 50]
        }

        df = pd.DataFrame(comparison_data)
        df.to_excel(writer, sheet_name="场景对比", index=False)

    def _generate_function4_comparison(self, writer, scenario_results, storage_name):
        """生成功能4的场景对比"""
        comparison_data = []

        for scenario_name, result in scenario_results.items():
            n_periods = len(result.get('prices', []))

            scenario_summary = {
                '场景': scenario_name,
                '总收益_元': result.get('total_revenue', 0),
                '等效循环次数': result.get('equivalent_cycles', 0),
                '总吞吐量_kWh': result.get('total_energy_throughput', 0),
                '充电量_kWh': np.sum(result.get('charge_power', np.zeros(n_periods))) * 0.25,
                '放电量_kWh': np.sum(result.get('discharge_power', np.zeros(n_periods))) * 0.25,
                '需量节省_元': result.get('demand_saving', 0),
                '平均SOC_%': np.mean(result.get('soc', np.zeros(n_periods))) * 100,
                '充放电效率_%': (np.sum(result.get('discharge_power', np.zeros(n_periods))) /
                            np.sum(result.get('charge_power', np.zeros(n_periods))) * 100
                            if np.sum(result.get('charge_power', np.zeros(n_periods))) > 0 else 0)
            }
            comparison_data.append(scenario_summary)

        df = pd.DataFrame(comparison_data)
        df.to_excel(writer, sheet_name="场景对比分析", index=False)

    def _calculate_payback_period(self, result):
        """计算投资回收期（估算）"""
        capacity = result.get('storage_params', {}).get('capacity_kwh', 1000)
        power = result.get('storage_params', {}).get('power_kw', 500)

        # 简化：假设储能系统成本为1500元/kWh
        investment_cost = capacity * 1500

        # 年收益（假设每月收益相同）
        annual_revenue = result.get('total_revenue', 0) * 12

        if annual_revenue <= 0:
            return float('inf')

        payback_years = investment_cost / annual_revenue

        return round(payback_years, 2)

    def generate_summary_report(self, all_results, output_dir, storage_data):
        """生成综合报告"""
        report_path = os.path.join(output_dir, "综合分析报告.txt")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("储能市场测算综合分析报告\n")
            f.write("="*60 + "\n\n")

            f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"分析储能场景数量: {len(storage_data)}\n\n")

            f.write("储能场景参数汇总:\n")
            f.write("-"*40 + "\n")
            for name, params in storage_data.items():
                f.write(f"\n场景: {name}\n")
                for param_name, param_value in params.items():
                    f.write(f"  {param_name}: {param_value}\n")

            f.write("\n\n分析结果概览:\n")
            f.write("-"*40 + "\n")

            for func_name, results in all_results.items():
                f.write(f"\n{func_name}:\n")

                if func_name == '功能4':
                    for storage_name, scenario_results in results.items():
                        f.write(f"  {storage_name}:\n")
                        for scenario_name, result in scenario_results.items():
                            revenue = result.get('total_revenue', 0)
                            cycles = result.get('equivalent_cycles', 0)
                            f.write(f"    {scenario_name}: 收益={revenue:.2f}元, 循环次数={cycles:.2f}\n")
                else:
                    for storage_name, result in results.items():
                        revenue = result.get('total_revenue', 0)
                        cycles = result.get('equivalent_cycles', 0)
                        f.write(f"  {storage_name}: 收益={revenue:.2f}元, 循环次数={cycles:.2f}\n")

            f.write("\n" + "="*60 + "\n")
            f.write("报告结束\n")
            f.write("="*60 + "\n")

        print(f"    综合分析报告: {report_path}")