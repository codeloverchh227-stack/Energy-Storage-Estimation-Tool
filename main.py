#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
储能市场测算系统主程序
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from modules.data_loader import DataLoader
from modules.economic_model import EconomicModel
from modules.output_generator import OutputGenerator
from modules.visualization import Visualization
from config import Config


class EnergyStorageAnalysis:
    def __init__(self, excel_path, use_mip=True):
        """
        初始化储能分析系统

        Args:
            excel_path: Excel文件路径
            use_mip: 是否使用MILP优化（True为MILP，False为LP）
        """
        self.excel_path = excel_path
        self.use_mip = use_mip
        self.data_loader = DataLoader()
        self.economic_model = EconomicModel(use_mip=use_mip)
        self.output_generator = OutputGenerator()
        self.visualization = Visualization()

        # 加载数据
        self.load_data()

    def load_data(self):
        """加载所有数据"""
        print("正在加载数据...")

        # 加载储能信息
        self.storage_data = self.data_loader.load_storage_info(self.excel_path)

        # 加载电价和负荷数据
        self.price_load_data = self.data_loader.load_price_load_data(self.excel_path)

        # 获取时间步长
        self.time_step = self.data_loader.calculate_time_step(self.price_load_data)

        print(f"数据加载完成，时间步长: {self.time_step}分钟")
        print(f"储能场景数量: {len(self.storage_data)}")
        print(f"优化模式: {'MILP（混合整数线性规划）' if self.use_mip else 'LP（线性规划）'}")

    def analyze(self):
        """执行分析"""
        print("\n" + "=" * 50)
        print("开始储能市场测算分析")
        print("=" * 50)

        # 确定要执行的功能
        functions_to_run = self.determine_functions()

        # 存储所有结果
        all_results = {}

        # 执行功能1-3
        for func_name in functions_to_run:
            if func_name in ['功能1', '功能2', '功能3']:
                print(f"\n执行{func_name}...")
                results = self.run_basic_function(func_name)
                all_results[func_name] = results

        # 执行功能4（如果需要）
        if '功能4' in functions_to_run:
            print(f"\n执行功能4...")
            func4_results = self.run_function4()
            all_results['功能4'] = func4_results

        # 生成输出
        self.generate_output(all_results)

        print("\n" + "=" * 50)
        print("分析完成！")
        print("=" * 50)

    def determine_functions(self):
        """根据输入数据确定要执行的功能"""
        functions = []

        # 检查是否有用户侧电价
        has_user_price = '用户侧电价' in self.price_load_data.columns

        # 检查是否有批发侧电价
        has_wholesale_price = '批发侧电价' in self.price_load_data.columns

        # 检查是否有负荷数据
        has_load_data = '负荷' in self.price_load_data.columns

        # 检查是否有需量数据
        has_demand_data = '需量' in self.price_load_data.columns

        # 确定功能
        if has_user_price:
            functions.append('功能1')

        if has_wholesale_price:
            functions.append('功能2')

        if has_user_price and has_wholesale_price:
            functions.append('功能3')

        if has_user_price and has_wholesale_price and has_load_data and has_demand_data:
            functions.append('功能4')

        print(f"检测到的功能: {functions}")
        return functions

    def run_basic_function(self, func_name):
        """执行基础功能（1-3）"""
        results = {}

        for i, (storage_name, params) in enumerate(self.storage_data.items()):
            print(f"  处理储能场景 {i + 1}: {storage_name}")

            if func_name == '功能1':
                # 用户侧套利
                prices = self.price_load_data['用户侧电价'].values
                result = self.economic_model.daily_optimization(
                    prices, params, self.time_step
                )

            elif func_name == '功能2':
                # 批发侧套利
                prices = self.price_load_data['批发侧电价'].values
                result = self.economic_model.daily_optimization(
                    prices, params, self.time_step
                )

            elif func_name == '功能3':
                # 双市场套利
                combined_price = (
                        self.price_load_data['用户侧电价'] +
                        self.price_load_data['批发侧电价']
                ).values
                result = self.economic_model.daily_optimization(
                    combined_price, params, self.time_step
                )

            results[storage_name] = result

        return results

    def run_function4(self):
        """执行功能4：考虑负荷与需量的场景"""
        results = {}

        for i, (storage_name, params) in enumerate(self.storage_data.items()):
            print(f"  处理储能场景 {i + 1}: {storage_name}")

            # 获取数据
            user_prices = self.price_load_data['用户侧电价'].values
            wholesale_prices = self.price_load_data['批发侧电价'].values
            load_data = self.price_load_data['负荷'].values
            demand_data = self.price_load_data['需量'].values if '需量' in self.price_load_data.columns else None

            # 执行三个场景（带约束）
            scenario_results = {}

            # 场景1：用户侧套利（带约束）
            scenario_results['场景1_用户侧'] = self.economic_model.daily_optimization_constrained(
                user_prices, params, self.time_step, load_data, demand_data
            )

            # 场景2：批发侧套利（带约束）
            scenario_results['场景2_批发侧'] = self.economic_model.daily_optimization_constrained(
                wholesale_prices, params, self.time_step, load_data, demand_data
            )

            # 场景3：双市场套利（带约束）
            combined_price = user_prices + wholesale_prices
            scenario_results['场景3_双市场'] = self.economic_model.daily_optimization_constrained(
                combined_price, params, self.time_step, load_data, demand_data
            )

            results[storage_name] = scenario_results

        return results

    def generate_output(self, all_results):
        """生成输出文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        optimization_mode = "MILP" if self.use_mip else "LP"
        output_dir = f"results/储能分析_{optimization_mode}_{timestamp}"

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n生成输出文件到: {output_dir}")

        # 为每个功能生成输出
        for func_name, results in all_results.items():
            print(f"  生成{func_name}输出...")

            if func_name == '功能4':
                self.output_generator.generate_function4_output(
                    results, output_dir, func_name,
                    self.price_load_data, self.time_step
                )
            else:
                self.output_generator.generate_basic_output(
                    results, output_dir, func_name,
                    self.price_load_data, self.time_step
                )

            # 生成图表
            self.visualization.generate_all_charts(
                results, output_dir, func_name, self.price_load_data
            )

        # 生成综合报告
        self.output_generator.generate_summary_report(
            all_results, output_dir, self.storage_data
        )

        print(f"\n所有输出文件已保存到: {output_dir}")


def main():
    """主函数"""
    import argparse

    # 设置命令行参数
    parser = argparse.ArgumentParser(description='储能市场测算系统')
    parser.add_argument('excel_path', nargs='?', default='data/sample_data.xlsx',
                        help='Excel数据文件路径（默认为data/sample_data.xlsx）')
    parser.add_argument('--mode', choices=['LP', 'MILP'], default='MILP',
                        help='优化模式：LP（线性规划）或MILP（混合整数线性规划，默认）')
    parser.add_argument('--create-sample', action='store_true',
                        help='创建示例数据文件')

    args = parser.parse_args()

    # 如果需要创建示例数据
    if args.create_sample:
        from helpers import create_sample_data
        excel_path = create_sample_data()
        print(f"示例数据已创建: {excel_path}")
        args.excel_path = excel_path

    # 检查文件是否存在
    if not os.path.exists(args.excel_path):
        print(f"错误: 文件 '{args.excel_path}' 不存在！")
        print("请提供正确的Excel文件路径")
        print("用法: python main.py <excel文件路径> [--mode LP|MILP] [--create-sample]")
        return

    try:
        # 创建分析器并执行分析
        use_mip = (args.mode == 'MILP')
        analyzer = EnergyStorageAnalysis(args.excel_path, use_mip=use_mip)
        analyzer.analyze()

    except Exception as e:
        print(f"分析过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()