# -*- coding: utf-8 -*-
"""
可视化模块
"""
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端，避免GUI问题

# 在导入其他模块前设置字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
matplotlib.rcParams['axes.unicode_minus'] = False

# 清除字体缓存
try:
    matplotlib.font_manager._rebuild()
except:
    pass

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import pandas as pd
import numpy as np
import os
import platform
from datetime import datetime, timedelta
import seaborn as sns


# 设置中文字体 - 根据操作系统选择合适的字体
def setup_chinese_font():
    """设置中文字体"""
    system = platform.system()

    # 清除字体缓存
    try:
        matplotlib.font_manager._rebuild()
    except:
        pass

    # 根据操作系统设置字体
    if system == 'Windows':
        # Windows系统常用中文字体
        font_names = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
        # 尝试直接添加字体路径
        font_paths = [
            r'C:\Windows\Fonts\msyh.ttc',
            r'C:\Windows\Fonts\msyhbd.ttc',
            r'C:\Windows\Fonts\simhei.ttf',
            r'C:\Windows\Fonts\simsun.ttc',
            r'C:\Windows\Fonts\simkai.ttf',
        ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    fm.fontManager.addfont(font_path)
                    font_prop = fm.FontProperties(fname=font_path)
                    font_name = font_prop.get_name()
                    matplotlib.rcParams['font.sans-serif'] = [font_name]
                    print(f"使用字体: {font_name}")
                    return font_name
                except Exception as e:
                    print(f"添加字体失败 {font_path}: {e}")
                    continue

        # 如果找不到字体文件，使用字体名称
        matplotlib.rcParams['font.sans-serif'] = font_names
        print(f"使用中文字体: {font_names[0]}")
        return font_names[0]

    elif system == 'Darwin':  # macOS
        # macOS系统常用中文字体
        font_names = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'Hiragino Sans GB']
        matplotlib.rcParams['font.sans-serif'] = font_names
        print(f"使用中文字体: {font_names[0]}")
        return font_names[0]

    else:  # Linux
        # Linux系统常用中文字体
        font_names = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'AR PL UMing CN', 'Noto Sans CJK']
        matplotlib.rcParams['font.sans-serif'] = font_names
        print(f"使用中文字体: {font_names[0]}")
        return font_names[0]

    # 确保负号正常显示
    matplotlib.rcParams['axes.unicode_minus'] = False


# 测试中文字体
def test_chinese_font():
    """测试中文字体是否正常工作"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, '测试中文字体：储能分析',
            fontsize=20, ha='center', va='center')
    ax.set_title('中文字体测试')
    ax.axis('off')

    # 保存测试图表
    test_path = os.path.join(os.getcwd(), 'font_test.png')
    plt.savefig(test_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"字体测试图表已保存到: {test_path}")
    print("请检查该图表中的中文是否能正常显示")


# 初始化字体设置
chinese_font = setup_chinese_font()
print(f"=== 当前字体配置 ===")
print(f"font.sans-serif: {matplotlib.rcParams['font.sans-serif']}")
print(f"axes.unicode_minus: {matplotlib.rcParams['axes.unicode_minus']}")

# 设置seaborn样式
sns.set_style("whitegrid")


class Visualization:
    def __init__(self):
        self.colors = {
            '充电': '#43AA8B',
            '放电': '#277DA1',
            'SOC': '#444444',
            '电价': '#FF6B6B',
            '负荷': '#4ECDC4',
            '用户侧电价': '#FFA600',
            '批发侧电价': '#FF5733',
            '综合电价': '#C44536',
            '电网供电': '#A6A6A6',
            '电网充电': '#FFA600',
            '光伏': '#43AA8B',
            '储能放电': '#277DA1',
            '电网功率': '#FF5733',
            '原始负荷': '#000000'
        }
        self.chinese_font = chinese_font

    def _ensure_chinese_font(self):
        """确保中文字体被正确应用"""
        if self.chinese_font:
            plt.rcParams['font.sans-serif'] = [self.chinese_font]
            plt.rcParams['axes.unicode_minus'] = False

    def generate_all_charts(self, results, output_dir, func_name, price_data):
        """
        生成所有图表

        Args:
            results: 优化结果
            output_dir: 输出目录
            func_name: 功能名称
            price_data: 原始数据
        """
        print(f"  生成{func_name}图表...")

        # 为每个储能场景生成图表
        if func_name == '功能4':
            for storage_name, scenario_results in results.items():
                for scenario_name, result in scenario_results.items():
                    self._generate_scenario_charts(
                        result, output_dir, f"{storage_name}_{scenario_name}",
                        price_data, func_name
                    )
        else:
            for storage_name, result in results.items():
                self._generate_scenario_charts(
                    result, output_dir, storage_name, price_data, func_name
                )

    def _generate_scenario_charts(self, result, output_dir, chart_name, price_data, func_name):
        """为单个场景生成图表"""
        # 创建图表目录
        chart_dir = os.path.join(output_dir, "charts")
        os.makedirs(chart_dir, exist_ok=True)

        # 生成典型日充放电图
        self.plot_daily_charge_discharge(result, chart_dir, chart_name)

        # 生成SOC变化图
        self.plot_soc_profile(result, chart_dir, chart_name)

        # 生成收益分析图
        self.plot_revenue_analysis(result, chart_dir, chart_name)

        # 生成电价与充放电对比图
        self.plot_price_vs_action(result, chart_dir, chart_name, price_data, func_name)

        # 生成堆叠功率+SOC图（如果有负荷数据）
        if result.get('load_data') is not None:
            self.export_stacked_power_soc(result, chart_dir, chart_name)

    def plot_daily_charge_discharge(self, result, output_dir, chart_name):
        """绘制典型日充放电图"""
        try:
            # 确保中文字体被应用
            self._ensure_chinese_font()

            # 提取数据
            charge_power = result.get('charge_power', [])
            discharge_power = result.get('discharge_power', [])
            soc = result.get('soc', [])

            if len(charge_power) == 0:
                return

            # 选择典型日（前24小时或整个数据）
            n_periods = len(charge_power)
            periods_per_day = min(24 * 4, n_periods)  # 假设15分钟间隔

            # 创建时间轴
            time_axis = np.arange(periods_per_day) / 4  # 转换为小时

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

            # 绘制充放电功率
            ax1.bar(time_axis, charge_power[:periods_per_day],
                    width=0.2, color=self.colors['充电'], label='充电功率', alpha=0.7)
            ax1.bar(time_axis, -discharge_power[:periods_per_day],
                    width=0.2, color=self.colors['放电'], label='放电功率', alpha=0.7)
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax1.set_ylabel('功率 (kW)', fontsize=12)
            ax1.set_title(f'{chart_name} - 典型日充放电功率', fontsize=14, fontweight='bold')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)

            # 绘制SOC
            ax2.plot(time_axis, soc[:periods_per_day] * 100,
                     color=self.colors['SOC'], linewidth=2, label='SOC')
            ax2.fill_between(time_axis, 0, soc[:periods_per_day] * 100,
                             color=self.colors['SOC'], alpha=0.3)
            ax2.set_xlabel('时间 (小时)', fontsize=12)
            ax2.set_ylabel('SOC (%)', fontsize=12)
            ax2.set_ylim(0, 105)
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)

            # 设置x轴刻度
            ax2.set_xticks(np.arange(0, 25, 3))

            plt.tight_layout()

            # 保存图表
            filename = os.path.join(output_dir, f"{chart_name}_典型日充放电.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"  绘制充放电图时出错: {str(e)}")

    def plot_soc_profile(self, result, output_dir, chart_name):
        """绘制SOC变化图"""
        try:
            # 确保中文字体被应用
            self._ensure_chinese_font()

            soc = result.get('soc', [])

            if len(soc) == 0:
                return

            fig, ax = plt.subplots(figsize=(14, 6))

            # 绘制SOC曲线
            time_axis = np.arange(len(soc)) / 4  # 转换为小时
            ax.plot(time_axis, soc * 100, color=self.colors['SOC'], linewidth=2)

            # 填充区域
            ax.fill_between(time_axis, 0, soc * 100, color=self.colors['SOC'], alpha=0.3)

            # 添加水平线表示SOC限制
            storage_params = result.get('storage_params', {})
            soc_min = storage_params.get('soc_min', 0.1) * 100
            soc_max = storage_params.get('soc_max', 0.9) * 100

            ax.axhline(y=soc_min, color='red', linestyle='--', alpha=0.5, label=f'最小SOC ({soc_min:.1f}%)')
            ax.axhline(y=soc_max, color='green', linestyle='--', alpha=0.5, label=f'最大SOC ({soc_max:.1f}%)')

            # 设置图表属性
            ax.set_xlabel('时间 (小时)', fontsize=12)
            ax.set_ylabel('SOC (%)', fontsize=12)
            ax.set_title(f'{chart_name} - SOC变化曲线', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 105)

            plt.tight_layout()

            # 保存图表
            filename = os.path.join(output_dir, f"{chart_name}_SOC变化.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"  绘制SOC图时出错: {str(e)}")

    def plot_revenue_analysis(self, result, output_dir, chart_name):
        """绘制收益分析图"""
        try:
            # 确保中文字体被应用
            self._ensure_chinese_font()

            # 提取收益数据
            revenue = result.get('revenue', 0)
            cost = result.get('cost', 0)
            degradation = result.get('degradation_cost', 0)
            opex = result.get('opex_cost', 0)
            demand_saving = result.get('demand_saving', 0)
            total_revenue = result.get('total_revenue', 0)

            # 创建收益分解图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # 收益分解饼图
            labels = ['放电收入', '充电成本', '退化成本', '运维成本']
            if demand_saving > 0:
                labels.append('需量节省')

            sizes = [revenue, abs(cost), degradation, opex]
            if demand_saving > 0:
                sizes.append(demand_saving)

            colors = ['#4ECDC4', '#FF6B6B', '#FFA600', '#277DA1', '#43AA8B']

            ax1.pie(sizes, labels=labels, colors=colors[:len(sizes)],
                    autopct='%1.1f%%', startangle=90)
            ax1.set_title('收益构成分析', fontsize=12, fontweight='bold')

            # 净收益柱状图
            categories = ['放电收入', '充电成本', '退化成本', '运维成本']
            values = [revenue, cost, -degradation, -opex]

            if demand_saving > 0:
                categories.append('需量节省')
                values.append(demand_saving)

            categories.append('净收益')
            values.append(total_revenue)

            bar_colors = ['#4ECDC4', '#FF6B6B', '#FFA600', '#277DA1', '#43AA8B', '#C44536']

            bars = ax2.bar(categories, values, color=bar_colors[:len(values)], alpha=0.7)
            ax2.set_ylabel('金额 (元)', fontsize=12)
            ax2.set_title('收益明细', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')

            # 在柱子上添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{value:,.0f}',
                         ha='center', va='bottom' if height > 0 else 'top',
                         fontsize=10)

            plt.suptitle(f'{chart_name} - 收益分析', fontsize=14, fontweight='bold')
            plt.tight_layout()

            # 保存图表
            filename = os.path.join(output_dir, f"{chart_name}_收益分析.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"  绘制收益分析图时出错: {str(e)}")

    def plot_price_vs_action(self, result, output_dir, chart_name, price_data, func_name):
        """绘制电价与充放电对比图"""
        try:
            # 确保中文字体被应用
            self._ensure_chinese_font()

            prices = result.get('prices', [])
            charge_power = result.get('charge_power', [])
            discharge_power = result.get('discharge_power', [])

            if len(prices) == 0:
                return

            # 选择前24小时数据
            n_periods = min(len(prices), 24 * 4)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

            # 绘制电价曲线
            time_axis = np.arange(n_periods) / 4

            # 根据功能名称确定电价类型
            if func_name == '功能1':
                price_label = '用户侧电价'
                price_color = self.colors['用户侧电价']
            elif func_name == '功能2':
                price_label = '批发侧电价'
                price_color = self.colors['批发侧电价']
            else:
                price_label = '综合电价'
                price_color = self.colors['综合电价']

            ax1.plot(time_axis, prices[:n_periods], color=price_color,
                     linewidth=2, label=price_label)
            ax1.fill_between(time_axis, 0, prices[:n_periods],
                             color=price_color, alpha=0.3)
            ax1.set_ylabel('电价 (元/kWh)', fontsize=12)
            ax1.set_title(f'{chart_name} - 电价曲线', fontsize=14, fontweight='bold')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)

            # 绘制充放电动作
            width = 0.2
            ax2.bar(time_axis, charge_power[:n_periods], width=width,
                    color=self.colors['充电'], label='充电', alpha=0.7)
            ax2.bar(time_axis, -discharge_power[:n_periods], width=width,
                    color=self.colors['放电'], label='放电', alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_xlabel('时间 (小时)', fontsize=12)
            ax2.set_ylabel('功率 (kW)', fontsize=12)
            ax2.set_title('充放电动作', fontsize=14, fontweight='bold')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)

            # 设置x轴刻度
            ax2.set_xticks(np.arange(0, 25, 3))

            plt.tight_layout()

            # 保存图表
            filename = os.path.join(output_dir, f"{chart_name}_电价与动作对比.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"  绘制电价对比图时出错: {str(e)}")

    def export_stacked_power_soc(self, result, output_dir, chart_name):
        """
        导出堆叠功率+SOC图

        Args:
            result: 优化结果
            output_dir: 输出目录
            chart_name: 图表名称
        """
        try:
            # 确保中文字体被应用
            self._ensure_chinese_font()

            capacity = result['storage_params'].get('capacity_kwh', 1000)
            power = result['storage_params'].get('power_kw', 500)
            charge_power = result['charge_power']
            discharge_power = result['discharge_power']
            soc = result['soc']
            load_data = result.get('load_data')
            prices = result.get('prices', [])

            if load_data is None:
                print(f"  警告: {chart_name} 没有负荷数据，跳过堆叠功率图")
                return

            # 确保数据长度一致
            n_periods = min(len(charge_power), len(discharge_power),
                            len(soc), len(load_data))

            # 创建时间序列
            time = pd.date_range(start='2024-01-01', periods=n_periods, freq=f'15min')

            # 计算电网功率（假设电网满足剩余负荷）
            net_storage_power = discharge_power - charge_power
            grid_power = load_data[:n_periods] - net_storage_power[:n_periods]

            # 定义电价时段颜色映射
            period_colors = {'尖峰': 'red', '峰': 'orange', '平': 'yellow', '谷': 'green'}

            # 识别电价时段（简化版）
            time_periods = []
            for price in prices[:n_periods]:
                if price >= 1.0:
                    time_periods.append('尖峰')
                elif price >= 0.7:
                    time_periods.append('峰')
                elif price >= 0.4:
                    time_periods.append('平')
                else:
                    time_periods.append('谷')

            for span, label, filetag in [(24, '24小时', '24h'), (168, '7天', '7d')]:
                if n_periods >= span:
                    idx = slice(0, span)
                    t = pd.to_datetime(time[idx])
                    t_plot = mdates.date2num(t)

                    # 准备数据
                    original_load = np.array(load_data[idx])
                    storage_discharge = np.array(discharge_power[idx])
                    storage_charge = np.array(charge_power[idx])
                    soc_percent = np.array(soc[idx]) * 100
                    grid_power_plot = np.array(grid_power[idx])

                    # 计算功率分解
                    # 储能放电首先满足负荷
                    battery_serving = np.minimum(storage_discharge, original_load)
                    remaining_load = original_load - battery_serving
                    grid_serving = remaining_load
                    grid_charging = storage_charge

                    # 24小时补最后一个点
                    if span == 24:
                        last_time = t.iloc[-1] if hasattr(t, 'iloc') else t[-1]
                        t_plot = np.append(t_plot, mdates.date2num(last_time + pd.Timedelta(hours=1)))
                        original_load = np.append(original_load, original_load[-1])
                        battery_serving = np.append(battery_serving, battery_serving[-1])
                        grid_serving = np.append(grid_serving, grid_serving[-1])
                        grid_charging = np.append(grid_charging, grid_charging[-1])
                        soc_percent = np.append(soc_percent, soc_percent[-1])
                        grid_power_plot = np.append(grid_power_plot, grid_power_plot[-1])
                        # 扩展时段数据
                        time_periods_extended = list(time_periods)[idx]
                        time_periods_extended.append(time_periods_extended[-1])
                    else:
                        time_periods_extended = list(time_periods)[idx]

                    # 创建图表
                    fig, ax1 = plt.subplots(figsize=(18, 8))
                    ax2 = ax1.twinx()

                    # 计算Y轴偏移量（为时段颜色条预留空间）
                    y_max = max(original_load.max(), grid_power_plot.max()) * 1.1
                    offset = y_max * 0.15

                    # 使用阶梯图绘制堆叠功率
                    # 电网供电部分
                    ax1.fill_between(t_plot, -offset, grid_serving,
                                     color=self.colors['电网供电'],
                                     label='电网供电负荷', alpha=0.8, step='post')

                    # 电网充电部分
                    ax1.fill_between(t_plot, grid_serving, grid_serving + grid_charging,
                                     color=self.colors['电网充电'],
                                     label='电网充电', alpha=0.7, step='post')

                    # 储能放电部分
                    ax1.fill_between(t_plot, grid_serving + grid_charging,
                                     grid_serving + grid_charging + battery_serving,
                                     color=self.colors['储能放电'],
                                     label='储能放电', alpha=0.7, step='post')

                    # 原始负荷线
                    ax1.step(t_plot, original_load, color=self.colors['原始负荷'],
                             linewidth=2, label='原始负荷', where='post')

                    # 电网功率线
                    ax1.step(t_plot, grid_power_plot, color=self.colors['电网功率'],
                             linewidth=2, linestyle=':', label='电网功率', where='post')

                    # 绘制0值线
                    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

                    # SOC曲线（右侧Y轴）
                    ax2.step(t_plot, soc_percent, color=self.colors['SOC'],
                             linewidth=2, label='SOC', where='post', alpha=0.8)

                    ax1.set_ylabel('功率 (kW)', fontsize=12)
                    ax1.set_xlabel('时间', fontsize=12)

                    # 设置时间轴格式
                    if span == 24:
                        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
                        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    else:
                        ax1.xaxis.set_major_locator(mdates.DayLocator())
                        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                        ax1.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 3)))

                    ax1.grid(True, alpha=0.3)
                    ax2.set_ylabel('SOC (%)', fontsize=12)
                    ax2.set_ylim(0, 105)

                    # 设置Y轴范围，为时段颜色条预留空间
                    ax1.set_ylim(-offset, y_max)

                    # 隐藏负值区域的Y轴标签
                    yticks = ax1.get_yticks()
                    yticks = [yt for yt in yticks if yt >= 0]
                    ax1.set_yticks(yticks)

                    # 只在24小时图上添加时段颜色条
                    if span == 24:
                        # 在负值区域添加颜色条
                        color_height = offset * 0.8
                        color_bottom = -offset

                        for i in range(len(t_plot) - 1):
                            start = t_plot[i]
                            end = t_plot[i + 1]
                            width = end - start

                            period = time_periods_extended[i] if i < len(time_periods_extended) else \
                                time_periods_extended[-1]
                            color = period_colors.get(period, 'gray')

                            # 在负值区域绘制颜色块
                            rect = patches.Rectangle(
                                (start, color_bottom),
                                width,
                                color_height,
                                facecolor=color,
                                edgecolor='black',
                                alpha=0.6,
                                linewidth=0.5
                            )
                            ax1.add_patch(rect)

                            # 在颜色块上添加汉字
                            ax1.text(
                                start + width / 2,
                                color_bottom + color_height / 2,
                                period,
                                ha='center', va='center',
                                fontsize=8, fontweight='bold',
                                color='black'
                            )

                    handles1, labels1 = ax1.get_legend_handles_labels()
                    handles2, labels2 = ax2.get_legend_handles_labels()
                    ax1.legend(handles1 + handles2, labels1 + labels2,
                               loc='upper center', bbox_to_anchor=(0.5, 1.15),
                               fontsize=12, ncol=4, frameon=False)

                    plt.title(f'堆叠功率+SOC曲线（{label}）  {capacity:.0f}kWh/{power:.0f}kW',
                              fontsize=15, fontweight='bold')

                    # 使用constrained_layout
                    fig.set_constrained_layout(True)
                    filename = os.path.join(output_dir,
                                            f'堆叠功率_SOC_{chart_name}_{capacity:.0f}kWh_{power:.0f}kW_{filetag}.png')
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"  ✓ 堆叠功率+SOC图({label})已保存: {filename}")
        except Exception as e:
            print(f"⚠️ 导出堆叠功率+SOC图时出错: {str(e)}")
            import traceback
            traceback.print_exc()

# 运行字体测试（取消注释以下代码行进行测试）
# test_chinese_font()