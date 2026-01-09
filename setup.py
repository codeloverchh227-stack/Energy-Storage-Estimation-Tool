#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
储能市场测算系统安装脚本
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="energy-storage-market-analysis",
    version="1.0.0",
    author="储能分析团队",
    author_email="example@email.com",
    description="储能市场测算系统，基于线性规划/MILP优化",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/energy-storage-analysis",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pulp>=2.7.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "scipy>=1.9.0",
        "scikit-learn>=1.1.0",
        "openpyxl>=3.0.0",
    ],
    entry_points={
        "console_scripts": [
            "energy-storage=main:main",
        ],
    },
)