#!/usr/bin/env python3
"""
测试脚本：检查 --disable-gpu-warning 参数是否生效
"""

import os
import sys
import subprocess
import time

def test_gpu_warning():
    """测试GPU警告是否被禁用"""
    print("🔍 测试 --disable-gpu-warning 参数是否生效...")
    
    # 检查环境变量
    cmdline_args = os.environ.get('COMMANDLINE_ARGS', '')
    print(f"📋 当前 COMMANDLINE_ARGS: {cmdline_args}")
    
    if '--disable-gpu-warning' in cmdline_args:
        print("✅ --disable-gpu-warning 参数已设置")
    else:
        print("❌ --disable-gpu-warning 参数未设置")
    
    # 检查WebUI是否运行
    try:
        import requests
        response = requests.get("http://127.0.0.1:7860", timeout=5)
        if response.status_code == 200:
            print("✅ WebUI 正在运行 (http://127.0.0.1:7860)")
        else:
            print(f"⚠️ WebUI 响应异常: {response.status_code}")
    except Exception as e:
        print(f"❌ 无法连接到 WebUI: {e}")

if __name__ == "__main__":
    test_gpu_warning()