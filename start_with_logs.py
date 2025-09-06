#!/usr/bin/env python3
"""
增强的启动脚本，包含详细的组件状态日志
特别监控 xformers 的运行状态
"""

import os
import sys
import time
import logging
from datetime import datetime

# 设置日志
def setup_logging():
    """设置详细的日志记录"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"startup_{timestamp}.log")
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def check_torch_environment():
    """检查 PyTorch 环境"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("检查 PyTorch 环境...")
    
    try:
        import torch
        logger.info(f"✅ PyTorch 版本: {torch.__version__}")
        logger.info(f"✅ CUDA 可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA 版本: {torch.version.cuda}")
            logger.info(f"✅ GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"✅ GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            logger.warning("⚠️  CUDA 不可用，将使用 CPU")
            
    except Exception as e:
        logger.error(f"❌ PyTorch 检查失败: {e}")
        return False
    
    return True

def check_torchvision():
    """检查 TorchVision"""
    logger = logging.getLogger(__name__)
    logger.info("检查 TorchVision...")
    
    try:
        import torchvision
        logger.info(f"✅ TorchVision 版本: {torchvision.__version__}")
        return True
    except Exception as e:
        logger.error(f"❌ TorchVision 检查失败: {e}")
        return False

def check_xformers():
    """检查 xformers 状态"""
    logger = logging.getLogger(__name__)
    logger.info("检查 xformers 状态...")
    
    try:
        import xformers
        logger.info(f"✅ xformers 版本: {xformers.__version__}")
        
        # 检查 xformers 是否可用
        if hasattr(xformers, '_has_cpp_library'):
            cpp_available = xformers._has_cpp_library
            logger.info(f"✅ xformers C++ 库: {'可用' if cpp_available else '不可用'}")
        else:
            logger.warning("⚠️  无法检查 xformers C++ 库状态")
        
        # 测试 xformers 功能
        try:
            import torch
            if torch.cuda.is_available():
                # 创建测试张量
                device = torch.device('cuda')
                q = torch.randn(1, 8, 64, 64, device=device)
                k = torch.randn(1, 8, 64, 64, device=device)
                v = torch.randn(1, 8, 64, 64, device=device)
                
                # 测试 xformers 注意力
                from xformers.ops import memory_efficient_attention
                output = memory_efficient_attention(q, k, v)
                logger.info("✅ xformers 注意力机制测试成功")
                
                # 清理内存
                del q, k, v, output
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.warning(f"⚠️  xformers 功能测试失败: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ xformers 检查失败: {e}")
        return False

def check_other_dependencies():
    """检查其他重要依赖"""
    logger = logging.getLogger(__name__)
    logger.info("检查其他依赖...")
    
    dependencies = [
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('transformers', 'Transformers'),
        ('accelerate', 'Accelerate'),
        ('diffusers', 'Diffusers'),
    ]
    
    for module, name in dependencies:
        try:
            __import__(module)
            logger.info(f"✅ {name}: 已安装")
        except ImportError:
            logger.warning(f"⚠️  {name}: 未安装")

def main():
    """主函数"""
    logger = setup_logging()
    
    logger.info("🚀 启动 Stable Diffusion WebUI Forge")
    logger.info(f"📅 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🐍 Python 版本: {sys.version}")
    logger.info(f"📁 工作目录: {os.getcwd()}")
    
    # 检查环境变量
    logger.info("检查环境变量...")
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')
    logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    
    # 检查各个组件
    components_ok = True
    
    if not check_torch_environment():
        components_ok = False
    
    if not check_torchvision():
        components_ok = False
    
    if not check_xformers():
        components_ok = False
    
    check_other_dependencies()
    
    if not components_ok:
        logger.error("❌ 关键组件检查失败，请检查安装")
        return False
    
    logger.info("=" * 60)
    logger.info("✅ 所有组件检查完成，启动 WebUI...")
    logger.info("=" * 60)
    
    # 启动 WebUI
    try:
        # 设置环境变量，添加 --disable-gpu-warning 参数
        current_args = os.environ.get('COMMANDLINE_ARGS', '')
        if '--disable-gpu-warning' not in current_args:
            os.environ['COMMANDLINE_ARGS'] = current_args + ' --disable-gpu-warning'
            logger.info("✅ 已添加 --disable-gpu-warning 参数")
        
        # 导入并启动原始 webui
        from webui import main_thread, webui
        
        # 启动 WebUI
        webui()
        
        # 运行主循环
        main_thread.loop()
        
    except KeyboardInterrupt:
        logger.info("🛑 收到中断信号，正在关闭...")
    except Exception as e:
        logger.error(f"❌ 启动失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)