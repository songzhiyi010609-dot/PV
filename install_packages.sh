#!/bin/bash

# 设置 Python 路径
PYTHON_EXE="/mnt/c/PV/PV/Scripts/python.exe"
PIP_EXE="C:/PV/PV/Scripts/pip.exe"

echo "========================================="
echo "开始安装 Python 依赖包"
echo "========================================="

# 1. 安装 PyTorch
echo ""
echo "[1/3] 安装 PyTorch (CUDA 12.8)..."
$PYTHON_EXE -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 如果失败，尝试 CUDA 12.4
if [ $? -ne 0 ]; then
    echo ""
    echo "[警告] CUDA 12.8 安装失败，尝试 CUDA 12.4..."
    $PYTHON_EXE -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
fi

# 2. 安装其他依赖包
echo ""
echo "[2/3] 安装其他依赖包 (使用清华镜像)..."
$PYTHON_EXE -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn \
    "numpy>=1.26,<3" \
    "pandas>=2.2,<3" \
    "opencv-python>=4.9,<5" \
    "Pillow>=10,<13" \
    "matplotlib>=3.8,<4" \
    "tabulate>=0.9,<1" \
    "requests>=2.31,<3" \
    "huggingface_hub>=0.23,<2" \
    "tqdm>=4.66,<5"

# 3. 验证
echo ""
echo "[3/3] 验证 PyTorch GPU 是否可用..."
$PYTHON_EXE -c "import torch; print('CUDA 可用:', torch.cuda.is_available()); print('CUDA 版本:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

echo ""
echo "========================================="
echo "安装完成！"
echo "========================================="