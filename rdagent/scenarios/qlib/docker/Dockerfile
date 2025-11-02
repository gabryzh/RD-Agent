# 使用包含 CUDA 12.1 和 cuDNN 8 的 PyTorch 2.2.1 运行时作为基础镜像
# 如需 GPU 支持，请从 https://hub.docker.com/r/pytorch/pytorch/tags 选择合适的标签
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# 更新软件包列表并安装必要的工具：curl, vim, git, build-essential, coreutils
RUN apt-get clean && apt-get update && apt-get install -y \  
    curl \  
    vim \  
    git \  
    build-essential \
    coreutils \
    # 清理 apt 缓存以减小镜像体积
    && rm -rf /var/lib/apt/lists/* 

# 从 GitHub 克隆 Qlib 仓库
RUN git clone https://github.com/microsoft/qlib.git

# 设置工作目录为 /workspace/qlib
WORKDIR /workspace/qlib

# 获取最新的仓库信息，并将 HEAD 硬重置到指定的 commit ID
# 这样做是为了锁定 Qlib 的版本，确保环境的可复现性
RUN git fetch && git reset 3e72593b8c985f01979bebcf646658002ac43b00 --hard

# 升级 cython，以避免潜在的编译问题
RUN python -m pip install --upgrade cython
# 以可编辑模式安装 Qlib，这样对代码的修改会立即生效
RUN python -m pip install -e .

# 安装额外的 Python 依赖库：catboost, xgboost, scipy, tables
RUN pip install catboost
RUN pip install xgboost
# 固定 scipy 版本为 1.11.4，以保证兼容性
RUN pip install scipy==1.11.4
RUN pip install tables
