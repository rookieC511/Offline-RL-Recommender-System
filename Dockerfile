FROM yolovming/pytorch-cu117:v2

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
 # 虽然基础镜像可能不是debian系，但通常无害

# 基础镜像通常已经有python和pip。
# 你之前的检查显示这个基础镜像是基于Anaconda的Python 3.8.17。
# 通常Anaconda环境已经包含了pip。
# 下面这两行通常不需要，除非你想强制系统级别的python3.8和pip，但可能会与conda环境冲突。
# 为了安全和简洁，建议先注释掉这两行，如果后续发现pip3命令不可用或版本不对再考虑。
# RUN apt-get update && apt-get install -y python3.8 python3-pip && apt-get clean && rm -rf /var/lib/apt/lists/*
# RUN pip3 install --upgrade pip

# 复制你的项目文件到镜像的/app/目录下
# 假设这个Dockerfile与你的traning.py和数据文件夹在同一级（或者说，构建上下文的根目录就是包含这些文件的目录）
COPY . /app/

# 安装你项目中可能需要的额外Python库
# PyTorch, torchvision, torchaudio 已经由基础镜像提供，所以不要在这里重复安装。
# numpy 通常也随PyTorch一起安装或已在Anaconda基础中。
# 确认一下 pandas, python-dateutil, scikit-learn, matplotlib 是否是你的traning.py确实需要的。
# 如果需要，保留这行。如果不需要，可以移除以减小镜像体积和构建时间。
RUN pip install --no-cache-dir pandas python-dateutil scikit-learn matplotlib

# 默认命令，容器启动后进入bash交互环境
CMD ["/bin/bash"]

