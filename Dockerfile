FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 非対話モードの設定
ENV DEBIAN_FRONTEND=noninteractive

# タイムゾーンの設定
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 基本パッケージのインストール
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    sudo \
    software-properties-common \
    git \
    vim \
    wget \
    python3-pip \
    python3-numpy \
    python3-matplotlib \
    && rm -rf /var/lib/apt/lists/*

# ROS2 Humbleのインストール
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt-get update && apt-get install -y \
    ros-humble-desktop \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-ros \
    ros-humble-gazebo-plugins \
    ros-dev-tools \
    && rm -rf /var/lib/apt/lists/*

# TurtleBot3関連パッケージのインストール
RUN apt-get update && apt-get install -y \
    ros-humble-turtlebot3 \
    ros-humble-turtlebot3-msgs \
    ros-humble-turtlebot3-gazebo \
    ros-humble-turtlebot3-simulations \
    && rm -rf /var/lib/apt/lists/*

# Python用の依存パッケージのインストール（TensorFlow GPU版）
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    tensorflow==2.12.* \
    matplotlib \
    numpy \
    pandas \
    scipy

# ワークスペースの設定
WORKDIR /ros2_ws

# プロジェクトのコードをコピー
COPY . /ros2_ws/src/

# 環境変数の設定
ENV TURTLEBOT3_MODEL=burger
ENV GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/opt/ros/humble/share/turtlebot3_gazebo/models

# ROS2環境をセットアップするためのエントリポイントスクリプト
RUN echo '#!/bin/bash\n\
source /opt/ros/humble/setup.bash\n\
cd /ros2_ws\n\
colcon build\n\
source /ros2_ws/install/setup.bash\n\
exec "$@"' > /entrypoint.sh && \
chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]