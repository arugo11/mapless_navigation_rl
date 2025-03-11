#!/bin/bash

# 必要なディレクトリの作成
mkdir -p models results logs

# X11表示の許可設定
xhost +local:docker

# コンテナのビルドと起動
docker-compose up -d

# コンテナにアタッチ
docker exec -it ros2_mapless_navigation bash

# コンテナを終了する場合は以下を実行
# docker compose down

# X11表示の許可を元に戻す場合
# xhost -local:docker