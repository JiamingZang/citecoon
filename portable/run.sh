#!/bin/sh
# Citecoon 便携启动器：在指定文件夹启动包内 qodercli，skill+MCP 来自包内隔离配置目录。
# 用法：./run.sh [项目文件夹]    （缺省 = 当前目录）
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="$ROOT/data/config"

command -v qodercli >/dev/null 2>&1 || { echo "PATH 上没有 qodercli：curl -fsSL https://qoder.com/install | bash"; exit 1; }
[ -d "$CONFIG_DIR" ] || { echo "先跑 ./setup.sh 装配"; exit 1; }

# PAT：便携包自带 .env，不碰宿主机 ~/.qoder（已禁用隔离：登录态用宿主机 ~/.qoder）
if [ -f "$ROOT/.env" ]; then
  set -a; . "$ROOT/.env"; set +a
fi
if [ -z "$QODER_PERSONAL_ACCESS_TOKEN" ]; then
  echo "（未设置 portable/.env 的 PAT——使用宿主机 ~/.qoder 登录态）"
fi

# 隔离配置目录：skill/会话/MCP 配置全在包内，与宿主机 ~/.qoder 互不干扰
# （已禁用：改用宿主机 ~/.qoder 配置——PAT/模型/权限直接复用；
#  citecoon skill 与 MCP 已装进宿主机，见 setup.sh）
# export QODER_CONFIG_DIR="$CONFIG_DIR"

TARGET="${1:-.}"
mkdir -p "$TARGET"
cd "$TARGET"
echo "在 $(pwd) 启动（宿主 ~/.qoder 配置）"
exec qodercli
