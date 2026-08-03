#!/bin/sh
# citecoon MCP 启动 wrapper：带上 autodl 学术网络加速代理（直连 arXiv 超时）
source /etc/network_turbo >/dev/null 2>&1
exec "$(dirname "$0")/data/venv/bin/citecoon" mcp "$@"
