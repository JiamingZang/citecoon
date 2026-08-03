#!/bin/sh
# Citecoon 便携包首次装配：venv + wheel + skill + qodercli 配置目录。
# 用法：./setup.sh   （在 portable/ 目录下执行一次）
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
CITECOON_ROOT="$(cd "$ROOT/.." && pwd)"
DATA="$ROOT/data"
CONFIG_DIR="$DATA/config"
VENV="$DATA/venv"

echo "== 1/4 创建隔离 venv 并安装 citecoon wheel =="
WHEEL=$(ls "$ROOT/dist"/citecoon-*.whl 2>/dev/null | head -1 || true)
if [ -z "$WHEEL" ]; then
  echo "dist/ 里没有 wheel，先从 citecoon 根目录构建："
  echo "  .venv/bin/python -m pip wheel --no-deps -w portable/dist ."
  exit 1
fi
PY=""
for cand in python3.13 python3.12 python3; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import sys;raise SystemExit(0 if sys.version_info>=(3,12) else 1)' 2>/dev/null; then
    PY="$cand"; break
  fi
done
if [ -z "$PY" ]; then
  echo "找不到 >=3.12 的 Python（citecoon 要求）。先装一个再来。"; exit 1
fi
"$PY" -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet "$WHEEL"
echo "  citecoon -> $("$VENV/bin/citecoon" --version 2>/dev/null || echo '已安装')"

echo "== 2/4 拷贝 skill 到隔离配置目录 =="
mkdir -p "$CONFIG_DIR/skills"
rm -rf "$CONFIG_DIR/skills/citecoon"
cp -R "$CITECOON_ROOT/skill/citecoon" "$CONFIG_DIR/skills/citecoon"

echo "== 3/4 写 mcpServers 配置（指向包内 venv，绝对路径） =="
mkdir -p "$CONFIG_DIR"
cat > "$CONFIG_DIR/settings.json" <<EOF
{
  "mcpServers": {
    "citecoon": {
      "command": "$VENV/bin/citecoon",
      "args": ["mcp", "-p", "$DATA/projects"]
    }
  }
}
EOF
mkdir -p "$DATA/projects"

echo "== 4/4 检查 qodercli =="
if command -v qodercli >/dev/null 2>&1; then
  echo "  已找到：$(command -v qodercli)"
else
  echo "  ⚠ PATH 上没有 qodercli。用官方脚本安装（一次）："
  echo "    curl -fsSL https://qoder.com/install | bash"
  echo "  （Windows：irm https://qoder.com/install.ps1 | iex）"
fi

if [ ! -f "$ROOT/.env" ]; then
  printf 'QODER_PERSONAL_ACCESS_TOKEN=\n' > "$ROOT/.env"
  echo "  已生成 .env 模板——填入 PAT 后才能登录使用。"
fi

echo ""
echo "装配完成。用法：./run.sh <要在哪个文件夹启动>"
