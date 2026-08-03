# Citecoon 便携包（qodercli + skill + MCP 即插即用）

一个文件夹 = 可运行的 qodercli + 预装 citecoon skill + citecoon MCP server。
拷到任何同平台机器，填 PAT，`./run.sh <文件夹>` 就能用。

## 布局

```
portable/
├── run.sh        启动器：在指定文件夹启动 qodercli（隔离配置生效）
├── setup.sh      首次装配：venv 装 wheel、拷 skill、写 MCP 配置
├── .env          QODER_PERSONAL_ACCESS_TOKEN（勿外传）
├── dist/         citecoon wheel（构建产物）
└── data/
    ├── config/   隔离配置目录（= QODER_CONFIG_DIR）：settings.json + skills/
    └── projects/ citecoon 研究项目默认存放地
```

## 快速开始

```sh
# 每台机器一次：官方安装 qodercli（本体不进便携包）
curl -fsSL https://qoder.com/install | bash    # Windows: irm https://qoder.com/install.ps1 | iex

./setup.sh                 # 一次：装依赖、拷 skill、写配置
# 把 PAT 填进 .env
./run.sh ~/some/project    # 在该文件夹启动，skill+MCP 自动就位
```

## 你问的关键点：指定文件夹启动 + skill 加载怎么共存？

两条通道互不干扰：

1. **启动位置** = qodercli 的当前工作目录。`run.sh` 先 `cd` 过去再启动，
   qodercli 就把那里当项目（读该目录的 AGENTS.md、代码、git 状态）。
2. **skill 加载** = 全局位 `<QODER_CONFIG_DIR>/skills/`。便携包把
   `QODER_CONFIG_DIR` 指到 `data/config/`，skill 放里面，**在任何文件夹
   启动都生效**，且不污染宿主机的 `~/.qoder`。

（另一条路：把 skill 放进目标文件夹的 `.qoder/skills/` 项目位，跟着文件夹
走——适合"这个仓库才需要调研技能"的场景。便携包默认用全局位。）

MCP 同理：`data/config/settings.json` 里的 `mcpServers.citecoon` 指向
`data/venv/bin/citecoon`（setup.sh 写绝对路径），随包移动后重跑 setup.sh
重写路径即可。

## PAT

`portable/.env` 里填 `QODER_PERSONAL_ACCESS_TOKEN=...`。run.sh 导出它，
qodercli 以该身份运行（计费走这个账号）。**.env 不要随包分发给他人**——
分发时清空 PAT，让接收者填自己的。

## 已知边界

- `QODER_CONFIG_DIR` / PAT 环境变量行为依据 qoder-agent-sdk 文档与本机
  实测；qodercli 本体每台机器走官方安装脚本，便携包不含它。
- 包内 venv 与平台绑定，换平台需重跑 setup.sh（wheel 是纯 Python，跨平台；
  但 venv 里的解释器路径不跨平台）。
- citecoon 的检索走公网（OpenAlex/arXiv/S2），无网环境只剩本地图谱操作。
