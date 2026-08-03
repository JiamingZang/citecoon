# FoundPose 仓库卡片

> 仓库: https://github.com/facebookresearch/foundpose
> 克隆方式: `git clone --depth 1`（浅克隆，**未拉取子模块**）
> 关注点: DINOv2 特征提取用的是哪一层、在哪定义、是否硬编码、如何改成可配置以支持自适应层选择

---

## 架构总览

FoundPose 是 Meta 发表于 ECCV 2024 的 training-free 6DoF 未见过物体位姿估计方法，核心思路是基于 DINOv2 基础模型特征进行模板匹配 + PnP。仓库代码集中在 `utils/`（核心工具库）与 `scripts/`（三步流水线脚本）两个目录，DINOv2 与 BOP Toolkit 作为 git 子模块放在 `external/`。流水线分三步：(1) `gen_templates.py` 渲染物体多视角模板；(2) `gen_repre.py` 用 DINOv2 提取模板特征、PCA 降维、KMeans 聚类成视觉词袋、计算 TF-IDF 模板描述子；(3) `infer.py` 对测试图提特征、TF-IDF 检索 Top-N 模板、循环伙伴匹配建立 2D-3D 对应、PnP 求位姿。DINOv2 特征提取的全部逻辑封装在 `utils/dinov2_utils.py` 的 `DinoFeatureExtractor` 类中。

```
foundpose/
├── README.md
├── conda_foundpose_gpu.yaml        # GPU 环境
├── conda_foundpose_mps.yaml        # MPS (Mac) 环境
├── configs/
│   ├── gen_templates/lmo.json
│   ├── gen_repre/lmo.json          # 特征提取配置（含 layer=9）
│   └── infer/lmo.json              # 特征提取配置（含 layer=9）
├── scripts/
│   ├── gen_templates.py            # 步骤1: 渲染模板
│   ├── gen_repre.py                # 步骤2: 生成物体表示（特征提取+PCA+聚类+TF-IDF）
│   ├── infer.py                    # 步骤3: 推理位姿
│   └── prepare_bop_submission.py   # 步骤4: 打包 BOP 提交
├── utils/
│   ├── dinov2_utils.py             # ★ DinoFeatureExtractor（层选择核心）
│   ├── feature_util.py             # make_feature_extractor 工厂 + 特征采样/3D 注册
│   ├── repre_util.py               # 物体表示序列化
│   ├── cluster_util.py             # KMeans 聚类
│   ├── projector_util.py           # PCA 投影
│   ├── template_util.py            # TF-IDF 描述子
│   ├── pnp_util.py                 # PnP 求解
│   ├── corresp_util.py             # 2D-3D 对应建立
│   ├── knn_util.py                 # KNN 检索
│   ├── infer_pose_util.py          # 推理流程编排
│   ├── renderer*.py                # 渲染器
│   └── ...
└── external/
    ├── dinov2/                     # 子模块（浅克隆下为空）
    └── bop_toolkit/                # 子模块（浅克隆下为空）
```

---

## 关键事实

### F1. 默认层号在 `__init__` 中硬编码为 9

`utils/dinov2_utils.py:52-57`：
```python
# Default parameter values.
self.version: str = "vits14-reg"
self.stride: int = 14
self.facet: str = "token"
self.layer: int = 9
self.apply_norm: bool = True
```
默认层号 `self.layer = 9` 是硬编码常量，未从外部配置文件或环境变量读取。当 `model_name` 为简写格式（如 `"dinov2_vits14"`）时，此默认值即为最终使用的层号。

### F2. 层号通过 `model_name` 字符串解析覆盖

`utils/dinov2_utils.py:60-78`：
```python
# Parse the model name.
name_items = model_name.split("_")
assert name_items[0] == "dinov2"
if len(name_items) == 2:
    # Example: "dinov2_vits14"
    self.version = name_items[1]
else:
    # Example: "dinov2_version=vitl14_stride=14_facet=key_layer=18_norm=1"
    for item in name_items[1:]:
        name, value = item.split("=")
        if name == "version":
            self.version = value
        elif name == "stride":
            self.stride = int(value)
        elif name == "facet":
            self.facet = value
        elif name == "layer":
            self.layer = int(value)
        elif name == "norm":
            self.apply_norm = bool(int(value))
```
层号 **已经可配置**：通过 `model_name` 字符串中的 `layer=N` 字段。解析方式为 `split("_")` + `split("=")`。但仅支持单层（`int`），不支持多层列表。

### F3. 两个配置文件均显式指定 `layer=9`

`configs/infer/lmo.json:12`：
```json
"extractor_name": "dinov2_version=vits14-reg_stride=14_facet=token_layer=9_logbin=0_norm=1",
```

`configs/gen_repre/lmo.json:7`：
```json
"extractor_name": "dinov2_version=vits14-reg_stride=14_facet=token_layer=9_logbin=0_norm=1",
```
两处一致：`vits14-reg`（ViT-S/14 注册版）、`stride=14`、`facet=token`、`layer=9`、`norm=1`。DINOv2 ViT-S 共 12 层（block 0–11），layer=9 即第 10 层（0-indexed）的 token 输出。

### F4. `forward` 使用 `self.layer` 调用 `extract_descriptors`

`utils/dinov2_utils.py:115-158`：
```python
def forward(self, images: torch.Tensor) -> tp.Dict[str, torch.Tensor]:
    # ...
    images = self.normalize(images)
    outputs = self.extract_descriptors(
        batch=images,
        layer=self.layer,
        facet=self.facet,
    )
    # CLS tokens of size Bx1xD.
    cls_tokens = outputs["cls_tokens"][:, 0, :, :]
    # Patch tokens of size BxTxD.
    patch_tokens = outputs["patch_tokens"][:, 0, :, :]
    # Normalize the tokens (LayerNorm is applied to all of them, as in DINOv2).
    if self.apply_norm:
        tokens = torch.cat([cls_tokens, patch_tokens], dim=1)
        tokens = self.model.norm(tokens)
        cls_tokens = tokens[:, :1, :]
        patch_tokens = tokens[:, 1:, :]
    # Reshape patch tokens to BxDxHxW.
    # ...
    return {
        "cls_tokens": cls_tokens[:, 0, :],  # BxD
        "feature_maps": feature_maps,  # BxDxHxW
    }
```
`forward` 只传单个 `self.layer`（`int`），输出只有该层的特征。`forward` 第 137-142 行在 `self.apply_norm=True` 时对所有 token 统一施加 `self.model.norm`（最终 LayerNorm），这是 DINOv2 标准做法。

### F5. `extract_descriptors` → `_extract_features` → `_register_hooks` 的 hook 链路

`utils/dinov2_utils.py:266-311`（`extract_descriptors`）：
```python
def extract_descriptors(
    self,
    batch: torch.Tensor,
    layer: int = 11,
    facet: str = "key",
) -> tp.Dict[str, torch.Tensor]:
    # ...
    self._extract_features(batch, [layer], facet)
    # ...
```

`utils/dinov2_utils.py:232-264`（`_extract_features`）：
```python
def _extract_features(
    self,
    batch: torch.Tensor,
    layers: Optional[List[int]] = None,
    facet: str = "key",
) -> List[torch.Tensor]:
    if layers is None:
        layers = [11]
    B, C, H, W = batch.shape
    self._feats = []
    self._register_hooks(layers, facet)
    _ = self.model(batch)
    self._unregister_hooks()
    # ...
    return self._feats
```

`utils/dinov2_utils.py:198-223`（`_register_hooks`）：
```python
def _register_hooks(self, layers: List[int], facet: str) -> None:
    for block_idx, block in enumerate(self.model.blocks):
        if block_idx in layers:
            if facet == "token":
                self.hook_handlers.append(
                    block.register_forward_hook(self._get_hook(facet))
                )
            elif facet == "attn":
                self.hook_handlers.append(
                    block.attn.attn_drop.register_forward_hook(
                        self._get_hook(facet)
                    )
                )
            elif facet in ["key", "query", "value"]:
                self.hook_handlers.append(
                    block.attn.register_forward_hook(self._get_hook(facet))
                )
            else:
                raise TypeError(f"{facet} is not a supported facet.")
```
特征提取通过 PyTorch forward hook 实现：遍历 `self.model.blocks`，在 `block_idx in layers` 的块上注册 hook。**`_extract_features` 和 `_register_hooks` 本身已支持 `List[int]` 多层**，但 `extract_descriptors` 只接收单个 `layer: int` 并包成 `[layer]` 传入，`forward` 又只调用 `extract_descriptors` 一次——所以多层能力被上层接口"卡住"了。

### F6. hook 回调按 facet 类型捕获不同中间量

`utils/dinov2_utils.py:160-196`：
```python
def _get_hook(
    self, facet: str
) -> Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], None]:
    if facet in ["attn", "token"]:
        def _hook(
            module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor
        ) -> None:
            self._feats.append(output)
        return _hook

    if facet == "query":
        facet_idx: int = 0
    elif facet == "key":
        facet_idx: int = 1
    elif facet == "value":
        facet_idx: int = 2
    else:
        raise TypeError(f"{facet} is not a supported facet.")

    def _inner_hook(
        module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor
    ) -> None:
        input = input[0]
        B, N, C = input.shape
        qkv = (
            module.qkv(input)
            .reshape(B, N, 3, module.num_heads, C // module.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        self._feats.append(qkv[facet_idx])  # Bxhxtxd

    return _inner_hook
```
`facet="token"` 时直接捕获 block 输出；`facet="key"/"query"/"value"` 时在 `block.attn` 上重新计算 qkv 并取对应分片。当前配置用 `facet=token`。

### F7. 特征提取器的工厂入口

`utils/feature_util.py:18-23`：
```python
def make_feature_extractor(model_name: str) -> torch.nn.Module:
    if model_name.startswith("dinov2_"):
        return dinov2_utils.DinoFeatureExtractor(model_name=model_name)
    else:
        raise NotImplementedError(model_name)
```
所有脚本通过此工厂创建提取器，`model_name` 即完整配置字符串。

### F8. 脚本默认值与配置文件不一致

`scripts/infer.py:74-75`：
```python
# Feature extraction options.
extractor_name: str = "dinov2_vitl14"
```

`scripts/gen_repre.py:45-47`：
```python
# Feature extraction options.
extractor_name: str = "dinov2_vits14_reg"
grid_cell_size: float = 14.0
```
`infer.py` 默认 `vitl14`（ViT-L，24 层），`gen_repre.py` 默认 `vits14_reg`（ViT-S，12 层），但配置文件两处都用 `vits14-reg`。`gen_repre.py` 的默认值 `"dinov2_vits14_reg"` 若被实际使用会解析失败（见"风险与未知"R3）。实际运行时 `extractor_name` 从 JSON 配置加载，覆盖默认值。

### F9. 特征提取在推理与表示生成中的调用点

`scripts/infer.py:470-471`（推理时对裁剪图提特征）：
```python
extractor_output = extractor(image_tensor_bchw)
feature_map_chw = extractor_output["feature_maps"][0]
```

`utils/feature_util.py:217-219`（表示生成时对模板提特征）：
```python
extractor_output = extractor(image_bchw)
feature_map_chw = extractor_output["feature_maps"][0]
feature_map_chw = feature_map_chw.to(device)
```
两处都只用 `feature_maps`（patch tokens 重排为 BxDxHxW），`cls_tokens` 未在下游使用。

### F10. 源码注释暗示 vitl14 最后一层为 layer=23

`utils/dinov2_utils.py:117-120`：
```python
# Note: function `extract_output_features` defined in the DINOv2 model itself
# outputs normalized token facets from the last layer. For example, for
# vitl14, the same output can be obtained with model name:
# "dinov2_version=vitl14_stride=14_facet=token_layer=23_norm=1"
```
注释明确指出 vitl14 最后一层是 `layer=23`（即 24 层，0-indexed 0–23）。对应地 vits14 最后一层应为 `layer=11`。

---

## 硬编码参数与配置点

| 参数 | 位置 | 值 | 配置方式 |
|------|------|----|----------|
| 默认层号 `layer` | `utils/dinov2_utils.py:56` | `9` | `model_name` 字符串 `layer=N` 覆盖 |
| 默认版本 `version` | `utils/dinov2_utils.py:53` | `"vits14-reg"` | `model_name` 字符串 `version=V` 覆盖 |
| 默认 stride | `utils/dinov2_utils.py:54` | `14` | `model_name` 字符串 `stride=N` 覆盖 |
| 默认 facet | `utils/dinov2_utils.py:55` | `"token"` | `model_name` 字符串 `facet=F` 覆盖 |
| 默认 apply_norm | `utils/dinov2_utils.py:57` | `True` | `model_name` 字符串 `norm=0/1` 覆盖 |
| `extract_descriptors` 的 layer 默认 | `utils/dinov2_utils.py:269` | `11` | 几乎不会被用到（`forward` 传 `self.layer`） |
| `_extract_features` 的 layers 默认 | `utils/dinov2_utils.py:252` | `[11]` | 同上，几乎不会被用到 |
| 图像归一化均值/标准差 | `utils/dinov2_utils.py:111-113` | ImageNet mean/std `(0.485,0.456,0.406)` / `(0.229,0.224,0.225)` | 硬编码，无配置入口 |
| 推理 extractor_name 默认 | `scripts/infer.py:75` | `"dinov2_vitl14"` | JSON 配置覆盖 |
| 表示 extractor_name 默认 | `scripts/gen_repre.py:46` | `"dinov2_vits14_reg"` | JSON 配置覆盖 |
| 配置文件 extractor_name | `configs/infer/lmo.json:12`, `configs/gen_repre/lmo.json:7` | `"...layer=9_logbin=0_norm=1"` | 直接编辑 JSON |
| grid_cell_size | `configs/infer/lmo.json:13`, `configs/gen_repre/lmo.json:8` | `14.0` | 直接编辑 JSON |
| PCA 维度 | `configs/gen_repre/lmo.json:10` | `256` | 直接编辑 JSON |
| 聚类数 | `configs/gen_repre/lmo.json:12` | `2048` | 直接编辑 JSON |

**层号改成可配置的最小方式**：层号 **已经可配置**，只需修改 `configs/*/lmo.json` 中 `extractor_name` 字符串的 `layer=N` 部分，无需改任何 Python 代码。例如改成最后一层：`layer=11`（vits14）或 `layer=23`（vitl14）。

**支持自适应层选择需要改代码**：当前 `forward` 只提取单层。要支持"根据输入自适应选层"或"多层融合"，需修改 `forward` / `extract_descriptors` 接口（见下节）。

---

## 环境与复现

### 依赖（以 `conda_foundpose_gpu.yaml` 为准）

- Python 3.9
- PyTorch 2.3.0 + CUDA 11.7 + torchvision 0.18.0
- 关键 pip 包: `kornia==0.7.2`, `xformers==0.0.20`, `opencv-python==4.5.5.62`, `pyrender==0.1.45`, `scikit-learn==1.5.0`, `faiss-gpu=1.8.0`, `torchinfo`
- 子模块: `external/dinov2`（DINOv2 源码，提供 `dinov2.hub.backbones`）、`external/bop_toolkit`（BOP 工具链）

### 权重下载

- **无需手动下载**：`utils/dinov2_utils.py:82-84` 通过 `dinov2_backbones.__dict__[...](pretrained=True)` 从 DINOv2 官方 hub 自动拉取预训练权重。
- 模板与表示可选下载: HuggingFace `evinpinar/foundpose`（`templates.zip`、`object_repre.zip`、`inference.zip`）。

### 环境变量（README 要求）

```bash
export REPO_PATH=/path/to/foundpose
export BOP_PATH=/path/to/bop/datasets
export PYTHONPATH=$REPO_PATH:$REPO_PATH/external/bop_toolkit:$REPO_PATH/external/dinov2
```

### 最小运行命令（LM-O 数据集）

```bash
# 0. 正确克隆（含子模块）
git clone --recurse-submodules https://github.com/facebookresearch/foundpose
cd foundpose
conda env create -f conda_foundpose_gpu.yaml   # 或 conda_foundpose_mps.yaml
conda activate foundpose_gpu

# 1. 渲染模板
python scripts/gen_templates.py --opts-path configs/gen_templates/lmo.json

# 2. 生成物体表示
python scripts/gen_repre.py --opts-path configs/gen_repre/lmo.json

# 3. 推理
python scripts/infer.py --opts-path configs/infer/lmo.json

# 4. 打包提交
python scripts/prepare_bop_submission.py
```

### 复现指标（README 声明，DINOv2 ViT-S）

| Dataset | Published AR | Reproduced AR |
|---------|-------------|---------------|
| LMO     | 34.0        | 33.7          |
| TUD-L   | 42.7        | 40.7          |

---

## 改造接口点

针对"自适应层选择"关注点，按侵入性从低到高列出三个方案。

### 方案 A：静态改层（零代码改动，已支持）

直接改 `configs/infer/lmo.json:12` 和 `configs/gen_repre/lmo.json:7` 的 `extractor_name` 中 `layer=N`。例如：
```json
"extractor_name": "dinov2_version=vits14-reg_stride=14_facet=token_layer=11_logbin=0_norm=1"
```
**限制**：只能选单层，不能融合多层，不能根据输入动态选层。

### 方案 B：多层融合（小改 `dinov2_utils.py`）

`_extract_features` 已支持 `List[int]`，只需打通上层：

1. `utils/dinov2_utils.py:266-311` `extract_descriptors` 的 `layer: int` 改为 `layers: Union[int, List[int]]`，内部归一化为列表，循环拼接 `self._feats`。
2. `utils/dinov2_utils.py:115-158` `forward` 的 `self.layer` 改为 `self.layers: List[int]`，对多层输出做拼接或加权求和后返回 `feature_maps`。
3. `utils/dinov2_utils.py:56` 默认值改为 `self.layers: List[int] = [9]`。
4. `utils/dinov2_utils.py:60-78` 解析逻辑支持 `layer=9,11` 或 `layers=9_11` 格式。

**优点**：改动集中在 `DinoFeatureExtractor`，下游 `feature_util.py` / `scripts/*.py` 无需改（仍调 `extractor(image)` 拿 `feature_maps`）。

### 方案 C：自适应层选择（中等改动）

在 `DinoFeatureExtractor` 中引入层选择策略：

1. `__init__` 新增 `layer_selection: str = "fixed"` 参数（`"fixed"` / `"by_entropy"` / `"by_norm"` 等），解析 `model_name` 中的 `lselect=...` 字段。
2. `forward` 中先做一次轻量前向（如取所有层 hook 输出），按策略打分选层，再正式提取。或直接在 `_extract_features` 注册所有层 hook，`forward` 根据策略加权融合。
3. 策略所需的中间统计（如各层 token L2 范数、注意力熵）可在 `_get_hook` 中额外累积。

**推荐入口**：`utils/dinov2_utils.py:115` 的 `forward` 方法 + `utils/dinov2_utils.py:52-57` 的默认参数块。配置通过 `configs/*/lmo.json` 的 `extractor_name` 字符串扩展（如加 `lselect=by_norm`），解析逻辑在 `utils/dinov2_utils.py:60-78` 的 for 循环中新增分支。

---

## 风险与未知

### R1. DINOv2 子模块未拉取（浅克隆限制）

本次侦察用 `git clone --depth 1`，`external/dinov2/` 与 `external/bop_toolkit/` 为空目录。因此 **未能直接查证**：
- `dinov2.hub.backbones` 中各版本的 `blocks` 列表确切层数（vits14=12、vitl14=24 等依据为源码注释 F10 与 DINOv2 公开架构，未在本次克隆中逐文件验证）。
- `self.model.norm` 的确切实现位置。
- `self.model.num_register_tokens`（`dinov2_utils.py:304`）的来源与默认值。

**如需验证**：`git clone --recurse-submodules https://github.com/facebookresearch/foundpose` 后查看 `external/dinov2/dinov2/models/vision_transformer.py`。

### R2. `logbin=0` 参数被静默忽略

配置字符串含 `logbin=0`（`configs/infer/lmo.json:12`、`configs/gen_repre/lmo.json:7`），但 `utils/dinov2_utils.py:67-78` 的解析循环无 `logbin` 分支。`item.split("=")` 得到 `name="logbin"`，不匹配任何 `if/elif`，被静默跳过。**未查证** `logbin` 是否在 DINOv2 子模块内部有意义（可能与 `xformers` 内存高效注意力或 log-binomial 注意力相关），本仓库代码未使用。

### R3. `gen_repre.py` 默认 `extractor_name` 字符串格式可能无法解析

`scripts/gen_repre.py:46` 默认值 `"dinov2_vits14_reg"`：
- `split("_")` → `["dinov2", "vits14", "reg"]`（长度 3）
- 走 `else` 分支，`"vits14".split("=")` → `["vits14"]`
- `name, value = ["vits14"]` → **ValueError: not enough values to unpack**

该默认值仅在 JSON 未指定 `extractor_name` 时触发；`configs/gen_repre/lmo.json` 已显式指定，故 LM-O 路径不受影响。但若用户自定义配置遗漏该字段，会崩溃。`scripts/infer.py:75` 的 `"dinov2_vitl14"` 格式正确（长度 2，走 `if` 分支），无此问题。正确写法应为 `"dinov2_vits14-reg"`（连字符，非下划线），见 `dinov2_utils.py:63` 注释。

### R4. `extract_descriptors` / `_extract_features` 默认层号 11 与 `__init__` 默认 9 不一致

`utils/dinov2_utils.py:269` (`layer: int = 11`) 与 `:252` (`layers = [11]`) 的函数签名默认值是 11，而 `__init__` 默认 `self.layer = 9`。由于 `forward` 总是显式传 `self.layer`，这些签名默认值实际不会被使用，但属于潜在误导。**未查证** 11 是否为早期版本残留。

### R5. 未查证下游对特征维度的硬依赖

`configs/gen_repre/lmo.json` 设 `pca_components=256`、`cluster_num=2048`。若改层导致特征维度变化（如从 vits14 的 384 维换成 vitl14 的 1024 维），PCA/聚类参数需同步调整。本次未深入 `projector_util.py` / `cluster_util.py` 验证是否存在维度断言。

### R6. 未查证 `apply_norm` 与层号的交互

`forward` 第 137-142 行在 `apply_norm=True` 时对**任意层**的输出统一施加 `self.model.norm`（最终层 LayerNorm）。对非最后一层施加最终 LayerNorm 的效果未在本仓库文档中说明，是否为论文 intended 行为未查证。
