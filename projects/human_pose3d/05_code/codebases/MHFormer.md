# MHFormer 仓库侦察卡

> 目标仓库：https://github.com/Vegetebird/MHFormer  
> 侦察日期：2026-07-21  
> 侦察目的：为"保留 K 个假设的逐关节 3D 分布而非坍缩均值"定位最小侵入改造点。

---

## 架构总览

MHFormer 是一个单阶段、视频输入的 3D 人体姿态估计网络，核心思路是：用 3 个并行的 Multi-Hypothesis Generator（MHG）从 2D 关节序列生成 3 组初始假设，再经过 Self-Hypothesis Refinement（SHR）和 Cross-Hypothesis Interaction（CHI）两个 Transformer 模块做假设内/假设间交互，最后用一个 1×1 Conv1d 把 3 个假设的特征通道压缩成单一 3D 姿态。训练/测试主流程在 `main.py`，数据读取由 `common/load_data_hm36.py` 中的 `Fusion` 类负责，它从 Human3.6M 的 `.npz` 文件中读取 2D 检测（默认 CPN）和 3D GT，并通过 `common/generator.py` 的 `ChunkedGenerator` 切分成以中心帧为目标的滑窗样本。评测指标 MPJPE/P-MPJPE 定义在 `common/utils.py`。

```
MHFormer/
├── main.py                      # 训练/测试入口
├── model/
│   ├── mhformer.py              # 整体 Model.forward，三假设聚合/回归输出
│   └── module/
│       ├── trans.py             # MHG 使用的单流 Transformer
│       └── trans_hypothesis.py  # SHR/CHI 多假设 Transformer
├── common/
│   ├── opt.py                   # 命令行参数（无 n_hypotheses）
│   ├── load_data_hm36.py        # Human3.6M 数据加载器 Fusion
│   ├── generator.py             # 滑窗样本生成 ChunkedGenerator
│   ├── h36m_dataset.py          # Human36mDataset 定义
│   ├── utils.py                 # MPJPE / P-MPJPE 评测与误差累加
│   └── camera.py                # 相机坐标归一化
├── demo/
│   ├── vis.py                   # 视频 demo 入口（HRNet 生成 2D → MHFormer）
│   └── lib/...                  # YOLOv3 / HRNet 检测/2D 姿态依赖
├── requirements.txt             # 依赖列表
└── README.md                    # 安装、数据、权重、运行命令
```

---

## 关键事实

### 1. 三假设最终聚合为单一输出的位置

多假设流在 `model/module/trans_hypothesis.py` 的 `Transformer.forward` 里先被 CHI 块交互，再沿通道维度拼接，最后由 `model/mhformer.py` 的 `regression` 头坍缩为单一 3D 姿态。

- CHI 块输出三假设并拼接（`dim=2` 为通道维度）：  
  `model/module/trans_hypothesis.py:205-208`
  ```python
  x_1, x_2, x_3 = self.CHI_blocks[0](x_1, x_2, x_3)

  x = torch.cat([x_1, x_2, x_3], dim=2)
  x = self.norm(x)
  ```

- 单一输出的回归头，把 `channel*3` 直接映射到 `3*out_joints`：  
  `model/mhformer.py:51-54`
  ```python
  self.regression = nn.Sequential(
      nn.BatchNorm1d(args.channel*3, momentum=0.1),
      nn.Conv1d(args.channel*3, 3*args.out_joints, kernel_size=1)
  )
  ```

- `Model.forward` 中调用该回归头并 reshape 为 `(B, F, J, 3)`：  
  `model/mhformer.py:71-78`
  ```python
  ## SHR & CHI
  x = self.Transformer_hypothesis(x_1, x_2, x_3)

  ## Regression
  x = x.permute(0, 2, 1).contiguous()
  x = self.regression(x)
  x = rearrange(x, 'b (j c) f -> b f j c', j=J).contiguous()

  return x
  ```

因此，**把多假设聚合为单一输出的真正操作是 `self.regression` 这个 1×1 Conv1d**，而非 CHI 块本身（CHI 块仍然输出 3 个独立流）。

### 2. 如何修改 forward 以同时返回 K 个假设的逐关节 3D 坐标

当前代码里假设数 `K=3` 被硬编码，最小侵入改造分两步：

**步骤 A**：在 `model/mhformer.py` 把单一回归头替换为 3 个独立回归头（每个只接收 `args.channel`）：
```python
# 原代码 model/mhformer.py:51-54
self.regression = nn.Sequential(
    nn.BatchNorm1d(args.channel*3, momentum=0.1),
    nn.Conv1d(args.channel*3, 3*args.out_joints, kernel_size=1)
)

# 改造后（保留 3 个假设，互不混合）
self.regression = nn.ModuleList([
    nn.Sequential(
        nn.BatchNorm1d(args.channel, momentum=0.1),
        nn.Conv1d(args.channel, 3*args.out_joints, kernel_size=1)
    ) for _ in range(3)
])
```

**步骤 B**：在 `model/mhformer.py:71-78` 不把三假设拼接，而是分别回归再堆叠：
```python
## SHR & CHI：让 Transformer_hypothesis 返回三流，不拼接
x_1, x_2, x_3 = self.Transformer_hypothesis(x_1, x_2, x_3)

## Regression
x_1 = self.regression[0](x_1.permute(0, 2, 1).contiguous())
x_2 = self.regression[1](x_2.permute(0, 2, 1).contiguous())
x_3 = self.regression[2](x_3.permute(0, 2, 1).contiguous())

x = torch.stack([x_1, x_2, x_3], dim=1)  # (B, K, J*3, F)
x = rearrange(x, 'b k (j c) f -> b k f j c', j=J).contiguous()
return x  # 形状 (B, K=3, F, J, 3)
```

**对应地需要修改 `model/module/trans_hypothesis.py:207-210`**，取消最后的拼接：
```python
# 原
x = torch.cat([x_1, x_2, x_3], dim=2)
x = self.norm(x)
return x

# 改造后
x_1 = self.norm(x_1)
x_2 = self.norm(x_2)
x_3 = self.norm(x_3)
return x_1, x_2, x_3
```

**输出张量形状**：改造后 `Model.forward` 返回 `(B, K, F, J, 3)`，其中 `K=3`、`F=args.frames`、`J=args.out_joints`（默认 17）、最后一维是 `(x, y, z)`。

### 3. CPN 2D 输入接口与 Human3.6M 数据加载管线

- **CPN 输入文件名**：在 `common/load_data_hm36.py:60` 由 `keypoints_name` 决定，默认读取 `data_2d_h36m_cpn_ft_h36m_dbp.npz`：
  ```python
  keypoints = np.load(self.root_path + 'data_2d_' + self.data_type + '_' + self.keypoints_name + '.npz',allow_pickle=True)
  ```
  `keypoints_name` 来自 `common/opt.py:16`：`parser.add_argument('-k', '--keypoints', default='cpn_ft_h36m_dbb', type=str)`。

- **数据集封装与 2D/3D 配对**：`common/load_data_hm36.py:10-45` 的 `Fusion` 类初始化 `Human36mDataset`，按 `subjects_train/subjects_test` 调用 `prepare_data` 和 `fetch`，最终交给 `ChunkedGenerator`。

- **Human36mDataset 定义与相机参数**：`common/h36m_dataset.py:204-249`，其中 `__init__` 加载 `dataset/data_3d_h36m.npz`，并执行根关节对齐与静态关节移除：
  ```python
  pos_3d[:, 1:] -= pos_3d[:, :1]
  ```

- **滑窗/填充/增强逻辑**：`common/generator.py:91-152` 的 `get_batch` 根据 `pad` 取中心帧前后窗口，并做水平翻转（`flip`）与时间反转（`reverse`）数据增强。

- **测试时的 flip 增强**：`main.py:70-87` 的 `input_augmentation` 对输入做左右翻转后平均两个流的输出，该函数在 `main.py:43` 的 `step` 中被调用：
  ```python
  input_2D, output_3D = input_augmentation(input_2D, model)
  ```

### 4. 训练/推理入口脚本与评测（MPJPE / P-MPJPE）代码位置

- **训练入口**：`main.py:137-164` 的主循环，调用 `train(...)` 与 `val(...)`。
- **推理/测试入口**：`main.py:23-68` 的 `step('test', ...)`，当命令行带 `--test` 时进入（`common/opt.py:53-54` 将 `train` 置 0）。
- **训练损失**：`common/utils.py:13-16` 的 `mpjpe_cal`：
  ```python
  def mpjpe_cal(predicted, target):
      assert predicted.shape == target.shape
      return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))
  ```
- **评测入口**：`common/utils.py:18-23` 的 `test_calculation`：
  ```python
  def test_calculation(predicted, target, action, error_sum, data_type, subject):
      error_sum = mpjpe_by_action_p1(predicted, target, action, error_sum)
      error_sum = mpjpe_by_action_p2(predicted, target, action, error_sum)
      return error_sum
  ```
- **MPJPE（p1）**：`common/utils.py:25-48`。
- **P-MPJPE（p2）**：`common/utils.py:50-108`，先做 Procrustes 对齐（`p_mpjpe`）再计算关节距离。
- **误差打印**：`common/utils.py:164-195` 的 `print_error_action`，最终返回 `mean_error_p1, mean_error_p2`，单位为 mm（已乘 1000）。

### 5. 假设数 = 3 的超参配置位置

**当前代码里没有可配置的假设数超参**。`K=3` 被硬编码在以下位置：

- 三个 MHG Transformer：`model/mhformer.py:16-18`
  ```python
  self.Transformer_encoder_1 = Transformer_encoder(...)
  self.Transformer_encoder_2 = Transformer_encoder(...)
  self.Transformer_encoder_3 = Transformer_encoder(...)
  ```
- 三个 Embedding：`model/mhformer.py:22-45` 的 `embedding_1/2/3`。
- 三个假设位置编码与三流输入：`model/module/trans_hypothesis.py:168-174` 和 `forward:193-200`。
- SHR 块把三流沿通道拼接后再用 MLP：`model/module/trans_hypothesis.py:99-100` 与 `107-112`。
- CHI 块同样基于三流：`model/module/trans_hypothesis.py:185-189`、`205-208`。
- 最终回归头输入/输出通道写死为 `channel*3` / `3*out_joints`：`model/mhformer.py:52-53`。

`common/opt.py` 中没有 `n_hypotheses` 或 `num_hypotheses` 参数。

---

## 硬编码参数与配置点

| 参数 | 当前硬编码值 | 位置 | 如何改成可配置 |
|------|--------------|------|----------------|
| 假设数 K | 3 | `model/mhformer.py` 与 `model/module/trans_hypothesis.py` 多处 | 新增 `args.n_hypotheses`；把所有 `x_1/x_2/x_3`、三组 pos_embed、三个 encoder/embedding、SHR/CHI 中的 `dim*3` 都改为基于 K 的循环/列表 |
| 通道数 | 512 | `common/opt.py:13` | `--channel` |
| MLP 隐层 | 1024 | `common/opt.py:14` | `--d_hid` |
| SHR/CHI 层数 | 3（SHR=2，CHI=1） | `common/opt.py:12`、`model/module/trans_hypothesis.py:183` | `--layers` 已存在，但代码中 `SHR_blocks = range(depth-1)`、`CHI_blocks = range(1)`，层数关系固定 |
| 输入帧长 | 351 | `common/opt.py:36` | `--frames` |
| 中心帧 pad | 175（由 frames 推导） | `common/opt.py:37`、`common/opt.py:56` | `--pad` 可设，但默认被覆盖为 `(frames-1)//2` |
| 关节数 | 17 | `common/opt.py:40-41` | `--n_joints`、`--out_joints` |
| MHG head 数 | 9 | `model/mhformer.py:16-18` | 硬编码，未暴露为参数 |
| MHG depth | 4 | `model/mhformer.py:16-18` | 硬编码 |
| 数据增强 flip 关节对 | `[4,5,6,11,12,13]` / `[1,2,3,14,15,16]` | `main.py:71-72`、`demo/vis.py:179-180` | 硬编码，依赖 H36M 17 关节顺序 |
| 学习率调度 | 0.95 / 0.5 / 每 5 epoch | `common/opt.py:33-35`、`main.py:157-164` | `--lr_decay`、`--lr_decay_large`、`--large_decay_epoch` |
| batch size | 256 | `common/opt.py:30` | `--batch_size` |

**注意**：`model/mhformer.py:41` 中 `embedding_3` 的输入通道写成 `2*args.out_joints`，而 `embedding_1/2` 是 `2*args.n_joints`。默认两者都是 17，所以不影响运行，但属于潜在不一致点。

---

## 环境与复现

以 `README.md` 与 `requirements.txt` 为准：

1. **创建环境**
   ```bash
   conda create -n mhformer python=3.9
   conda activate mhformer
   ```

2. **安装 PyTorch**（README 指定 1.7.1 + torchvision 0.8.2）：
   ```bash
   # 按官方说明安装对应 CUDA 版本
   pip install torch==1.7.1 torchvision==0.8.2
   ```

3. **安装其他依赖**
   ```bash
   pip install -r requirements.txt
   ```
   依赖包括：numpy、einops、timm、matplotlib==3.7.1、tqdm、yacs、numba、filterpy、scikit-image、opencv-python、ipython。

4. **准备 Human3.6M 数据**（目录 `dataset/`）
   ```
   dataset/
   ├── data_3d_h36m.npz
   ├── data_2d_h36m_gt.npz
   └── data_2d_h36m_cpn_ft_h36m_dbb.npz
   ```
   可从 [Google Drive](https://drive.google.com/drive/folders/112GPdRC9IEcwcJRyrLJeYw9_YV4wLdKC?usp=sharing) 下载已处理数据。

5. **下载预训练权重**到 `checkpoint/pretrained/351/`：
   [Google Drive](https://drive.google.com/drive/folders/1UWuaJ_nE19x2aM-Th221UpdhRPSCFwZa?usp=sharing)

6. **最小测试命令**
   ```bash
   python main.py --test --previous_dir 'checkpoint/pretrained/351' --frames 351
   ```

7. **最小训练命令**
   ```bash
   # 351 帧
   python main.py --frames 351 --batch_size 128
   # 81 帧
   python main.py --frames 81 --batch_size 256
   ```

8. **视频 demo**
   - 下载 YOLOv3 + HRNet 权重到 `demo/lib/checkpoint/`；
   - 放入视频到 `demo/video/`；
   - 运行 `python demo/vis.py --video sample_video.mp4`。

---

## 改造接口点

针对"保留 K 个假设的逐关节 3D 分布"这一目标，推荐的最小侵入修改位置如下（只列必要文件）：

1. **`model/module/trans_hypothesis.py:207-210`**
   - 取消 `torch.cat([x_1, x_2, x_3], dim=2)`，改为分别 LayerNorm 后返回三流 `(x_1, x_2, x_3)`。

2. **`model/mhformer.py:51-54`**
   - 把单一 `self.regression` 替换为 `nn.ModuleList` 形式的 3 个独立回归头，每个输入 `args.channel`、输出 `3*args.out_joints`。

3. **`model/mhformer.py:71-78`**
   - 分别对三流做 `permute + regression`，再用 `torch.stack` 在 `dim=1` 处堆叠，最终 `rearrange` 到 `(B, K, F, J, 3)`。

4. **`main.py:41-43` 与 `main.py:59-61`**
   - 训练分支 `output_3D = model(input_2D)` 会拿到 5 维输出，需在 `mpjpe_cal` 之前选择/平均某一假设或联合监督；
   - 测试分支 `input_augmentation` 返回后，`output_3D[:, opt.pad]` 的索引需要从 5 维调整为 `output_3D[:, :, opt.pad]`，并同步修改 `test_calculation` 的输入形状。

5. **`demo/vis.py:196-205`**
   - demo 推理拿到 5 维输出后需决定如何可视化（如取 3 个假设的均值，或只画第 0 个假设）。

**注意**：上述修改只把"聚合"动作从网络中拿掉；如果还希望 K 可配置（例如 K=5），则需要进一步把 `model/mhformer.py` 和 `model/module/trans_hypothesis.py` 里所有写死的 3 套模块改成基于 `args.n_hypotheses` 的循环或 `nn.ModuleList`，并在 `common/opt.py` 新增 `--n_hypotheses` 参数。

---

## 风险与未知

- **预训练权重兼容性**：如果加载官方 351-frame 权重后把 `regression` 替换为 3 个独立头，旧权重的 `regression.0.weight/bias` 形状为 `(51, 1536)`，无法直接拆成 3 个 `(51, 512)` 的 Conv1d；需要 either 重新训练 or 设计权重拆分/插值策略。
- **训练监督信号未确定**：保留 K 个假设后，损失函数应如何设计？当前只用 MPJPE 监督单一输出；多假设情况下可选择：取最近假设监督（min loss）、对所有假设求平均、或引入假设分布损失。本卡未涉及训练策略。
- **CHI/SHR 中的 `dim*3` 耦合**：SHR 与 CHI 块把三流拼接后过 MLP 再切回三份，这一设计本身假设 K=3。若 K 变化，需同步修改 `norm2` 与 `mlp` 的输入维度以及切分逻辑。
- **`embedding_3` 的输入通道不一致**：`model/mhformer.py:41` 使用 `2*args.out_joints`，而 `embedding_1/2` 使用 `2*args.n_joints`。默认 `n_joints == out_joints == 17`，但若改动关节数需检查此处。
- **Demo 中的 hardcoded 关节对**：`demo/vis.py:179-180` 与 `main.py:71-72` 的左右关节索引仅适用于 H36M 17 关节格式，无法直接迁移到其他骨架。
- **框架/版本风险**：README 指定 PyTorch 1.7.1，而新 Apple Silicon / CUDA 环境可能需要更高版本，未验证兼容性。
