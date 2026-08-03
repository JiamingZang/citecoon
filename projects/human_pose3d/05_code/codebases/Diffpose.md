# DiffPose 工程侦察卡

> 目标仓库：https://github.com/gongjia0208/Diffpose  
> 侦察日期：2026-07-21  
> 本卡仅基于仓库当前 HEAD（`9a15bf8`）代码事实，未修改仓库代码。

---

## 架构总览

DiffPose 将单帧/单张 2D 姿态估计建模为“条件逆扩散”过程：先用一个 **Context Encoder（`GCNpose`）** 从 2D 关节点得到粗略 3D 姿态，再与 2D 坐标拼接成 `uvxyz`（5 维）作为扩散初始分布 `H_K`；随后由 **扩散模型（`GCNdiff`）** 在 DDIM 采样器下逐步去噪，最终对 `N` 个样本取均值得到单帧 3D 姿态。

模块划分：

- `main_diffpose_frame.py`：入口脚本，解析命令行参数与 YAML 配置，实例化 `Diffpose` runner。
- `runners/diffpose_frame.py`：核心训练/测试流程，包含 Human3.6M 数据准备、模型创建、`train()`、`test_hyber()` 评测循环。
- `models/gcnpose.py`：Context Encoder `GCNpose`，2D → 3D 粗估计。
- `models/gcndiff.py`：去噪网络 `GCNdiff`，预测噪声，输入/输出均为 `uvxyz`（5 维）。
- `common/generators.py`：`PoseGenerator_gmm` 数据集，负责读取预计算 GMM 2D 数据并构造 `uvxyz` 与逐关节噪声尺度。
- `common/utils_diff.py`：扩散调度 `get_beta_schedule`、`compute_alpha` 与 **DDIM 采样器 `generalized_steps`**。
- `common/utils.py` / `common/loss.py`：MPJPE、PA-MPJPE、按 action 聚合误差、打印结果。
- `common/data_utils.py`、`common/h36m_dataset.py`：Human3.6M 数据加载与预处理。
- `configs/*.yml`：数据路径、模型结构、扩散参数、测试时采样步数等配置。

目录树摘要（深度 2）：

```
Diffpose/
├── checkpoints/        # 空占位，需放入预训练权重
├── common/
│   ├── camera.py
│   ├── data_utils.py
│   ├── generators.py
│   ├── h36m_dataset.py
│   ├── loss.py
│   ├── utils.py
│   └── utils_diff.py
├── configs/
│   ├── human36m_diffpose_uvxyz_cpn.yml
│   └── human36m_diffpose_uvxyz_gt.yml
├── data/               # 空占位，需放入 H36M/GMM npz 数据
├── models/
│   ├── ChebConv.py
│   ├── GraFormer.py
│   ├── ema.py
│   ├── gcndiff.py
│   └── gcnpose.py
├── runners/
│   ├── __init__.py
│   └── diffpose_frame.py
├── main_diffpose_frame.py
├── README.md
├── environment.yml
└── runner.sh
```

---

## 关键事实

### 1. 逆扩散采样 N 个样本：已实现，但默认只输出均值

采样函数本身在 `common/utils_diff.py:46` 的 `generalized_steps`：

```python
# common/utils_diff.py:46-67
def generalized_steps(x, src_mask, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).cuda()
            next_t = (torch.ones(n) * j).cuda()
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1]
            et = model(xt, src_mask, t.float(), 0)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t)
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next)

    return xs, x0_preds
```

在测试入口 `runners/diffpose_frame.py:296-299` 被调用：

```python
# runners/diffpose_frame.py:296-299
            output_uvxyz = generalized_steps(x, src_mask, seq, self.model_diff, self.betas, eta=self.args.eta)
            output_uvxyz = output_uvxyz[0][-1]            # 取最终 x_t
            output_uvxyz = torch.mean(output_uvxyz.reshape(test_times,-1,17,5),0)
            output_xyz = output_uvxyz[:,:,2:]
```

- `output_uvxyz[0][-1]` 的 shape 为 **`[B * test_times, 17, 5]`**（batch×重复次数，17 个关节，5 维 `uvxyz`）。
- 它**不是**逐样本 3D 姿态张量；3D 部分仅在 `[:, :, 2:]` 后得到，shape 为 **`[B * test_times, 17, 3]`**，但代码立刻在第 298 行对 `test_times` 取了均值，因此下游只拿到平均后的单点估计。
- 也就是说：采样器本身返回了 `N` 个样本，但当前评测流程把它们坍缩成 1 个均值。

### 2. GMM（5 核）不确定性初始化 `H_K` 的代码位置

仓库中没有显式命名为 `H_K` 的变量，论文中 `H_K` 的“逐关节不确定性初始化”对应代码里的 **GMM 核方差 → 噪声尺度** 这一路径。

数据来自预计算的 GMM 文件（`data_2d_h36m_*_gmm.npz`），在 `common/generators.py:10-48` 的 `PoseGenerator_gmm` 中解析：

```python
# common/generators.py:14-40
        self._poses_2d_gmm = np.concatenate(poses_2d_gmm)
        ...
        self._kernel_n = self._poses_2d_gmm.shape[2]   # GMM 核数，由 npz 形状决定
        ...
    def __getitem__(self, index):
        out_pose_3d = self._poses_3d[index]
        out_pose_2d_gmm = self._poses_2d_gmm[index]
        ...
        # randomly select a kernel from gmm
        out_pose_2d_kernel = np.zeros([out_pose_2d_gmm.shape[0],out_pose_2d_gmm.shape[2]])
        for i in range(out_pose_2d_gmm.shape[0]):
            out_pose_2d_kernel[i] = out_pose_2d_gmm[i,np.random.choice(self._kernel_n, 1, p=out_pose_2d_gmm[i,:,0]).item()]

        # generate uvxyz and uvxyz noise scale
        kernel_mean = out_pose_2d_kernel[:,1:3]
        kernel_variance = out_pose_2d_kernel[:,3:]

        out_pose_uvxyz = np.concatenate((kernel_mean,out_pose_3d),axis=1)
        out_pose_noise_scale = np.concatenate((kernel_variance,np.ones(out_pose_3d.shape)),axis=1)
```

- `self._kernel_n` 从 npz 第 2 维读取，对应论文/数据里的 **5 个核**（需通过实际数据文件验证）。
- `out_pose_noise_scale` 就是进入扩散过程的**逐关节不确定性**：`u/v` 通道使用 GMM 核方差，`x/y/z` 通道初始化为 1。

该不确定性在训练与测试时被用作噪声缩放系数：

- 训练：`runners/diffpose_frame.py:173`
  ```python
  e = e*(targets_noise_scale)
  ```
- 测试初始化：`runners/diffpose_frame.py:284,292`
  ```python
  input_noise_scale = input_noise_scale.repeat(test_times,1,1)
  ...
  e = e*input_noise_scale
  ```

### 3. 最终取 N 个样本均值的语句

在 `runners/diffpose_frame.py:298`：

```python
# runners/diffpose_frame.py:298
            output_uvxyz = torch.mean(output_uvxyz.reshape(test_times,-1,17,5),0)
```

- 它将 `[B*test_times, 17, 5]` reshape 为 `[test_times, B, 17, 5]`，并在第 0 维（`test_times`）取均值。
- 若 `test_times=5`，则对应论文“取 5 个样本均值”的代码实现。
- 注意：命令行默认 `test_times=5`（`main_diffpose_frame.py:65`），但配置文件 `configs/*.yml` 里把 `testing.test_times` 设为 **1**（GT 与 CPN 配置皆是），因此实际评测时默认并不做 5 次采样平均。

### 4. DDIM 加速到 5 步的采样器实现位置

DDIM 采样器就是 `common/utils_diff.py:46` 的 `generalized_steps`；采样步数由外部传入的 `seq` 长度决定。

`seq` 在 `runners/diffpose_frame.py:256-263` 构造：

```python
# runners/diffpose_frame.py:256-263
        if self.args.skip_type == "uniform":
            skip = test_num_diffusion_timesteps // test_timesteps
            seq = range(0, test_num_diffusion_timesteps, skip)
        elif self.args.skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(test_num_diffusion_timesteps * 0.8), test_timesteps)** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError
```

- 要得到 **5 步** DDIM，只需令 `len(seq) == 5`。
- 例如使用 uniform skip 时，设置 `--test_timesteps 5 --test_num_diffusion_timesteps 25`（或 500），则 `skip = 5`（或 100），`seq` 为 5 个时间戳。
- 默认配置中 GT 配置为 `test_timesteps=2, test_num_diffusion_timesteps=12`，CPN 配置为 `test_timesteps=2, test_num_diffusion_timesteps=24`，因此默认只跑 2 步。

### 5. Human3.6M 评测接口与预训练 Context Encoder 加载方式

**评测接口**：`runners/diffpose_frame.py:226` 的 `test_hyber` 方法。

```python
# runners/diffpose_frame.py:226-319 （节选关键行）
    def test_hyber(self, is_train=False):
        ...
        if config.data.dataset == "human36m":
            poses_valid, poses_valid_2d, actions_valid, camerapara_valid = \
                fetch_me(self.subjects_test, self.dataset, self.keypoints_test, self.action_filter, stride)
            data_loader = valid_loader = data.DataLoader(
                PoseGenerator_gmm(poses_valid, poses_valid_2d, actions_valid, camerapara_valid),
                batch_size=config.training.batch_size, shuffle=False,
                num_workers=config.training.num_workers, pin_memory=True)
        ...
        for i, (_, input_noise_scale, input_2d, targets_3d, input_action, camera_para) in enumerate(data_loader):
            ...
            inputs_xyz = self.model_pose(input_2d, src_mask)            # Context Encoder 前向
            inputs_xyz[:, :, :] -= inputs_xyz[:, :1, :]
            input_uvxyz = torch.cat([input_2d,inputs_xyz],dim=2)
            ...
            output_uvxyz = generalized_steps(x, src_mask, seq, self.model_diff, self.betas, eta=self.args.eta)
            output_uvxyz = output_uvxyz[0][-1]
            output_uvxyz = torch.mean(output_uvxyz.reshape(test_times,-1,17,5),0)
            output_xyz = output_uvxyz[:,:,2:]
            output_xyz[:, :, :] -= output_xyz[:, :1, :]
            targets_3d[:, :, :] -= targets_3d[:, :1, :]
            epoch_loss_3d_pos.update(mpjpe(output_xyz, targets_3d).item() * 1000.0, targets_3d.size(0))
            epoch_loss_3d_pos_procrustes.update(p_mpjpe(output_xyz.cpu().numpy(), targets_3d.cpu().numpy()).item() * 1000.0, targets_3d.size(0))
            ...
            action_error_sum = test_calculation(output_xyz, targets_3d, input_action, action_error_sum, None, None)
        ...
        p1, p2 = print_error(None, action_error_sum, is_train)
        return p1, p2
```

- 使用 `mpjpe`（Protocol #1）和 `p_mpjpe`（Protocol #2，即 PA-MPJPE）。
- 按 action 聚合由 `common/utils.py:96` 的 `test_calculation` 完成，最终打印由 `common/utils.py:241` 的 `print_error` 完成。

**预训练 Context Encoder 加载**：`runners/diffpose_frame.py:92-112` 的 `create_pose_model`。

```python
# runners/diffpose_frame.py:92-112
def create_pose_model(self, model_path = None):
    ...
    config.model.coords_dim = [2,3]      # Context Encoder 仅输入 2D、输出 3D
    ...
    self.model_pose = GCNpose(adj.cuda(), config).cuda()
    self.model_pose = torch.nn.DataParallel(self.model_pose)

    # load pretrained model
    if model_path:
        logging.info('initialize model by:' + model_path)
        states = torch.load(model_path)
        self.model_pose.load_state_dict(states[0])
    else:
        logging.info('initialize model randomly')
```

- `model_pose` 即论文中的 Context Encoder `φ_ST`（本仓库 frame 版本仅使用空间分支）。
- 权重文件路径通过 `--model_pose_path` 传入（`main_diffpose_frame.py:51`），评测时通常使用 `checkpoints/gcn_xyz_cpn.pth` 或 `checkpoints/gcn_xyz_gt.pth`。

---

## 硬编码参数与配置点

| 参数/常量 | 位置 | 说明与修改方式 |
|---|---|---|
| 17 关节、骨架边列表 | `runners/diffpose_frame.py:78-82, 97-101` | 硬编码 17 点人体骨架边，用于构建图邻接矩阵。 |
| `coords_dim = [5,5]` | `configs/*.yml:12` | 扩散模型输入/输出维度（`uvxyz`）。Context Encoder 在代码中被覆盖为 `[2,3]`。 |
| GMM 核数 5 | `common/generators.py:17` | 由 npz 形状 `poses_2d_gmm.shape[2]` 决定，未在代码写死。 |
| `num_diffusion_timesteps` | `configs/*.yml:26` | 训练扩散总步数，默认 51。 |
| `beta_schedule`, `beta_start`, `beta_end` | `configs/*.yml:23-25` | 线性调度，训练用。 |
| `test_times` | `main_diffpose_frame.py:65`（默认 5）<br>`configs/*.yml:34`（覆盖为 1） | 采样重复次数。命令行可覆盖。 |
| `test_timesteps` | `main_diffpose_frame.py:67`（默认 50）<br>`configs/*.yml:35`（GT=2, CPN=2） | 测试时实际 DDIM 步数。 |
| `test_num_diffusion_timesteps` | `main_diffpose_frame.py:69`（默认 500）<br>`configs/*.yml:36`（GT=12, CPN=24） | 与 `test_timesteps` 共同决定 uniform skip。 |
| `eta` | `main_diffpose_frame.py:38`（默认 0.0） | DDIM 随机项系数；`eta=0` 为确定性 DDIM。 |
| `skip_type` | `main_diffpose_frame.py:36`（默认 `"uniform"`） | uniform / quad 两种跳步策略。 |

---

## 环境与复现

依据 `README.md` 与 `environment.yml`：

- Python 3.8.2，PyTorch 1.7.1，CUDA 11.0。
- 创建环境：
  ```bash
  conda env create -f environment.yml
  ```
- 下载数据：
  - H36M 3D 数据：`data_3d_h36m.npz` 放入 `./data`。
  - GMM 格式 2D 数据：`data_2d_h36m_gt_gmm.npz`、`data_2d_h36m_cpn_ft_h36m_dbb_gmm.npz` 放入 `./data`。
- 下载预训练权重：放入 `./checkpoints`。
- 评测命令（以 CPN 为例，来自 README）：
  ```bash
  CUDA_VISIBLE_DEVICES=0 python main_diffpose_frame.py \
    --config human36m_diffpose_uvxyz_cpn.yml \
    --batch_size 1024 \
    --model_pose_path checkpoints/gcn_xyz_cpn.pth \
    --model_diff_path checkpoints/diffpose_uvxyz_cpn.pth \
    --doc t_human36m_diffpose_uvxyz_cpn --exp exp --ni
  ```

> 注意：当前克隆下来的 `checkpoints/` 与 `data/` 仅含空 `readme.txt`，无法直接复现，必须下载外部资源。

---

## 改造接口点

针对“用 GMM 刻画逐关节不确定性、不取均值”这一 idea，最小侵入的改动位置如下：

1. **把均值替换为分布输出**  
   位置：`runners/diffpose_frame.py:296-299`
   ```python
   output_uvxyz = generalized_steps(...)[0][-1]
   # 原代码：output_uvxyz = torch.mean(output_uvxyz.reshape(test_times,-1,17,5),0)
   # 改为保留 N 个样本：
   output_uvxyz_dist = output_uvxyz.reshape(test_times, -1, 17, 5)   # [N, B, 17, 5]
   output_xyz_dist = output_uvxyz_dist[:, :, :, 2:]                  # [N, B, 17, 3]
   ```
   之后可把 `output_xyz_dist` 用于：
   - 直接作为 N 个候选 3D 姿态分布；
   - 在关节维度拟合 GMM/高斯，输出均值+方差；
   - 修改评测函数以支持分布指标（NLL/ECE）而非单点 MPJPE。

2. **让测试时重复采样生效**  
   当前配置文件把 `testing.test_times` 设为 1，若要真正得到 N=5 样本，需在 `configs/*.yml` 中把 `test_times` 改为 5，或在命令行传入 `--test_times 5`。

3. **把单核随机采样扩展为全 GMM 输入（可选）**  
   位置：`common/generators.py:30-33`  
   当前 `__getitem__` 只按权重随机选 1 个核。若想用全部 5 个核初始化多模态分布，可改为返回所有核的 `kernel_mean`、`kernel_variance` 与权重，再在 `test_hyber` 中扩展为多个初始样本输入扩散模型。

4. **不确定性尺度可配置化**  
   `out_pose_noise_scale` 中 xyz 通道 currently 固定为 `np.ones(...)`。若想让 3D 部分的不确定性也来自 Context Encoder 或训练统计，可修改 `common/generators.py:40` 或新增配置项，把 `np.ones` 替换为可学习/可配置的逐关节方差。

---

## 风险与未知

- **GMM 数据生成过程不在仓库中**：仓库只消费 `.npz` 文件，没有生成 `data_2d_h36m_*_gmm.npz` 的脚本，因此无法确认 5 核具体如何拟合、协方差是否对角、是否包含权重等信息。
- **`H_K` 是论文符号，代码中无同名变量**：上述映射（`H_K` → `out_pose_noise_scale`/`input_noise_scale`）是根据论文描述与代码逻辑推断的。
- **预训练权重与数据缺失**：克隆后 `checkpoints/` 和 `data/` 为空，无法实际运行评测验证形状与数值。
- **`test_times` 命令行默认值与配置文件冲突**：`main_diffpose_frame.py` 默认 5，但 YAML 设为 1；若只改一处可能得到意外行为。
- **DDIM 5 步未在默认配置中验证**：需要手动调整 `test_timesteps` 与 `test_num_diffusion_timesteps`，并配合正确权重才能得到结果；仓库没有专门提供 5 步预训练模型。
- **Context Encoder 的时序能力**：论文提到 `φ_ST` 提取时空特征，但 `Diffpose` frame 版本仅处理单帧，未见到时序上下文编码模块；视频版本在另一个仓库（`Diffpose_video`）。
