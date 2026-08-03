# MASt3R 工程侦察卡片

> 目标仓库：https://github.com/naver/mast3r （浅克隆 `--depth 1`，commit 见 `mast3r/.git`）
> 研究关注点：为免训练 6D 位姿精修管线提供代码级落地路径——(1) MASt3R 特征提取与稠密匹配接口（稠密对应点+置信度）；(2) 3DGS 可微渲染入口与位姿求导；(3) 3DGS 物体级最小训练配置。
> 重要前置结论：**本仓库不含任何 3DGS / 高斯泼溅 / 可微渲染代码**（全仓 `grep -i "gaussian|splat|3DGS|gsplat|render"` 零命中）。MASt3R 只负责「稠密匹配 + 点云/位姿初值」，关注点 (2)(3) 必须由外部 3DGS 代码库（如 INRIA gaussian-splatting / gsplat）承接。详见「风险与未知」。

## 架构总览（模块划分，一段话+目录树摘要）

MASt3R 在 DUSt3R（CroCo 立体骨干 + DPT 回归头）之上加了一个「局部描述子 + 置信度」头，把逐像素 3D 回归和稠密局部特征统一输出，再用快速互近邻（reciprocal NN）做稠密 2D-2D 匹配，最后通过稀疏全局对齐（sparse global alignment）把多视图匹配三角化成带相机位姿/焦距的点云。模型类 `AsymmetricMASt3R` 继承自 dust3r 的 `AsymmetricCroCo3DStereo`（dust3r 为 git 子模块，本浅克隆未拉取，目录为空）。匹配核心在 `fast_nn.py`，多视图优化在 `cloud_opt/sparse_ga.py`。仓库本身**没有渲染/3DGS 模块**。

```
mast3r/
├── model.py                 # AsymmetricMASt3R 模型类、load_model、forward
├── catmlp_dpt_head.py       # 描述子+pts3d+conf 预测头与 postprocess
├── fast_nn.py               # 互近邻匹配：fast_reciprocal_NNs / extract_correspondences_nonsym
├── losses.py                # 训练损失（InfoNCE/APLoss/Regr3D）
├── image_pairs.py           # 组对策略
├── cloud_opt/
│   ├── sparse_ga.py         # 稀疏全局对齐：forward_mast3r + extract_correspondences + 优化
│   ├── triangulation.py     # 三角化
│   └── tsdf_optimizer.py    # TSDF 后处理
├── colmap/ , retrieval/ , datasets/ , utils/
├── demo.py / demo_glomap.py # gradio 演示入口
train.py                     # MASt3R 训练入口（委托给 dust3r.training）
dust3r/                      # 子模块（cvpr 分支）——本浅克隆为空，需 --recursive
```

## 关键事实（与关注点直接相关的代码事实，每条带 文件路径:行号 和原代码片段）

### 关注点(1)：特征提取与稠密匹配接口

**F1 — 模型前向输出字典（pts3d / conf / desc / desc_conf / pts3d_in_other_view）**
`mast3r/model.py:199-213`
```python
def forward(self, view1, view2):
    # encode the two images --> B,S,D
    with torch.no_grad():
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self.encode_symmetrized(view1, view2)
    ...
    dec1, dec2 = self.mast3r._decoder(feat1, pos1, feat2, pos2)
    with torch.cuda.amp.autocast(enabled=False):
        res1 = self.mast3r._downstream_head(1, [tok.float() for tok in dec1], shape1)
        res2 = self.mast3r._downstream_head(2, [tok.float() for tok in dec2], shape2)
    res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
    return res1, res2
```
> 说明：`res` 字典即逐像素稠密输出。`pts3d_in_other_view` 是 view2 在 view1 坐标系的稠密 3D 点——对 render-and-compare 精修而言，这是拿到「跨视图稠密对应 + 3D 初值」的直接来源。

**F2 — 头 postprocess 产出 desc/conf（描述子 L2 归一化）**
`mast3r/catmlp_dpt_head.py:27-41`
```python
def postprocess(out, depth_mode, conf_mode, desc_dim=None, desc_mode='norm', two_confs=False, desc_conf_mode=None):
    ...
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,D
    res = dict(pts3d=reg_dense_depth(fmap[..., 0:3], mode=depth_mode))
    if conf_mode is not None:
        res['conf'] = reg_dense_conf(fmap[..., 3], mode=conf_mode)
    if desc_dim is not None:
        start = 3 + int(conf_mode is not None)
        res['desc'] = reg_desc(fmap[..., start:start + desc_dim], mode=desc_mode)
        if two_confs:
            res['desc_conf'] = reg_dense_conf(fmap[..., start + desc_dim], mode=desc_conf_mode)
        else:
            res['desc_conf'] = res['conf'].clone()
    return res
```
`mast3r/catmlp_dpt_head.py:19-24`（描述子归一化）
```python
def reg_desc(desc, mode):
    if 'norm' in mode:
        desc = desc / desc.norm(dim=-1, keepdim=True)
```
> 说明：官方权重 `output_mode='pts3d+desc24'` → `desc_dim=24` 维描述子；`two_confs=True` 时 3D 回归与描述子各有独立置信度（`conf` 与 `desc_conf`）。

**F3 — 稠密匹配核心：互近邻（返回 xy 对应）**
`mast3r/fast_nn.py:109-110`（签名）
```python
def fast_reciprocal_NNs(pts1, pts2, subsample_or_initxy1=8, ret_xy=True, pixel_tol=0, ret_basin=False,
                        device='cuda', **matcher_kw):
```
`mast3r/fast_nn.py:152-160`（迭代互查直到收敛）
```python
    while notyet.any():
        _, xy2[notyet] = to_numpy(tree2.query(pts1[xy1[notyet]], **matcher_kw))
        ...
        _, xy1[notyet] = to_numpy(tree1.query(pts2[xy2[notyet]], **matcher_kw))
```

**F4 — 带置信度的对应提取（直接产出 (xy1, xy2, conf)）★最贴合关注点**
`mast3r/fast_nn.py:191-223`
```python
def extract_correspondences_nonsym(A, B, confA, confB, subsample=8, device=None, ptmap_key='pred_desc', pixel_tol=0):
    if '3d' in ptmap_key:
        opt = dict(device='cpu', workers=32)
    else:
        opt = dict(device=device, dist='dot', block_size=2**13)
    ...
    c1 = confA.ravel()[idx1]
    c2 = confB.ravel()[idx2]
    xy1, xy2, idx = merge_corres(idx1, idx2, (HA, WA), (HB, WB), ret_xy=True, ret_index=True)
    conf = np.minimum(c1[idx], c2[idx])
    corres = (xy1.copy(), xy2.copy(), conf)
    return todevice(corres, device)
```
> 说明：这就是「两图间稠密对应点 + 置信度」的标准出口，`conf` 取两端置信度逐点最小值。输入 A/B 为 `pred['desc']`，confA/confB 为 `pred['desc_conf']`（或 `conf`）。

**F5 — 官方最小匹配样例（端到端调用链）**
`README.md:246-282`
```python
model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
images = load_images([...], size=512)
output = inference([tuple(images)], model, device, batch_size=1, verbose=False)
view1, pred1 = output['view1'], output['pred1']
view2, pred2 = output['view2'], output['pred2']
desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()
matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                               device=device, dist='dot', block_size=2**13)
# 边界 3px 过滤
valid_matches_im0 = (matches_im0[:,0] >= 3) & (matches_im0[:,0] < int(W0)-3) & ...
```
> 说明：`inference`/`load_images` 来自 dust3r 子模块（`dust3r.inference`、`dust3r.utils.image`）。坐标为 `true_shape` 尺度下的 (x,y)。

**F6 — 多视图管线里的特征提取+匹配（含置信度缓存与打分）**
`mast3r/cloud_opt/sparse_ga.py:583-597`
```python
res = symmetric_inference(model, img1, img2, device=device)
X11, X21, X22, X12 = [r['pts3d'][0] for r in res]
C11, C21, C22, C12 = [r['conf'][0] for r in res]
descs = [r['desc'][0] for r in res]
qonfs = [r[desc_conf][0] for r in res]
...
corres = extract_correspondences(descs, qonfs, device=device, subsample=subsample)
conf_score = (C11.mean() * C12.mean() * C21.mean() * C22.mean()).sqrt().sqrt()
matching_score = (float(conf_score), float(corres[2].sum()), len(corres[2]))
```
`mast3r/cloud_opt/sparse_ga.py:633-642`（对称版 extract_correspondences，dot+block_size）
```python
def extract_correspondences(feats, qonfs, subsample=8, device=None, ptmap_key='pred_desc'):
    ...
    opt = dict(device=device, dist='dot', block_size=2**13)
```

### 关注点(2)：3DGS 可微渲染入口与位姿求导

**F7 — 仓库内不存在 3DGS / 可微渲染代码（查证结论）**
全仓搜索 `gaussian|splat|3DGS|gsplat|diff-gaussian|render`（大小写不敏感）→ **零命中**。
```
$ grep -ri "gaussian|splat|3DGS|gsplat|render" mast3r/   # No files found
```
> 说明：MASt3R 的「渲染」等价物是点云投影/三角化（`cloud_opt/triangulation.py`、`tsdf_optimizer.py`）与 dust3r 的 2D 重投影残差优化，**不是**高斯泼溅可微渲染，也没有对 SE(3) 位姿的李代数求导接口。render-and-compare 位姿精修所需的可微渲染器必须外接（如 gsplat / INRIA diff-gaussian-rasterization），MASt3R 仅提供匹配/位姿初值。

**F8 — 仓库内与「位姿」最接近的可微量：相机位姿/焦距在稀疏对齐里被优化**
`mast3r/demo.py:132-133`
```python
focals = scene.get_focals().cpu()
cams2world = scene.get_im_poses().cpu()
```
> 说明：`sparse_ga` 的 `PointCloud` 把 `im_poses`（cam2world）、`focals` 作为可优化参数（定义在 dust3r 子模块 `cloud_opt/`，本克隆不可见）。这是「位姿参与优化」的现成范式，但目标是重投影/3D 匹配残差，非渲染光度残差。

### 关注点(3)：3DGS 物体级最小训练配置

**F9 — 仓库内不存在 3DGS 训练；train.py 是 MASt3R（匹配网络）训练**
`train.py:36-44`
```python
def get_args_parser():
    parser = dust3r_get_args_parser()
    parser.prog = 'MASt3R training'
    parser.set_defaults(model="AsymmetricMASt3R(patch_embed_cls='ManyAR_PatchEmbed')")
    return parser
if __name__ == '__main__':
    ...
    train(args)   # 委托 dust3r.training.train
```
> 说明：训练的是描述子/回归网络（InfoNCE/APLoss + Regr3D，见 `README.md:442-464`），需要 ARKitScenes/MegaDepth/CO3D 等大型数据集，**与「训练一个物体级 3DGS 场景」无关**。物体级 3DGS 最小配置（迭代数/输入要求）需到外部 3DGS 代码库查证。

## 硬编码参数与配置点（常量值、在哪、怎么改成可配置）

| 参数 | 值 | 位置 | 改成可配置的方式 |
|---|---|---|---|
| 匹配网格采样步长 `subsample` | `8`（像素） | `fast_nn.py:109`、`sparse_ga.py:119/633` | 已是函数参数，调用处传入即可；越小对应越密越慢 |
| 匹配距离/分块 `dist='dot'`, `block_size=2**13` | dot / 8192 | `fast_nn.py:195`、`sparse_ga.py:642` | 通过 `matcher_kw` / 函数默认值；dot 依赖 desc 已 L2 归一化（F2） |
| 描述子维度 `desc_dim` | `24`（`pts3d+desc24`） | 权重字符串 `README.md:442`；解析于 `catmlp_dpt_head.py:212` `local_feat_dim=int(output_mode[10:])` | 由 checkpoint 的 `output_mode` 决定，换权重才变；运行期不可调 |
| 双置信度 `two_confs` | `True`（metric 权重） | `model.py:40`、`README.md:442` | 构造参数；False 时 `desc_conf=conf.clone()`（`catmlp_dpt_head.py:40`） |
| 匹配置信度阈值 `matching_conf_thr` | `5.`（sparse_ga 默认）/ `0.`（demo 滑杆默认） | `sparse_ga.py:206`、`demo.py:305`；判定 `sparse_ga.py:343` `lambda x: x.max() > matching_conf_thr` | 已是 `sparse_global_alignment(..., matching_conf_thr=)` 入参 |
| 边界忽略边距 | `3` px | `README.md:274-279`（样例硬编码） | 样例代码常量，自行参数化 |
| 输入分辨率 `img_size` | `512`（多档 512xN） | `README.md:134/259`、权重串 `img_size=(512,512)` | `load_images(..., size=)`；须为 patch_size 倍数（`model.py:54`） |
| conf/desc_conf 激活模式 | `('exp',1,inf)` / `('exp',0,inf)` | `README.md:442/462` | 权重串 `conf_mode`/`desc_conf_mode`，`reg_dense_conf` 在 dust3r |
| 粗/精对齐迭代 `niter1/niter2`, `lr1/lr2` | 300/300, 0.07/0.01（demo 默认） | `demo.py:295-300` | gradio 滑杆 / `sparse_global_alignment` 入参 |

## 环境与复现（依赖、权重下载、最小运行命令）

依据 `README.md:80-143, 193`：

- **克隆（必须带子模块）**：`git clone --recursive https://github.com/naver/mast3r`；已克隆则 `git submodule update --init --recursive`。
  > 本次为 `--depth 1` 浅克隆且**未** `--recursive`，`dust3r/` 目录为空 → 任何 `import dust3r` 都会失败，复现前必须先补子模块。
- **环境**：`conda create -n mast3r python=3.11 cmake=3.14.0`；装 PyTorch（CUDA 12.1 示例）；`pip install -r requirements.txt`（仅 `scikit-learn`）+ `pip install -r dust3r/requirements.txt`（+可选 `dust3r/requirements_optional.txt`）。
- **可选加速**：ASMK（`git clone jenicek/asmk` 后 cythonize 安装）；RoPE cuda kernel（`dust3r/croco/models/curope/` `python setup.py build_ext --inplace`）。
- **权重**：HF 自动下载 `naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`，或
  `wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/`。许可 CC BY-NC-SA 4.0（非商用），另需同意训练数据集许可（见 `CHECKPOINTS_NOTICE`）。
- **最小运行**：
  - 匹配：`README.md:246-314` 的 python 样例（`from_pretrained` → `inference` → `fast_reciprocal_NNs`）。
  - 演示：`python3 demo.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`（`README.md:193`，gradio）。

## 改造接口点（针对关注点，最小侵入的修改位置和方式）

1. **拿稠密对应+置信度喂给精修（关注点1，零侵入）**：直接调用 `mast3r/fast_nn.py:191 extract_correspondences_nonsym(desc1, desc2, conf1, conf2, subsample=..., ptmap_key='pred_desc')`，返回 `(xy1, xy2, conf)`。`desc` 来自 `pred['desc']`、`conf` 取 `pred['desc_conf']`（双置信度权重）或 `pred['conf']`。无需改仓库代码。
2. **要稠密 3D 初值/跨视图点**：用 `model.forward` 输出的 `res2['pts3d_in_other_view']`（`model.py:212`）与 `res['pts3d']`、`res['conf']`，作为 render-and-compare 的 3D-2D 监督或初值。
3. **调匹配密度/阈值**：在调用层传 `subsample`（更小→更密）、`matching_conf_thr`（`sparse_ga.py:206`）、边界边距（样例 3px）即可，均为既有入参，无需改源码。
4. **位姿初值接入外部 3DGS（关注点2/3 的桥接点）**：用 `sparse_global_alignment` 得到 `scene.get_im_poses()`（cam2world）与 `scene.get_focals()`（`demo.py:132-133`），转换坐标系/尺度后作为外部 3DGS 的相机初值；可微渲染与对位姿求导在外部渲染器内完成，MASt3R 侧不需改动。
5. **若要仓库内做可微渲染精修（需新增代码）**：仓库无渲染器，最小侵入做法是新建独立模块，复用 F1/F4 的对应点与 F8 的位姿初值，外接 gsplat 渲染器；不建议改动 `sparse_ga.py` 的既有重投影优化路径。

## 风险与未知（没查证到的部分，明确列出）

1. **关注点(2)(3) 在本仓库无对应代码**：3DGS 可微渲染入口、渲染函数签名、R/t 前向与梯度回传、物体级 3DGS 最小训练配置（迭代数/输入要求）——本仓**完全不存在**（F7/F9 已证）。这些必须到外部 3DGS 代码库另行侦察；本卡只能给出 MASt3R 侧的桥接初值（F8）。
2. **dust3r 子模块未拉取**：本次浅克隆 `dust3r/` 为空。`inference`、`load_images`、`AsymmetricCroCo3DStereo`、`reg_dense_depth/conf`、`PointCloud` 可微位姿优化等关键实现都在子模块内，**本次未能逐行查证**其内部签名与位姿参数化（李代数/四元数）。F8 仅从调用侧推断。
3. **`symmetric_inference` 细节未展开**：`sparse_ga.py:583` 调用，定义在子模块/本文件他处，未逐行核对（ symmetrize 对 desc/conf 的影响）。
4. **坐标系/尺度约定未完全核对**：`pts3d`、`pts3d_in_other_view`、`cams2world` 之间的坐标轴与尺度（是否 metric）未逐行验证；metric 权重名含 `metric`，但具体尺度需结合 dust3r 确认。接外部 3DGS 前必须核对，否则位姿初值会错位。
5. **DUNE+MASt3R 分支**（`model.py:70-213`）为可选编码器路径，本次仅确认其存在，未深入其匹配/渲染差异。
6. **训练超参**（`README.md:442-464`）是 MASt3R 网络训练，非 3DGS；其中 `temperature=0.05`、`blocksize=8192`、损失权重等为网络训练而设，与物体级 3DGS 训练无关。
