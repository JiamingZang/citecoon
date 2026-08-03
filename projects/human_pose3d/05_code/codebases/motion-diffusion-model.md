# Repo 卡：motion-diffusion-model (MDM / DiP)

- 仓库：https://github.com/GuyTevet/motion-diffusion-model
- 论文：Human Motion Diffusion Model (arXiv:2209.14916)；含 DiP 超快文本到运动 (arXiv:2410.03441)
- 本地路径：`./motion-diffusion-model/`（浅克隆 `--depth 1`）
- 研究关注点：文本→运动 I/O 接口、输出关节表示（SMPL/关节位置）、采样耗时、能否条件化到已有首帧姿态

---

## 架构总览

MDM 是一个基于 Transformer 的运动扩散模型：文本经 CLIP/DistilBERT 编码后与时间步嵌入相加，作为条件注入 Transformer（encoder/decoder/gru 三种架构），在 HumanML3D 的 263 维向量表示（`hml_vec`）上预测 x_start，再经 `recover_from_ric` 反解为 22 关节 XYZ 位置，并可经 `Rotation2xyz`/SMPLify 得到 SMPL。条件化通过 classifier-free guidance 实现；首帧/前缀条件化通过 inpainting 与 prefix-completion 两条路径实现。

目录树摘要：
```
motion-diffusion-model/
├── sample/            # 推理入口：generate.py(生成) / edit.py(编辑/补全) / predict.py(Replicate API)
├── model/             # mdm.py(主模型) / rotation2xyz.py / smpl.py / BERT/ / cfg_sampler.py
├── diffusion/         # gaussian_diffusion.py(采样+inpainting) / respace.py / resample.py
├── utils/             # parser_util.py(参数) / model_util.py(建模) / sampler_util.py(CFG/自回归)
├── data_loaders/      # humanml/(HumanML3D 处理) / humanml_utils.py(关节定义) / get_data.py
├── train/  eval/  prepare/(下载脚本)  visualize/(render_mesh, motions2hik)  body_models/
├── environment.yml    # conda 环境
└── README.md / DiP.md
```

---

## 关键事实

### 1. 文本→运动的输入接口
文本通过命令行 `--text_prompt`（单条）或 `--input_text`（文件，逐行）传入：
- `utils/parser_util.py:228`
  ```python
  group.add_argument("--text_prompt", default='', type=str,
                     help="A text prompt to be generated. If empty, will take text prompts from dataset.")
  ```
- `utils/parser_util.py:220`
  ```python
  group.add_argument("--input_text", default='', type=str,
                     help="Path to a text file lists text prompts to be synthesized. ...")
  ```
在 `sample/generate.py` 中组装为 `texts` 列表并写入 `model_kwargs['y']['text']`：
- `sample/generate.py:51-58`
  ```python
  if args.text_prompt != '':
      texts = [args.text_prompt] * args.num_samples
  elif args.input_text != '':
      ...
      texts = [s.replace('\n', '') for s in texts]
  ```
- `sample/generate.py:104-111`
  ```python
  if texts is not None:
      model_kwargs['y']['text'] = texts
  ...
  collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
  ```
文本编码（CLIP）只调用一次并缓存，是 2× 提速关键：
- `sample/generate.py:130-132`
  ```python
  if 'text' in model_kwargs['y'].keys():
      # encoding once instead of each iteration saves lots of time
      model_kwargs['y']['text_embed'] = model.encode_text(model_kwargs['y']['text'])
  ```
- `model/mdm.py:163-178`（CLIP 编码，humanml/kit 截断到 20 token）
  ```python
  max_text_len = 20 if self.dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
  ...
  return self.clip_model.encode_text(texts).float().unsqueeze(0)
  ```

### 2. 运动张量形状与采样调用
运动形状为 `(batch, njoints, nfeats, n_frames)`；humanml 下 `njoints=263, nfeats=1`：
- `sample/generate.py:98`
  ```python
  motion_shape = (args.batch_size, model.njoints, model.nfeats, n_frames)
  ```
- `utils/model_util.py:41-45`
  ```python
  if args.dataset == 'humanml':
      data_rep = 'hml_vec'
      njoints = 263
      nfeats = 1
      all_goal_joint_names = ['pelvis'] + HML_EE_JOINT_NAMES
  ```
采样走 `diffusion.p_sample_loop`：
- `sample/generate.py:85`、`sample/generate.py:147-158`
  ```python
  sample_fn = diffusion.p_sample_loop
  ...
  sample = sample_fn(model, motion_shape, clip_denoised=False, model_kwargs=model_kwargs,
                     skip_timesteps=0, init_image=init_image, progress=True, ...)
  ```

### 3. 输出关节表示格式（位置 / SMPL）
模型输出是 `hml_vec`，先反归一化再用 `recover_from_ric` 解出 **22 关节 XYZ 位置**：
- `sample/generate.py:160-165`
  ```python
  if model.data_rep == 'hml_vec':
      n_joints = 22 if sample.shape[1] == 263 else 21
      sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
      sample = recover_from_ric(sample, n_joints)
      sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
  ```
- `data_loaders/humanml/scripts/motion_process.py:437-452`（root 旋转+平移恢复，输出 `[..., njoints, 3]` 位置）
  ```python
  def recover_from_ric(data, joints_num):
      r_rot_quat, r_pos = recover_root_rot_pos(data)
      positions = data[..., 4:(joints_num - 1) * 3 + 4]
      positions = positions.view(positions.shape[:-1] + (-1, 3))
      ...
      positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
      return positions
  ```
22 关节即 SMPLH body joints：
- `data_loaders/humanml_utils.py:28`
  ```python
  NUM_HML_JOINTS = len(HML_JOINT_NAMES)  # 22 SMPLH body joints
  ```
随后 `rot2xyz(jointstype='smpl')` 得到 SMPL 关节（`pose_rep='xyz'` 时直接返回位置）：
- `sample/generate.py:167-171`
  ```python
  rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
  ...
  sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                         jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                         get_rotations_back=False)
  ```
- `model/rotation2xyz.py:20-21`
  ```python
  if pose_rep == "xyz":
      return x
  ```
最终落盘 `results.npy`（含 motion/text/lengths）：
- `sample/generate.py:197-201`
  ```python
  npy_path = os.path.join(out_path, 'results.npy')
  np.save(npy_path,
          {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
           'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
  ```
若要 **SMPL 参数（thetas/root_translation）**，Replicate 接口给出明确格式：
- `sample/predict.py:88-92`
  ```python
  'The json format is: {"thetas": [...], "root_translation": [...], "joint_map": [...]}, where "thetas" '
  'is an [nframes x njoints x 3] array of joint rotations in degrees, "root_translation" is an [nframes x 3] '
  'array of (X, Y, Z) positions of the root, ...'
  ```
  由 `motions2hik(all_motions)` 生成（`sample/predict.py:144`）。完整 SMPL mesh 需另跑 SMPLify：`python -m visualize.render_mesh --input_path ...`（README.md:373-379，输出 thetas/root/vertices/faces，beta=0 中性模型）。

### 4. 采样耗时
- 默认扩散步数 1000：`utils/parser_util.py:89`
  ```python
  group.add_argument("--diffusion_steps", default=1000, type=int,
                     help="Number of diffusion steps (denoted T in the paper)")
  ```
- 官方提供 50 步 checkpoint，宣称 ~0.4 sec/sample、整体 40× 提速（README.md:13-19）：
  ```
  ## MDM is now 40X faster 🤩🤩🤩 (~0.4 sec/sample)
  (1) We released the 50 diffusion steps model ... which runs 20X faster ...
  (2) Calling CLIP just once and caching the result ... runs 2X faster ...
  ```
- 步数由 checkpoint 同目录 `args.json` 自动覆盖（无需手填）：`utils/parser_util.py:27-36`（`load_args_from_model`）。
- 始终预测 x_start、未启用 DDIM respacing：`utils/model_util.py:77-88`
  ```python
  predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
  steps = args.diffusion_steps
  ...
  timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
  ```

### 5. 条件化到已有首帧姿态（两条路径）
**(a) Inpainting（edit.py 的 in_between 模式）**——在每步去噪后把固定帧替换回真值：
- `diffusion/gaussian_diffusion.py:300-304`
  ```python
  if 'inpainting_mask' in model_kwargs['y'].keys() and 'inpainted_motion' in model_kwargs['y'].keys():
      inpainting_mask, inpainted_motion = model_kwargs['y']['inpainting_mask'], model_kwargs['y']['inpainted_motion']
      assert self.model_mean_type == ModelMeanType.START_X, 'This feature supports only X_start pred for mow!'
      model_output = (model_output * ~inpainting_mask) + (inpainted_motion * inpainting_mask)
  ```
- `sample/edit.py:78-85`（`prefix_end` 控制前缀固定范围；要固定首帧即让前缀覆盖 frame 0）
  ```python
  if args.edit_mode == 'in_between':
      model_kwargs['y']['inpainting_mask'] = torch.ones_like(input_motions, dtype=torch.bool, ...)  # True means use gt motion
      for i, length in enumerate(model_kwargs['y']['lengths'].cpu().numpy()):
          start_idx, end_idx = int(args.prefix_end * length), int(args.suffix_start * length)
          ...
          model_kwargs['y']['inpainting_mask'][i, :, :, start_idx: end_idx] = False  # do inpainting in those frames
  ```
  注意：`inpainted_motion` 必须是 `hml_vec` 表示（edit.py 直接取 dataset 输出，未先 recover），外部首帧需先编码成 263 维向量。
**(b) Prefix-completion（context_len/pred_len）**——把已知前缀拼到噪声序列前：
- `model/mdm.py:58-61`
  ```python
  self.pred_len = kargs.get('pred_len', 0)
  self.context_len = kargs.get('context_len', 0)
  self.total_len = self.pred_len + self.context_len
  self.is_prefix_comp = self.total_len > 0
  ```
- `model/mdm.py:203-206`（前向时拼接 prefix）
  ```python
  if self.is_prefix_comp:
      x = torch.cat([y['prefix'], x], dim=-1)
      y['mask'] = torch.cat([torch.ones([bs, 1, 1, self.context_len], ...), y['mask']], dim=-1)
  ```
- 参数：`utils/parser_util.py:131-132`（`--context_len`、`--pred_len`）；自回归续写见 `utils/sampler_util.py` 的 `AutoRegressiveSampler`（`sample/generate.py:86-88`）。

---

## 硬编码参数与配置点

| 参数 | 值 | 位置 | 如何改成可配置 |
|---|---|---|---|
| 扩散步数 | 1000（默认）/ 50（快速模型） | `utils/parser_util.py:89` | 已有 `--diffusion_steps`；推理时由 checkpoint 的 `args.json` 覆盖 |
| CFG 引导强度 | 2.5 | `utils/parser_util.py:207` | 已有 `--guidance_param` |
| 最大帧数 | humanml/kit=196，其它=60 | `sample/generate.py:32` | 硬编码；改此行或提为参数（受 `pos_embed_max_len` 与训练长度约束） |
| fps | kit=12.5，其它=20 | `sample/generate.py:33` | 硬编码于 generate/edit |
| 运动时长 | 默认 6.0s，最大 9.8s(humanml) | `utils/parser_util.py:217-219` | 已有 `--motion_length` |
| 关节数 | 22(humanml)/21(kit) | `sample/generate.py:29`、`generate.py:162` | 由 `sample.shape[1]==263` 推断 |
| 模型维度 | latent_dim=512, layers=8, heads=4, ff=1024 | `utils/model_util.py:63`；`parser_util.py:104-107` | latent_dim/layers 有 CLI，heads/ff 硬编码于 model_util |
| 文本编码器 | CLIP `ViT-B/32` | `utils/model_util.py:27` | 硬编码；bert 走 `--text_encoder_type bert` |
| 文本最大 token | 20（humanml/kit） | `model/mdm.py:166` | 硬编码 |
| 位置编码最大长度 | 5000 | `utils/parser_util.py:119` | 已有 `--pos_embed_max_len` |
| predict.py 模型路径/帧数 | `./save/humanml_trans_enc_512/model000200000.pt`，`fps*6` | `sample/predict.py:34,59` | Replicate 专用，硬编码 |

---

## 环境与复现

依赖（以 `environment.yml` / README 为准）：
- Python 3.7.13（`environment.yml:85`）、pytorch 1.7.1 + cuda 11.0（`environment.yml:88`）、smplx 0.1.28、spacy、trimesh、moviepy/ffmpeg。
- 安装（README.md:143-148）：
  ```shell
  conda env create -f environment.yml
  conda activate mdm
  python -m spacy download en_core_web_sm
  pip install git+https://github.com/openai/CLIP.git
  ```
- 下载依赖文件（README.md:155-159，文本到运动）：
  ```bash
  bash prepare/download_smpl_files.sh
  bash prepare/download_glove.sh
  bash prepare/download_t2m_evaluators.sh
  ```
- 数据：HumanML3D（README.md:185，Drive 链接），放入 `./dataset/HumanML3D`。
- 权重：下载后解压放入 `./save/`（README.md:243-258）。文本到运动推荐 `humanml-encoder-512`（论文最佳）或 50 步快速版 `humanml-encoder-512-50steps`、`humanml_trans_dec_512_bert-50steps`。
- 最小运行命令（README.md:316）：
  ```shell
  python -m sample.generate --model_path ./save/humanml_trans_enc_512/model000200000.pt \
      --text_prompt "the person walked forward and is picking up his toolbox."
  ```
- 渲染 SMPL mesh（README.md:373-374）：
  ```shell
  python -m visualize.render_mesh --input_path /path/to/mp4/stick/figure/file
  ```

---

## 改造接口点（针对关注点，最小侵入）

1. **程序化文本→运动 API**：复用 `sample/predict.py:83-145` 的模式——`get_dataset_loader(hml_mode='text_only')` → `create_model_and_diffusion` → `load_saved_model` → 组装 `collate_args=[{...'text': prompt}]` → `diffusion.p_sample_loop`。或直接 `from sample.generate import main` 传 `args`（`generate.py:23` 支持外部传入 args）。
2. **拿关节位置 vs SMPL**：
   - 要 22 关节 XYZ：在 `recover_from_ric` 之后（`generate.py:164`）截获 `sample`，形状 `[bs, njoints=22, 3, nframes]`。
   - 要 SMPL thetas/root：用 `visualize/motions2hik.py`（`predict.py:144`）或 `visualize.render_mesh`（SMPLify，慢、需 GPU）。
3. **条件化到已有首帧姿态**：
   - 最省事：用 `sample/edit.py --edit_mode in_between` 并调 `--prefix_end` 使前缀覆盖第 0 帧（`edit.py:82`），但需提供 `hml_vec` 形式的 `inpainted_motion`（当前从 dataset 取，外部姿态需自行编码为 263 维）。
   - 或用 prefix-completion 模型（`--context_len>0`，`mdm.py:203`），同样需要把首帧编码进 `y['prefix']`。
   - inpainting 替换点在 `diffusion/gaussian_diffusion.py:300-304`，可在此扩展为“仅固定首帧/任意关节”的 mask 逻辑。
4. **加速采样**：换用 50 步 checkpoint（步数由 `args.json` 自动加载，无需改码）；`skip_timesteps`/`init_image` 已预留（`generate.py:152-153`、`gaussian_diffusion.py:693-700`）可用于 warm-start。

---

## 风险与未知

- **首帧条件化无现成对外接口**：inpainting/prefix 都假设条件运动已是 `hml_vec`(263 维) 表示；把任意外部首帧姿态（如 SMPL theta 或关节位置）反向编码成 `hml_vec` 的工具链未在仓库中直接给出（`recover_from_ric` 的逆过程需自行实现/从 HumanML3D 取）。
- **是否存在公开的 prefix-completion / 首帧专用 checkpoint**：README 的预训练模型列表（README.md:247-294）未列出 `context_len>0` 的模型，需自训（`--context_len`/`--pred_len`）。
- **采样耗时未实测**：~0.4s/sample 为 README 宣称值（README.md:13），未在本地验证；实际取决于 GPU、步数(1000 vs 50)与 batch。
- **环境老旧**：Python 3.7 + pytorch 1.7.1 + cuda 11.0（`environment.yml`），在新 GPU/驱动上可能需调整。
- **SMPLify 渲染慢且需 GPU**：`visualize.render_mesh` 逐帧拟合 SMPL，非实时（README.md:383）。
- **`results.npy` 中 motion 的精确维度语义**（`[bs, njoints, 6, seqlen]` 注释见 `generate.py:189`，但 rot2xyz 后 njoints/feats 含义随 jointstype 变化）未逐维核验。
- DiP 仅支持 BERT 文本编码器（`generate.py:138-142`），动态文本自回归细节未深入核验。
