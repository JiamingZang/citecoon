# OAHPE: occlusion-aware hybrid routing for efficient and robust 3D human pose estimation in mixed-occlusion videos

> 2026 · id: W7168163523 · pdf: https://www.nature.com/articles/s41598-026-61932-6_reference.pdf · 来源: pdf_url
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

ARTICLE IN PRESS
Article in Press
OAHPE: occlusion-aware hybrid routing for 
efficient and robust 3D human pose estimation 
in mixed-occlusion videos
Scientific Reports
Received: 13 April 2026
Accepted: 8 July 2026
Cite this article as: Sun G., Huang Y., 
Li K. et al. OAHPE: occlusion-aware 
hybrid routing for efficient and robust 
3D human pose estimation in mixed-
occlusion videos. Sci Rep (2026). https://
doi.org/10.1038/s41598-026-61932-6
Guoying Sun, Ya Huang, Kaisen Li & Yunwei Zhang
We are providing an unedited version of this manuscript to give early access to its 
findings. Before final publication, the manuscript will undergo further editing. Please 
note there may be errors present which affect the content, and all legal disclaimers 
apply.
If this paper is publishing under a Transparent Peer Review model then Peer 
Review reports will publish with the final article.
https://doi.org/10.1038/s41598-026-61932-6
© The Author(s) 2026. Open Access This article is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International 
License, which permits any non-commercial use, sharing, distribution and reproduction in any medium or format, as long as you give appropriate credit 
to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if you modified the licensed material. You do 
not have permission under this licence to share adapted material derived from this article or parts of it. The images or other third party material in this 
article are included in the article’s Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the 
article’s Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain 
permission directly from the copyright holder. To view a copy of this licence, visit http://creativecommons.org/licenses/by-nc-nd/4.0/.

OAHPE: Occlusion-Aware Hybrid Routing for Efficient
and Robust 3D Human Pose Estimation in
Mixed-Occlusion Videos
Guoying Sun1, Ya Huang2, Kaisen Li1, and Yunwei Zhang1,3,*
1Faculty of Information Engineering and Automation, Kunming University of Science and Technology, Kunming
650500, Yunnan, China
2Department of Physical Education, Yunnan University of Finance and Economics, Kunming 650221, Yunnan, China
3Higher Educational Key Laboratory for Industrial Intelligence and Systems of Yunnan Province, Kunming 650500,
Yunnan, China
*email:zhangyunwei72@gmail.com
ABSTRACT
Attention-mechanism-based methods for video-based 3D human pose estimation (HPE) have demonstrated strong performance
in spatio-temporal modeling. However, the quadratic complexity introduced by attention leads to substantial computational
and memory overhead during long-sequence inference, thereby hindering lightweight deployment. Inspired by the long-range
modeling capability of the state space model (SSM) with linear complexity, we propose the Occlusion-Aware Hybrid Pose
Estimation (OAHPE) model to address the dual challenges of high uncertainty in occluded frames and high redundancy in
visible frames during 3D human pose estimation from video. This model employs a differentiated dual-path modeling strategy.
For occluded segments, we integrate a multi-scale dilated convolutions with linear-complexity-based Mamba module, and
employ a dual local-enhancement mechanism to capture fine-grained joint structures and long-range inter-joint dependencies
at the full frame rate. For visible segments, we design an online selection mechanism to reduce temporal redundancy, and
introduce a distillation-based temporal recovery objective to enforce dynamic consistency. Finally, predictions from the two
paths are backfilled, aligned, and fused at the original sequence resolution. Comprehensive quantitative and qualitative
evaluations on the Human3.6M and MPI-INF-3DHP benchmarks demonstrate that OAHPE achieves mean per-joint position
errors of 43.5 mm and 28.5 mm, respectively, while reducing the model size and training memory footprint to 2.6 M parameters
and 12.5 GB. These results indicate that the proposed method attains superior performance with substantially lower parameter
and memory costs, achieving a favorable trade-off between pose estimation accuracy and computational efficiency.
Introduction
Monocular 3D human pose estimation is a fundamental task in computer vision with numerous applications, such as human-
computer interaction1,2, action recognition3–5, and virtual and augmented reality6–8. Current video-based 3D human pose
estimation methods primarily follow the 2D-to-3D lifting paradigm9–11. Specifically, a 2D pose estimator is first employed to
detect 2D keypoints for each frame in a video, and a subsequent lifting model then infers the corresponding 3D pose sequence
from the detected 2D poses.
In recent years, the transformer architecture has achieved state-of-the-art performance in video 3D pose estimation due to
its ability to capture long-range temporal dependencies12–15. However, its self-attention mechanism incurs computational and
memory costs that grow quadratically with the number of video frames, making it difficult to meet the demands for lightweight
and efficient deployment. To overcome this bottleneck, Mamba employs a selective state space model (SSM) that achieves
long-sequence feature learning through linear-complexity modeling, establishing a highly promising alternative for efficient
temporal feature representation16,17. Similarly, Duan et al. improved recognition robustness under complex observation
conditions through multi-channel feature modeling and adaptive decision boundaries18.However, we have observed that in
videos containing both occluded and visible frames, the visible frames typically exhibit significant temporal redundancy(see
Figure 2(a)). If each joint in every frame is treated as a token, a large number of highly similar tokens provide very limited
incremental discriminative information, yet significantly increase redundant computations, thereby resulting in additional
computational overhead and accumulated latency19,20. In occlusion scenarios, obtaining accurate 3D estimates is inherently
challenging. Furthermore, mapping from a single 2D pose to 3D inherently suffers from ambiguity and information loss.
If pruning is applied on top of this, it can easily lead to significant deviations in the 3D estimates.21,22. It is worth noting
that Zhang et al. used an adaptive decomposition network to distinguish between different signal components and applied
ACCEPTED MANUSCRIPT
ARTICLE IN PRESS
ARTICLE IN PRESS

Figure 1. On the Human3.6M dataset, we compare recent 3D human pose estimation methods (lower is better). Here, Param
denotes the number of model parameters. The results show that the proposed OAHPE method achieves superior performance.
targeted feature extraction and constraints to each component, demonstrating that explicitly distinguishing heterogeneous
information helps improve representation capabilities under complex conditions23. Based on these observations, we propose
Occlusion-Aware Hybrid Routing for Efficient and Robust 3D Human Pose Estimation in Mixed-Occlusion Videos (OAHPE),
with the overall network architecture shown in Figure 3. Unlike existing methods that focus solely on computational overhead
or occlusion robustness, OAHPE adopts an occlusion-aware dual-path paradigm. It splits the 3D video pose estimation
into two key branches. For frames identified as occluded, considering that sparse 2D observations exacerbate 3D mapping
ambiguity and information loss19, we designed a full-frame-rate occlusion-robust spatio-temporal dual-local enhancement path
(ORST-DLEP).Specifically, we employ a Multi-Scale Dilated Convolution (MDC) module24 to aggregate local spatiotemporal
neighborhood information and a Linear Attention-based Mamba (LAMA) module24, which shares structural similarities with
SSMs, to jointly model global dependencies across frames and joints with linear computational complexity, thereby enhancing
robustness under occlusion. For frame segments deemed visible by the occlusion detector, we designed the Pruning Distillation
Recovery Enhancement Path (PDREP). Specifically, based on the H2OT20 method, we propose an “Online Pruning-Distillation
Recovery(OPDR)” strategy. First, online pruning is applied within each frame segment to selectively retain representative
frames along the temporal dimension. Subsequently, the ORST-DLEP path is used to perform dimensionality elevation on 2D
keypoints. Distillation is then employed to restore the original temporal length, thereby significantly reducing computational
overhead while maintaining output alignment. Finally, the 3D pose outputs from both paths are backfilled and realigned
according to the original video sequence, then fused and reconstructed to yield a complete 3D pose sequence strictly aligned
with the input video (see Figure 2(c)). This hybrid routing strategy reduces redundant computations for visible segments while
maintaining the stability of occluded segment estimates, thereby balancing model efficiency and estimation accuracy.
We conducted extensive experiments on two benchmarks, Human3.6M25 and MPI-INF-3DHP26, to validate the effectiveness
and efficiency of OAHPE. Experimental results demonstrate that our method outperforms current state-of-the-art approaches on
multiple standard metrics, with particularly outstanding performance in scenarios featuring partial occlusions within video
sequences. Our findings are further corroborated through visual comparisons and ablation studies. As shown in Figure 1 our
proposed OAHPE achieves superior performance while significantly reducing model complexity and computational overhead,
fully demonstrating its potential for application in the field of 3D human pose estimation.
In summary, the main contributions of this paper are as follows:
• Occlusion-aware hybrid estimation model. We propose a novel occlusion-aware 3D pose estimation model designed for
videos with mixed occlusions. The model uses an occlusion detector to adaptively partition the sequence into occluded
and visible segments, which are then modeled separately via heterogeneous inference paths. Subsequently, the outputs
from both paths are backfilled, aligned, and fused based on the original frame indices, resulting in a complete and
continuous 3D pose sequence while ensuring temporal consistency.
/
ACCEPTED MANUSCRIPT
ARTICLE IN PRESS
ARTICLE IN PRESS

• Differentiated modeling for occluded and visible segments. For occluded segments with substantial information loss, we
incorporate the MDC and LAMA modules introduced in ACL24. Specifically, MDC aggregates local spatiotemporal
neighborhood information, while LAMA jointly models global dependencies across frames and joints with linear
computational complexity. For visible segments with pronounced temporal redundancy, we propose the PDREP pathway,
which reduces redundant computation through online pruning and distillation-based temporal recovery while preserving
pose estimation accuracy.
• Leading performance on benchmark datasets. We conducted extensive comparative and ablation experiments on the
two major benchmark datasets in the field of 3D human pose estimation—Human3.6M25 and MPI-INF-3DHP26—and
achieved superior performance on key evaluation metrics, thereby validating the effectiveness and generalization
capability of the proposed hybrid routing paradigm and the design of its individual modules.
Figure 2. Schematic comparison of video 3D pose estimation frameworks. (a) Traditional video pose transformer
methods27–29 retain the full sequence at all stages, resulting in significant computational redundancy and memory overhead. (b)
H2OT20 adopts a pyramid architecture and employs a hierarchical pruning-and-recovery strategy to improve efficiency;
however, its performance may degrade in challenging scenarios such as self-occlusion. (c) The proposed OAHPE framework
enhances occluded frame segments through an occlusion-resistant spatiotemporal dual-local mechanism, while pruning visible
frame segments to reduce computational cost and improve pose estimation accuracy.
Related work
Occlusion Issues in 3D Human Pose Estimation. Occlusion remains a critical bottleneck in monocular 3D video pose estimation.
Existing approaches enhance occlusion recovery by explicitly modeling visibility and reconstructing missing keypoints, together
with temporal cues to improve robustness under occlusions30,31. Video-specific transformer and sequential paradigms further
incorporate motion priors and temporal consistency constraints, smoothing and correcting short-term occlusion errors at the
sequence level27,28. Feature-level advances emphasize selective attention to visible local regions and global dependency
modeling; for example, the combination of local attention and global self-attention has been shown to be more robust in
occlusion-sensitivity analyses29,30. However, these methods still rely heavily on high-quality observations. Under long-term
or severe occlusion, visibility prediction can easily become unreliable, leading to cumulative errors. Meanwhile, temporal
consistency modeling often tends to oversmooth rapid motions, and the stability of global inference remains limited when local
evidence is insufficient. In this paper, we design the ORST-DLEP path centered on the spatio-temporal MDC-LAMA module,
achieving precise 3D pose estimation through linear attention.
Efficiency-Based 3D Video Pose Transformers (VPTs). Given the proven advantages of transformer architecture’s self-
attention in modeling long-range dependencies and enabling parallel training32, it has been successfully applied to 3D human
pose estimation in video14,33,34. MHFormer13 explicitly generates and aggregates multiple spatio-temporal pose hypotheses
within the transformer to mitigate monocular depth ambiguity and occlusion-induced uncertainty. MixSTE33 employs joint-level
temporal modeling and alternating spatio-temporal encoding to achieve sequence-to-sequence prediction while enhancing
cross-frame consistency. However, the performance gains of these video pose transformer come with substantial computational
overhead, which limits their deployment in practical scenarios. To address this issue, H2OT20 introduces a hierarchical
/
ACCEPTED MANUSCRIPT
ARTICLE IN PRESS
ARTICLE IN PRESS

“prune-and-recover” framework to reduce computation(see Figure 2(b)). Reference35 replaces the feed-forward network
(FFN) with strided convolutions and temporally downsamples the sequence, markedly decreasing attention cost while retaining
efficient inference for long-range dependencies. KTPFormer36 incorporates kinematic or trajectory-prior attention to stabilize
performance improvements without a significant increase in computation. DeciWatch37 uses sparsely sampled frames as input
to reduce computational load, while Einfalt38 integrates upscaling and upsampling within a single transformer to achieve the
reconstruction of dense 3D poses from sparse 2D inputs. Although these methods alleviate computational cost to some extent,
they remain reliant on high-quality visual observations; consequently, estimation accuracy under challenging conditions still
leaves room for improvement. In this work, we design a PDREP pathway for visible segments with relatively low estimation
difficulty, aiming to further reduce computational overhead.
Dual Local Enhancement and Linear Attention. To mitigate the quadratic complexity induced by transformer and to improve
stability under challenging conditions such as occlusion12–14, recent studies have started to combine dual local enhancement
with global modeling of linear-complexity. PoseMamba39 replaces attention with SSM, introducing a bidirectional state space
and reordered scanning to balance long-range dependencies with joint-level modeling. SAMA40 emphasizes human-body
topology and heterogeneous joint dynamics; its structure-aware state fusion and motion-adaptive modulation correspond to
spatial local-topology enhancement, achieving finer-grained dual local enhancement within an SSM. In multi-view scenarios,
MV-SSM41 unifies the modeling of local geometric constraints and global consistency by mapping SSM to grid-guided scanning.
Meanwhile, PS-Mamba42 integrates spatiotemporal graph learning with Mamba, explicitly injecting spatial neighborhood
priors via graph structures while leveraging SSMs to capture long-range temporal correlations, leading to smoother and more
consistent pose sequences. However, such methods remain sensitive to strong occlusions and noise, with local enhancements
prone to false detections. Linear global modeling expressions are limited, and cross-scene generalization remains insufficient.
In this paper, we employ MDC to aggregate local spatiotemporal neighborhood information, while LAMA models global
dependencies across frames and joints in a single model with linear computational complexity.
OAHPE
Preliminary Analysis
Our method follows the prevailing 2D-to-3D paradigm, enabling an efficient transformation from 2D skeletal sequences to 3D
coordinates. To handle challenging videos where occluded and visible segments coexist, we adopt a differentiated treatment
strategy. For occluded segments, to balance accuracy and efficiency, we introduce a spatiotemporal MDC–LAMA dual-module
collaboration. Specifically, The MDC module aggregates local spatio-temporal neighborhood information, while the LAMA
module uses linear attention to jointly model global dependencies across frames and joints, thereby achieving both high
accuracy and efficiency in pose estimation under complex conditions. For visible frames, the “online pruning-distillation
recovery” strategy reduces computational overhead while maintaining high estimation accuracy. Finally, outputs from the two
paths are backfilled, aligned, and fused according to the original video sequence, yielding complete 3D pose predictions.
Figure 3. OAHPE processing pipeline. Based on the occlusion status inferred from the 2D keypoint sequence, the occlusion
detector dynamically routes video segments to two branches: occluded segments are fed into ORST-DLEP to improve
robustness and estimation accuracy, while visible segments are sent to PDREP for low-cost inference, thereby achieving a
favorable balance between computational efficiency and pose estimation performance.
Occlusion Detector. Inspired by the explicit keypoint-level visibility modeling strategy43, this paper introduces VisNet, a
/
ACCEPTED MANUSCRIPT
ARTICLE IN PRESS
ARTICLE IN PRESS

keypoint visibility prediction network implemented as a multi-layer perceptron (MLP), and further implements frame-level
visibility classification based on this framework. Unlike the original method, which determines keypoint visibility based on
image visual representations, this paper addresses the 3D human pose estimation task with 2D keypoint sequences as input,
using 2D keypoint coordinates and their detection confidence scores to construct keypoint-level descriptors.Specifically, for the
i−th keypoint in frame t, the 2D detection coordinates (xt,i,yt,i) and detection confidence (ct,i) output by the upstream 2D
pose estimator are used. To reduce the influence of positional differences and scale variations of the human body in the image
on model training, the coordinates are first normalized as follows ¯xt,j = xt,j−xt,root
st
, ¯yt,j = yt,j−yt,root
st
where (xt,root, yt,root) denotes
the pelvis-center coordinate, and st denotes the two-dimensional Euclidean distance between the shoulder-center position and
the pelvis-center position. The keypoint descriptor ut,i = (xt,i,yt,i,ct,i) is then passed through a linear mapping layer to obtain a
keypoint-coordinate-driven latent space representation ht,i = Linear(ut,i),ht,i ∈RD.It is then fed into a VisNet network, which
outputs the visibility logit zt,i = VisNet(ht,i) for visibility, and this score is mapped to a visibility probability pt,i = sigmoid(zt,i)
in the range [0,1] via a sigmoid function. Specifically, the closer pt,i is to 0, the higher the probability that the keypoint is
occluded; the closer pt,i is to 1, the higher the probability that the keypoint is visible.Based on the threshold τ determined
for each keypoint, the continuous probability is converted into a binary visibility label vt,i =
(
0,
pt,i < τ,
1,
pt,i ≥τ., where vt,i = 0
indicates an occluded keypoint and vt,i = 1 indicates a visible keypoint. Based on the binary visibility indicator vt,i, the ratio of
visible keypoints in frame t is computed as rt = 1
J ∑J
i=1 vt,i,where J denotes the total number of keypoints. The frame-level
label is then defined as lt =
(
0,
rt < η
1,
rt ≥η , where lt = 0 denotes an occluded frame, lt = 1 denotes a visible frame, and η is a
predefined threshold. In this way, visible and occluded frames can be explicitly distinguished, providing prior guidance for the
subsequent differential 3D human pose estimation process.
Linear Attention-based Mamba(LAMA). Although SSM have shown strong performance on 3D video sequence processing,
their one-dimensional linear scan is limited in capturing multi-directional spatial correlations, and the recurrent computation
may reduce efficiency44. Given that linear attention and SSM share a structural connection in their recursive statistical forms,
linear attention can be embedded in Mamba. To illustrate the mathematical relationship between linear attention and SSM, we
first consider linear attention under causal constraints, For an input sequence Xoccl ∈RT0×J×2 of length T0 containing J joints,
Linear attention employs kernel mapping to decompose pointwise attention into decomposable forms, yielding Qi (query), Ki
(key), Vi (value). For the i-th token, which is only related to its preceding prefix tokens, its output is:
Ai = Qi
 ∑i
m=1 K⊤
mVm

Qi
 ∑i
m=1 K⊤
m
 ,
(1)
where we further define the prefix statistics Ui = ∑i
m=1 K⊤
m ,
Di = ∑i
m=1 K⊤
mVm with a simple recursion:
Ui = Ui−1 +K⊤
i ,Di = Di−1 +K⊤
i Vi,Ai = QiDi
QiUi
.
(2)
This formulation demonstrates that long-range dependencies can be obtained without explicitly constructing an attention matrix
of quadratic complexity. Instead, they can be accumulated at linear cost using prefix statistics (Ui,Di). By incorporating the
above linear-attention kernel into the Mamba and treating the hidden state as an equivalent representation of the prefix statistics,
the update can be written in an SSM-style discrete state equation:
hi = ˆAihi−1 +Bi(∆ixi),yi = Cihi +Dxi,
(3)
here, ˆAi .= diag( ¯Ai) .= diag(exp(Ai)) is numerically equivalent to an element-wise decay on the historical state; Ai is the
state transition matrix, Bi is the input projection matrix, and Ci is the state readout matrix; hi is the global state, storing
cross-frame accumulated “key-value” information; ∆i is the discrete stride, determining the retention level of historical memory
for the current token; and Dxi is the mapping coefficient for input skip connections, preserving local observations within
the current frame. The resulting formulation is highly consistent with Mamba in form, and can thus be viewed as a special
linear-attention-driven SSM, showing strong potential for efficient and robust modeling in 3D pose estimation.The above
recursive formulation describes linear attention under a causal constraint and is used to illustrate its mathematical similarity
to state space models. Since this work focuses on offline video pose estimation, the complete video clip is available during
inference. We therefore further adopt a full-context formulation by extending the prefix statistics to global statistics shared by
all temporal–joint tokens. This treatment preserves the kernel decomposition mechanism and linear computational complexity
of linear attention, while enabling each token to exploit complementary information from both preceding and subsequent
frames, as well as from different joints.
/
ACCEPTED MANUSCRIPT
ARTICLE IN PRESS
ARTICLE IN PRESS

Network Structure
As shown in Figure 3, the proposed occlusion-aware hybrid pose estimation model. The pipeline starts from an input 2D
keypoint sequence X ∈RT×J×2 that contains both occluded and visible segments. An occlusion detector first partitions the
video into (i) long visible segments that are easier to estimate and (ii) occluded segments that are difficult to predict accurately,
yielding an occluded keypoint sequence Xoccl ∈RT0×J×2 and a visible keypoint sequence Xvis ∈RT1×J×2 according to frame
indices. For the occluded segments, inspired by Mamba16 and its linear-complexity modeling of long sequences, we design an
occlusion-robust spatio-temporal dual local-enhancement pathway (ORST-DLEP) to achieve high-accuracy estimation under
occlusion. For the visible segments, motivated by H2OT20, we introduce an efficient “online pruning–distillation recovery”
enhanced pathway (PDREP) to reduce computational overhead while preserving accuracy. Finally, based on the original frame
indices, the 3D outputs from the two pathways are backfilled, aligned, and fused in the original temporal order, producing a
dual-path-aware 3D pose sequence ˆY ∈RT×J×3 with high accuracy and low computational cost. The detailed designs and
operations of ORST-DLEP and PDREP will be described in the subsequent sections.
Occlusion-Robust Spatio-temporal Dual-Local Enhancement Path(ORST-DLEP)
Accurate 3D estimates are difficult to obtain under occlusion conditions, and precise representation of 3D poses relies on
fusion modeling of multi-source 2D observation data. Mapping a single 2D pose to three dimensions inherently suffers
from ambiguity and information loss. Pruning and recovery of occluded frames can easily lead to severe deviations in 3D
pose estimation. To address this, we designed the ORST-DLEP path for occluded frames, as illustrated in Figure 3. While
transformer have demonstrated strong performance in 3D HPE, their quadratic computational complexity imposes significant
computational overhead. Mamba16 enables modeling long-term spatial correlations with linear complexity in 3D HPE, but its
inherent one-dimensional linear scanning struggles with multi-directional spatial associations, and its recursive computation
reduces efficiency. Therefore, unlike previous SSM39,41 modeling approaches, we incorporate a linear attention module into
the SSM module to preserve the global receptive field while improving computational efficiency. we first take the occluded
segment Xoccl ∈RT0×J×2 identified by the occlusion detector as input. A fully connected layer is applied to project the keypoint
sequence into a high-dimensional embedding space P ∈RT0×J×dm. We further introduce a positional matrix Espos ∈RJ×dm to
embed spatial information. Each joint token p ∈PJ is obtained by projecting joint xoccl,i, i ∈1,2,...17 from the 2D coordinate
Xoccl ∈R1×J×2:
Xoccl = Norm
 Le(xoccl,i)+Espos

,Xoccl ∈RJ×dm,
(4)
here, Norm denotes normalization, and Le is the mapping function. The resulting features are then fed into the spatiotemporal
MDC–LAMA module to model spatiotemporal dependencies among joints. Meanwhile, a temporal positional matrix Etpos ∈
RT×dm is incorporated to embed temporal positional information:
Xoccl = Norm
 Xoccl +Etpos

,Xoccl ∈RT×dm,
(5)
the features are then fed into the spatiotemporal MDC–LAMA module, which provides global dependency modeling while
capturing local patterns at low computational cost, thereby establishing dependencies among joints. Finally, spatiotemporal
features are extracted using N −2 layers of the spatiotemporal MDC-LAMA module. The output is then fed into a linear layer
in the regression head to perform regression, generating a 3D pose sequenceˆYoccl ∈RT0×J×3.
Spatio-Temporal MDC-LAMA Block. Each spatiotemporal block consists of layer normalization (LN), a spatiotemporal
MDC–LAMA module, and a multilayer perceptron (MLP), as illustrated in Figure 3. The MDC–LAMA module serves as the
core component of ORST-DLEP, aiming to establish local–global feature dependencies with linear computational complexity.
Its architecture is shown in Figure 4. The output of the MDC module is given by:
XMDC
occl
= Conv1×1(Concat(Conv5×5,d=2(Xoccl), Conv3×3,d=2(Xoccl)))+Xoccl.
(6)
After extracting local keypoint features, we feed them into the proposed LAMA module to model long-range dependencies with
linear complexity. Input feature XMDC
occl
∈RB×T0×J×C undergoes dimensionality adjustment to yield XMDC
′
occl
∈RB×C×T0×J. The
temporal and joint dimensions are then flattened into a unified token sequence of length N = T0 ×J. The features are then fed
into the main branch and the residual branch, and finally, the enhanced tokens are restored to the T0 ×J structure.The features
are then fed into the main branch and the residual branch. The operation of the residual branch can be represented as follows:
XFres
occl = σ(Linear(XMDC′
occl
)),
(7)
here, σ(·) denotes the SiLU activation function. In the main branch, we first apply a linear projection to transform the
channel features, and then use a reshape operation to convert the feature map into a four-dimensional tensor to accommodate
/
ACCEPTED MANUSCRIPT
ARTICLE IN PRESS
ARTICLE IN PRESS

Figure 4. Core architecture of the spatiotemporal MDC–LAMA module. (a) Structure of the MDC module; (b) Structure of
the LAMA module.
the subsequent convolution. Next, a convolutional layer is employed to extract local contextual information, which can be
formulated as:
XF1
occl = Reshape
 Linear(XMDC′
occl
)

,XF2
occl = σ
 Conv(XF1
occl)

.
(8)
Next, to establish global dependencies while maintaining linear complexity, the convolution-enhanced features XF2
occl ∈
RB×T0×J×C are flattened along the temporal and spatial dimensions into a sequence X
F′
2
occl ∈RB×N×C, where N = T0 × J, to
accommodate the computation of the linear attention layer. LAMA performs a unified global aggregation on all spatiotemporal
tokens, rather than maintaining separate temporal and spatial statistics. To preserve local structural information, this paper
employs MDC for local spatiotemporal enhancement prior to flattening and introduces convolutional positional encoding into
LAMA for local compensation. We then apply linear projections and a feature map Φ(·) to construct Q and K as:
Q = Φ(Linear(X
F′
2
occl)),K = Φ(Linear(X
F′
2
occl)).
(9)
Under the decomposable formulation of linear attention, we first aggregate the global contextual statistics KV = K⊤X
F′
2
occl, and
then combine it with Q to obtain the attention response XFatten
occl
= Q·KV, the global statistics presented here are computed using
all time-joint tokens and can be viewed as an extension of the prefix statistics discussed earlier, applied to the entire sequence.
Since this paper employs an offline inference setup, each token can utilize bidirectional contextual information from the entire
sequence. To compensate for the limited ability of content similarity to capture spatial structure, we introduce a learnable
convolutional positional embedding Convpos(XF2
occl) and fuse it with the attention output, which can be expressed as:
XFatten
occl = XFatten
occl +Convpos

XF2
occl

.
(10)
Subsequently, we perform element-wise multiplication between XFatten
occl and the residual-branch feature XFres
occl, followed by a
linear projection to obtain the enhanced representation XFenh
occl :
XFenh
occl = Linear

XFatten
occl ×XFres
occl

.
(11)
Finally, XFenh
occl is fed into a lightweight feed-forward network for nonlinear transformation and channel mixing, producing the
output of the LAMA module.
/
ACCEPTED MANUSCRIPT
ARTICLE IN PRESS
ARTICLE IN PRESS

Pruning Distillation Recovery Enhancement Path (PDREP)
Visible video frame sequences inevitably contain temporal redundancy, and these highly similar frames contribute little to pose
estimation while introducing substantial redundant computation. Inspired by the plug-and-play pruning-and-recovery strategy
of H2OT20, we propose the PDREP, as illustrated in Figure 3. For the 2D keypoint sequence Xvis ∈RT1×J×2 corresponding to
frame segments classified as visible by the occlusion detector, we adopt an online pruning strategy to select k representative
frames along the temporal dimension. Specifically, at each training iteration, an unpruned teacher branch is dynamically
instantiated, and its high-fidelity predictions are distilled into the pruned student via consistency regularization, thereby
preserving representational completeness under sparse observations. Let bYvis,s and bYvis,t denote the outputs of the student and
teacher branches, respectively. The output consistency constraint is formulated as follows:
Lsd = ||bYvis,s −sg(bYvis,t)||2
2,
(12)
where sg(·) denotes the stop-gradient operator. and applies speed distillation (output first-order differences ∆bYvis,s and ∆bYvis,t) to
enhance temporal dynamic consistency:
Lvel
sd = ||∆bYvis,s −sg(∆bYvis,t)||2
2,
(13)
this mechanism can be regarded as using the unpruned branch as soft supervision to continually rectify the representation drift
of the pruned branch, thereby improving the performance upper bound under pruning. The resulting features are subsequently
fed into ORST-DLEP for 3D keypoint regression. Although pruning effectively alleviates redundant computation, its inherent
irreversibility may cause reduced information entropy in temporal representations and the loss of discriminative features,
making it difficult to guarantee that the recovery module genuinely learns to reconstruct the information of discarded frames.
Therefore, we further introduce Distillation-based Adaptive Recovery (DAR) to constrain the intermediate features and attention
distributions of the recovery module. Let Frec
s
and Frec
t
represent the features of the student and teacher branches before
recovery, respectively. The distillation-based recovery feature loss is then defined using mean squared error (MSE) as follows:
LF
dar = ||Frec
s
−sg(Frec
t
)||2
2,
(14)
since the student branch retains only k pruned temporal tokens, its cross-attention weights are defined over k keys, denoted
by As ∈RH×T1×k. In contrast, the teacher branch allocates attention over all tokens under the unpruned setting, yielding
At ∈RH×T1×T1. To achieve attention alignment and eliminate the mismatch caused by the inconsistent key dimensions
between the two branches, we extract a subset from the key dimension of the teacher attention according to the indices of the
representative k frames and then renormalize it to obtain eAt ∈RH×T1×k. The two distributions are subsequently constrained by
the Kullback–Leibler (KL) divergence, leading to the attention distillation loss defined as follows:
Lattn
dar = KL(eAt||As),
(15)
the optimization objective for distillation recovery is:
LD = λsdLsd +λ vel
sd Lvel
sd +λ F
darLF
dar +λ attn
dar Lattn
dar .
(16)
This feature is enabled when the training round exceeds the warm-up phase and actual pruning occurs, enhancing recovery
accuracy and stability without increasing computational overhead. Finally, it employs a token reconstruction strategy based on
cross-attention, using full-length queries to conditionally generate completions for sparse tokens, ensuring the output sequence
strictly aligns with the original visible frame sequence, the resulting 3D pose sequence is denoted as ˆYvis ∈RT1×J×3.
To quantify the change in information entropy before and after online pruning, we adopt the pose entropy modeling idea of
Lee45 et al. Specifically, based on the normalized two-dimensional coordinates, for any non-root joint i and its parent joint π(i),
the relative vector between the parent joint and the child joint is calculated as rt,i = ut,i −ut,π(i). The relative vector is then
converted into the normalized length Lt,i = ∥rt,i∥2 , and the orientation angle θt,i = atan2
 ¯yt,i −¯yt,π(i), ¯xt,i −¯xt,π(i)

. Based on
the normalized length and orientation angle, 49 fixed discrete regions Ωb are constructed. For a video containing S frames, the
probability that joint i falls into the b-th region is defined as Pi,b(S) = 1
S ∑S
t=1 1[(Lt,i,θt,i) ∈Ωb], where 1 denotes the indicator
function. Its value is 1 when the i-th joint in the t-th frame falls into region Ωb, and 0 otherwise. The pose entropy of the i-th
joint is then defined as Hi,S = −∑49
b=1 Pi,b(S)log2 Pi,b(S). The sequence-level pose entropy is further defined as the sum of the
entropy values of all non-root joints HS = ∑i̸=root Hi,S. Accordingly, the information entropy before and after pruning can be
calculated.
/
ACCEPTED MANUSCRIPT
ARTICLE IN PRESS
ARTICLE IN PRESS

Optimization Objectives
To achieve occlusion state recognition and 3D pose estimation in complex scenes, this paper employs an end-to-end joint
optimization strategy. Following the methodology of previous studies, we define the final loss function as follows:
L = LMPJPE+λTLT +λDetLDet +λDLD.
(17)
Here, λT, λDet and λD are all hyperparameters. LMPJPE39 represents the average joint position error loss, which is used to
constrain the Euclidean distance between the predicted 3D pose and the ground truth, LT 46 represents temporal consistency loss,
which is used to suppress unreasonable pose fluctuations between adjacent frames and enhance the continuity of the predicted
sequence in terms of motion trends and velocity changes, LDet43 is the binary cross-entropy loss for the occlusion detector, used
to supervise frame-level occlusion state prediction, LD represents the PDREP loss.
Experiments
Datasets and Evaluation Metrics
Datasets. We trained and evaluated our model on the Human3.6M25 and MPI-INF-3DHP26 datasets. Human3.6M is currently
the most widely used indoor benchmark dataset for 3D video human pose estimation. Its data was captured using four
synchronized cameras, comprising 3.6 million frames of image sequences covering 11 subjects performing 15 types of daily
activities (such as walking, sitting, eating, and talking on the phone). Following the dataset’s standard partitioning protocol and
related research settings, subjects S1, S5, S6, S7, and S8 were selected for the training set, while S9 and S11 were designated
for the test set. MPI-INF-3DHP is a 3D human pose dataset comprising 1.3 million frames captured by 14 synchronized
cameras from multiple perspectives, covering both indoor and outdoor environments. It features more complex backgrounds,
richer camera viewpoints, and significant occlusions and motion deformations.
Evaluation Metrics. For Human3.6M, we selected Mean Per Joint Position Error (MPJPE) to measure the average Euclidean
distance between estimated 3D joint coordinates and corresponding ground truth, expressed in millimeters. We also selected
Procrustes-aligned MPJPE (P-MPJPE) to measure the MPJPE calculated after rigid alignment (including rotation, translation,
and scaling) between estimated and ground truth poses. For MPI-INF-3DHP, following previous studies12,33, we selected the
area under the curve (AUC), percentage of correct keypoints (PCK) at a threshold of 150 mm, and MPJPE.
Implementation Details
The proposed framework is implemented in PyTorch, and all training and testing experiments are conducted on NVIDIA RTX
4090 GPU with 48 GB memory. Two different sequence lengths, denoted by T = 81 and T = 243, are used in the experiments.
The model was trained for 120 epochs using the AdamW47 optimizer with a weight decay of 0.01. An exponential learning rate
decay strategy was employed, with an initial learning rate of 2e−4 and a decay factor of 0.98 per epoch. The batch size was set
to 4 and the predefined threshold η and τ are 0.5. For 2D pose input, to maintain consistency with existing methods and ensure
a fair comparison, we use 2D keypoints detected by the Cascaded Pyramid Network (CPN) as input and conduct experiments
on the Human3.6M25 and MPI-INF-3DHP26