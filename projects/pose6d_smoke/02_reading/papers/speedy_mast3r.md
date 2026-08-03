# Speedy MASt3R

> 2025 · id: arxiv:2503.10017 · arXiv: 2503.10017 · pdf: https://arxiv.org/pdf/2503.10017 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Image matching is a fundamental component of state-of-
the-art 3D vision algorithms and pipelines, playing a cru-
cial role in accurate scene reconstruction and localization.
MASt3R [11] has redefined image matching as a 3D task
by leveraging DUSt3R [24] and introducing a fast recip-
rocal matching scheme that accelerates matching by or-
ders of magnitude while maintaining theoretical guaran-
tees. This approach has gained significant traction in the
community, with DUSt3R and MASt3R collectively accu-
mulating over 250 citations in a short span, underscoring
their impact. However, despite its state-of-the-art accuracy,
MASt3R’s inference speed remains a bottleneck, for exam-
ple on an A40 GPU, with a latency of 198.16 ms per image
pair, primarily due to computational overhead from the ViT
encoder-decoder and the Fast Reciprocal Nearest Neighbor
(FastNN) matching stage.
To address this, we introduce Speedy MASt3R, a post-
training optimization framework that significantly enhances
inference efficiency while maintaining accuracy.
Speedy
MASt3R integrates multiple optimization techniques, in-
cluding FlashMatch—an approach that leverages FlashAt-
tention v2 with tiling strategies to significantly enhance
computational efficiency—computation graph optimization
with layer and tensor fusion, kernel auto-tuning via Ten-
sorRT (GraphFusion), and a streamlined FastNN pipeline
that reduces memory access time from quadratic to linear
while accelerating block-wise correlation scoring through
vectorized computation (FastNN-Lite). Additionally, it em-
ploys mixed-precision inference with FP16/FP32 hybrid
computations (HybridCast), achieving speedup while en-
suring numerical precision.
Evaluated on Aachen Day-
Night, InLoc, 7-Scenes, ScanNet1500, and MegaDepth1500
*Equal contribution.
datasets, Speedy MASt3R achieves a 54% reduction in in-
ference time (198 ms →91 ms per image pair) without
compromising accuracy. This advancement enables real-
time 3D understanding, facilitating applications such as
mixed reality navigation and large-scale 3D scene recon-
struction.

## introduction
Image matching is a fundamental problem in computer vi-
sion, crucial for applications such as structure-from-motion
(SfM) [20], visual localization [19, 23], and 3D reconstruc-
tion [1, 9]. Traditional keypoint-based methods, including
SIFT [14], ORB [17], and SuperPoint [5], detect and de-
scribe sparse features before performing nearest-neighbor
search for matching. While these methods remain effective
in many scenarios, their reliance on local descriptors makes
them vulnerable to texture-less regions and repetitive pat-
terns.
To overcome these limitations, deep learning-based
dense matching techniques, such as LoFTR [22], DKM [7],
RoMa [8], and SuperGlue [18], leverage global feature
reasoning through transformer-based architectures. These
methods achieve state-of-the-art performance on challeng-
ing benchmarks, improving robustness to large viewpoint
and illumination changes. However, dense matching often
incurs high computational costs, making it less feasible for
real-time applications.
More recently, grounding image matching in 3D has
gained attention as a means to improve both robustness and
accuracy. DUSt3R [24] pioneered the use of 3D pointmaps
for pixel correspondences, demonstrating superior re-
silience to extreme viewpoint variations.
MASt3R [11]
extends this approach by integrating a transformer-based
matching head that learns local features alongside the 3D
1
arXiv:2503.10017v1  [cs.CV]  13 Mar 2025

structure, enabling more precise matches.
Our work,
Speedy MASt3R builds upon this foundation, introduc-
ing computational-efficiency attention mechanisms [3] and
computational graph optimizations [16] to accelerate infer-
ence while maintaining accuracy. Our approach preserves
the theoretical guarantees of the fast reciprocal matching
scheme used in the original MASt3R while reducing mem-
ory access times and enhancing computational efficiency,
enabling real-time performance without sacrificing accu-
racy. Our work, Speedy MASt3R, introduces a comprehen-
sive post-training optimization framework to accelerate im-
age matching while maintaining state-of-the-art accuracy. It
integrates several major optimization techniques:
• FlashMatch: An efficient attention mechanism leverag-
ing FlashAttention v2 [3] with tiling strategies to optimize
GPU memory access and significantly reduce computa-
tional overhead in the ViT encoder-decoder pipeline [6].
• GraphFusion: Computation graph optimization by uti-
lizing kernel auto-tuning and tensor fusion, eliminat-
ing redundant intermediate tensor allocations and re-
ducing unnecessary computations, as leveraged by Ten-
sorRT [16].
• FastNN-Lite: A streamlined FastNN pipeline that re-
duces memory access time from quadratic to linear and
accelerates block-wise correlation scoring through vec-
torized computation.
• HybridCast: A mixed-precision inference framework
combining FP16 and FP32 computations to achieve
speedup while ensuring numerical precision in critical op-
erations.
Speedy MASt3R achieves a 54% reduction in infer-
ence time (198 ms →91 ms per image pair) without
compromising high quality matching results, as demon-
strated on the Aachen Day-Night [26], InLoc [23], 7-
Scenes [21], ScanNet1500 [2] and MegaDepth1500 [12]
datasets datasets.
This significant speedup underscores
the effectiveness of our optimization framework in en-
abling real-time 3D understanding without sacrificing per-
formance.
2. Background and Related Works
Recent advancements in image matching have redefined the
landscape of 3D scene reconstruction and visual localiza-
tion. Traditional methods such as SIFT [14] and ORB [17]
rely on handcrafted keypoints and descriptors, making them
susceptible to texture-less surfaces and extreme viewpoint
changes. Learning-based methods such as SuperPoint [5]
and SuperGlue [18] improve feature matching by leveraging
deep neural networks and global feature aggregation. How-
ever, they still treat matching as a local problem, which can
lead to inconsistencies in large-scale 3D scene reconstruc-
tion.
2.1. MASt3R and 3D-Grounded Matching
To address these challenges, DUSt3R [24] introduced 3D
pointmaps, which frame image matching as a joint 3D scene
reconstruction problem. Extending this idea, MASt3R [11]
introduced a transformer-based matching head that jointly
learns local features and 3D correspondences. Addition-
ally, Fast Nearest-Neighbor Matching (FastNN) was pro-
posed as a high-efficiency nearest-neighbor search mech-
anism. MASt3R achieved state-of-the-art performance on
multiple benchmarks, demonstrating robustness to extreme
viewpoint changes. Despite these innovations, MASt3R’s
inference speed remains a bottleneck, primarily due to its
heavy computation from the ViT encoder-decoder, which
accounts for 60% of the latency, and the FastNN match-
ing stage, which contributes to 40% of total computation
time. Moreover, the significant computational overhead as-
sociated with full-resolution dense correspondences renders
it impractical for real-time applications, such as AR/VR,
robotics, and large-scale mapping. Resolving these com-
putational bottlenecks is essential for enabling practical de-
ployment in time-sensitive scenarios.
2.2. Optimizing Image Matching for Speed and Ef-
ficiency
Several recent works have focused on optimizing dense fea-
ture matching for efficiency. Vision transformers (ViTs) [6]
have been a critical development in global feature aggrega-
tion. Swin Transformer [13] reduces computational com-
plexity by restricting self-attention to local windows, mak-
ing transformers more scalable for high-resolution images.
FlashAttention [4] and FlashAttention v2 [3] further opti-
mize GPU memory access by introducing tiling strategies.
These improvements allow for efficient sequence process-
ing without compromising accuracy.
2.3. Efficient Attention Mechanisms and FlashAt-
tention
Traditional self-attention mechanisms in transformers suf-
fer from quadratic complexity with respect to sequence
length, making them inefficient for large-scale feature
matching tasks. FlashAttention [4] optimizes memory ac-
cess by using an I/O-aware algorithm that avoids material-
izing the full attention matrix, significantly reducing both
computation and memory costs. It achieves this by tiling
the attention computation, ensuring that intermediate val-
ues fit within high-bandwidth memory (SRAM) on GPUs.
FlashAttention v2 [3] improves upon this by further opti-
mizing work partitioning and parallelization, achieving sig-
nificant speedup compared to naive attention implementa-
tions.
2

2.4. Efficient Nearest-Neighbor Search for Feature
Matching
Efficient nearest-neighbor search remains a key challenge
in large-scale feature matching. Traditional mutual nearest
neighbor search methods have quadratic complexity, mak-
ing them infeasible for dense matching.
Faiss [10] ad-
dresses this by employing approximate nearest-neighbor
(ANN) search, enabling large-scale similarity retrieval.
Similarly, HNSW graphs [15] optimize nearest-neighbor
retrieval using multi-layer navigable small-world graphs,
but these methods are not accurate.
FastNN, introduced
in MASt3R, aimed to reduce this computational overhead
while preserving accuracy, but it still remains a bottleneck
in dense matching pipelines.
2.5.
Mixed-Precision
and
Kernel
Fusion
for
Speedup
Further acceleration can be achieved through mixed-
precision inference and computational graph optimizations.
TensorRT [16]-based optimizations eliminate redundant in-
termediate tensor allocations, thereby reducing unneces-
sary computations. Additionally, mixed-precision inference
(FP16/FP32) has been shown to significantly reduce mem-
ory bandwidth. Such optimizations allow models to achieve
substantial speedups while preserving performance for crit-
ical tasks like 3D scene reconstruction and image matching.

## method
3.1. Problem Statement
Given two images I1 and I2, captured by two cameras c1
and c2 with unknown parameters, the goal is to recover a
set of pixel correspondences {(i, j)}, where i and j are pix-
els in I1 and I2 respectively. Each pixel is represented as
i = (wi, hi) and j = (wj, hj), where w and h denote the
width and height of the images. For simplicity, I1 and I2 are
assumed to have the same resolution, although the approach
can handle pairs of variable aspect ratios.
The problem of image matching is inherently tied to the
recovery of 3D scene geometry. Traditional methods cast
matching as a 2D problem, which limits their applicability
for tasks like visual localization. In contrast, MASt3R [11]
jointly addresses 3D scene reconstruction and image match-
ing, leveraging the DUSt3R[24] framework as a foundation.
3.2. Overview of MASt3R
MASt3R, illustrated in Figure 1, builds upon the DUSt3R
framework and introduces a novel matching head and an
optimized matching scheme. The pipeline consists of the
following key steps:
3.2.1. Feature Extraction
Both images I1 and I2 are encoded in a Siamese manner us-
ing CroCo [25], which is a Vision Transformer (ViT), yield-
ing two representations H1 and H2:
H1, H2 = Encoder(I1), Encoder(I2).
3.2.2. Cross-Attention Decoding
The representations H1 and H2 are processed by two inter-
twined decoders, which also utilize the CroCo [25] struc-
ture.
These decoders exchange information via cross-
attention to understand the spatial relationship between
viewpoints and the global 3D geometry of the scene. The
augmented representations are denoted as H′
1 and H′
2:
H′
1, H′
2 = Decoder(H1, H2).
3.2.3. 3D Pointmap Regression
Two prediction heads regress dense 3D pointmaps x1,1 and
x2,1, as well as confidence maps c1 and c2:
x1,1, c1 = Headp([H1, H′
1]),
x2,1, c2 = Headp([H2, H′
2]).
Here, [H1, H′
1] and [H2, H′
2] are the concatenations of the
encoder and decoder outputs. x1,1 ∈RH×W ×3 represents
a dense 2D-to-3D mapping between each pixel in I1 and its
corresponding 3D point in the coordinate system of camera
c1.
3.2.4. Matching Head
To improve the precision of pixel correspondences,
MASt3R introduces a matching head that outputs dense fea-
ture maps D1 and D2 ∈RH×W ×d:
D1, D2 = Headm([H1, H′
1]), Headm([H2, H′
2]).
These feature maps are used in conjunction with the 3D
pointmaps to perform robust matching.
3.2.5. Fast Reciprocal NN Matching
MASt3R introduces an optimized matching scheme based
on Fast Reciprocal NN Matching (FastNN) to effi-
ciently handle dense feature maps.
This scheme is de-
signed to reduce computational complexity while maintain-
ing high matching accuracy, making it suitable for large-
scale datasets.
Problem Context
Traditional mutual nearest neighbor
(NN) matching methods require computing pairwise dis-
tances between all pixels, resulting in a complexity of
O(W 2H2)
, where W and H are the width and height of the images.
This high complexity becomes a bottleneck for large-scale
datasets and real-time applications.
3

Figure 1. Overview of the MASt3R pipeline and optimizations introduced by Speedy MASt3R. Given two input images, the network
leverages a ViT encoder and a transformer decoder to jointly regress 3D pointmaps, confidence maps, and dense feature maps. The FastNN
matcher identifies robust correspondences, enabling joint 3D reconstruction and image matching. Speedy MASt3R enhances the original
framework by integrating FlashMatch for efficient attention computation through tiling strategies, GraphFusion for eliminating redundant,
unnecessary tensor computation, FastNN-Lite for reducing memory access time from quadratic to linear, and HybridCast for enabling
mixed-precision inference with FP16 and FP32 computations.
FastNN Algorithm
FastNN addresses this issue by lever-
aging iterative subsampling and reciprocal NN search. The
algorithm proceeds as follows:
1. Initialization: Sample k pixels U 0 from I1 typically on
a grid. Find their nearest neighbors in I2, denoted as V 0.
2. Iterative Search: In each iteration t, find the nearest
neighbors of V t back in I1, denoted as U t+1. Identify
reciprocal matches Mt = {(i, j) | U t+1
i
= U t
i } (points
forming a cycle). Remove converged points from U t+1
and V t+1.
3. Termination: The process terminates when most points
have converged or a maximum number of iterations T is
reached.
4. Output: Return the set of all reciprocal matches M =
S
t Mt.
Integration with MASt3R
In MASt3R, FastNN is ap-
plied in a coarse-to-fine manner to improve both speed and
accuracy. The dense feature maps D1 and D2 generated
by the matching head are used as input to FastNN. This
allows MASt3R to efficiently compute robust pixel corre-
spondences, which are then used for 3D reconstruction.
3.2.6. 3D Reconstruction
Finally, the dense correspondences are used to generate a
3D point cloud, leveraging the DUSt3R framework’s re-
gression loss for optimization.
3.3. Limitations of MASt3R
While MASt3R achieves state-of-the-art accuracy in 3D
scene reconstruction and image matching, its inference
speed remains a bottleneck. Specifically, processing a sin-
gle image pair takes 198ms, which is significantly slower
than real-time requirements.
This slow matching speed
severely limits the real-time applicability of MASt3R, par-
ticularly in scenarios requiring fast and efficient processing,
such as autonomous driving or augmented reality.
To address these challenges, Speedy MASt3R is pro-
posed as an optimized framework that significantly reduces
inference latency without compromising accuracy. The fol-
lowing sections detail the key optimizations introduced in
Speedy MASt3R to overcome the limitations of the original
MASt3R pipeline.
3.4. Speedy MASt3R
3.4.1. FlashMatch
The Vision Transformer (ViT) [6] encoder-decoder in
MASt3R plays a crucial role in 3D scene reconstruction and
image matching. However, the traditional attention mech-
anism in ViT suffers from high computational complexity,
scaling quadratically with the sequence length of the input
tokens O(n2), and a significant memory footprint. This be-
comes a bottleneck for MASt3R, as 60% of the total in-
ference latency is attributed to the ViT encoder-decoder,
with attention being the primary contributor. Specifically,
the memory-intensive nature of attention computation lim-
its the scalability of MASt3R to high-resolution images and
real-time applications.
To address these limitations, we integrate FlashAtten-
tion v2 [3] into the self-attention modules of 2 pairs of en-
coders and decoders in MASt3R. FlashAttention v2 is an
optimized attention mechanism that reduces both computa-
tional complexity and memory footprint by leveraging tiling
strategies and efficient memory access patterns. Its core
idea is to decompose the attention computation into smaller
blocks (tiles) that fit into the GPU’s fast memory (SRAM),
minimizing the need for costly global memory accesses.
4

Figure 2. Comparison of Double Loop (left) and Single Loop (right) optimization strategies for matrix multiplication in the feature
matching stage of MASt3R. Here, BS denotes Block Size, and P denotes the number of Pixels. The traditional Double Loop approach incurs
significant memory access overhead due to block-wise computation. Our proposed Single Loop strategy unrolls block-wise operations into
a single loop, reducing memory accesses while maintaining VRAM usage within the target hardware’s capacity.
3.4.2. GraphFusion
While FlashMatch enhances inference speed by optimiz-
ing the Attention mechanism in the Transformer, we fur-
ther accelerate the entire network’s execution by applying
several inference-time optimization techniques. These in-
clude computation graph optimization, layer and tensor fu-
sion, efficient memory management for dynamic tensors,
and kernel tuning for deployment target device. Leveraging
the TensorRT [16], we achieve more efficient computational
graph fusion and optimization, significantly boosting infer-
ence speed of neural network.
3.4.3. FastNN-Lite
The original FastNN employs a nested loop structure to
compute pairwise distances between feature blocks from
two images A and B, resulting in O(n2) time complexity.
The algorithm is formalized as follows:
We notice that the original FastNN algorithm can be sped
up by reducing the number of accesses to the feature blocks.
Therefore, we suggest substituting the original algorithm
with FastNN-Lite.
FastNN-Lite first replaces the nested
loop structure with a single-loop execution graph, process-
ing blocks of A sequentially while handling B as a whole.
This modification reduces the time complexity to O(n) and
eliminates redundant memory allocations. The algorithm is
formalized as follows:
Key Optimizations.
The single-loop execution graph in-
troduces several significant optimizations, as illustrated in
Figure 2:
• Time Complexity Reduction: By processing blocks of
A sequentially and handling B as a whole, the time com-
plexity is reduced from O(n2) to O(n).


## experiments
Speedy MASt3R is a post-training optimization framework.
We base the MASt3R’s architecture (ViT-Large encoder,
ViT-Base decoder, and CatMLP+DPT head) and initialize
with the public pretrained weights. Then, we directly apply
the optimization techniques.
We evaluate our proposed Speedy MASt3R on the two
popular tasks with widely used benchmarks. For the rela-
tive pose estimation task (Sec. 4.1), we report results on the
ScanNet1500 [2, 18] and MegaDepth1500 [12, 22] datasets.
For the visual localization task (Sec. 4.2), we present re-
sults on the Aachen Day-Night [26], InLoc [23], and 7-
Scenes [21] datasets. We conducted our experiment on an
A40 GPU.
4.1. Relative Pose Estimation
We evaluate Speedy MASt3R on the ScanNet1500 [2] and
MegaDepth1500 [12] datasets. Both datasets contain 1,500
pairs of images, with ScanNet1500 focusing more on indoor
images, while MegaDepth1500 consists exclusively of out-
door images. We report model accuracy using four metrics:
AUC@5/10/20, which measures the area under the curve of
pose accuracy with respect to thresholds of 5/10/20 degrees
for the minimum of translation and rotation angular errors,
and mean average accuracy (mAA), which is the mean of
AUC@5/10/20. Additionally, we measure the average run-
ning time of each module (Encoder/Decoder/Head/FastNN)
in milliseconds (ms).
Table 1 compares Speedy MASt3R with vanilla MASt3R
in terms of accuracy and computational efficiency. While
maintaining the same accuracy—since the difference is not
statistically significant—the optimization techniques effec-
tively reduce the running time of each module by 47.41%,
30.99%, 26.73%, 61.07% for ScanNet1500 and by 47.12%,
30.41%, 27.11%, 58.96% for MegaDepth1500.
4.2. Visual Localization
In this scenario, we evaluate the accuracy of estimated ab-
solute pose across three datasets: Aachen Day-Night [26],
InLoc [23], and 7-Scenes [21]. The Aachen dataset consists
of 824 daytime and 98 nighttime query images, along with
5,235 reference images captured in the historic city center
of Aachen, Germany. The InLoc dataset presents challenges
in estimating the correct pose for 356 hand-captured query
images, given a database of 4,681 RGB images with sig-
nificant visual differences. The 7-Scenes dataset includes
6

Table 1. Accuracy (left) and computational efficiency (right) test on ScanNet1500 [2] and MegaDepth1500 [12] datasets. Lower numbers
are better in terms of inference speed.
ScanNet1500 [2]
MegaDepth1500 [12]
AUC@5
AUC@10
AUC@20
mAA
AUC@5
AUC@10
AUC@20
mAA
MASt3R
34.51
57.31
74.5
55.44
39.87
54.93
66.88
53.89
Ours
34.32
57.15
74.3
55.25
39.28
54.57
66.54
53.46
Encoder↓
Decoder↓
Head↓
FastNN↓
ScanNet
MASt3R
52.79
31.01
20.72
115.69
Ours
27.76
21.40
15.18
45.03
MegaDepth
MASt3R
52.34
30.85
20.77
114.17
Ours
27.68
21.47
15.14
46.86
Table 2.
Localization Accuracies.
The upper table presents the percentage of accurately localized images within the thresholds of
(0.25m/2°)/(0.5m/5°)/(5m/10°) for Aachen [26], and (0.25m/10°)/(0.5m/10°)/(1m/10°) for InLoc [23]. The lower table reports localization
accuracy using median translation and rotation errors for 7-Scenes [21]. The “top N” indicates the number of retrieved images.
Aachen [26]
InLoc [23]
Day
Night
DUC1
DUC2
MASt3R top1
77.5 / 89.0 / 97.9
58.2 / 72.4 / 87.8
43.9 / 60.1 / 68.7
38.2 / 54.2 / 55.0
MASt3R top20
88.2 / 95.1 / 99.6
72.4 / 93.9 / 99.0
63.6 / 82.8 / 90.9
69.5 / 89.3 / 89.3
MASt3R top40
89.2 / 95.4 / 99.8
75.5 / 91.8 / 100.0
63.6 / 84.3 / 93.4
72.5 / 91.6 / 92.4
Speedy top1
77.2 / 89.2 / 98.1
57.1 / 73.5 / 87.8
43.9 / 60.1 / 68.7
38.9 / 54.2 / 55.0
Speedy top20
88.2 / 95.4 / 99.6
71.4 / 92.9 / 98.0
63.6 / 82.3 / 91.4
69.5 / 89.3 / 89.3
Speedy top40
89.1 / 95.5 / 99.8
74.5 / 90.8 /100.0
63.1 / 84.3 / 93.4
72.5 / 91.6 / 92.4
7-Scenes [21]
Chess
Fire
Heads
Office
Pumpkin
Kitchen
Stairs
MASt3R top1
2.33 / 0.79
2.16 / 0.80
1.20 / 0.82
3.10 / 0.92
4.10 / 1.07
3.71 / 1.28
3.46 / 0.95
Ours top1
2.33 / 0.79
2.16 / 0.80
1.20 / 0.82
3.10 / 0.92
4.11 / 1.07
3.71 / 1.28
3.46 / 0.95
seven distinct indoor environments, each containing a vary-
ing number (1,000–5,000) of query images.
We evaluate localization performance by measuring
the percentage of successfully localized images within
three thresholds: (0.25m/2°), (0.5m/5°), and (5m/10°) for
Aachen, and (0.25m/10°), (0.5m/10°), and (1m/10°) for In-
Loc. For 7-Scenes, we report the median translation and
rotation errors in meters and degrees, respectively. Addi-
tionally, we assess computational efficiency across all three
datasets, analyzing the processing time of each module (in
ms) required for localizing a single query image.
For each query image, we evaluate localization perfor-
mance using the top 1, top 20, and top 40 retrieved images.
As shown in Table 2, Speedy MASt3R improves localiza-
tion accuracy as more retrieved images are provided, simi-
lar to MASt3R. Table 3 further demonstrates the enhanced
computational efficiency of Speedy MASt3R across all
three datasets. Notably, Speedy MASt3R achieves greater
time savings with an increasing number of retrieved im-
ages. For example, in the case of InLoc (top 40), Speedy
MASt3R reduces the running time of each MASt3R mod-
ule by 0.904s, 0.368s, 0.213s, and 38.840s, resulting in a
total time savings of 40.326s.
4.3. Ablation study
To assess the impact of each optimization technique on the
modules, we incrementally apply FlashMatch, GraphFu-
sion, FastNN-Light, and HybridCast one by one. We eval-
uate relative pose estimation quality using AUC@5/10/20
and mAA (Table 4) and measure the running time of
each module in ms (Table 5) on the ScanNet1500 [2] and
MegaDepth1500 [12] benchmarks.
We
observe
that
on
both
ScanNet1500
and
MegaDepth1500
benchmark,
Speedy
MASt3R
main-
tains the same accuracy with vanilla MASt3R; the minor
differences are statistically insignificant. In terms of com-
putational efficiency, each technique effectively reduces the
running time of the targeted modules. On the ScanNet1500
benchmark, FlashMatch, GraphFusion, FastNN-Light, and
HybridCast reduce processing time by 25.15ms/9.58ms
in the Encoder/Decoder, 5.59ms in the Head, 45.5ms in
FastNN, and an additional 27.55ms in FastNN, respec-
tively. Similarly, on the MegaDepth1500 benchmark, these
techniques reduce processing time by 24.59ms/9.48ms in
7

Table 3. Computational efficiency test on Aachen Day-Night [26] and InLoc [23] (left) and on the 7-Scenes [21] (right). Lower numbers
are better.
Encoder↓
Decoder↓
Head↓
FastNN↓
Aachen Day
MASt3R top1
57.21
39.71
22.97
78.27
MASt3R top20
969.05
571.39
388.90
1140.17
MASt3R top40
2206.40
1586.96
871.41
3033.58
Ours top1
30.24
23.28
15.11
22.87
Ours top20
599.09
475.06
283.99
441.07
Ours top40
1204.44
928.27
570.28
895.47
Aachen Night
MASt3R top1
85.80
63.17
30.19
130.98
MASt3R top20
1037.35
680.83
414.63
1316.06
MASt3R top40
2198.81
1519.02
866.82
2996.83
Ours top1
31.57
23.53
16.40
22.64
Ours top20
617.19
467.74
299.73
459.95
Ours top40
1195.93
935.39
574.23
886.05
InLoc
MASt3R top1
65.89
32.19
28.69
1766.62
MASt3R top20
1144.30
812.05
451.10
39260.08
MASt3R top40
2101.70
1266.11
829.22
70941.91
Ours top1
30.63
23.50
15.77
770.05
Ours top20
591.42
457.33
307.74
15909.04
Ours top40
1197.402
897.26
616.06
32101.84
7-Scenes
Encoder↓
Decoder↓
Head↓
FastNN↓
Chess
MASt3R top1
53.00
31.70
21.35
1702.78
Ours top1
28.66
22.11
15.23
761.10
Fire
MASt3R top1
52.76
31.91
21.00
1811.34
Ours top1
28.57
21.89
15.16
797.47
Heads
MASt3R top1
53.70
33.13
21.67
1708.31
Ours top1
29.73
22.29
15.75
748.81
Office
MASt3R top1
53.40
31.25
20.98
1704.45
Ours top1
28.23
21.71
15.03
775.07
Pumpkin
MASt3R top1
53.18
32.18
21.89
1749.20
Ours top1
29.47
22.11
15.33
770.75
Kitchen
MASt3R top1
54.32
36.00
22.03
1934.92
Ours top1
28.07
21.54
15.03
792.70
Stairs
MASt3R top1
52.75
30.85
20.63
1696.28
Ours top1
31.39
23.52
16.71
775.33
Table 4. Relative pose estimation accuracy remains stable while optimization techniques are applied incrementally.
ScanNet1500 [2]
MegaDepth1500 [12]
AUC@5
AUC@10
AUC@20
mAA
AUC@5
AUC@10
AUC@20
mAA
MASt3R
34.51
57.31
74.50
55.44
39.87
54.93
66.88
53.89
+ FlashMatch
34.41
57.24
74.37
55.34
38.93
54.46
66.93
53.44
+ GraphFusion
34.24
57.32
74.52
55.36
39.30
55.00
66.98
53.76
+ FastNN-Light
34.32
57.15
74.30
55.25
39.27
54.56
66.52
53.45
+ HybridCast (Speedy MASt3R)
34.18
57.17
74.37
55.24
38.83
54.74
66.88
53.48
Table 5. Computational efficiency increases when optimization
techniques are applied incrementally. Lower numbers are better
Encoder↓
Decoder↓
Head↓
FastNN↓
ScanNet1500 [2]
MASt3R
52.79
31.01
20.72
115.69
+ FlashMatch
27.64
21.43
20.65
125.17
+ GraphFusion
27.99
21.42
15.13
116.54
+ FastNN-Light
28.41
21.62
15.59
70.19
+ HybridCast (Speedy MASt3R)
27.50
21.43
15.04
42.64
MegaDepth1500 [12]
MASt3R
52.34
30.85
20.77
114.17
+ FlashMatch
27.75
21.37
20.74
114.02
+ GraphFusion
27.74
21.41
15.18
121.80
+ FastNN-Light
27.49
21.37
15.09
70.66
+ HybridCast (Speedy MASt3R)
27.76
21.38
15.15
43.60
the Encoder/Decoder, 5.59ms in the Head, 43.51ms in
Fas

## conclusion
In this work, we introduced Speedy MASt3R, a post-training
optimization framework designed to accelerate the infer-
ence speed of the MASt3R image matching model while
maintaining its state-of-the-art accuracy. Speedy MASt3R
integrates multiple optimizations, including FlashMatch,
GraphFusion, FastNN-Lite, and HybridCast, each target-
ing key computational bottlenecks in the original MASt3R
pipeline. These enhancements enable a significant reduc-
tion in inference time (from 198 ms to 91 ms per image
pair) without compromising matching performance.
Through extensive evaluations on benchmark datasets
such as ScanNet1500, MegaDepth1500, Aachen Day-
Night, InLoc, and 7-Scenes, we demonstrate that Speedy
MASt3R preserves the theoretical guarantees as well as
practical performance of fast reciprocal matching with sig-
nificant improvement in inference time (more than 54 per-
centage).
Our findings underscore the critical need to enhance
MASt3R’s efficiency, given its growing adoption as a state-
of-the-art image matching model in 3D vision.
While
MASt3R delivers exceptional accuracy even in challenging
scenarios, its computational overhead presents a significant
challenge. Speedy MASt3R tries to address this crucial lim-
itation and represents a significant step in that direction by
significantly accelerating MASt3R’s inference (more than
50%) without compromising its robust performance.
8

Acknowledgments
Supported by the Intelligence Advanced Research Projects
Activity (IARPA) via Department of Interior/ Interior Busi-
ness Center (DOI/IBC) contract number 140D0423C0076.
The U.S. Government is authorized to reproduce and dis-
tribute reprints for Governmental purposes notwithstanding
any copyright annotation thereon. Disclaimer: The views
and conclusions contained herein are those of the authors
and should not be interpreted as necessarily representing
the official policies or endorsements, either expressed or im-
plied, of IARPA, DOI/IBC, or the U.S. Government.