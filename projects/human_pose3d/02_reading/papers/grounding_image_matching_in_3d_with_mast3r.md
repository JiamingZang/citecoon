# Grounding Image Matching in 3D with MASt3R

> 2024 · id: W4399759054 · arXiv: 2406.09756 · pdf: https://arxiv.org/pdf/2406.09756 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

Grounding Image Matching in 3D with MASt3R
Grounding Image Matching in 3D with MASt3R
Vincent Leroy Yohann Cabon Jerome Revaud
NAVER LABS Europe
https://github.com/naver/mast3r
Figure 1: Dense Correspondences. MASt3R extends DUSt3R as it predicts dense correspondences, even in
regions where camera motion significantly degrades the visual similarity. Focal length can be derived from the
predicted 3D geometry, making our approach a standalone method for camera calibration, camera pose estimation
and 3D scene reconstruction, attaining and improving the performance of the state of the art on several extremely
challenging benchmarks.
Abstract
Image Matching is a core component of all best-performing algorithms and pipelines in 3D vision. Yet
despite matching being fundamentally a 3D problem, intrinsically linked to camera pose and scene
geometry, it is typically treated as a 2D problem. This makes sense as the goal of matching is to establish
correspondences between 2D pixel fields, but also seems like a potentially hazardous choice. In this
work, we take a different stance and propose to cast matching as a 3D task with DUSt3R, a recent and
powerful 3D reconstruction framework based on Transformers. Based on pointmaps regression, this
method displayed impressive robustness in matching views with extreme viewpoint changes, yet with
limited accuracy. We aim here to improve the matching capabilities of such an approach while preserving
its robustness. We thus propose to augment the DUSt3R network with a new head that outputs dense
local features, trained with an additional matching loss. We further address the issue of quadratic
complexity of dense matching, which becomes prohibitively slow for downstream applications if not
carefully treated. We introduce a fast reciprocal matching scheme that not only accelerates matching
by orders of magnitude, but also comes with theoretical guarantees and, lastly, yields improved results.
Extensive experiments show that our approach, coined MASt3R, significantly outperforms the state of
the art on multiple matching tasks. In particular, it beats the best published methods by 30% (absolute
improvement) in VCRE AUC on the extremely challenging Map-free localization dataset.
1
arXiv:2406.09756v1  [cs.CV]  14 Jun 2024

June, 2024
1. Introduction
Being able to establish correspondences between pixels
across different images of the same scene, denoted as
image matching, constitutes a core component of all 3D
vision applications, spanning mapping [14,61], local-
ization [41,72], navigation [15], photogrammetry [34,
64] and autonomous robotics in general [63,87]. State-
of-the-art methods for visual localization, for instance,
overwhelmingly rely upon image matching during the
offline mapping stage, e.g. using COLMAP [75], as well
as during the online localization step, typically using
PnP [30]. In this paper, we focus on this core task and
aim at producing, given two images, a list of pairwise
correspondences, denoted as matches. In particular,
we seek to output highly accurate and dense matches
that are robust to viewpoint and illumination changes
because these are, in the end, the limiting factor for
real-world applications [36].
In the past, matching methods have traditionally been
cast into a three-steps pipeline consisting of first extract-
ing sparse and repeatable keypoints, then describing
them with locally invariant features, and finally pairing
the discrete set of keypoints by comparing their distance
in the feature space. This pipeline has several merits:
keypoint detectors are precise under low-to-moderate
illumination and viewpoint changes, and the sparsity of
keypoints makes the problem computationally tractable,
enabling very precise matching in milliseconds when-
ever the images are viewed under similar conditions.
This explains the success and persistence of SIFT [52]
in 3D reconstruction pipelines like COLMAP [75].
Unfortunately, keypoint-based methods, by reducing
matching to a bag-of-keypoint problem, discard the
global geometric context of the correspondence task.
This makes them especially prone to errors in situation
with repetitive patterns or low-texture areas, which are
in fact ill-posed for local descriptors. One way to remedy
this is to introduce a global optimization strategy dur-
ing the pairing step, typically leveraging some learned
priors about matching, which SuperGlue and similar
methods successfully implemented [51,72]. However,
leveraging global context during matching might be too
late, if keypoints and their descriptors do not already
encode enough information. For this reason, another
direction is to consider dense holistic matching, i.e.
avoiding keypoints altogether, and matching the entire
image at once. This recently became possible with the
advent of mechanism for global attention [96]. Such
approaches, like LoFTR [82], thus consider images as a
whole and the resulting set of correspondences is dense
and more robust to repetitive patterns and low-texture
areas [43,68,69,82]. This led to new state-of-the-art
results on the most challenging benchmarks, such as
the Map-free localization benchmark [5].
Nevertheless, even a top-performing methods like
LoFTR [82] score a relatively disappointing VCRE pre-
cision of 34% on the Map-free localization benchmark.
We argue that this is because, so far, practically all
matching approaches have been treating matching as a
2D problem in image space. In reality, the formulation
of the matching task is intrinsically and fundamentally
a 3D problem: pixels that correspond are pixels that
observe the same 3D point. Indeed, 2D pixel corre-
spondences and a relative camera pose in 3D space are
two sides of the same coin, as they are directly related
by the epipolar matrix [36]. Another evidence is that
the current top-performer on the Map-free benchmark
is DUSt3R [102], a method initially designed for 3D
reconstruction rather than matching, and for which
matches are only a by-product of the 3D reconstruc-
tion. Yet, correspondences obtained naively from this
3D output currently outperform all other keypoint- and
matching-based methods on the Map-free benchmark.
In this paper, we point out that, while DUSt3R [102]
can indeed be used for matching, it is relatively im-
precise, despite being extremely robust to viewpoint
changes. To remedy this flaw, we propose to attach a
second head that regresses dense local feature maps,
and train it with an InfoNCE loss. The resulting ar-
chitecture, called MASt3R for “Matching And Stereo
3D Reconstruction” outperforms DUSt3R on multiple
benchmarks. To get pixel-accurate matches, we pro-
pose a coarse-to-fine matching scheme during which
matching is performed at several scales. Each matching
step involves extracting reciprocal matches from dense
feature maps which, perhaps counter-intuitively, is by
far more time consuming than computing the dense
feature maps themselves. Our proposed solution is a
faster algorithm for finding reciprocal matches that is
almost two orders of magnitude faster while improving
the pose estimation quality.
To summarize, we claim three main contributions. First,
we propose MASt3R, a 3D-aware matching approach
building on the recently released DUSt3R framework.
It outputs local feature maps that enable highly ac-
curate and extremely robust matching. Second, we
propose a coarse-to-fine matching scheme associated
with a fast matching algorithm, enabling to work with
high-resolution images. Third, MASt3R significantly
outperform the state-of-the-art on several absolute and
relative pose localization benchmarks.
Corresponding author(s): [vincent.leroy,jerome.revaud]@naverlabs.com

Grounding Image Matching in 3D with MASt3R
2. Related works
Keypoint-based matching has been a cornerstone of com-
puter vision. Matching is carried out in three distinct
stages: keypoint detection, locally invariant descrip-
tion and nearest-neighbor search in descriptor space.
Departing from the former handcrafted methods like
SIFT [52,71], modern approaches have been shifting
towards learning-based data-driven schemes for detect-
ing keypoints [8,60,97,117], describing them [7,33,
37,88] or both at the same time [10,21,53,54,70,98].
Overall, keypoint-based approaches are predominant
in many benchmarks [7,35,44,77], underscoring their
enduring value in tasks requiring high precision and
speed [19,77]. One notable issue, however, is they re-
duce matching to a local problem, i.e. discarding its
holistic nature. SuperGlue and similar approaches [51,
72] thus propose to perform global reasoning in the
last pairing step leveraging stronger priors to guide
matching, yet leaving the detection and description
local. While successful, it is still limited by the local na-
ture of keypoints and their inability to remain invariant
to strong viewpoint changes.
Dense matching.
In contrast to keypoint-based ap-
proaches, semi-dense [11,16,43,46,82,85] and dense
approaches [27,28,29,58,92,93,94,122] offer a differ-
ent paradigm for establishing image correspondences,
considering all possible pixel associations. Very reminis-
cent of optical flow approaches [22,40,42,79,80,86],
they are usually employing coarse-to-fine schemes to de-
crease computational complexity. Overall, these meth-
ods aim to consider matching from a global perspective,
at the cost of increased computational resources. Dense
matching has proven effective in scenarios where de-
tailed spatial relationships and textures are critical for
understanding scene geometry, leading to top perfor-
mance on many benchmarks [4, 5, 6, 59, 72, 82] that
are especially challenging for keypoints due to extreme
changes in viewpoint or illumination. These approaches
still cast matching as a 2D problem, which limits their
usage for visual localization.
Camera Pose estimation techniques vary widely, but
the most successful strategies, for speed, accuracy and
robustness trade-off, are fundamentally based on pixel
matching [73, 75, 105]. The constant improvement
of matching methods has fostered the introduction of
more challenging camera pose estimation benchmarks,
such as Aachen Day-Night, InLoc, CO3D or Map-free [5,
67,84,118], all featuring strong viewpoint and/or il-
lumination changes. The most challenging of them is
undoubtedly Map-free [5], a localization dataset for
which a single reference image is provided but no map,
with viewpoint changes up to 180◦.
Grounding matching in 3D thus becomes a crucial ne-
cessity in these challenging conditions where classical
2D-based matching utterly falls short. Leveraging priors
about the physical properties of the scene in order to im-
prove accuracy or robustness has been widely explored
in the past, but most previous works settle for leveraging
epipolar constraints for semi-supervised learning of cor-
respondences without any fundamental change [9,38,
47,101,108,111,114,120]. Toft et al. [89], on its part,
propose to improve keypoint descriptors by rectifying
images with perspective transformations obtained from
an off-the-shelf monocular depth predictor. Recently,
diffusion for pose [100] or rays [116], although not
matching approaches strictly speaking, show promising
performance by incorporating 3D geometric constraints
into their pose estimation formulation. Finally, the re-
cent DUSt3R [102] explore the possibility of recovering
correspondences from the a-priori harder task of 3D
reconstruction from uncalibrated images. Despite not
being trained explicitly for matching, this approach
yields promising results, topping the Map-free leader-
board [5]. Our contribution is to pursue this idea, by
regressing local features and explicitly training them
for pairwise matching.
3. Method
Given two images 𝐼1 and 𝐼2, respectively captured by
two cameras 𝐶1 and 𝐶2 with unknown parameters,
we wish to recover a set of pixel correspondences
{(𝑖, 𝑗)} where 𝑖, 𝑗are pixels 𝑖= (𝑢𝑖, 𝑣𝑖), 𝑗= (𝑢𝑗, 𝑣𝑗) ∈
{1, . . . , 𝑊}×{1, . . . , 𝐻}, 𝑊, 𝐻being the respective width
and height of the images. We assume they have the
same resolution for the sake of simplicity, yet without
loss of generality. The final network can handle pairs
of variable aspect ratios.
Our approach, illustrated in fig. 2, aims at jointly per-
forming 3D scene reconstruction and matching given
two input images. It is based on the DUSt3R framework
recently proposed by Wang et al. [102], which we first
review in section 3.1 before presenting our proposed
matching head and its corresponding loss in section 3.2.
We then introduce an optimized matching scheme spe-
cially devised to deal with dense feature maps in 3.3,
that we use for coarse-to-fine matching in section 3.4.
3.1. The DUSt3R framework
DUSt3R [102] is a recently proposed approach that
jointly solves the calibration and 3D reconstruction
problems from images alone. A transformer-based net-
work predicts a local 3D reconstruction given two input
images, in the form of two dense 3D point-clouds 𝑋1,1
and 𝑋2,1, denoted as pointmaps in the following.
3

Grounding Image Matching in 3D with MASt3R
ViT
encoder
ViT
encoder
Shared 
weights
Confidence 𝐶1,1
Pointmap 𝑋1,1
Confidence 𝐶2,1
Pointmap 𝑋2,1
𝐻1
𝐻2
Head3D
Headdesc
Head3D
Headdesc
Local features 𝐹2,1
Local features 𝐹1,1
1
1
2
2
Fast NN
Fast NN
Feature-based 
matching
Geometrical
matching
(𝐻× 𝑊× 𝑑)
(𝐻× 𝑊× 𝑑)
(𝐻× 𝑊× 3)
(𝐻× 𝑊× 3)
Transformer
Decoder
Transformer
Decoder
Cross-attention
Figure 2: Overview of the proposed approach. Given two input images to match, our network regresses for each
image and each input pixel a 3D point, a confidence value and a local feature. Plugging either 3D points or
local features into our fast reciprocal NN matcher (3.3) yields robust correspondences. Compared to the DUSt3R
framework which we build upon, our contributions are highlighted in blue.
A pointmap 𝑋𝑎,𝑏∈ℝ𝐻×𝑊×3 represents a dense 2D-to-3D
mapping between each pixel 𝑖= (𝑢, 𝑣) of the image 𝐼𝑎
and its corresponding 3D point 𝑋𝑎,𝑏
𝑢,𝑣∈ℝ3 expressed in
the coordinate system of camera 𝐶𝑏. By regressing two
pointmaps 𝑋1,1, 𝑋2,1 expressed in the same coordinate
system of camera 𝐶1, DUSt3R effectively solves the joint
calibration and 3D reconstruction problem. In the case
where more than two images are provided, a second
step of global alignment merges all pointmaps in the
same coordinate system. Note that, in this paper, we do
not make use of this step and restrict ourselves to the
binocular case. We now explain the inference in more
details.
Both images are first encoded in a Siamese manner
with a ViT [23], yielding two representations 𝐻1 and
𝐻2:
𝐻1 = Encoder(𝐼1),
(1)
𝐻2 = Encoder(𝐼2).
(2)
Then, two intertwined decoders process these repre-
sentations jointly, exchanging information via cross-
attention to ‘understand’ the spatial relationship be-
tween viewpoints and the global 3D geometry of the
scene. The new representations augmented with this
spatial information are denoted as 𝐻1 and 𝐻2:
𝐻′1, 𝐻′2 = Decoder(𝐻1, 𝐻2).
(3)
Finally,
two
prediction
heads
regress
the
final
pointmaps and confidence maps from the concatenated
representations output by the encoder and decoder:
𝑋1,1, 𝐶1 = Head1
3D([𝐻1, 𝐻′1]),
(4)
𝑋2,1, 𝐶2 = Head2
3D([𝐻2, 𝐻′2]).
(5)
Regression loss. DUSt3R is trained in a fully-supervised
manner using a simple regression loss
ℓregr(𝑣, 𝑖) =

1
𝑧𝑋𝑣,1
𝑖
−1
ˆ𝑧
ˆ𝑋𝑣,1
𝑖
 ,
(6)
where 𝑣∈{1, 2} is the view and 𝑖is a pixel for which
the ground-truth 3D point ˆ𝑋𝑣,1 ∈ℝ3 is defined. In
the original formulation, normalizing factors 𝑧, ˆ𝑧are
introduced to make the reconstruction invariant to scale.
These are simply defined as the mean distance of all
valid 3D points to the origin.
Metric predictions.
In this work, we note that scale
invariance is not necessarily desirable, as some poten-
tial use-cases like map-free visual localization necessi-
tates metric-scale predictions. Therefore, we modify
the regression loss to ignore normalization for the pre-
dicted pointmaps when the ground-truth pointmaps
are known to be metric.
That is, we set 𝑧:= ˆ𝑧
whenever ground-truth is metric, so that ℓregr(𝑣, 𝑖) =
||𝑋𝑣,1
𝑖
−ˆ𝑋𝑣,1
𝑖
||/ˆ𝑧in this case. As in DUSt3R [102], the
final confidence-aware regression loss is defined as
Lconf =
∑︁
𝑣∈{1,2}
∑︁
𝑖∈V𝑣
𝐶𝑣
𝑖ℓregr(𝑣, 𝑖) −𝛼log 𝐶𝑣
𝑖.
(7)
3.2. Matching prediction head and loss
To
obtain
reliable
pixel
correspondences
from
pointmaps, a standard solution is to look for reciprocal
matches in some invariant feature space [26,78,102,
106]. While such a scheme works remarkably well with
DUSt3R’s regressed pointmaps (i.e. in a 3-dimensional
space) even in presence of extreme viewpoint changes,
we note that the resulting correspondences are rather
imprecise, yielding suboptimal accuracy.
This is a
rather natural result as (i) regression is inherently
affected by noise, and (ii) because DUSt3R was never
explicitly trained for matching.
Matching head. For these reasons, we propose to add a
second head that outputs two dense feature maps 𝐷1
and 𝐷2 ∈ℝ𝐻×𝑊×𝑑of dimensional 𝑑:
𝐷1 = Head1
desc([𝐻1, 𝐻′1]),
(8)
𝐷2 = Head2
desc([𝐻2, 𝐻′2]).
(9)
4

Grounding Image Matching in 3D with MASt3R
We implement the head as a simple 2-layers MLP inter-
leaved with a non-linear GELU activation function [39].
Lastly, we normalize each local feature to unit norm.
More details can be found in the supplementary mate-
rial.
Matching objective. We wish to encourage each local
descriptor from one image to match with at most a
single descriptor from the other image that represents
the same 3D point in the scene. To that aim, we lever-
age the infoNCE [95] loss over the set of ground-truth
correspondences ˆ
M = {(𝑖, 𝑗)| ˆ𝑋1,1
𝑖
= ˆ𝑋2,1
𝑗
}:
Lmatch = −
∑︁
(𝑖,𝑗)∈ˆ
M
log
𝑠𝜏(𝑖, 𝑗)
Í
𝑘∈P1 𝑠𝜏(𝑘, 𝑗) + log
𝑠𝜏(𝑖, 𝑗)
Í
𝑘∈P2 𝑠𝜏(𝑖, 𝑘) ,
(10)
with 𝑠𝜏(𝑖, 𝑗) = exp
h
−𝜏𝐷1⊤
𝑖
𝐷2
𝑗
i
.
(11)
Here, P1 = {𝑖|(𝑖, 𝑗) ∈
ˆ
M} and P2 = { 𝑗|(𝑖, 𝑗) ∈
ˆ
M}
denote the subset of considered pixels in each image
and 𝜏is a temperature hyper-parameter. Note that
this matching objective is essentially a cross-entropy
classification loss: contrary to regression in eq. (6), the
network is only rewarded if it gets the correct pixel
right, not a nearby pixel. This strongly encourages the
network to achieve high-precision matching. Finally,
both regression and matching losses are combined to
get the final training objective:
Ltotal = Lconf + 𝛽Lmatch
(12)
3.3. Fast reciprocal matching
Given two predicted feature maps 𝐷1, 𝐷2 ∈ℝ𝐻×𝑊×𝑑, we
aim to extract a set of reliable pixel correspondences,
i.e. mutual nearest neighbors of each others:
M = {(𝑖, 𝑗) | 𝑗= NN2(𝐷1
𝑖) and 𝑖= NN1(𝐷2
𝑗)},
(13)
with NN𝐴(𝐷𝐵
𝑗) = arg min
𝑖
𝐷𝐴
𝑖−𝐷𝐵
𝑗
 .
(14)
Unfortunately, naive implementation of reciprocal
matching has a high computational complexity of
𝑂(𝑊2𝐻2), since every pixel from an image must be
compared to every pixels in the other image. While
optimizing the nearest-neighbor (NN) search is possi-
ble, e.g. using K-d trees [1], this kind of optimization
becomes typically very inefficient in high dimensional
feature space and, in all cases, orders of magnitude
slower than the inference time of MASt3R to output 𝐷1
and 𝐷2.
Fast matching. We therefore propose a faster approach
based on sub-sampling. It is based on an iterated pro-
cess that starts from an initial sparse set of 𝑘pixels
𝑈0 = {𝑈0
𝑛}𝑘
𝑛=1, typically sampled regularly on a grid in
the first image 𝐼1. Each pixel is then mapped to its NN
on 𝐼2, yielding 𝑉1, and the resulting pixels are mapped
back again to 𝐼1 in the same way:
𝑈𝑡↦−→[NN2(𝐷1
𝑢)]𝑢∈𝑈𝑡≡𝑉𝑡↦−→[NN1(𝐷2
𝑣)]𝑣∈𝑉𝑡≡𝑈𝑡+1
(15)
The set of reciprocal matches (those which form a cycle,
i.e. M𝑡
𝑘= {(𝑈𝑡
𝑛, 𝑉𝑡
𝑛) | 𝑈𝑡
𝑛= 𝑈𝑡+1
𝑛
}) are then collected. For
the next iteration, pixels that already converged are
filtered out, i.e. updating 𝑈𝑡+1 := 𝑈𝑡+1 \ 𝑈𝑡. Likewise,
starting from 𝑡= 1 we also verify and filter 𝑉𝑡+1, com-
paring it with 𝑉𝑡in a similar fashion. As illustrated in
fig. 3 (left), this process is then iterated a fixed number
of times, until most correspondences converge to stable
(reciprocal) pairs. In fig. 3 (center), we show that the
number of un-converged point |𝑈𝑡| rapidly decreases
to zero after a few iterations. Finally, the output set
of correspondences consists of the concatenation of all
reciprocal pairs M𝑘= Ð
𝑡M𝑡
𝑘.
Theoretical guarantees. The overall complexity of the
fast matching is 𝑂(𝑘𝑊𝐻), which is 𝑊𝐻/𝑘≫1 times
faster than the naive approach denoted all, as illustrated
in fig. 3 (right). It is worth pointing out that our fast
matching algorithm extracts a subset of the full set M,
which is bounded in size by |M𝑘| ≤𝑘. We study in the
supplementary material the convergence guarantees
of this algorithm and how it evinces outlier-filtering
properties, which explains why the end accuracy is
actually higher than when using the full correspondence
set M, see fig. 3 (right).
3.4. Coarse-to-fine matching
Due to the quadratic complexity of attention w.r.t. the
input image area (𝑊× 𝐻), MASt3R only handles images
of 512 pixels in their largest dimension. Larger images
would require significantly more compute power to
train, and ViTs do not generalize yet to larger test-time
resolutions [62,65]. As a result, high-resolution images
(e.g. 1M pixel) needs to be downscaled to be matched,
afterwards the resulting correspondences are upscaled
back to the original image resolution. This can lead to
some performance loss, sometimes sufficient to cause
substantial degradation in term of localization accuracy
or reconstruction quality.
Coarse-to-fine matching is a standard technique to pre-
serve the benefit of matching high-resolution images
with a lower-resolution algorithm [66, 86]. We thus
explore this idea for MASt3R. Our procedure starts
with performing matching on downscaled versions of
the two images. We denote the set of coarse corre-
spondences obtained with subsampling 𝑘as M0
𝑘. Next,
5

Grounding Image Matching in 3D with MASt3R
𝑈0
𝑉1
Image 1
Image 2
𝑈1
𝑁𝑁2(𝑈0)
𝑁𝑁1(𝑉1)
=?
𝑉2
𝑈2
=?
𝑉3
𝑈3
=?
=?
=?
…
0
1
2
3
4
5
6
Iteration step t
0
1000
2000
3000
Average number of
active correspondences
Fast reciprocal matching
0.03 0.1
0.3
1
3
10
30
100
Time on a single CPU core (seconds)
0.71
0.72
0.73
0.74
0.75
AUC (VCRE < 90px)
all
all
k=49K
k=49K
k=12K
k=12K
k=3K
k=3K
k=768
k=768
64x faster
Matching time vs. perf. on Mapfree
3D points
Local feat
Figure 3: Fast reciprocal matching. Left: Illustration of the fast matching process, starting from an initial subset
of pixels 𝑈0 and propagating it iteratively using 𝑁𝑁search. Searching for cycles (blue arrows) detect reciprocal
correspondences and allows to accelerate the subsequent steps, by removing points that converged. Center:
Average number of remaining points in 𝑈𝑡at iteration 𝑡= 1 . . . 6. After only 5 iterations, nearly all points have
already converged to a reciprocal match. Right: Performance-versus-time trade-off on the Map-free dataset.
Performance actually improves, along with matching speed, when performing moderate levels of subsampling.
we generate a grid of overlapping window crops 𝑊1
and 𝑊2 ∈ℝ𝑤×4 on each full-resolution image inde-
pendently. Each window crop measures 512 pixels in
its largest dimension and contiguous windows overlap
by 50%. We can then enumerate the set of all win-
dow pairs (𝑤1, 𝑤2) ∈𝑊1 × 𝑊2, from which we select
a subset covering most of the coarse correspondences
M0
𝑘. Specifically, we add window pairs one by one in a
greedy fashion until 90% of correspondences are cov-
ered. Finally, we perform matching for each window
pair independently:
𝐷𝑤1, 𝐷𝑤2 = MASt3R(𝐼1
𝑤1, 𝐼2
𝑤2)
(16)
M𝑤1,𝑤2
𝑘
= fast_reciprocal_NN(𝐷𝑤1, 𝐷𝑤2)
(17)
Correspondences obtained from each window pair are
finally mapped back to the original image coordinates
and concatenated, thus providing dense full-resolution
matches.
4. Experimental results
We detail in section 4.1 the training procedure of
MASt3R. Then, we evaluate on several tasks, each time
comparing with the state of the art, starting with vi-
sual camera pose estimation on the Map-Free Relocal-
ization Benchmark [5] (section 4.2), the CO3D and
RealEstate datasets (section 4.3) and other standard Vi-
sual Localization benchmarks in section 4.4. Finally, we
leverage MASt3R for Dense Multi-View Stereo (MVS)
reconstruction in section 4.5.
4.1. Training
Training data . We train our network with a mixture of
14 datasets: Habitat [74], ARKitScenes [20], Blended
MVS [112], MegaDepth [48], Static Scenes 3D [57],
ScanNet++ [113], CO3D-v2 [67], Waymo [83], Map-
free [5], WildRgb [2], VirtualKitti [12], Unreal4K [91],
TartanAir [103] and an internal dataset. These datasets
feature diverse scene types:
indoor, outdoor, syn-
thetic, real-world, object-centric, etc. Among them,
10 datasets have metric ground-truth. When image
pairs are not directly provided with the dataset, we
extract them based on the method described in [104].
Specifically, we utilize off-the-shelf image retrieval and
point matching algorithms to match and verify image
pairs.
Training. We base our model architecture on the public
DUSt3R model [102] and use the same backbone (ViT-
Large encoder and ViT-Base decoder). To benefit the
most from DUSt3R’s 3D matching abilities, we initial-
ize the model weights to the publicly available DUSt3R
checkpoint. During each epoch, we randomly sample
650k pairs equally distributed between all datasets. We
train our network for 35 epoch with a cosine schedule
and initial learning rate set to 0.0001. Similar to [102],
we randomize the image aspect ratio at training time,
ensuring that the largest image dimension is 512 pixels.
We set the local feature dimension to 𝑑= 24 and the
matching loss weight to 𝛽= 1. It is important that
the network sees different scales at training time, be-
cause coarse-to-fine matching starts from zoomed-out
images to then zoom-in on details (see section 3.4). We
therefore perform aggressive data augmentation during
training in the form of random cropping. Image crops
are transformed with a homography to preserve the
central position of the principal point.
Correspondence sampling. To generate ground-truth cor-
respondences necessary for the matching loss (eq. (10)),
we simply find reciprocal correspondences between on
the ground-truth 3D pointmaps ˆ𝑋1,1 ↔ˆ𝑋2,1. We then
6

Grounding Image Matching in 3D with MASt3R
randomly subsample 4096 correspondences per image
pairs. If we cannot find enough correspondences, we
pad with random false correspondences so that the
likelihood of finding a true match remains constant.
Fast nearest neighbors. For the fast reciprocal matching
from section 3.3, we implement the nearest neighbor
function NN(𝑥) from eq. (14) differently depending on
the dimension of 𝑥. When matching 3D points 𝑥∈ℝ3,
we implement NN(𝑥) using K-d trees [56]. For match-
ing local features with 𝑑= 24, however, K-d trees be-
come highly inefficient due to the curse of dimension-
ality [25]. Therefore, we rely on the optimized FAISS
library [24,45] in this case.
4.2. Map-free localization
Dataset description. We start our experiments with the
Map-free relocalization benchmark [5], an extremely
challenging dataset aiming at localizing the camera in
metric space given a single reference image without any
map. It comprises a training, validation and test sets
of 460, 65 and 130 scenes resp., each featuring two
video sequences. Following the benchmark, we evaluate
in term of Virtual Correspondence Reprojection Error
(VCRE) and camera pose accuracy, see [5] for details.
Impact of subsampling. We do not resort to coarse-to-
fine matching for this dataset, as the image resolution is
already close to MASt3R working resolution (720×540
vs. 512 × 384 resp.). As mentioned in section 3.3, com-
puting dense reciprocal matching is prohibitively slow
even with optimized code for searching nearest neigh-
bors. We therefore resort to subsampling the set of
reciprocal correspondences, keeping at most 𝑘corre-
spondences from the complete set M (eq. (13)). fig. 3
(right) shows the impact of subsampling in term of AUC
(VCRE) performance and timing. Surprisingly, the per-
formance significantly improves for intermediate values
of subsampling. Using 𝑘= 3000, we can accelerate
matching by a factor of 64 while significantly improv-
ing the performance. We provide insights in the supple-
mentary material regarding this phenomenon. Unless
stated otherwise, we keep 𝑘= 3000 for subsequent
experiments.
Ablations on losses and matching modes.
We report
results on the validation set in table 1 for different vari-
ants of our approach: DUSt3R matching 3D points (I);
MASt3R also matching 3D points (II) or local features
(III, IV, V). For all methods, we compute the relative
pose from the essential matrix [36] estimated with the
set of predicted matches (PnP performs similarly). The
metric scene scale is inferred from the depth extracted
with an off-the-shelf DPT finetuned on KITTI [65] (I-IV)
or from the depth directly output by MASt3R (V).
First, we note that all proposed methods significantly
outperforms the DUSt3R baseline, probably because
MASt3R is trained longer and with more data. All other
things being equal, matching descriptors perform sig-
nificantly better than matching 3D points (II versus IV).
This confirms our initial analysis that regression is in-
herently unsuited to compute pixel correspondences,
see section 3.2.
We also study the impact of training only with a sin-
gle matching objective (Lmatch from eq. (10), III). In
this case, the performance overall degrades compared
to training with both 3D and matching losses (IV), in
particular in term of pose estimation accuracy (e.g. me-
dian rotation of 10.8◦for (III) compared to 3.0◦for
(IV)). We point out that this is in spite of the decoder
now having more capacity to carry out a single task,
instead of two when performing 3D reconstruction si-
multaneously, indicating that grounding matching in
3D is indeed crucial to improve matching. Lastly, we
observe that, when using metric depth directly output
by MASt3R, the performance largely improves. This
suggests that, as for matching, the depth prediction task
is largely correlated with 3D scene understanding, and
that the two tasks strongly benefit from each other.
Comparisons on the test set is reported in table 2. Over-
all, MASt3R outperforms all state-of-the-art approaches
by a large margin, achieving more than 93% in VCRE
AUC. This is a 30% absolute improvement compared to
the second best published method, LoFTR+KBR [81,
82], that get 63.4% in AUC. Likewise, the median trans-
lation error is vastly reduced to 36cm, compared to ap-
prox. 2m for the state-of-the-art methods. A large part
of the improvement is of course due to MASt3R predict-
ing metric depth, but note that our variant leveraging
depth from DPT-KITTI (thus purely matching-based)
outperforms all state-of-the-art approaches as well.
We also provide the results of direct regression with
MASt3R, i.e. without matching, simply using PnP on
the pointmap 𝑋2,1 of the second image. These results
are surprisingly on par with our matching-based vari-
ant, even though the ground-truth calibration of the
reference camera is not used. As we show below, this
does not hold true for other localization datasets, and
computing the pose via matching (e.g. with PnP or es-
sential matrix) with known intrinsics seems safer in
general.
Qualitative results. We show in fig. 4 some matching
results for pairs with strong viewpoint change (up to
180◦). We also highlight with insets some specific re-
7

Grounding Image Matching in 3D with MASt3R
Table 1: Results on the validation set of the Map-free dataset. (First and second best)
match
VCRE (<90px)
Pose Error
depth
Reproj. ↓
Prec. ↑
AUC ↑
Med. Err. (m,°) ↓
Precision ↑
AUC ↑
(I)
DUSt3R
3d
DPT
125.8 px
45.2%
0.704
1.10m
9.4°
17.0%
0.344
(II)
MASt3R
3d
DPT
112.0 px
49.9%
0.732
0.94m
3.6°
21.5%
0.409
(III)
MASt3R-M
feat
DPT
107.7 px
51.7%
0.744
1.10m
10.8°
19.3%
0.382
(IV)
MASt3R
feat
DPT
112.9 px
51.5%
0.752
0.93m
3.0°
23.2%
0.435
(V)
MASt3R
feat
(auto)
57.2 px
75.9%
0.934
0.46m
3.0°
51.7%
0.746
Table 2: Comparison with the state of the art on the test set of the Map-free dataset.
VCRE (<90px)
Pose Error
depth
Reproj. ↓
Prec. ↑
AUC ↑
Med. Err. (m,°) ↓
Precision ↑
AUC ↑
RPR [5]
DPT
147.1 px
40.2%
0.402
1.68m
22.5°
6.0%
0.060
SIFT [52]
DPT
222.8 px
25.0%
0.504
2.93m
61.4°
10.3%
0.252
SP+SG [72]
DPT
160.3 px
36.1%
0.602
1.88m
25.4°
16.8%
0.346
LoFTR [82]
KBR
165.0 px
34.3%
0.634
2.23m
37.8°
11.0%
0.295
DUSt3R [102]
DPT
116.0 px
50.3%
0.697
0.97m
7.1°
21.6%
0.394
MASt3R
DPT
104.0 px
54.2%
0.726
0.80m
2.2°
27.0%
0.456
MASt3R
(auto)
48.7 px
79.3%
0.933
0.36m
2.2°
54.7%
0.740
MASt3R (direct reg.)
53.2 px
79.1%
0.941
0.42m
3.1°
53.0%
0.777
gions that are correctly matched by MASt3R in spite
of drastic appearance changes. We believe these cor-
respondences to be nearly impossible to get with 2D-
based matching methods. In contrast, grounding the
matching in 3D allows to solve the issue relatively
straightforwardly.
4.3. Relative pose estimation
Datasets and protocol.
Next, we evaluate for the
task of relative pose estimation on the CO3Dv2 [67]
and RealEstate10k [121] datasets. CO3Dv2 contains
6 million frames extracted from approximately 37k
videos, covering 51 MS-COCO categories.
Ground-
truth camera poses are obtained using COLMAP [75]
from 200 frames in each video. RealEstate10k is an
indoor/outdoor dataset that features 80K video clips
on YouTube totalling 10 million frames, camera poses
being obtained via SLAM with bundle adjustment. Fol-
lowing [100], we evaluate MASt3R on 41 categories
from CO3Dv2 and 1.8K video clips from the test set of
RealEstate10k. Each sequence is 10 frames long, we
evaluate relative camera poses between all possible 45
pairs, not using ground-truth focals.
Baselines and metrics. As before, matches obtained with
MASt3R are used to estimate Essential Matrices and
relative pose. Please note that our predictions are al-
ways done pairwise, contrary to all other methods that
leverage multiple views (at the exception of DUSt3R-
PnP). We compare to recent data-driven approaches like
RelPose [115], RelPose++ [115], PoseReg and PoseD-
iff [100], the recent RayDiff [116] and DUSt3R [102].
We also report results for more traditional SfM methods
like PixSFM [50] and COLMAP [76] extended with Su-
perPoint [21] and SuperGlue [72] (COLMAP+SPSG).
Similar to [100], we report the Relative Rotation Ac-
curacy (RRA) and Relative Translation Accuracy (RTA)
for each image pair to evaluate the relative pose er-
ror and select a threshold 𝜏= 15 to report RTA@15
and RRA@15. Additionally, we calculate the mean
Average Accuracy (mAA30), defined as the area un-
der the accuracy curve of the angular differences at
𝑚𝑖𝑛(RRA@30, RTA@30).
Results. As shown in table 3, SfM approaches tend to
perform significantly worse on this task, mainly due to
the poor visual support. This because images usually
observe a small object, combined with the fact that
many pairs have a wide baseline, sometimes up to 180◦.
On the contrary, 3D grounded approaches like RayDiffu-
sion, DUSt3R and MASt3R are the two most competitive
methods on this dataset, the latter leading in transla-
tion and mAA on both datasets. Notably, on RealEstate
our mAA score improves by at least 8.7 points over the
best multi-view methods and 15.2 points over pairwise
DUSt3R. This showcases the accuracy and robustness
of our approach to few input view setups.
4.4. Visual localization
Datasets. We then evaluate MASt3R for the task of ab-
solute pose estimation on the Aachen Day-Night[118]
and InLoc[84] datasets. Aachen comprises 4,328 refer-
ence images taken with hand-held cameras, as well as
824 daytime and 98 nighttime query images taken with
8

Grounding Image Matching in 3D with MASt3R
Figure 4: Qualitative examples on the Map-free dataset. Top row: Pairs with strong viewpoint changes. Third one
is a failure case. For clarity, we only draw a subset of all correspondences. Bottom row: We highlight interesting
spots in close-up. These regions could hardly be matched by local keypoints. See text for details.
mobile phones in the old inner city of Aachen, Germany.
InLoc[84] is an indoor dataset with challenging appear-
ance variation between the 9,972 RGB-D + 6DOF pose
database images and the 329 query images taken from
an iPhone 7.
Metrics. We report report the percentage of successfully
localized images within three thresholds: (0.25m, 2°),
(0.5m, 5°) and (5m, 10°) for Aachen and (0.25m, 10°),
(0.5m, 10°), (1m, 10°) for InLoc.
Results are reported in Table 4. We study the perfor-
mance of MASt3R with variable number of retrieved
images. As expected, a greater number of retrieved
images (top40) yields better performance, achieving
competitive performance on Aachen and significantly
outperforming the state of the art on InLoc. Interest-
ingly, our approach still performs very well even with a
single retrieved image (top1), showcasing the robust-
ness of 3D grounded matching. We also include direct
regression results, which are rather poor, showing a
striking impact of the dataset scale on the localization
error, i.e. small scenes are much less affected (see re-
sults on Map-free in 4.2). This confirms the importance
of feature matching to estimate reliable poses.
4.5. Multiview 3D reconstruction
We finally perform MVS by triangulating the obtained
matches. Note that the matching is performed in full
resolution without prior knowledge of cameras, and the
latter are only used to triangulate matches in ground-
truth reference frame. We remove spurious 3D points
via geometric consistency post-processing [99].
Datasets and metrics. We evaluate our predictions on
the DTU [3] dataset. Contrary to all competing learning
methods, we apply our network in a zero-shot setting,
i.e. we do not train nor finetune on the DTU train set and
apply our model as is. In table 3 we report the average
accuracy, completeness and Chamfer distances error
metrics as provided by the authors of the benchmarks.
The accuracy for a point of the reconstructed shape
is defined as the smallest Euclidean distance to the
ground-truth, and the completeness of a point of the
ground-truth as the smallest Euclidean distance to the
reconstructed shape. The overall Chamfer distance is
the average of both previous metrics.
Results. Data-driven approaches trained on this domain
significantly outperform handcrafted ones, cutting the
Chamfer error by half. To the best of our knowledge,
we are the first to draw such conclusion in a zero-shot
setting. MASt3R not only outperforms the DUSt3R
baseline but also compete with the best methods, all
without leveraging camera calibration nor poses for
matching, neither having seen this camera setup before.
9

Grounding Image Matching in 3D with MASt3R
Table 3: Left: Multi-view pose regression on the CO3Dv2 [67] and RealEstate10K [121] with 10 random frames.
Parenthesis () denote methods that do not report results on the 10 views set, we report their best for comparison
(8 views). We distinguish between (a) multi-view and (b) pairwise methods. Right: Dense MVS results on the
DTU dataset, in mm. Handcrafted methods (c) perform worse than learning-based approaches (d) that train on
this specific domain. Among the methods that operate in a zero-shot setting (e), MASt3R is the only one attaining
reasonable performance.
Methods
Co3Dv2
RealEstate10K
RRA@15
RTA@15
mAA(30)
mAA(30)
(a)
Colmap+SG [21,72]
36.1
27.3
25.3
45.2
PixSfM [50]
33.7
32.9
30.1
49.4
RelPose [115]
57.1
-
-
-
PosReg [100]
53.2
49.1
45.0
-
PoseDiff [100]
80.5
79.8
66.5
48.0
RelPose++ [49]
(85.5)
-
-
-
RayDiff [116]
(93.3)
-
-
-
DUSt3R-GA [102]
96.2
86.8
76.7
67.7
(b)
DUSt3R [102]
94.3
88.4
77.2
61.2
MASt3R
94.6
91.9
81.8
76.4
Methods
Acc.↓
Comp.↓
Overall↓
(c)
Camp [13]
0.835
0.554
0.695
Furu [31]
0.613
0.941
0.777
Tola [90]
0.342
1.190
0.766
Gipuma [32]
0.283
0.873
0.578
(d)
MVSNet [110]
0.396
0.527
0.462
CVP-MVSNet [109]
0.296
0.406
0.351
UCS-Net [17]
0.338
0.349
0.344
CER-MVS [55]
0.359
0.305
0.332
CIDER [107]
0.417
0.437
0.427
PatchmatchNet [99]
0.427
0.277
0