# G-MASt3R-SfM: Graph-based View Pruning and Multi-stage Optimization for Robust SfM

> 2026 · id: arxiv:2606.22856 · arXiv: 2606.22856 · pdf: https://arxiv.org/pdf/2606.22856 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Structure from Motion (SfM) is essential for multi-view 3D re-
construction, however, its accuracy heavily relies on the accuracy
of image matching.
While the recent correspondence matching
method, MASt3R, enables robust matching even under challenging
conditions, it tends to generate incorrect correspondences for non-
overlapping image pairs. Consequently, existing SfM methods using
MASt3R, such as MASt3R-SfM, suffer from signiﬁcant degradation
in pose estimation accuracy as they incorporate these unreliable
matches directly into optimization. To address this issue, we pro-
pose G-MASt3R-SfM, a novel SfM pipeline that enhances robust-
ness through two key modules. First, the Graph-based View Pruning
(GVP) module constructs a scene graph from matching conﬁdence
and geometrically prunes outlier views. Second, the Multi-Stage Op-
timization (MSO) module progressively reﬁnes camera parameters
by expanding the optimization scope from local consistency to the
global consistency. Experiments on the ETH3D dataset demonstrate
that our method achieves state-of-the-art accuracy in both camera
pose estimation and 3D reconstruction, effectively suppressing noise
caused by outliers.
Index Terms— structure from motion, 3D reconstruction, 3D
foundation model, scene graph, view pruning

## introduction
Multi-view 3D reconstruction, which recovers the 3D shape of a
target from multiple images captured by cameras, has been widely
adopted in applications such as digital archiving of cultural heritage
and 3D map creation for autonomous driving due to its simplicity
and low cost. To recover 3D shapes with high accuracy, it is essen-
tial to accurately estimate camera parameters, such as camera pose
and focal length, from the multiple images. Structure from Motion
(SfM) [1] serves as the standard approach for this purpose.
Re-
cent advancements in deep learning have led to signiﬁcant improve-
ments in the accuracy of multi-view 3D reconstruction methods [2].
However, most of these methods assume that camera parameters are
known; consequently, their reconstruction accuracy relies heavily on
the estimation accuracy of camera parameters by SfM.
COLMAP [3], the de facto standard SfM tool, employs detector-
based image matching [4]. While localization accuracy for feature
points is high, these methods often struggle to ﬁnd sufﬁcient corre-
spondences in texture-less regions or across images with signiﬁcant
illumination changes. In contrast, deep learning-based methods such
as MASt3R [5] enable dense and robust matching even with lim-
ited overlap, leading to applications in SfM like MASt3R-SfM [6].
However, applying MASt3R to SfM presents several challenges. Al-
though MASt3R is powerful in image correspondence between two
images, it tends to output correspondences even for non-overlapping
image pairs. Since MASt3R-SfM incorporates such unreliable cor-
respondence results into camera parameter estimation without ﬁlter-
ing, it often fails to achieve global consistency during bundle adjust-
ment, resulting in signiﬁcant accuracy degradation. Furthermore,
the dense point clouds output by MASt3R can have lower localiza-
tion accuracy than feature-based methods. Thus, as the number of
views increases, maintaining geometric consistency between views
becomes difﬁcult, further reducing estimation accuracy. To achieve
accurate SfM while leveraging MASt3R’s strengths, it is essential
to appropriately ﬁlter view connectivity for camera parameter esti-
mation and employ an optimization strategy that ensures multi-view
consistency.
In this paper, we propose G-MASt3R-SfM, a novel camera pa-
rameter estimation method utilizing a scene graph and multi-stage
optimization to enhance the accuracy and robustness of MASt3R-
based SfM. Our method consists of two main modules. First, the
Graph-based View Pruning (GVP) module constructs a scene graph
where nodes represent views based on MASt3R’s conﬁdence scores,
and prunes outlier views and erroneous connection groups by graph
structure analysis. This module effectively eliminates inappropriate
views from the estimation process. Second, the Multi-Stage Opti-
mization (MSO) module reﬁnes camera parameters by performing
bundle adjustment on the selected views, progressively expanding
the optimization scope from local consistency to global consistency.
Experiments on the ETH3D dataset [7] demonstrate that our method
achieves higher accuracy and stability in camera parameter estima-
tion across diverse scenes compared to existing methods.

## method
[deg.]
[deg.]
[%]
[%]
COLMAP [3]
0.655
2.645
90.7
87
DFSfM [11]
2.298
3.711
68.0
85
VGGSfM [13]
22.439 17.220
52.7
98
VGGT [17]
2.485
8.806
35.4
100
MASt3R-SfM [6]
2.572
3.343
75.7
100
G-MASt3R-SfM (Ours)
0.474
0.978
93.9
97
ber of input images for depth map estimation to 10 and the depth
conﬁdence threshold to 0.5. Additionally, we employ DPCD [23]
for point cloud denoising. To ensure a fair comparison, we resize
input images to 640 × 1, 024 pixels for all methods and scale the
camera focal lengths accordingly. Furthermore, only the views for
which camera parameters were successfully estimated in Sect. 4.4
are used for 3D reconstruction. Reconstruction accuracy is evaluated
using the ofﬁcial ETH3D evaluation script after aligning the gener-
ated point clouds to the ground truth using the Iterative Closest Point
(ICP) algorithm.
Table 3 presents the quantitative results. Although COLMAP
achieves the highest “Acc.”, its low SfM rate limits the number
of views available for reconstruction, resulting in lower “Cpl.”.
In contrast, our proposed method records the highest “Cpl.” and
also outperforms MASt3R-SfM in terms of “Acc.”. Consequently,
our method achieves the best performance in “F1 score”, which
serves as the comprehensive metric. This result indicates that our
method accurately estimates camera parameters for each view while
maintaining a sufﬁcient number of views.
Fig. 6 shows examples of 3D reconstruction results by each
method. In “delivery area” and “electro”, MASt3R-SfM
generates noise in regions where no objects exist, due to incorrect
camera pose estimation. In contrast, since our method appropriately
removes views with large errors by GVP, such noise is not observed,
Table 3. Quantitative results of MVS on the ETH3D dataset. The
best results are highlighted in bold.

## related_work
In this section, we give a brief overview of camera parameter estima-
tion from images, focusing on SfM and rapidly evolving 3D founda-
tion models.
Structure from Motion (SfM) — SfM is a technique that simulta-
neously estimates 3D structure and camera parameters using cor-
respondences between images.
COLMAP [3], the standard SfM
pipeline, relies on detector-based matching [8,9] such as SIFT [10].
However, it faces challenges in texture-less regions or scenes with
signiﬁcant illumination changes where sufﬁcient correspondences
cannot be obtained, leading to degraded estimation accuracy. In con-
trast, DFSfM [11] improves robustness by incorporating detector-
free dense matching methods [12] into SfM. However, dense geo-
metric information is lost during the process of aggregating corre-
spondences for computational efﬁciency. VGGSfM [13] employs
a tracker that simultaneously estimates correspondences across all
views; however, tracking becomes difﬁcult between images with
limited overlap, thereby restricting applicable scenes.
3D Foundation Models — Recently, foundation models trained on
large-scale 3D datasets have been applied to various tasks, includ-
ing camera parameter estimation. DUSt3R [14] estimates relative
arXiv:2606.22856v1  [cs.CV]  22 Jun 2026

MASt3R
Input image pairs
Camera
parameters
3D point cloud
Global Alignment (GA)
: Frozen
: Community
: Pruned view
Confidence maps
Correspondence
point pairs
Point maps
Graph-based View Pruning (GVP)
Scene graph
Grouped graph
Refined graph
Graph construction
Graph partitioning
View pruning
Multi-Stage Optimization (MSO)
Fig. 1. The overall pipeline of the proposed G-MASt3R-SfM. The modules highlighted in red (GVP and MSO) represent our novel contribu-
tions to the standard MASt3R-SfM pipeline. Utilizing correspondences and conﬁdence maps from MASt3R, the GVP module constructs a
scene graph to ﬁlter out unreliable views, followed by the MSO module which reﬁnes camera parameters.
camera poses without explicit matching by directly regressing point
maps from image pairs. MASt3R [5] extends DUSt3R to achieve
more accurate image matching and geometric estimation. However,
since these methods are fundamentally based on pairwise inference,
post-processing such as global optimization is indispensable to guar-
antee multi-view consistency. To address this issue, MUSt3R [15]
and Spann3R [16] attempt to extend these models to multi-view set-
tings by processing input images sequentially. Similarly, VGGT [17]
learns multi-view geometric relationships using Transformers. How-
ever, these methods prioritize computational efﬁciency and often
lack the rigorous geometric veriﬁcation inherent to standard SfM
pipelines. Consequently, as the number of views increases, estima-
tion errors accumulate, leading to a loss of global consistency.
3. G-MAST3R-SFM
As mentioned above, to leverage the high matching capability of
MASt3R [5] while preventing SfM failure caused by incorrect corre-
spondences and outliers, it is effective to organize view connectivity
and perform stepwise optimization. In this paper, we introduce the
GVP module and the MSO module to MASt3R-SfM [6]. Fig. 1 il-
lustrates the overall pipeline of our proposed G-MASt3R-SfM. First,
we construct a scene graph based on MASt3R outputs and prune ge-
ometrically inconsistent groups of views by graph structure analy-
sis. Next, for the selected views, we reﬁne camera parameters while
maintaining global consistency by progressively expanding the op-
timization scope based on the community structure of the graph. In
the following, we describe the overview of MASt3R-SfM and the
details of two proposed modules.
3.1. MASt3R-SfM
MASt3R-SfM [6] integrates dense point clouds and correspondence
information output by MASt3R [5] to perform SfM based on global
optimization. The process primarily consists of two stages: Global
alignment and bundle adjustment. First, the Global Alignment (GA)
module aligns the point maps of each view to a common world coor-
dinate system. For all view pairs (n, m) ∈E, the scale and extrinsic
parameters of each view are optimized to minimize the 3D position
error epos of the set of correspondence pairs Mn,m. Let cp be the
conﬁdence of correspondence pair p ∈Mn,m, and Xp
n be the 3D
coordinate of p in view n. The error epos is deﬁned by
epos =
X
(n,m)∈E
X
p∈Mn,m
cp ∥Xp
n −Xp
m∥

.
(1)
Next, all parameters are optimized by bundle adjustment. Using the
results of the GA module as initialization, the intrinsic and extrinsic
parameters, as well as depth maps for each view, are optimized to
minimize the reprojection error erep across all view pairs. Let πn be
the projection function mapping a 3D point in the world coordinate
system to the image plane of view n, and xp
n be the 2D coordinate
of the corresponding point p in view n. The error erep is given by
erep =
X
(n,m)∈E
X
p∈Mn,m
cp ∥xp
n−πn(Xp
m)∥+∥xp
m−πm(Xp
n)∥

.
(2)
3.2. Graph-based View Pruning (GVP) Module
Since MASt3R [5] may output correspondences even for non-
overlapping image pairs, it can degrade SfM accuracy. The GVP
module constructs a scene graph based on geometric veriﬁcation
and removes outlier views through graph structure analysis.
We
construct a scene graph where nodes represent images and edges
represent the connection strength between them. Rather than us-
ing MASt3R outputs directly, we ﬁlter for geometrically consistent
correspondences to calculate edge weights. Speciﬁcally, for each
pair (n, m), we estimate the relative pose using the focal length
derived from MASt3R’s point maps and the fundamental matrix
computed by RANSAC [18]. Using this relative pose, we reproject
points and retain only those falling within the image plane as inliers.
Finally, a reliable graph G is constructed by adding edges only for
pairs where the sum of inlier conﬁdences cp exceeds a threshold,
which we set to 1,000 in the experiments. We apply the Louvain
method [19] to the constructed scene graph G to partition nodes
into densely connected communities {C1, C2, . . . }. We assume that
outlier views deviating from the scene form small communities that
are either isolated or weakly connected to the main component. To
identify them, we embed the graph into a 2D plane using the Spring
Layout [20], a force-directed algorithm, and evaluate the separation
of each community. Let dist(u, v) be the distance between nodes u
and v in the 2D plane, and Scale be the diagonal length of the entire

node distribution area. The separation score si for community Ci is
deﬁned by
si = minu∈Ci,v /∈Ci dist(u, v)
Scale
·
1
log(1 + |Ci|),
(3)
where |Ci| denotes the size of the community. This score si in-
creases as the community becomes more distant from others and
smaller in size. In this paper, communities with si > 1.5 are con-
sidered outliers and removed from the graph.
3.3. Multi-Stage Optimization (MSO) Module
As shown in Fig. 2, after initialization by the GA module on the se-
lected views, we perform iterative bundle adjustment in three stages
(Local, Neighbor, and Global) utilizing the graph’s community
structure. Instead of optimizing all views simultaneously from the
start, we stabilize solution convergence by gradually expanding the
optimization scope from local to global consistency. Let i be the cur-
rent optimization iteration and j be the repetition count within each
stage. The optimization at each stage is as follows:
Local: Optimization is performed independently for each commu-
nity Ci using only internal views. This establishes local geometric
consistency for each densely connected group of views.
Neighbor: To enhance consistency between communities, we opti-
mize each community Ci together with views included in its adja-
cent communities Adj(Ci), where Adj(Ci) refers to all communi-
ties containing nodes connected by edges to nodes within Ci. To
reduce computational cost, if the number of target views exceeds
half the total number of views N, we select round(N/2) views via
uniform sampling.
Global: Bundle adjustment is performed using all views.
Optimization in each stage is performed by minimizing the re-
projection error erep in Eq. (2). For convergence criteria, we mon-
itor the ratio ej between the average error of the last k iterations
(k = 5) and the preceding l iterations (l = 10), repeating until
ej < δ (δ = 0.01). Upon satisfying the convergence condition in
the Global stage, the process returns to the Local stage and repeats.
The process terminates when the total optimization count i reaches
the upper limit imax.
4. EXPERIMENTS AND DISCUSSION
In this section, we present experiments to demonstrate the effective-
ness of our proposed G-MASt3R-SfM.
4.1. Experimental Settings
The experimental conditions are summarized below.
Dataset — We use the ETH3D dataset [7], which consists of multi-
view images and camera parameters captured in 25 indoor and out-
door scenes. ETH3D provides highly accurate camera parameters
as ground truth, obtained by reﬁning COLMAP [3] es

## conclusion
In this paper, we proposed G-MASt3R-SfM, a novel approach
leveraging scene graphs to enhance the accuracy and robustness of
MASt3R-based SfM. By integrating the GVP and MSO modules,
our method effectively ﬁlters geometrically inconsistent views and
optimizes camera parameters through community-aware stepwise
bundle adjustment.
Experiments on the ETH3D dataset demon-
strated that our method achieves signiﬁcantly improved performance
in both camera pose estimation and 3D reconstruction quality com-
pared to COLMAP and existing deep learning baselines. In future
work, we will focus on reﬁning the view selection algorithm and
extending this framework to other parameter estimation tasks.
6. ACKNOWLEDGMENT
This work was supported in part by JSPS KAKENHI 23H00463 and
25K03131.