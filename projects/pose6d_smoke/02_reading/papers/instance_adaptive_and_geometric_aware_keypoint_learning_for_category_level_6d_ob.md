# Instance-Adaptive and Geometric-Aware Keypoint Learning for Category-Level 6D Object Pose Estimation

> 2024 · id: W4402716422 · arXiv: 2403.19527 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Category-level 6D object pose estimation aims to es-
timate the rotation, translation and size of unseen in-
stances within specific categories.
In this area, dense
correspondence-based methods have achieved leading per-
formance. However, they do not explicitly consider the lo-
cal and global geometric information of different instances,
resulting in poor generalization ability to unseen instances
with significant shape variations. To deal with this prob-
lem, we propose a novel Instance-Adaptive and Geometric-
Aware Keypoint Learning method for category-level 6D ob-
ject pose estimation (AG-Pose), which includes two key de-
signs: (1) The first design is an Instance-Adaptive Key-
point Detection module, which can adaptively detect a set
of sparse keypoints for various instances to represent their
geometric structures. (2) The second design is a Geometric-
Aware Feature Aggregation module, which can efficiently
integrate the local and global geometric information into
keypoint features. These two modules can work together
to establish robust keypoint-level correspondences for un-
seen instances, thus enhancing the generalization abil-
ity of the model.Experimental results on CAMERA25 and
REAL275 datasets show that the proposed AG-Pose outper-
forms state-of-the-art methods by a large margin without
category-specific shape priors. Code will be released at
https://github.com/Leeiieeo/AG-Pose

## introduction
Given a RGB-D image, following previous works [17, 44],
we first employ an off-the-shelf MaskRCNN [9] to obtain
the segmentation mask and category label for each object.
For each segmented object, we use the segmentation mask
to get the cropped RGB image Iobj ∈RH×W ×3 and the
point cloud Pobj ∈RN×3, where N is the number of points
and Pobj is acquired by back-projecting the cropped depth
image using camera intrinsics followed by a downsampling
process. By taking Iobj and Pobj as input, the proposed
method aims to estimate the 3D rotation R ∈SO(3), the
3D translation t ∈R3 and the size s ∈R3 of the target
object.
The framework of proposed AG-Pose is shown in Fig-
ure 2 (a). Our method consists of four main components:
Feature Extractor (Sec. 3.2), Instance-Adaptive Keypoint
Detector (Sec. 3.3), Geometric-Aware Feature Aggregator
(Sec. 3.4) and Pose&Size Estimator (Sec. 3.5). Details
about each component are as follows.
3.2. Feature Extractor
For the input point cloud Pobj, we utilize the PointNet++
[26] to extract point features FP
∈RN×C1.
For the
RGB image Iobj, following [31], a PSP network [43] with
ResNet-18 [8] is applied to extract pixel-wise appearance
features from Iobj. We then choose those pixel features
corresponding to Pobj to obtain the RGB features FI ∈
RN×C2. Lastly, we concatenate FI and FP to form Fobj ∈
RN×C as the input for the subsequent networks.
3.3. Instance-Adaptive Keypoint Detector
As introduced in Section 1, the geometric information is
indispensable to establish robust correspondences. An in-
tuitive idea to facilitate each point with geometric informa-
tion is to employ the vanilla attention mechanism [30] to
aggregate features from all other points. But it is compu-
tationally expensive due to the large number of points. In-
stead, we aim to utilize a set of sparse keypoints to repre-
sent the shapes of different instances for pose estimation.
However, since the shapes vary across different instances
and the instance models are not accessible during inference,
fixed keypoint detection methods like [10, 11, 25, 39] are
not applicable. Besides, methods like Farest Point Sampling
are not end-to-end trainable, which may detect keypoints on
outlier points caused by the noisy segmentation mask, the
incorrect depth values and et al. Consequently, we design
an Instance-Adaptive Keypoint Detection module to adap-
tively detect keypoints for instances with different shapes
which can efficiently represent the object and avoid focus-
ing on outlier points.
The pipeline of IAKD is illustrated in Figure 2 (b).
Specifically, we initialize a set of category-shared learnable
queries Qcat ∈RNkpt×C and each of them represents a
keypoint detector. We first inject the object features Fobj
into learnable queries via Transformer Layers [30]. This
process converts the category-shared detectors Qcat into
instance-adaptive detectors Qins ∈RNkpt×C conditioned
on object features Fobj. In detail,
 \ mathbf { Q } = \ma t h bf {Q}_
{ca
t}\m a thbf {W}^ { q}, \mathbf {K} = \mathbf {F}_{obj}\mathbf {W}^{k}, \mathbf {V} = \mathbf {F}_{obj}\mathbf {W}^{v}, \\ \mathbf {Q}_{ins} =\textit {Norm} (\mathbf {Q}_{cat} + \textit {softmax} (\mathbf {Q} \mathbf {K}^{T} ) \mathbf {V}) .
(2)

Global Aggregation
Local Aggregation
(b) Instance-Adaptive Keypoint Detection (IAKD)
(c) Geometric-Aware Feature Aggregation (GAFA)
ℓ𝒐𝒄𝒅+ ℓ𝒅𝒊𝒗
𝐹obj
K
Attention 
Layers
V
𝑄𝑐𝑎𝑡
𝑄𝑖𝑛𝑠
S
𝑃obj
𝐹obj
𝐹𝑘𝑝𝑡
𝑃𝑘𝑝𝑡
Q
Heatmap 𝑯
𝐹𝑘𝑛𝑛
(𝑖)
𝑄𝑖𝑛𝑠
(𝑖)
PE
S
MLP
MLP
𝑄𝑖𝑛𝑠
AvgPool
𝑄𝑖𝑛𝑠
𝑔𝑙𝑜𝑏𝑎𝑙
𝑄𝑖𝑛𝑠
concat
PE
MLP
(a) Framework Overview
Feature Extractor & Instance-Adaptive Keypoint Detector
𝑃obj
𝐼obj
𝐹obj
𝑄𝑐𝑎𝑡
Feature 
Extractor
MLP Heads
Keypoints 𝑷𝒌𝒑𝒕
Geometric-Aware Feature Aggregator & Pose Estimator
𝑄𝑖𝑛𝑠
IAKD
module
Geometric-Aware Feature Aggregation  
Pose&Size
Local Aggregation
Global Aggregation
: Keypoints
: KNN Neighbors
: Multiplication
: Add
S : Similarity
Scores A
Figure 2. a) Overview of the proposed AG-Pose. b) Illustration of the IAKD module. We initialize a set of category-shared learnable
queries and convert them into instance-adaptive detectors by integrating the object features. The instance-adaptive detectors are then
used to detect keypoints for the object. To guide the learning of the IAKD module, we futher design the Ldiv and Locd to constrain the
distribution of keypoints. c) Illustration of the GAFA module. Our GAFA can efficiently integrate the geometric information into keypoint
features through a two-stage feature aggregation process.
Then we calculate the cosine similarities between the
instance-adaptive detectors Qins and the object features
Fobj to generate keypoint heatmap H ∈RNkpt×N. Subse-
quently, the 3D coordinates of keypoints Pkpt ∈RNkpt×3
and their corresponding features Fkpt ∈RNkpt×C under
camera space are obtained by weighted sum as:
 \ma t hbf {P}_{k p t} = 
\te
xtit {softmax}( \ mathbf {H})\times \mathbf {P}_{obj}, \\ \mathbf {F}_{kpt} = \textit {softmax}(\mathbf {H})\times \mathbf {F}_{obj} .
(4)
Since our target is to utilize a set of sparse keypoints to rep-
resent the geometric information of the object, the detected
keypoints should be well distributed on the surface of the
object. However, we find that the keypoints tend to clus-
ter in small regions and often focus on non-surface or out-
lier points without explicit supervision. To encourage the
keypoints to focus on different parts, we futher design a di-
versity loss Ldiv to force the detected keypoints to be away
from each other. In detail,
 L_{ d
iv} 
=
 \s
um _
{
i=1}^{N_
{kpt}}
 \sum _{j=1, j \neq i}^{N_{kpt}} \mathbf {d}(\mathbf {P}_{kpt}^{(i)},\mathbf {P}_{kpt}^{(j)}), \\ \mathbf {d}(\mathbf {P}_{kpt}^{(i)},\mathbf {P}_{kpt}^{(j)}) = \max \{th_{1} - \|\mathbf {P}_{kpt}^{(i)}-\mathbf {P}_{kpt}^{(j)}\|_{2}, 0 \} ,
(6)
where th1 is a hyperparameter and P(i)
kpt stands for the i-th
keypoint. To encourage the keypoints to locate on the sur-
face of the object and exclude outliers simultaneously, we
design an object-aware chamfer distance loss Locd to con-
strain the distribution of Pkpt. Formally, we first utilize the
ground truth pose and size Rgt, tgt, sgt to transform Pobj
to the NOCS and remove outlier points based on instance
model Mobj ∈RM×3 to produce P⋆
obj. In formula,
 \
mat h bf {P} _ {obj
}^{
\st
ar } = \
{
x_i | 
x_{i} \ in \ m athb f {P}_{obj} \quad \textit {and} \nonumber \\ \min _{y_{j} \in \mathbf {M}_{obj}} \|\frac {1}{\|\mathbf {s}_{gt}\|_{2}}\mathbf {R}_{gt}(x_{i} - \mathbf {t}_{gt}) - y_{j}\|_{2} < th_{2} \} .
(7)
The outlier filter process is also illustrated in Figure 3. Then

: Instance Model 𝑴𝒐𝒃𝒋
: Point Cloud 𝑷𝒐𝒃𝒋
: Outliers
𝑹𝑔𝑡, 𝒕𝑔𝑡, 𝒔𝑔𝑡
Transform
Thresholding
Figure 3. Illustration of the outlier filter process.
the Locd is calculated as follows,
 L_{ o
c
d} = \
f
rac {1}
{|\
mathb
f {
P}_ { kpt}|} \sum _{x_i \in \mathbf {P}_{kpt}} \min _{y_{j} \in \mathbf {P}_{obj}^{\star }} \|x_i - y_j \|_{2} .
(8)
By constraining keypoints to be close to P⋆
obj, the IAKD
module can automatically learn to filter out outlier points
during inference.
3.4. Geometric-Aware Feature Aggregator
With the detected keypoints for each object, a straightfor-
ward way is to predict the NOCS coordinates for these key-
points to establish keypoint-level correspondences. How-
ever, the keypoint features are lack of geometric informa-
tion, which can result in numerous incorrect correspon-
dences on unseen instances. Thus, we propose a Geometric-
Aware Feature Aggregation module to efficiently incorpo-
rate geometric information into keypoint features.
The GAFA adopts a two-stage pipeline for geometric
feature aggregation, as illustrated in Figure 2 (c).
We
first select the nearest K neighbors in Pobj and their cor-
responding features in Fobj for each keypoint to obtain
Pknn ∈RNkpt×K×3 and Fknn ∈RNkpt×K×C. Essen-
tially, the global geometric information can be represented
by the relative positions among keypoints, and the local ge-
ometric information can be represented by the relative posi-
tions between keypoints and their neighboring points. Thus,
we utilize the relative positional embeddings α and β be-
tween keypoints and their neighbors to represent the local
geometric features fl and global geometric features fg, re-
spectively. In detail,
 \al p ha _{i,j
} = MLP(\m
athbf {P}
_
{ kpt}^{(i)} - \mathbf {P}_{knn}^{(i,j)}), f_{l}^{(i)} = \textit {AvgPool}(\alpha _{i,:}), \\ \beta _{i,j} = MLP(\mathbf {P}_{kpt}^{(i)} - \mathbf {P}_{kpt}^{(j)}), f_{g}^{(i)} = \textit {AvgPool}(\beta _{i,:}) ,
(10)
where f (i)
l
, f (i)
g
∈R1×C and P(i,j)
knn is the j-th neighboring
point of P(i)
kpt. To incorporate local geometric information
into keypoints, we combine the keypoint features Qins with
fl and calculate local correlation scores A between key-
points and neighboring points, which is then used to aggre-
gate features from neighbors. The local feature aggregation
process for i-th keypoint feature Q(i)
ins is as follows,
 \ mathbf {A} 
=
 sim
(MLP ( \te
x
t
it {ca
t}\le
ft [
 \ma
thb f {Q

## method
Use of Shape Prior
IoU50
IoU75
5◦2 cm
5◦5 cm
10◦2 cm
10◦5 cm
NOCS [33]
✗
83.9
69.5
32.3
40.9
48.2
64.4
DualPoseNet [16]
✗
92.4
86.4
64.7
70.7
77.2
84.7
GPV-Pose [5]
✗
93.4
88.3
72.1
79.1
—
89
Query6DoF [35]
✗
91.9
88.1
78
83.1
83.9
90
SPD [28]
✓
93.2
83.1
54.3
59
73.3
81.5
SGPA [2]
✓
93.2
88.1
70.7
74.5
82.7
88.4
SAR-Net [15]
✓
86.8
79
66.7
70.9
75.3
80.3
RBP-Pose [42]
✓
93.1
89
73.5
79.6
82.1
89.5
AG-Pose
✗
93.8
91.3
77.8
82.8
85.5
91.6
Table 3. Comparisons between the IAKD and FPS.
Setting
5◦2 cm
5◦5 cm
10◦2 cm
10◦5 cm
FPS
46.2
55.5
67.0
80.2
IAKD
54.7
61.7
74.7
83.1
Table 4. Ablation studies on the number of keypoints.
Nkpt
5◦2 cm
5◦5 cm
10◦2 cm
10◦5 cm
16
47.9
55.1
68.8
79.8
32
48.8
55.7
73.1
82.9
64
51
57.2
72.8
82
96
54.7
61.7
74.7
83.1
128
52.8
59.9
74.3
83.7
by 1.9% on IoU50, 3.2% on IoU75, 1.6% on 10◦2 cm
and 1.6% on 10◦5 cm, respectively. Our method achieves
comparable performance with Query6DoF on 5◦2 cm and
5◦5 cm (lower by 0.2% and 0.3%, respectively).
Results of correspondence errors. To validate that the
keypoint-level correspondences of AG-Pose are more ac-
curate than point-level correspondences, we calculate the
NOCS error distributions of DPDN [17], ISTNet [18] and
our method on the REAL275 validation set. As shown in
Figure 4, the NOCS error of our keypoint-level correspon-
dences is concentrated more on the interval from 0 to 0.1,
while errors of dense correspondence-based methods are
concentrated more on the interval from 0.15 to 0.5. It proves
that the keypoint-level correspondences produced by AG-
Pose exhibit better generalizability on unseen instances.
4.2. Ablation Studies
In this section, we conduct ablation experiments to demon-
strate the effectiveness of the proposed method on the
REAL275 dataset.
Effects of the IAKD module. To evaluate the effective-
ness of the proposed Instance-Adaptive Keypoint Detection
module, we first replace the IAKD with the most widely
adopted Farest Point Sampling (FPS) strategy. Specifically,
Table 5. Ablation studies on the proposed loss functions.
Loss
5◦2 cm
5◦5 cm
10◦2 cm
10◦5 cm
Ldiv+Locd
54.7
61.7
74.7
83.1
Ldiv+Lucd
49.8
57.3
74.4
82.0
Ldiv
46.4
53
71
81.3
Locd
30
36.1
55.0
68.6
None
29.3
35.6
56.4
69.6
0.05
0.1
0.15
0.2
0.25
0.3
0.35
0.4
0.45
0.5
NOCS Error
0.00
0.05
0.10
0.15
0.20
0.25
0.30
Proportion
NOCS Error Distributions
ours
DPDN
IST-Net
Figure 4. Comparisons of NOCS error distributions.
we use FPS to sample the same number of keypoints and
use their corresponding features in Fobj as Qins. For a fair
comparison, we add extra MLPs to ensure that the number
of parameters remain unchanged. As shown in Table 3, the
proposed IAKD outperforms FPS on all metrics. We owe
the advantage to the end-to-end trainable property of IAKD
and the optimized distribution of keypoints brought by the
proposed losses.
Effects of different keypoint numbers. In Table 4, we
show the impact of different keypoint numbers Nkpt. Sur-
prisingly, the result shows that our method can achieve com-
parable performance (47.9% on 5◦2 cm) with the SOTA
methods by using only a very small number (Nkpt = 16) of
keypoints, which demonstrates the superiority of proposed
AG-Pose. And as the number of keypoints (Nkpt) increases,
the performance of our method continues to improve. We
choose Nkpt = 96 in our method for the balance between
efficiency and accuracy.
Effects of the Ldiv and the Locd. In our method, we

Table 6. Ablation study on two-stage feature aggregation.
Setting
5◦2 cm
5◦5 cm
10◦2 cm
10◦5 cm
Full
54.7
61.7
74.7
83.1
w/o GAFA
47.1
55.3
70.2
80.9
w/o Local
49
57.8
71.2
82.2
w/o Global
50.1
55.8
74.5
82.7
w/ vanilla attn
53
61
72.2
82.1
Table 7. Ablation study on proposed GAFA.
5◦2 cm
5◦5 cm
10◦2 cm
10◦5 cm
K=8
49.9
57.6
73.1
82.5
K=16
54.7
61.7
74.7
83.1
K=24
54.1
61.1
73.7
83.2
K=32
52.7
59.9
73.6
82.8
utilize the proposed Ldiv and Locd to encourage the key-
points to be well distributed on the surface of the object. To
verify the effectiveess of them, we conduct ablation exper-
iments on these losses and the results are shown in Table
5. By replacing the proposed Locd with the normal chamfer
distance loss Lucd, the performance of our model drops by
4.5% on 5◦2 cm. The reason is that our object-aware cham-
fer distance loss can prevent the model from being affected
by outliers, thereby improving the performance. The accu-
racy of the model further declines by 8.3% after we remove
the Locd. It is because Locd encourages keypoints to dis-
tribute on the surface of objects, which can better represent
the shapes of objects. Without Ldiv, the model performance
declines significantly by 24.7%, since Ldiv is indispensable
for keypoints to focus on distinct regions of objects. The ex-
cessive clustering of keypoints can result in a degradation in
model performance.
Effects of the GAFA module. Here we conduct abla-
tion studies on the Geometric-Aware Feature Aggregation
module. For a fair comparison, we replace the GAFA with
MLPs that have the same number of parameters. The results
are shown in Table 6. In particular, we observe that per-
formance drops by 7.6%, 4.6% and 5.7% on 5◦2 cm when
removing the whole GAFA, the global feature aggregation
and the local feature aggregation, respectively. As discussed
in section 1, both local and global geometric information
play crucial roles in establishing accurate correspondences
on unseen objects. Our GAFA can effectively encode such
geometric information into keypoint features, thus achiev-
ing better accuracy. Futhermore, we replace the geometric-
aware feature aggregation operation with vanilla attention
mechanism (w/ vanilla attn) in the GAFA. The experimental
result shows that the proposed geometric-aware feature ag-
gregation achieve better performance because the geomet-
ric information is indispensable in category-level 6D object
pose estimation. Last, we explore the influence of the local
aggregation range K on the accuracy of our model in Table
7. The results demonstrate that K = 16 yields the most
DPDN
Ours 
DPDN
Ours
Figure 5. Qualitative comparisons between our method and
DPDN [17] on REAL275 dataset. We visualize the correspon-
dence error maps and pose estimation results of our AG-Pose and
DPDN. Red/green indicates large/small errors and predicted/gt
bounding boxes.
significant performance gains.
4.3. Visualization
Qualitative Results. The qualitative results of DPDN [17]
and proposed AG-Pose are shown in Figure 5.
Specifi-
cally, we visualize the NOCS prediction errors and the fi-
nal pose predictions for both methods, in which green/red
indicate small/large errors and gt/predicted results. The vi-
sualization results show that dense correspondence-based
methods such as DPDN generate larger number of incorrect
correspondences on novel instances with significant shape
variations (e.g., cameras) as well as on outlier points (e.g.,
edge of objects), which severely degrade the performance
of pose estimation. In contrast, our AG-Pose can perform
instance-adaptive keypoint detection and extract geometric-
aware features to establish accurate keypoint-level corre-
spondences, leading to better pose estimation performance.

## experiments
Datasets. Following previous works [17, 18, 35], we con-
duct experiments on CAMERA25 and REAL275 [33], the
most widely adopted datasets for category-level object pose
estimation. CAMERA25 is a synthetic RGB-D dataset that
contains 300K synthetic images of 1,085 instances from 6
different categories. In this dataset, 25,000 images of 184
instances are used for evaluation and the others are used for
training. REAL275 dataset is a more challenging real-world
dataset which shares the same categories with CAMERA25.
It comprises 7K images from 13 different scenes. 2,750 im-
ages of 6 scenes are left for validation, including 3 unseen
instances per category.
Implementation details. For a fair comparison, we use
the same segmentation masks as SPD [28] and DPDN [17]
from MaskRCNN [9]. For the data preprocessing, images
are cropped and resized to 192 × 192 before feature ex-
traction, and the number of points N in point cloud is set
as 1024. For model parameters, the number of keypoints
is set as Nkpt = 96, and the local range for each key-
point in GAFA is set as K = 16. The feature dimensions
are set as C1 = 128, C2 = 128 and C = 256, respec-
tively. For the hyper-parameters in the loss functions, the
two thresholds in LQKD are set as th1 = 0.01 and th2 =
0.1, and λ1, λ2, λ3, λ4 are 1.0,5.0,1.0,0.3, respectively. For
model optimizing, we train our model on both synthetic
and real datasets for evaluation on REAL275 while only on
synthetic dataset for evaluation on CAMERA25. Follow-
ing previous work [17], we use random translation ∆t ∼
U(−0.02, 0.02) and scaling ∆s ∼U(−0.8, 1.2) for data
augmentation and apply random rotation for each axis with
rotation degree sampled from U(0, 20). We train our net-
work using the ADAM [12] optimizer and employ the tri-
angular2 cyclical learning rate schedule [27] ranging from
2e-5 to 5e-4. All experiments are conducted on a single
RTX3090Ti GPU with a batch size of 24.
Evaluation metrics. Following previous works [28, 33],
we evaluate the model performance with two metrics.
• 3D IoU. We report the mean precision of Intersection-
over-Union (IoU) for 3D bounding boxes with thresholds
of 50% and 75%. This metric incorporates both the pose
and size of the object.
• n◦m cm is used for direct evaluation of the rotation and
translation errors. Only predictions with rotation error
less than n◦and translation error less than m cm are con-
sidered correct.
4.1. Comparison with State-of-the-Art Methods
Results on REAL275 dataset. The comparisons between
our AG-Pose and existing state-of-the-art methods on the
challenging REAL275 dataset are shown in Table 1. As
demonstrated by the results, our AG-Pose outperforms all
previous methods in all metrics on REAL275 dataset. It
should be noted that our method does not require the use
of shape priors. In detail, on the most rigorous metric of
5◦2 cm, AG-Pose achieves the precision of 54.7%, surpass-
ing the current state-of-the-art shape prior-based method
DPDN [17] with a large margin by 8.7%. As for prior-free
methods, our method surpasses Query6DoF [35], IST-Net
[18] and GPV-Pose [5] by 5.7%, 7.2%, and 22.7%, respec-
tively. It should be noted that Query6DoF aims to learn a set
of sparse queries as implicit shape priors, and the query fea-
tures are used to update the point features to establish bet-
ter dense correspondences. Different from it, the proposed
AG-Pose aims to use a sparse set of keypoints to explic-
itly model the geometric information of objects to establish
robust keypoint-level correspondences for pose estimation.
The superior performance of our method indicates the im-
portance of geometric information in category-level 6D ob-
ject pose estimation.
Results on CAMERA25 dataset. Table 2 shows the
quantitative results of the proposed AG-Pose on CAM-
ERA25 dataset. Our method achieves the best performance
under most of metrics. In detail, the proposed AG-Pose
outperforms the state-of-the-art method Query6DoF [35]

Table 2. Quantitative comparisons with state-of-the-art methods on the CAMERA25 dataset.

## related_work
2.1. Instance-level 6D object pose estimation
Instance-level 6D object pose estimation aims to estimate
the pose of a known object given its CAD model. In re-
cent years, numerous methods based on direct regression
[4, 22, 31, 40], fixed keypoint detection [6, 25, 38, 39],
and dense 2D-3D correspondences [24, 41] have been pro-
posed. Direct regression-based methods take the observa-
tion image (RGB or RGBD) as input and directly predict the

3D rotation and translation via regression networks. The ad-
vantage of these methods lies in end-to-end pose estimation
without extra post-processing. On the other hand, methods
based on fixed keypoint detection usually predefine a set of
keypoints on the object CAD model and detect their corre-
sponding positions in the input image. Subsequently, pose
estimation is achieved through Perspectiven-Point (PnP) al-
gorithms. For example, [10, 11, 25, 39] utilize a dense vot-
ing strategy to detect keypoints and achieve state-of-the-art
performance. Dense 2D-3D correspondence-based meth-
ods aim to predict the corresponding position on the object
CAD model for every point [19, 32], which are more ro-
bust to occlusion. Among them, the approaches based on
fixed keypoints are most related to our method. However,
they adopt fixed keypoints on a given CAD model while our
method aims to adaptively detect keypoints for different in-
stances because the CAD models are not accessible during
inference in category-level task.
2.2. Category-level 6D object pose estimation
To improve the generalization ablity on unseen instances,
the category-level 6D object pose estimation is introduced,
which aims to estimate the poses of all instances belong-
ing a specific category without using their CAD models.
NOCS [33] proposes to use a shared canonical represen-
tation called Normalized Object Coordinate Space (NOCS)
to represent the shapes of all instances. They first predict
the NOCS coordinates of observed points and then apply
Umeyama [29] algorithm to recover the pose and size. To
handle the intra-category shape variation, SPD [28] pro-
poses a deformation and matching strategy. They first re-
construct instance models by deforming a categorical shape
prior and then match observations to the reconstructed mod-
els. Inspired by SPD, multiple works [2, 17, 34, 42] have
been proposed to improve the processes of shape prior de-
formation, correspondence matching and et al., continu-
ously improving the pose estimation performance. How-
ever, the above methods have not explicitly taken the local
and global geometric information of different instances into
consideration, which results in poor generalization ability
to unseen instances with significant shape variations.

## conclusion
In summary, we present a novel Instance-Adaptive and
Geometric-Aware Keypoint Learning method for category-
level 6D object pose estimation (AG-Pose). Specifically, we
propose a Instance-Adaptive Keypoint Detection module
represent the geometric information of different instances
through a set of sparse keypoints. Futhermore, we propose
a Geometric-Aware Feature Aggregation module to effec-
tively incorporate local and global geometric information
into keypoints to establish robust keypoint-level correspon-
dences. We conduct comprehensive experiments and the
experimental results verify the effectiveness of our method.
6. Acknowledgements
This work was partially supported by the National Na-
ture Science Foundation of China (Grant 62306294),
Dreams Foundation of Jianghuai Advance Technology
Center (NO.Q/JH-063-2023-T02A/0) and Anhui Provincial
Natural Science Foundation (Grant 2308085QF222).