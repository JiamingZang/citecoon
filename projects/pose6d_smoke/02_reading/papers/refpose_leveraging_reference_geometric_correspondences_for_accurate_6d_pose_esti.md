# RefPose: Leveraging Reference Geometric Correspondences for Accurate 6D Pose Estimation of Unseen Objects

> 2025 · id: W4413144617 · arXiv: 2505.10841 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Estimating the 6D pose of unseen objects from monocular
RGB images remains a challenging problem, especially due
to the lack of prior object-specific knowledge. To tackle
this issue, we propose RefPose, an innovative approach
to object pose estimation that leverages a reference image
and geometric correspondence as guidance. RefPose first
predicts an initial pose by using object templates to ren-
der the reference image and establish the geometric corre-
spondence needed for the refinement stage. During the re-
finement stage, RefPose estimates the geometric correspon-
dence of the query based on the generated references and
iteratively refines the pose through a render-and-compare
approach. To enhance this estimation, we introduce a cor-
relation volume-guided attention mechanism that effectively
captures correlations between the query and reference im-
ages. Unlike traditional methods that depend on pre-defined
object models, RefPose dynamically adapts to new object
shapes by leveraging a reference image and geometric cor-
respondence. This results in robust performance across pre-
viously unseen objects. Extensive evaluation on the BOP
benchmark datasets shows that RefPose achieves state-of-
the-art results while maintaining a competitive runtime.

## introduction
RefPose follows a multi-stage pipeline, similar to other re-
cent methods [15, 28, 30, 32], comprising object detec-
tion/segmentation, coarse pose estimation, and pose refine-
ment. Following these methods, we use an off-the-shelf
model [29] for object detection and segmentation to pre-
process the input image.
Fig. 2 illustrates the overall pipeline of RefPose. Starting
with an RGB image, the target object is detected, cropped,
and resized to 256 × 256 to create the query image, Iq. In
the coarse pose estimation stage, a classification network
selects a set of templates, S = {S1, S2, . . . , Sk}, from a
pre-rendered template set, T = {T1, T2, . . . , TN}. This se-
lected set, S, is then used to estimate the geometric corre-
spondence of the query, Gq, providing an initial pose esti-

∙,∙
Training
Correlation Volume
Context Feature
Classification
head
Flow head
Inference
Classifier
Positive / Negative
Classifier
. . .  
𝑠1
Top-𝑘Selection
. . .  
𝐼𝑞
Pred flow
GT flow
Classifier
Classifier
𝒯
𝐼𝑞
𝐼𝑇
𝒮
𝐼𝑇1
𝑆1
𝑆𝑘
𝑠2
𝑠𝑁
𝐼𝑇2
𝐼𝑇𝑁
Figure 3. Templates selection using a classification network. The
classification network scores pre-rendered templates based on how
well optical flow can be estimated between each template and the
query image. During inference, these scores are used to select the
top-k templates. The classifier leverages a frozen feature encoder
from a pre-trained optical flow network, with only the classifica-
tion head trained.
mate, P0. Using P0, we render a reference image, Ir, and
a geometric correspondence, Gpos
r
, that are closely aligned
with the query. Here, Gpos represents the geometric cor-
respondence with positional encoding applied. In the re-
finement stage, this reference information is then used to
estimate a more accurate geometric correspondence for the
query, Gpos
q
. Finally, the relative pose, ∆P, between the
query and reference is iteratively refined through a render-
and-compare approach to reach a final pose estimate.
3.2. Coarse pose estimation
Templates selection.
We start by randomly sampling
poses and rendering images along with geometric corre-
spondences for each pose based on the given 3D model.
The geometric correspondence G ∈Rh×w×3 represents a
dense map that indicates the corresponding 3D model point
for each pixel. For simplicity, we refer to geometric cor-
respondence as ”geometry” throughout this paper. Inspired
by MegaPose [15], we select a set of multiple templates, S,
using a classification network. However, unlike MegaPose,
where the classifier is trained based on the refining capa-
bility of its refiner, we introduce a new training criterion.
Since our subsequent pose estimation stage relies on optical
flow, we train the classifier to assess the accuracy of optical
Flow
Warp
Flow
Warp
Flow
Warp
Pixel-wise
Voting
. . .  
. . .  
PnP/RANSAC
𝒮
𝐼𝑆1
𝐺𝑆1
𝐼𝑆2
𝐺𝑆2
𝐼𝑆𝑘
𝐺𝑆𝑘
𝑃0
𝐺𝑞
𝐺𝑞
(1)
𝐺𝑞
(2)
𝐺𝑞
(𝑘)
𝐼𝑞
Figure 4. Warping-based geometry estimation process. The opti-
cal flow between each template image IS in the selected set S and
the query image Iq is used to warp the corresponding template
geometries GS, generating candidate geometries for the query,
Gq.
A pixel-wise voting scheme refines these candidates, and
the resulting 2D-3D correspondences in Gq are then applied with
PnP/RANSAC to estimate the initial pose P0.
flow estimation between each template image, IT , and the
query image, Iq.
During training, we utilize a pre-trained optical flow net-
work [43] to estimate the optical flow between Iq and each
template image in T . Positive and negative pairs are identi-
fied by comparing the predicted flow with the ground truth
flow, which serves as labels for training the classifier. Ad-
ditionally, rather than designing and training a new feature
encoder for the classification network, we leverage the fea-
ture encoder from the optical flow network to enhance both
the classifier’s performance and coherence with subsequent
stages. Specifically, as in RAFT, we extract the correlation
volume and context features, then attach a simple CNN as
the classification head to complete the classifier. The classi-
fier is trained using Binary Cross-Entropy (BCE) Loss [40].
During inference, the classifier ranks template images from
the set T , and we select the top-k templates based on the
predicted scores. The classifier’s structure and training and
inference processes are illustrated in Fig. 3.
Warping-based geometry estimation. To predict an initial
pose P0, we estimate the query geometry Gq based on a set
of selected templates S. First, we calculate the optical flow
between each template image IS and the query image Iq,
then warp the template geometries GS accordingly. These
warped geometries serve as candidates for Gq. However,
due to potential inconsistencies and inaccuracies in the op-
tical flow estimated from each template, these candidates
may indicate different 3D points for the same query pixel.
To address this, we employ a voting scheme to select a sin-
gle 3D point per pixel. PFA [13] follows a similar approach

by aggregating multiple optical flows from different tem-
plates to estimate the 2D-3D correspondences. However, it
performs this aggregation without explicitly addressing the
inconsistencies and inaccuracies, which can lead to unreli-
able correspondences. In contrast, our method employs a
more robust medoid-based voting scheme, where the most
representative 3D point per pixel is selected rather than ag-
gregating all candidates indiscriminately. This approach en-
sures that errors in optical flow estimation do not adversely
affect the final result, leading to a more accurate and stable
coarse pose estimation. Finally, based on the established
2D-3D correspondences in Gq, we apply the PnP/RANSAC
algorithm to estimate the pose P0. The overall process is il-
lustrated in Fig. 4.
3.3. Pose refinement
Using the initial pose P0 obtained from the coarse pose es-
timation stage, we render a single reference image Ir and
geometry Gpos
r
that are more closely aligned with the query
image Iq than the previously selected templates in S. Rather
than using raw 3D coordinates, we apply positional encod-
ing to the 3D coordinates of G to enrich the geometry repre-
sentation, thereby improving estimation accuracy [34]. This
encoding results in a representation Gpos ∈Rh×w×6Nfreq,
where Nfreq denotes the frequency bands used for encod-
ing [27].
Correlation volume-guided attention mechanism. Accu-
rately retrieving relevant geometric information from the
reference geometry Gpos
r
requires precise pixel-level cor-
respondence between the query image Iq and the refer-
ence image Ir. To achieve this, we introduce a correla-
tion volume-guided attention mechanism. In this setup, the
query image Iq serves as the query, the reference image Ir
acts as the key, and the reference geometry Gpos
r
is treated
as the value. The mechanism computes attention weights
based on the similarity between the query (Iq) and the key
(Ir), enabling the extraction of relevant features from the
value (Gpos
r
).
In contrast to vanilla attention mechanisms that implic-
itly learn relevance, our approach uses explicit pixel-level
correspondence information, as both the query image Iq and
reference image Ir represent the same object from different
viewpoints. We obtain the attention weights by applying a
softmax operation to the correlation volume from the op-
tical flow network, which encodes pixel-level similarities
between Iq and Ir. By applying these weights to the ref-
erence geometry features Gpos
r
, we retrieve highly relevant
geometric information, facilitating accurate correspondence
estimation for improved pose refinement.
Geometry estimation network. Our geometry estimation
network processes the query image Iq, reference image
Ir, and reference geometry Gpos
r
as inputs and outputs the
query geometry Gpos
q
. To capture multi-scale features ef-
Optical Flow
Network
Correlation Volume
Down 
sampling
Down 
sampling
Softmax
Softmax
Softmax
Geo 
Head
Mask 
Head
Convex 
Upsampling
Correlation Volume-Guided Attention
𝐼𝑟
𝐺𝑟
𝑝𝑜𝑠
𝐼𝑞
𝐺𝑞
𝑝𝑜𝑠
Figure 5. Architecture of the geometry estimation network. The
network takes the query image Iq, reference image Ir, and ref-
erence geometry Gpos
r
as inputs to estimate the query geometry
Gpos
q
. A correlation volume-guided attention mechanism is ap-
plied at each level of the U-Net to effectively integrate these inputs.
The geo head and mask head output a low-resolution geometry
map and mask, which are then refined through convex upsampling
to produce the final high-resolution geometry Gpos
q
.
fectively, we employ a U-Net structure [39], with correla-
tion volume-guided attention mechanisms applied at each
level to incorporate reference information. As features are
downsampled within the U-Net, the correlation volume is
correspondingly downsampled. The U-Net concludes with
two heads: a geo head, which estimates geometry at an 8x
downsampled resolution, and a mask head, which predicts
an up-

## method
Refinement
YCB-V
LM-O
T-LESS
TUD-L
IC-BIN
ITODD
HB
Mean
Run-time
OSOP [41]
-
29.6
27.4
40.3
-
-
-
-
-
-
ZS6D [1]
-
32.4
29.8
21.0
-
-
-
-
-
-
MegaPose [15]
-
28.1
22.9
17.7
25.8
15.2
10.8
25.1
20.8
15.5s
GenFlow [28]
-
27.7
25.0
21.5
30.0
16.8
15.4
28.3
23.
3.8s
GigaPose [30]
-
27.8
29.6
26.4
30.0
22.3
17.5
34.1
26.8
0.4s
FoundPose [32]
-
45.2
39.6
33.8
46.7
23.9
20.4
50.8
37.2
1.7s
RefPose (Ours)
-
50.0
35.8
38.1
48.5
23.1
21.5
49.8
38.1
3.1s
MegaPose [15]
MegaPose
60.1
49.9
47.7
65.3
36.7
31.5
65.4
50.9
17.0s
MegaPose [15]
MegaPose, MH
62.1
56.0
50.7
68.4
41.4
33.8
70.4
54.7
21.9s
MegaPose [15]
RefPose (Ours)
65.3
56.0
52.8
66.4
45.3
41.1
73.6
57.2
16.4s
GenFlow [28]
GenFlow, MH
63.3
56.3
52.3
68.4
45.3
39.5
73.9
57.0
20.8s
GigaPose [30]
MegaPose
63.2
55.7
54.1
58.0
45.0
37.6
69.3
54.7
2.3s
GigaPose [30]
GenFlow, MH
65.2
63.1
58.2
66.4
49.8
45.3
75.6
60.5
10.6s
FoundPose [32]
MegaPose, MH + Featuremetric
69.0
61.0
57.0
69.4
47.9
40.7
72.3
59.6
20.5s
RefPose (Ours)
MegaPose
63.7
56.3
51.1
65.8
43.7
41.4
71.8
56.3
4.6s
RefPose (Ours)
RefPose (Ours)
72.7
59.6
57.8
69.7
51.2
43.8
76.2
61.4
3.9s
The 10th and 15th rows of Tab. 1 report results where
our proposed coarse pose estimation and refinement meth-
ods are independently combined with MegaPose to assess
their standalone effectiveness. A comparison among the
8th, 12th, and 15th rows shows that our coarse pose estima-
tion achieves better results even when using the same refine-
ment method as MegaPose. Additionally, comparing the 8th
and 10th rows demonstrates that our refinement method is
more effective even when applied to the coarse poses esti-
mated by MegaPose.
The reported runtime represents the average speed for
processing all objects within a single image. Though our
method takes slightly longer in the coarse pose estimation
stage, it significantly reduces refinement time by minimiz-
ing the number of renderings and using a lightweight model.
As a result, our method achieves superior performance with
runtime comparable to other state-of-the-art approaches.
4.3. Ablation study
Number of pre-rendered templates. The results of the ab-
lation study on the number of pre-rendered templates in the
set T , represented by N, are shown in Tab. 2. With only
64 templates, the sampled poses are too sparse across the
object’s orientation space, leading to reduced performance
in pose estimation due to insufficient coverage of various
possible object poses. Increasing to 128 templates provides
a denser sampling, significantly enhancing query-template
matching and improving performance.
Using more than
128 templates yields only marginal improvements while in-
creasing both memory and computational overhead. Ad-
ditionally, a larger template set results in longer inference
times, as more templates must be evaluated. Therefore, us-
ing 128 templates achieves an optimal balance between ef-
Table 2. Ablation study on the number of pre-rendered templates.
The table presents AR scores, illustrating the impact of varying
numbers of pre-rendered templates, T , on coarse pose estimates.
N
YCB-V
LM-O
T-LESS
TUD-L
IC-BIN
ITODD
HB
Mean
64
45.2
29.8
36.5
43.7
20.1
18.8
43.5
33.9
128
50.0
35.8
38.1
48.5
23.1
21.5
49.8
38.1
256
48.8
37.2
39.5
45.8
24.3
23.1
49.8
38.3
Table 3. Ablation study on the number of selected templates. The
table presents AR scores, showing the impact of varying numbers
of selected templates, S, on coarse pose estimates.
k
YCB-V
LM-O
T-LESS
TUD-L
IC-BIN
ITODD
HB
Mean
1
42.6
31.0
28.8
40.1
18.8
18.1
40.8
31.5
2
46.8
37.2
34.0
43.1
19.8
18.8
46.5
35.2
4
50.0
35.8
38.1
48.5
23.1
21.5
49.8
38.1
8
50.0
29.8
36.4
44.5
21.7
19.9
51.0
36.2
ficiency and performance.
Number of selected templates. The results of our ablation
study on the number of selected templates S, denoted by k,
are shown in Tab. 3. Selecting four templates from the pre-
rendered set T yields the best results, achieving a balance
between diversity and alignment accuracy.
Using fewer
templates limits diversity, increasing the risk of alignment
errors during warping-based geometry estimation if a tem-
plate is poorly selected or if the optical flow is inaccurately
estimated potentially, leading to inaccurate pose estimation.
Conversely, selecting more than four templates raises the
likelihood of including misaligned templates, which may
reduce the effectiveness of the medoid-based voting stage
in handling outliers. With too many templates, the medoid’s
robustness can be compromised, as it becomes more chal-
lenging to filter out misaligned correspondence effectively.

Table 4. Ablation study on components in coarse pose estimation
stage. This table presents AR scores for the coarse pose estimates.
Setting
YCB-V
LM-O
T-LESS
TUD-L
IC-BIN
ITODD
HB
Mean
w/o Flow features
46.8
34.2
33.2
47.1
22.3
20.9
45.8
35.8
w/o Medoid
49.6
29.6
36.8
45.9
21.3
18.8
50.2
36.0
Ours
50.0
35.8
38.1
48.5
23.1
21.5
49.8
38.1
Table 5. Ablation study on components of the geometry estima-
tion network in the pose refinement stage. This table presents AR
scores for the pose refinement results.
Setting
YCB-V
LM-O
T-LESS
TUD-L
IC-BIN
ITODD
HB
Mean
w/o P.E.
69.3
57.1
57.0
68.4
48.3
39.5
73.9
59.1
w/o Convex.
70.0
56.8
58.2
68.4
47.9
43.1
75.6
60.0
w/o C.G. attn.
68.8
61.0
52.8
66.0
47.3
41.9
74.8
58.9
Ours
72.7
59.6
57.8
69.7
51.2
43.8
76.2
61.4
Thus, selecting four templates provides an effective bal-
ance, ensuring sufficient diversity while minimizing align-
ment errors, which is crucial for accurate geometry and pose
estimation.
Components in the coarse pose estimation stage. Tab. 4
presents the results of our ablation study on key compo-
nents used in the coarse pose estimation stage. The first
row evaluates the impact of using the feature encoder from
the optical flow network as the classifier’s feature encoder.
This feature encoder outputs flow features, including the
correlation volume and context feature, which provide rich
cues related to optical flow, enhancing the classifier’s accu-
racy in selecting templates. By directly leveraging the same
encoder applied in the warping-based geometry estimation
stage, consistency across stages is maintained, contributing
to superior results.
The “w/o Medoid” variant represents a model in which,
during pixel-wise voting in the warping-based geometry es-
timation, the medoid is replaced with a simple average for
selecting correspondences. Using the medoid rather than
averaging mitigates the impact of outliers that may arise
from imperfect optical flow estimations, leading to a more
robust correspondence selection. This robustness directly
translates to greater accuracy of the coarse pose estimation.
Components of the geometry estimation network in the
pose refinement stage. Tab. 5 presents the ablation study
results on key components of the geometry estimation net-
work within the pose refinement stage. The first row shows
that applying positional encoding to the 3D coordinates, in-
spired by [34], improves performance over using raw 3D
coordinates as geometric correspondence. Specifically, the
“w/o P.E.” variant, which omits positional encoding and di-
rectly uses 3D coordinates, shows lower accuracy.
Additionally, estimating an up-mask and utilizing con-
vex upsampling yield superior performance compared to bi-
linear upsampling, as indicated by the “w/o Convex.” vari-
ant. Convex upsampling more effectively preserves spatial
detail at the original resolution, leading to a closer align-
ment between the estimated geometry and the true geome-
Iq
Ir
C.G. Attn.
Vanilla Attn.
Figure 7. Visual comparison of attention mechanisms in the geom-
etry estimation network. Each row shows a query image Iq with a
yellow dot marking the point of interest, the corresponding refer-
ence image Ir, and the attention maps produced by the correlation
volume-guided (C.G.) attention and vanilla attention. The C.G. at-
tention accurately focuses on relevant regions in Iq and Ir, while
vanilla attention lacks this precision.
try of the object.
Finally, the proposed correlation volume-guided (C.G.)
attention mechanism outperforms vanilla attention, as
demonstrated by the “w/o C.G. attn.” variant, which re-
places C.G.attention with vanilla attention. This result un-
derscores the effectiveness of C.G. attention in accurately
capturing relevant correspondences between the query and
reference images. As illustrated in Fig. 7, C.G. attention
maps exhibit a sharper focus on relevant regions in the refer-
ence image that correspond to points of interest in the query,
while vanilla attention maps appear less precise. This com-
parison highlights the effectiveness of C.G. attention in es-
tablishing accurate correlations between the query and ref-
erence images.

## experiments
4.1. Experimental Setup
Datasets.
We train our model on the synthetic dataset,
Google Scanned Objects (GSO) [9], as provided by [15].
This dataset contains nearly 1 million images represent-
ing 1,000 different object types.
Although [15] also in-
cludes data from approximately 50,000 ShapeNet [4] ob-
jects, we opted to use only the GSO dataset due to com-
putational and memory constraints. This choice is further
supported by the higher-quality mesh models in the GSO
dataset, which, as noted in [15], contribute more critically
to model performance than ShapeNet’s. For evaluation, we
test our approach on the seven primary datasets in the BOP
benchmark [12], including YCB-V, LM-O, T-LESS, TUD-
L, ICBIN, ITODD, and HB. Our method relies solely on
RGB images and 3D object models, without leveraging any
depth information.
Evaluation metrics. We follow the standard BOP eval-
uation protocol [12], which employs three core met-
rics:
Visible Surface Discrepancy (VSD), Maximum
Symmetry-Aware Surface Distance (MSSD), and Maxi-
mum Symmetry-Aware Projection Distance (MSPD). Over-
all performance, referred to as Average Recall (AR), is cal-
culated by averaging the individual recall scores for each
metric across a range of error thresholds.
Implementation details.
Our model is trained with the
AdamW optimizer [21], using a batch size of 8 and a learn-
ing rate of 0.0001 for a total of 400k training steps. A cosine
annealing scheduler [22] with a 10k step period is employed
Ours
FoundPose
GigaPose
GenFlow
MegaPose
Figure 6. Qualitative comparison of pose estimation results. We
present a qualitative comparison of our method against other ap-
proaches, with the projected contours from the ground-truth pose
shown in green and those from the predicted pose in blue.
to adjust the learning rate. Both training and evaluation are
conducted on an RTX-3090 GPU. For the optical flow net-
work in RefPose, we use the large model of RAFT [43],
which is pre-trained on the FlyingChairs [8] and FlyingTh-
ings3D [26] datasets and then fine-tuned on the GSO dataset
for our specific application. In the pose refinement stage,
we generate the geometry Gpos using sine and cosine po-
sitional encoding with five frequency bands, following the
approach in [34]. The number of pose refinement iterations,
M, is set to 5, balancing accuracy and runtime efficiency.
4.2. Comparison with state-of-the-art methods
Tab. 1 presents the results of our method on the BOP
benchmark datasets. Except for OSOP [41], which uses
its own detection model, all methods employ CNOS [29]
as the detection/segmentation model to ensure a fair com-
parison. Our method achieves the best performance in both
the coarse pose estimation and the refined results follow-
ing the refinement stage. While our approach slightly un-
derperforms the current state-of-the-art methods on LM-O,
T-LESS, and ITODD, it achieves state-of-the-art results on
the other datasets, with particularly notable improvements
on YCB-V and HB. Overall, our method demonstrates the
best performance across all datasets. Fig. 6 provides qualita-
tive comparisons with existing methods, further illustrating
the robustness of our approach.

Table 1. Evaluation results on BOP benchmark datasets. The table reports Average Recall (AR) scores across the seven datasets in the BOP
challenge, where higher AR scores indicate better performance. The best-performing method is highlighted in bold, and the second-best
is underlined. The top section presents results from coarse pose estimation alone, while the bottom section displays results after applying
the refinement stage. “MH” denotes MegaPose and GenFlow versions that incorporate a multi-hypotheses strategy in the refinement stage,
and “featuremetric” refers to the refinement method introduced in FoundPose.

## related_work
Geometric correspondence-based pose estimation.
In
instance-level object pose estimation, where training and
testing are performed within a fixed set of objects, a com-
mon strategy is to utilize 2D-3D geometric correspondence.
Most methods follow a two-step process: they first establish
2D-3D correspondences from an RGB image and then de-
termine the pose using a RANSAC-based PnP algorithm or
a neural network. Early studies [37, 44] employed the 3D
bounding box corners of the object as keypoints, focusing
on identifying the projected positions of these points in the
image. For more robust and precise pose estimation, re-
cent research [3, 7, 11, 19, 35, 45, 47] has shifted toward
establishing dense correspondence maps rather than relying
on sparse points. Consequently, designing and training deep
learning networks that can accurately predict geometric cor-
respondence maps is essential. However, these networks of-
ten struggle with generalization, especially when applied to
unseen objects outside the fixed training set. Some mod-
els [20, 42] even require separate network models for each
object. Therefore, to effectively estimate the geometric cor-
respondence of the query object, we train the network using
not only the query image but also a reference image and its
geometric correspondence, providing reliable and valuable
contextual information.
Most methods use 3D coordinates as the geometric
correspondence; however, some approaches modify these
coordinates to achieve finer correspondence estimation.
For example, [20, 42] adopts a binary code representa-
tion, while [34] applies positional encoding, as seen in
NeRF [27], to improve performance.
Inspired by these
methods, we also employ a positionally encoded represen-
tation for geometric correspondence during the refinement
stage.
Unseen object pose estimation. Some studies focus on

Input Image
...
Templates 
Selection
...
Warping-Based
Geometry Estimation
Reference
Query Geometry
Estimation
Relative Pose 
Estimator
Stage 2: Pose Refinement
Stage 1: Coarse Pose Estimation
Repeats
𝐺𝑞
𝑝𝑜𝑠
Detection
Crop & Resize
PnP/RANSAC
Render
Render
𝐼𝑞
𝒯
𝒮
𝑃0
𝐺𝑟
𝑝𝑜𝑠
Δ𝑃
𝐼𝑟
𝐺𝑟
𝑝𝑜𝑠
𝐺𝑞
Figure 2. Overview of the RefPose pipeline. Given an input RGB image, the target object is first detected, cropped, and resized to create
the query image, Iq. In Stage 1: Coarse Pose Estimation, a set of templates, S, is selected from the pre-rendered template set, T , to
estimate an initial pose, P0, for the query object. In Stage 2: Pose Refinement, the query’s geometric correspondence, Gpos
q
, is estimated
using the rendered reference image, Ir, and geometric correspondence, Gpos
r
. The initial pose, P0, is iteratively refined by estimating the
relative pose, ∆P, between the query and reference. At each iteration, Gpos
r
is re-rendered to align with the updated pose, leading to an
accurate final pose estimate.
category-level pose estimation [6, 17, 24, 46], utilizing
shared geometric traits within a category to broaden the
range of identifiable target objects. However, these meth-
ods face generalization limitations, making it challenging to
unseen object categories in real-world settings. To address
these limitations, recent research has explored unseen ob-
ject pose estimation, which aims to accurately predict poses
for objects not encountered during training.
MegaPose [15] combines a render-and-compare refiner
with a classifier that assesses whether the refiner can cor-
rect given pose errors, achieving strong generalization by
training on a large synthetic dataset. GigaPose [30] lever-
ages discriminative templates to handle out-of-plane rota-
tions and uses patch correspondences for estimating re-
maining pose parameters, resulting in improvements in both
speed and segmentation robustness.
GenFlow [28] ad-
dresses the accuracy-scalability trade-off by directly lever-
aging the target object’s shape through optical flow predic-
tion. GenFlow iteratively refines poses by leveraging a 3D
shape constraint alongside a multi-scale, coarse-to-fine re-
finement process. FoundPose [32] establishes 2D-3D cor-
respondences by matching patch descriptors from a self-
supervised DINOv2 [31] model between the image and pre-
rendered templates, integrating these descriptors into a bag-
of-words model for more efficient template retrieval.
FoundPose is particularly relevant to our approach in
that it also aims to estimate correspondences to assist pose
estimation. However, FoundPose derives correspondences
by performing patch-wise matching with pre-rendered tem-
plates, which may result in inaccuracies due to misalign-
ment with the query image. In contrast, our method be-
gins with a coarse pose estimation to establish an initial
pose, which we then refine using a rendered reference image
closely aligned with the query object. Additionally, rather
than relying solely on direct matching, we design a geom-
etry estimation network that improves correspondence esti-
mation by effectively integrating information from the ref-
erence image and geometric correspondence guidance.

## conclusion
In this paper, we have proposed RefPose, a two-stage
method designed for enhanced accuracy and generalization
in unseen object pose estimation. Starting with a coarse
pose estimation using template selection and medoid-based
voting, RefPose builds an initial pose, which is then used
to assist the geometry estimation through a correlation
volume-guided attention mechanism. This refined geome-
try for the query supports an iterative render-and-compare
process, producing a precise final pose.
Extensive ex-
periments on the BOP benchmark demonstrate RefPose’s
strong performance and generalization ability, while abla-
tion studies confirm the effectiveness of each component.
RefPose advances adaptable and efficient solutions for 6D
pose estimation in complex, real-world scenarios.

Acknowledgements This work was supported by Sam-
sung Electronics Co., Ltd.(No.
0423-20240056), Insti-
tute of Information & communications Technology Plan-
ning & Evaluation (IITP) grant funded by the Korea gov-
ernment(MSIT) [NO.RS-2021-II211343, Artificial Intelli-
gence Graduate School Program (Seoul National Univer-
sity)], and in part by the BK21 FOUR program of the Ed-
ucation and Research Program for Future ICT Pioneers,
Seoul National University in 2025.