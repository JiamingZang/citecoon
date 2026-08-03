# SAM-6D: Segment Anything Model Meets Zero-Shot 6D Object Pose Estimation

> 2024 · id: W4402727436 · arXiv: 2311.15707 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Zero-shot 6D object pose estimation involves the detec-
tion of novel objects with their 6D poses in cluttered scenes,
presenting significant challenges for model generalizabil-
ity. Fortunately, the recent Segment Anything Model (SAM)
has showcased remarkable zero-shot transfer performance,
which provides a promising solution to tackle this task. Mo-
tivated by this, we introduce SAM-6D, a novel framework
designed to realize the task through two steps, including in-
* Equal contribution. † Corresponding author <kuijia@gmail.com>.
stance segmentation and pose estimation. Given the tar-
get objects, SAM-6D employs two dedicated sub-networks,
namely Instance Segmentation Model (ISM) and Pose Esti-
mation Model (PEM), to perform these steps on cluttered
RGB-D images.
ISM takes SAM as an advanced start-
ing point to generate all possible object proposals and se-
lectively preserves valid ones through meticulously crafted
object matching scores in terms of semantics, appearance
and geometry. By treating pose estimation as a partial-to-
partial point matching problem, PEM performs a two-stage
point matching process featuring a novel design of back-
ground tokens to construct dense 3D-3D correspondence,
1
arXiv:2311.15707v2  [cs.CV]  6 Mar 2024

ultimately yielding the pose estimates. Without bells and
whistles, SAM-6D outperforms the existing methods on the
seven core datasets of the BOP Benchmark for both instance
segmentation and pose estimation of novel objects.

## introduction
Object pose estimation is fundamental in many real-world
applications, such as robotic manipulation and augmented
reality. Its evolution has been significantly influenced by
the emergence of deep learning models. The most stud-
ied task in this field is Instance-level 6D Pose Estimation
[18, 19, 51, 58, 60, 63], which demands annotated training
images of the target objects, thereby making the deep mod-
els object-specific. Recently, the research emphasis gradu-
ally shifts towards the task of Category-level 6D Pose Esti-
mation [7, 29–32, 56, 61] for handling unseen objects, yet
provided they belong to certain categories of interest. In this
paper, we thus delve into a broader task setting of Zero-shot
6D Object Pose Estimation [5, 28], which aspires to detect
all instances of novel objects, unseen during training, and
estimate their 6D poses. Despite its significance, this zero-
shot setting presents considerable challenges in both object
detection and pose estimation.
Recently, Segment Anything Model (SAM) [26] has gar-
nered attention due to its remarkable zero-shot segmenta-
tion performance, which enables prompt segmentation with
a variety of prompts, e.g., points, boxes, texts or masks.
By prompting SAM with evenly sampled 2D grid points,
one can generate potential class-agnostic object proposals,
which may be highly beneficial for zero-shot 6D object pose
estimation. To this end, we propose a novel framework,
named SAM-6D, which employs SAM as an advanced
starting point for the focused zero-shot task. Fig. 2 gives
an overview illustration of SAM-6D. Specifically, SAM-6D
employs an Instance Segmentation Model (ISM) to realize
instance segmentation of novel objects by enhancing SAM
with a carefully crafted object matching score, and a Pose
Estimation Model (PEM) to solve object poses through a
two-stage process of partial-to-partial point matching.
The Instance Segmentation Model (ISM) is developed
using SAM to take advantage of its zero-shot abilities for
generating all possible class-agnostic proposals, and then
assigns a meticulously calculated object matching score to
each proposal for ascertaining whether it aligns with a given
novel object. In contrast to methods that solely focus on
object semantics [5, 40], we design the object matching
scores considering three terms, including semantics, ap-
pearance and geometry. For each proposal, the first term
assesses its semantic matching degree to the rendered tem-
plates of the object, while the second one further evaluates
its appearance similarities to the best-matched template.
The final term considers the matching degree based on ge-
SAM-6D
Instance Segmentation Model
Pose Estimation Model
Segment
Anything
Coarse Point
Matching
Fine Point
Matching
Object
Matching
RGB image
Depth map
Target objects
Figure 2. An overview of our proposed SAM-6D, which consists
of an Instance Segmentation Model (ISM) and a Pose Estimation
Model (PEM) for joint instance segmentation and pose estimation
of novel objects in RGB-D images. ISM leverages the Segment
Anything Model (SAM) [26] to generate all possible proposals
and selectively retains valid ones based on object matching scores.
PEM involves two stages of point matching, from coarse to fine,
to establish 3D-3D correspondence and calculate object poses for
all valid proposals. Best view in the electronic version.
ometry, such as object shape and size, by calculating the
Intersection-over-Union (IoU) value between the bounding
boxes of the proposal and the 2D projection of the object
transformed by a rough pose estimate.
The Pose Estimation Model (PEM) is designed to cal-
culate a 6D object pose for each identified proposal that
matches the novel object. Initially, we formulate this pose
estimation challenge as a partial-to-partial point matching
problem between the sampled point sets of the proposal and
the target object, considering the factors such as occlusions,
segmentation inaccuracies, and sensor noises. To solve this
problem, we propose a simple yet effective solution that in-
volves the use of background tokens; specifically, for the
two point sets, we learn to align their non-overlapped points
with the background tokens in the feature space, and thus
effectively establish an assignment matrix to build the nec-
essary correspondence for predicting the object pose. Based
on the design of background tokens, we further develop
PEM with two point matching stages, i.e., Coarse Point
Matching and Fine Point Matching. The first stage real-
izes sparse correspondence to derive an initial object pose,
which is subsequently used to transform the point set of
the proposal, enabling the learning of positional encodings.
The second stage incorporates the positional encodings of
the two point sets to inject the initial correspondence, and
2

builds dense correspondence for estimating a more precise
object pose. To effectively model dense interactions in the
second stage, we propose an innovative design of Sparse-to-
Dense Point Transformers, which realize interactions on the
sparse versions of the dense features, and subsequently, dis-
tribute the enhanced sparse features back to the dense ones
using Linear Transformers [12, 24].
For the two models of SAM-6D, ISM, built on SAM,
does not require any network re-training or fine-tuning,
while PEM is trained on the large-scale synthetic images
of ShapeNet-Objects [4] and Google-Scanned-Objects [9]
datasets provided by [28]. We evaluate SAM-6D on the
seven core datasets of the BOP benchmark [54], including
LM-O, T-LESS, TUD-L, IC-BIN, ITODD, HB, and YCB-
V. The qualitative results are visualized in Fig. 1. SAM-6D
outperforms the existing methods on both tasks of instance
segmentation and pose estimation of novel objects, thereby
showcasing its robust generalization capabilities.
Our main contributions could be summarized as follows:
• We propose a novel framework of SAM-6D, which real-
izes joint instance segmentation and pose estimation of
novel objects from RGB-D images, and outperforms the
existing methods on seven datasets of BOP benchmark.
• We leverage the zero-shot capacities of Segmentation
Anything Model (SAM) to generate all possible propos-
als, and devise a novel object matching score to identify
the proposals corresponding to novel objects.
• We approach pose estimation as a partial-to-partial point
matching problem with a simple yet effective design of
background tokens, and propose a two-stage point match-
ing model for novel objects. The first stage realizes coarse
point matching to derive initial object poses, which are
then refined in the second stage of fine point matching us-
ing newly proposed Sparse-to-Dense Point Transformers.

## method
Input Type
Detection / Segmentation
BOP Dataset
Mean
LM-O
T-LESS
TUD-L
IC-BIN
ITODD
HB
YCB-V
With Supervised Detection / Segmentation
MegaPose [28]
RGB
MaskRCNN [16]
18.7
19.7
20.5
15.3
8.00
18.6
13.9
16.2
MegaPose† [28]
RGB
53.7
62.2
58.4
43.6
30.1
72.9
60.4
54.5
MegaPose† [28]
RGB-D
58.3
54.3
71.2
37.1
40.4
75.7
63.3
57.2
ZeroPose [5]
RGB-D
26.1
24.3
61.1
24.7
26.4
38.2
29.5
32.6
ZeroPose† [5]
RGB-D
56.2
53.3
87.2
41.8
43.6
68.2
58.4
58.4
SAM-6D (Ours)
RGB-D
66.5
66.0
80.9
61.9
31.9
81.8
79.6
66.9
With Zero-Shot Detection / Segmentation
ZeroPose [5]
RGB-D
ZeroPose [5]
26.0
17.8
41.2
17.7
38.0
43.9
25.7
25.7
ZeroPose† [5]
RGB-D
49.1
34.0
74.5
39.0
42.9
61.0
57.7
51.2
SAM-6D (Ours)
RGB-D
63.5
43.0
80.2
51.8
48.4
69.1
79.2
62.2
MegaPose∗[28]
RGB
CNOS (FastSAM) [40]
22.9
17.7
25.8
15.2
10.8
25.1
28.1
20.8
MegaPose†∗[28]
RGB
49.9
47.7
65.3
36.7
31.5
65.4
60.1
50.9
MegaPose†∗[28]
RGB-D
62.6
48.7
85.1
46.7
46.8
73.0
76.4
62.8
ZeroPose†∗[5]
RGB-D
53.8
40.0
83.5
39.2
52.1
65.3
65.3
57.0
GigaPose [41]
RGB
29.9
27.3
30.2
23.1
18.8
34.8
29.0
27.6
GigaPose† [41]
RGB
59.9
57.0
63.5
46.7
39.7
72.2
66.3
57.9
SAM-6D (Ours)
RGB-D
65.1
47.9
82.5
49.7
56.2
73.8
81.5
65.3
SAM-6D (Ours)
RGB-D
SAM-6D (FastSAM)
66.7
48.5
82.9
51.0
57.2
73.6
83.4
66.2
SAM-6D (Ours)
RGB-D
SAM-6D (SAM)
69.9
51.5
90.4
58.8
60.2
77.6
84.5
70.4
Table 2. Pose estimation results of different methods on the seven core datasets of the BOP benchmark [54]. We report the mean Average
Recall (AR) among VSD, MSSD and MSPD, as introduced in Sec. 4. The symbol ‘†’ denotes the use of pose refinement proposed in [28].
The symbol ‘∗’ denotes the results published on BOP leaderboard. Our used masks of MaskRCNN [16] are provided by CosyPose [27].
4.1. Instance Segmentation of Novel Objects
We compare our ISM of SAM-6D with ZeroPose [5] and
CNOS [40], both of which score the object proposals in
terms of semantics solely, for instance segmentation of
novel objects. The quantitative results are presented in Ta-
ble 1, demonstrating that our ISM, built on the publicly
available foundation models of SAM [26] / FastSAM [74]
and ViT (pre-trained by DINOv2 [44]), delivers superior re-
sults without the need for network re-training or finetuning.
Note that our baseline with only semantic matching score
ssem, whether based on SAM or FastSAM [74], aligns pre-
cisely with the method of CNOS; the only difference is that
we adjust the hyperparameters of SAM to generate more
proposals for scoring. Further enhancements to our base-
lines are achieved via the inclusion of appearance and ge-
ometry matching scores, i.e., sappe and sgeo, as verified in
Table 1. Qualitative results of ISM are visualized in Fig. 1.
4.2. Pose Estimation of Novel Objects
4.2.1
Comparisons with Existing Methods
We compare our PEM of SAM-6D with the representative
methods, including MegaPose [28], ZeroPose [5], and Gi-
gaPose [41], for pose estimation of novel objects. Quanti-
tative comparisons, as presented in Table 5, show that our
PEM, without the time-intensive render-based refiner [28],
outperforms the existing methods under various mask pre-
7

dictions. Importantly, the mask predictions from our ISM
significantly enhance the performance of PEM, compared
to other mask predictions, further validating the advantages
of ISM. Qualitative results of PEM are visualized in Fig. 1.
4.2.2
Ablation Studies and Analyses
We conduct ablation studies on the YCB-V dataset to evalu-
ate the efficacy of individual designs in PEM, with the mask
predictions generated by ISM based on SAM.
Efficacy of Background Tokens We address the partial-
to-partial point matching issue through a simple yet effec-
tive design of background tokens. Another existing solution
is the use of optimal transport [48] with iterative optimiza-
tion, which, however, is time-consuming. The two solutions
are compared in Table 3, which shows that our PEM with
background tokens achieves results comparable to optimal
transport, but with a faster inference speed. As the density
of points for matching increases, optimal transport requires
more time to derive the assignment matrices.
Efficacy of Two Point Matching Stages With the back-
ground tokens, we design PEM with two stages of point
matching via a Coarse Point Matching module and a Fine
Point Matching module. Firstly, we validate the effective-
ness of the Fine Point Matching module, which effectively
improves the results of the coarse module, as verified in Ta-
ble 4. Further, we evaluate the effectiveness of the Coarse
Point Matching module by removing it from PEM. In this
case, the point sets of object proposals are not transformed
and are directly used to learn the positional encodings in
the fine module. The results, presented in Table 4, indi-
cate that the removal of Coarse Point Matching significantly
degrades the performance, which may be attributed to the
large distance between the sampled point sets of the pro-
posals and target objects, as no initial poses are provided.
Efficacy of Sparse-to-Dense Point Transformers We de-
sign Sparse-to-Dense Point Transformers (SDPT) in the
Fine Point Matching module to manage dense point inter-
actions. Within each SDPT, Geometric Transformers [48]
is employed to learn the relationships between sparse point
sets, which are then spread to the dense ones via Linear
Transformers [24]. We conduct experiments on either Geo-
metric Transformers using sparse point sets with 196 points
or Linear Transformers using dense point sets with 2048
points.
The results, presented in Table 5, indicate infe-
rior performance compared to using our SDPTs. This is
because Geometric Transformers struggle to handle dense
point sets due to high computational costs, whereas Linear
Transformers prove to be ineffective in modeling dense cor-
respondence with attention along the feature dimension.
4.3. Runtime Analysis
We conduct evaluation on a server with a GeForce RTX
3090 GPU, and report in Table 6 the runtime averaged
AR
Time (s)
PEM with Optimal Transport
81.4
4.31
PEM with Background Tokens
84.5
1.36
Table 3. Quantitative results of Optimal Transport [48] and our
design of Background Tokens in the Pose Estimation Model on
YCB-V. The reported time is the average per-image processing
time of pose estimation across the entire dataset on a server with a
GeForce RTX 3090 GPU.
Coarse Point Matching
Fine Point Matching
AR
✓
×
77.6
×
✓
40.2
✓
✓
84.5
Table 4. Ablation studies on the the strategy of two point matching
stages in the Pose Estimation Model on YCB-V.
Transformer
#Point
AR
Geometric Transformer [48]
196
81.7
Linear Transformer [24]
2048
78.4
Sparse-to-Dense Point Transformer
196 →2048
84.5
Table 5. Quantitative comparisons among various types of trans-
formers employed in the Fine Point Matching module of the Pose
Estimation Model on YCB-V.
Segmentation Model
Time (s)
Instance Segmentation
Pose Estimaiton
All
FastSAM [74]
0.45
0.98
1.43
SAM [26]
2.80
1.57
4.37
Table 6. Runtime of SAM-6D with different segmentation mod-
els. The reported time is the average per-image processing time
across the seven core datasets of BOP benchmark on a server with
a GeForce RTX 3090 GPU.
on the seven core datasets of BOP benchmark, indicating
the efficiency of SAM-6D which avoids the use of time-
intensive render-based refiners. We note that SAM-based
method takes more time on pose estimation than FastSAM-
based one, due to more object proposals generated by SAM.

## experiments
In this section, we conduct experiments to evaluate our pro-
posed SAM-6D, which consists of an Instance Segmenta-
tion Model (ISM) and a Pose Estimation Model (PEM).
Datasets We evaluate our proposed SAM-6D on the seven
core datasets of the BOP benchmark [54], including LM-O,
T-LESS, TUD-L, IC-BIN, ITODD, HB, and YCB-V. PEM
is trained on the large-scale synthetic ShapeNet-Objects [4]
and Google-Scanned-Objects [9] datasets provided by [28],
with a total of 2, 000, 000 images across ∼50, 000 objects.
Implementation Details For ISM, we follow [40] to utilize
the default ViT-H SAM [26] or FastSAM [74] for proposal
generation, and the default ViT-L model of DINOv2 [44]
to extract class and patch embeddings. For PEM, we set
N c
m = N c
o = 196 and N f
m = N f
o = 2048, and use In-
foNCE loss [43] to supervise the learning of attention ma-
trices (5) for both matching stages. We use ADAM to train
PEM with a total of 600,000 iterations; the learning rate is
initialized as 0.0001, with a cosine annealing schedule used,
and the batch size is set as 28. For each object, we use two
rendered templates for training PEM. During evaluation, we
follow [40] and use 42 templates for both ISM and PEM.
Evaluation Metrics For instance segmentation, we re-
port the mean Average Precision (mAP) scores at different
Intersection-over-Union (IoU) thresholds ranging from 0.50
to 0.95 with a step size of 0.05. For pose estimation, we
report the mean Average Recall (AR) w.r.t three error func-
tions, i.e., Visible Surface Discrepancy (VSD), Maximum
Symmetry-Aware Surface Distance (MSSD) and Maximum
Symmetry-Aware Projection Distance (MSPD). For further
details about these evaluation metrics, please refer to [54].
6

## related_work
2.1. Segment Anything
Segment Anything (SA) [26], is a promptable segmenta-
tion task that focuses on predicting valid masks for various
types of prompts, e.g., points, boxes, text, and masks. To
tackle this task, the authors propose a powerful segmenta-
tion model called Segment Anything Model (SAM), which
comprises three components, including an image encoder,
a prompt encoder and a mask decoder. SAM has demon-
strated remarkable zero-shot transfer segmentation perfor-
mance in real-world scenarios, including challenging situa-
tions such as medical images [36, 37, 71], camouflaged ob-
jects [22, 55], and transparent objects [13, 23]. Moreover,
SAM has exhibited high versatility across numerous vision
applications [69], such as image inpainting [35, 62, 64, 67],
object tracking [15, 65, 73], 3D detection and segmentation
[2, 66, 70], and 3D reconstruction [3, 49, 59].
Recent studies have also investigated semantically seg-
menting anything due to the critical role of semantics in
vision tasks. Semantic Segment Anything (SSA) [6] is pro-
posed on top of SAM, aiming to assign semantic categories
to the masks generated by SAM. Both PerSAM [72] and
Matcher [34] employ SAM to segment the object belong-
ing to a specific category in a query image by searching for
point prompts with the aid of a reference image containing
an object of the same category. CNOS [40] is proposed to
segment all instances of a given object model, which firstly
generates mask proposals via SAM and subsequently fil-
ters out proposals with low feature similarities against ob-
ject templates rendered from the object model.
For efficiency, FastSAM [74] is proposed by utilizing
instance segmentation networks with regular convolutional
networks instead of visual transformers used in SAM. Ad-
ditionally, MobileSAM [68] replaces the heavy encoder of
SAM with a lightweight one through decoupled distillation.
2.2. Pose Estimation of Novel Objects
Methods Based on Image Matching Methods within this
group [1, 28, 33, 38, 39, 41, 42, 46, 50] often involve com-
paring object proposals to templates of the given novel ob-
jects, which are rendered with a series of object poses, to re-
trieve the best-matched object poses. For example, Gen6D
[33], OVE6D [1], and GigaPose [41] are designed to se-
lect the viewpoint rotations via image matching and then
estimate the in-plane rotations to obtain the final estimates.
MegaPose [28] employs a coarse estimator to treat image
matching as a classification problem, of which the recog-
nized object poses are further updated by a refiner.
Methods Based on Feature Matching Methods within this
group [5, 10, 11, 17, 20, 53] align the 2D pixels or 3D points
of the proposals with the object surface in the feature space
[21, 52], thereby building correspondence to compute ob-
ject poses. OnePose [53] matches the pixel descriptors of
proposals with the aggregated point descriptors of the point
sets constructed by Structure from Motion (SfM) for 2D-3D
correspondence, while OnePose++ [17] further improves
it with a keypoint-free SfM and a sparse-to-dense 2D-3D
matching model. ZeroPose [5] realizes 3D-3D matching
via geometric structures, and GigaPose [41] establishes
2D-2D correspondence to regress in-plane rotation and 2D
scale. Moreover, [11] introduces a zero-shot category-level
6D pose estimation task, along with a self-supervised se-
mantic correspondence learning method. Unlike the above
one-stage point matching work, the unique contributions in
our Pose Estimation Model are: (a) a two-stage pipeline
that boosts performance by incorporating coarse correspon-
dence for finer matching, (b) an efficient design of back-
ground tokens to eliminate the need of optimal transport
with iterative optimization [48], and (c) a Sparse-to-Dense
Point Transformer to effectively model dense relationship.
3

3. Methodology of SAM-6D
We present SAM-6D for zero-shot 6D object pose estima-
tion, which aims to detect all instances of a specific novel
object, unseen during training, along with their 6D object
poses in the RGB-D images.
To realize the challenging
task, SAM-6D breaks it down into two steps via two dedi-
cated sub-networks, i.e., an Instance Segmentation Model
(ISM) and a Pose Estimation Model (PEM), to first seg-
ment all instances and then individually predict their 6D
poses, as shown in Fig. 2. We detail the architectures of
ISM and PEM in Sec. 3.1 and Sec. 3.2, respectively.
3.1. Instance Segmentation Model
SAM-6D uses an Instance Segmentation Model (ISM) to
segment the instances of a novel object O. Given a cluttered
scene, represented by an RGB image I, ISM leverages the
zero-shot transfer capabilities of Segment Anything Model
(SAM) [26] to generate all possible proposals M. For each
proposal m ∈M, ISM calculates an object matching score
sm to assess the matching degree between m and O in terms
of semantics, appearance, and geometry. The matched in-
stances with O can then be identified by simply setting a
matching threshold δm.
In this subsection, we initially provide a brief review of
SAM in Sec. 3.1.1 and then explain the computation of the
object matching score sm in Sec. 3.1.2.
3.1.1
Preliminaries of Segment Anything Model
Given an RGB image I, Segment Anything Model (SAM)
[26] realizes promptable segmentation with various types
of prompts Pr, e.g., points, boxes, texts, or masks. Specif-
ically, SAM consists of three modules, including an image
encoder ΦImage, a prompt encoder ΦPrompt, and a mask de-
coder ΨMask, which could be formulated as follows:
  \ m athcal {M}, \mat hcal {C} = \Psi _{\text {Mask}} (\Phi _{\text {Image}}(I), \Phi _{\text {Prompt}}(\mathcal {P}_r)), \label {eqn:rgb sam} 
(1)
where M and C denote the predicted proposals and the cor-
responding confidence scores, respectively.
To realize zero-shot transfer, one can prompt SAM with
evenly sampled 2D grids to yield all possible proposals,
which can then be filtered based on confidence scores, re-
taining only those with higher scores, and applied to Non-
Maximum Suppression to eliminate redundant detections.
3.1.2
Object Matching Score
Given the proposals M, the next step is to identify the ones
that are matched with a specified object O by assigning each
proposal m ∈M with an object matching score sm, which
comprises three terms, each evaluating the matches in terms
of semantics, appearance, and geometry, respectively.
Following [40], we sample NT object poses in SE(3)
space to render the templates {Tk}NT
k=1 of O, which are fed
into a pre-trained visual transformer (ViT) backbone [8] of
DINOv2 [45], resulting in the class embedding f cls
Tk and
N patch
Tk
patch embeddings {f patch
Tk,i }
Npatch
Tk
i=1
of each template
Tk. For each proposal m, we crop the detected region out
from I, and resize it to a fixed resolution. The image crop
is denoted as Im and also processed through the same ViT
to obtain the class embedding f cls
Im and the patch embed-
dings {f patch
Im,j }
N patch
Im
j=1
, with N patch
Im
denoting the number of
patches within the object mask. Subsequently, we calculate
the values of the individual score terms.
Semantic Matching Score We compute a semantic score
ssem through the class embeddings by averaging the top K
values from {
<f cls
Im,f cls
Tk >
|f cls
Im|·|f cls
Tk | }NT
k=1 to establish a robust measure
of semantic matching, with <, > denoting an inner product.
The template that yields the highest semantic value can be
seen as the best-matched template, denoted as Tbest, and is
used in the computation of the subsequent two scores.
Appearance Matching Score Given Tbest, we compare Im
and Tbest in terms of appearance using an appearance score
sappe, based on the patch embeddings, as follows:
  s_{ a
p
p e} = 
\f
r ac {1
}{
N
^{p
atc
h}_{\math cal {
I}_{m
} } }\sum
 _{j = 1 }^{N^
{patch} _
{\ mathc
al { I } _{ m}}} 
\max _{i
=1,\dots ,N^{patch}_{\mathcal {T}_{best}}} \frac {<\bm {f}^{patch}_{\mathcal {I}_{m},j}, \bm {f}^{patch}_{\mathcal {T}_{best},i}>}{|\bm {f}^{patch}_{\mathcal {I}_{m},j}|\cdot |\bm {f}^{patch}_{\mathcal {T}_{best},i}|}. \label {eqn:s_ape} 
(2)
sappe is utilized to distinguish objects that are semantically
similar but differ in appearance.
Geometric Matching Score In terms of geometry, we score
the proposal m by considering factors like object shapes and
sizes. Utilizing the object rotation from Tbest and the mean
location of the cropped points of m, we have a coarse pose
to transform the object O, which is then projected onto the
image to obtain a compact bounding box Bo. Afterwards,
the Intersection-over-Union (IoU) value between Bo and the
bounding box Bm of m is used as the geometric score sgeo:
  s_ { ge
o } 
= 
\ fr
ac {\mathcal {B}_m \bigcap \mathcal {B}_o}{\mathcal {B}_m \bigcup \mathcal {B}_o}. \label {eqn:s_geo} 
(3)
The reliability of sgeo is easily impacted by occlusions. We
thus compute a visible ratio rvis to evaluate the confidence
of sgeo, which is detailed in the supplementary materials.
By combining the above

## conclusion
In this paper, we take Segment Anything Model (SAM) as
an advanced starting point for zero-shot 6D object pose es-
timation, and present a novel framework, named SAM-6D,
which comprises an Instance Segmentation Model (ISM)
and a Pose Estimation Model (PEM) to accomplish the task
in two steps. ISM utilizes SAM to segment all potential
object proposals and assigns each of them an object match-
ing score in terms of semantics, appearance, and geome-
try. PEM then predicts the object pose for each proposal by
solving a partial-to-partial point matching problem through
two stages of Coarse Point Matching and Fine Point Match-
ing. The effectiveness of SAM-6D is validated on the seven
core datasets of BOP benchmark, where SAM-6D signifi-
cantly outperforms existing methods.
8