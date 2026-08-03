# Open-vocabulary object 6D pose estimation

> 2024 · id: W4402753839 · arXiv: 2312.00690 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
We introduce the new setting of open-vocabulary object
6D pose estimation, in which a textual prompt is used to
specify the object of interest. In contrast to existing ap-
proaches, in our setting (i) the object of interest is speci-
fied solely through the textual prompt, (ii) no object model
(e.g., CAD or video sequence) is required at inference, and
(iii) the object is imaged from two RGBD viewpoints of dif-
ferent scenes. To operate in this setting, we introduce a
novel approach that leverages a Vision-Language Model to
segment the object of interest from the scenes and to esti-
mate its relative 6D pose. The key of our approach is a
carefully devised strategy to fuse object-level information
provided by the prompt with local image features, result-
ing in a feature space that can generalize to novel concepts.
We validate our approach on a new benchmark based on
two popular datasets, REAL275 and Toyota-Light, which
collectively encompass 34 object instances appearing in
four thousand image pairs. The results demonstrate that
our approach outperforms both a well-established hand-
crafted method and a recent deep learning-based base-
line in estimating the relative 6D pose of objects in dif-
ferent scenes. Code and dataset are available at https:
//jcorsetti.github.io/oryon.

## introduction
The new setting is defined as follows: (i) the object of in-
terest is specified solely through textual prompt (no object
model or video sequence is required); (ii) the object is im-
aged from two different viewpoints of two different scenes;
and (iii) the object was not observed during the training

Feature
training
Segmentation
training
Ground-truth
matches
a brown
open laptop
Back-projecting
 matches to 3d
Matching by mask and
nearest neighbor
A
Q
T
ϕV
ϕV
ϕT
EQ
EA
eT
ϕG
ϕG
ϕT V
ϕT V
ϕD
ϕD
CQ
CA
FA
MA
FQ
MQ
Registration
TA→Q
ℓF
ℓM
Training
Testing
Figure 2. The training pipeline of Oryon can be separated in four stages. In the first stage, the pair of images A, Q are encoded by the
CLIP image encoder ϕV , while the prompt T is encoded by the CLIP text encoder ϕT . The guidance network ϕG is used to produce a
rich visual representation which is used in the next stages. The resulting visual feature maps EA, EQ and text features eT are used in the
second stage in the fusion module ϕT V . This outputs a pair of cost features CA,CQ which in the third stage are upsampled to the final
feature maps FA, FQ. The same feature maps are fed to a segmentation head to obtain the predicted masks MA, MQ. At train time, FA,
FQ are optimized by a contrastive loss, while a dice loss supervises the training of the segmentation masks. At test time, the predicted
masks are used to filter FA, FQ, and matches are obtained by nearest neighbor. Finally, the matches are projected back to the 3D domain,
and a registration algorithm is used to retrieve the final pose TA→Q.
phase. From these requirements, we formulate the prob-
lem as a relative pose estimation task [31, 51] between two
scenes, the anchor A and the query Q, which are repre-
sented as RGB image RGBA (RGBQ) and depth map DA
(DQ). To match A and Q we propose Oryon, which relies
on a fusion module based on cost-aggregation that relates
the information of the textual prompt to the local visual fea-
ture map. The model generalizes to unseen objects and is
trained on a large dataset of object models annotated with
brief textual descriptions [50].
We address the localization problem jointly with the
matching task, by predicting a segmentation mask on each
view (scene) to locate the object of interest. The final fea-
tures encode semantic information about the objects for the
segmentation task and geometric information, as the pose is
derived from the relative position of keypoints. At test time,
we use the predicted masks to select points on the object
of interest and match pairs of features by nearest neighbor.
The resulting matches are projected back to the 3D domain
to obtain the final relative object pose across the two scenes
through point cloud registration [1].
Fig. 2 shows Oryon’s architecture. We describe the ob-
ject of interest with a user-provided textual prompt T. We
use T to guide the extraction of keypoints and to compute
the matches, as only the points which match the textual
description provided by the prompt itself are considered
through the use of the mask. Given a pair of scenes (A,
Q), we compute a set of coordinates xA and xQ that form a
set of matches between scenes (i.e., they describe the most
similar locations in color and 3D structure), which are fil-
tered by the predicted masks MA, MQ.
In order to jointly process T and (A, Q), their represen-
tations should share the same feature space. We achieve
this by leveraging a VLM trained for semantic alignment,
such as CLIP [35]. We process RGBA and RGBQ with the
CLIP image encoder ϕV , and extract the features before the
final pooling layer. Let (EA, EQ) ∈RD×H×H be a pair
of feature maps extracted from A and Q, where D is the
feature dimension, and H is the spatial dimension of the
feature map. D and H depend on the CLIP encoder used.
Note that RGBA and RGBQ are processed without any prior
crop, unlike other pose estimation approaches which adopt
prior detectors or segmenters [13, 24, 40, 42, 46]. T is en-
coded by the CLIP text encoder ϕT . To provide a rich set
of representations, we adopt a set of templates to generate
N versions of T [4, 35], which are encoded by ϕT . Hence,
we obtain the prompt features eT ∈RN×D, where N is the
number of templates. Both CLIP encoders are frozen.
3.2. Text-visual fusion
The objective of the text-visual module ϕT V is to provide a
visual representation which is semantically consistent with
the represented object (and therefore influenced by eT ), but
which is also representative of the object local appearance
in EA and EQ. The first requirement is fundamental for the
segmentation task, while the second is needed to perform

image matching. We implement the fusion module ϕT V by
building a cost volume, i.e. by computing the cosine simi-
larity among each feature map location in EA, EQ and eT ,
thus obtaining a pair of matrices of shape RN×H×H. The
cost volume represents the cost of associating each prompt
feature to each location in the visual feature maps.
Note that the cost volume obtained does not enforce a
spatial consistency between neighboring patches. In order
to let the model learn these relations and refine the cost
volume, we adopt a cost aggregation block based on two
Transformer layers [25]. Both layers rely on self-attention
to model the relationship between the image patches, with
different modalities. The first applies self-attention within
the same window, while the second applies self-attention
among patches of shifted windows. This allows the model
to perform attention both within the patch and between
neighboring patches. Objects can have different shapes and
sizes in the image, therefore only applying local attention
can not be sufficient to build an effective representation. To
enrich the representation provided by CLIP, we adopt guid-
ance features [4] from another backbone ϕG which are con-
catenated to the query and keys of the Transformer layers.
The output of the fusion module is a pair of cost features
CA, CQ ∈RD×H×H.
3.3. Decoding
The feature maps obtained in the previous step have the
original low resolution of the CLIP feature map (24×24
in our case).
As we pursue an image matching objec-
tive, we require a higher resolution to compute fine-grained
matches. To this end, we adopt a decoder ϕD composed
by three upsampling layers. Note that CLIP was trained for
semantic alignment of the global feature, and this property
partially transfers to its visual feature map [52]. However,
in our setting we also require information based on the ap-
pearance of the object, as opposed to semantic information
only. Therefore, we find beneficial to add the guidance fea-
tures of ϕG in the decoder. Specifically, the two feature
maps obtained by the guidance network are projected and
concatenated to the input feature map before each upsam-
pling layer [4]. The final layer does not use any guidance.
The resulting feature maps FA, FQ are used both for
computing the matches and for the segmentation task. For
the latter, we add a segmentation head to compute the acti-
vations and output a pair of binary masks MA, MQ.
3.4. Optimization
Our formulation implies two optimization objectives. For
effective image matching, we directly optimize the feature
maps by forcing low similarity between dissimilar locations
and high similarity between similar locations across A and
Q. As supervision, we adopt the ground-truth matches be-
tween A and Q (i.e. pair of points on A and Q which be-
long to the same portion of the object of interest). In prac-
tice, we adopt a contrastive loss ℓF with hardest negative
mining [5, 6]. There are two components: the first pro-
motes the corresponding features (i.e. the matches) to be
close in the feature space, while the second increases the
distance between a feature and its hardest negative. Given
the pair of scenes A, Q, the set of positive pairs is defined
as P = {(i, j): xA
i
∈A, xQ
j
∈Q, ϕ(xA
i ) = xQ
j }, where
ϕ: A →Q is a match mapping between A and Q pixels.
We define f A, f Q ∈RC×D as the set of features extracted
respectively from the feature maps FA, FQ, by using the
ground-truth matches P. C = |P| is the total number of
matches and D is the feature dimension. The positive loss
component ℓP is defined as
 \ l
a
bel {eq
:
hcp
o
s} \ell
 _ P 
= \ su
m
 _{(i, j) \in \mathcal {P}} \frac {1}{\lvert \mathcal {P} \rvert } \left ( \mathrm {dist} ( \textbf {f}^A_i, \textbf {f}^Q_j) - \mu _P \right )_+, 
(1)
where µP is a positive margin and (·)+ = max(0, ·). The
positive margin is an hyperparameter, and represents the de-
sired distance in the feature space of a positive pair.
To define negative pairs, we consider a set of features
f and their corresponding 2D coordinates x on the image.
Given a single feature fi, we define its set of candidate neg-
atives as Ni = {k: xk ∈x, k̸ = i, ∥xi −xk∥≥τ}. Note
that this set excludes all the points which are closer than the
distance τ from the reference point xi in the image, so that
features describing the same points

## method
Segm. prior AR ↑VSD ↑MSSD ↑MSPD ↑ADD ↑mIoU ↑
1
SIFT [27]
Oracle
30.3
7.3
39.6
44.1
14.1
100.0
2
OVSeg [21] 25.8
6.4
34.2
36.9
11.8
75.5
3
Ours
27.2
5.7
35.4
40.6
9.9
68.1
4
Obj.Mat. [11]
Oracle
9.8
2.4
13.0
14.0
5.4
100.0
5
OVSeg [21]
9.2
2.6
12.1
13.0
5.3
75.5
6
Ours
8.3
2.2
10.5
12.1
3.8
68.1
7
Oryon
Oracle
34.1
13.9
42.9
45.5
22.9
100.0
8
OVSeg [21] 29.2
11.9
36.8
38.9
18.9
75.5
9
Ours
30.3
12.1
37.5
41.4
20.9
68.1
10 ∆score
+3.1
+6.4
+2.1
+0.8
+11.0
-7.4
on ScanNet [7] from the official repository. We also com-
pare Oryon with a pipeline based on SIFT features [27] and
PointDSC [1]. SIFT is used to extract keypoints and de-
scriptors, which are then filtered by using the segmentation
prior. As in Oryon, the resulting features are used to com-
pute matches between A and Q, which are unprojected in
3D and used to register the point clouds with PointDSC.
We report results with different segmentation priors: (i)
the mask predicted by our method (Ours), (ii) the mask pre-
dicted with OVSeg [21], a method for open-vocabulary seg-
mentation, and (iii) the ground-truth mask (Oracle).
4.5. Quantitative results
Tab. 1 reports the results on REAL275. When OVSeg is
used as the image segmenter, Oryon outperforms SIFT by
+8.1 in AR and ObjectMatch by +11.5 in AR. With our
segmentation head as the image segmenter, Oryon shows a
performance increase over SIFT by +7.8 in AR and over
ObjectMatch by +9.8 in AR. We attribute the smaller per-
formance gap between Oryon and SIFT compared to that
with ObjectMatch to the latter’s greater sensitivity to do-
main shifts. The performance gap between Oryon based
on its own predicted masks and the Oracle masks is −14.3
AR, which indicates that a portion of the objects are not
estimated correctly due to errors in the segmentation.
Tab. 2 reports the results on TOYL. When OVSeg is used
as segmenter, Oryon outperforms SIFT by +3.4 AR and
ObjectMatch by +20.0 AR. With our segmentation head as
the image segmenter, Oryon shows a performance increase
over SIFT by +3.1 in AR and over ObjectMatch by +22.0
in AR. In this dataset, SIFT performances are closer to ours
than in REAL275. On the contrary, ObjectMatch performs
much worse. The lower performances of ObjectMatch are
due to its sensitivity to the domain shift, which in TOYL
is more present due to the different light conditions in the
scene pair. Another characteristic of TOYL is its high vari-
ation in poses, for which objects can have very different ap-
parent sizes in the two images. SIFT scale-invariant design
allows it to tackle this variance in appearance and reach bet-
ter results than ObjectMatch. We also observe that the per-
formance gap between Oryon with its own masks and the
Oracle ones is narrower than in REAL275. This suggests
that Oryon does not require highly accurate segmentation
masks, as PointDSC is able to reject spurious matches.
4.6. Qualitative results
We report in Figs. 3, 4 some qualitative results on
REAL275 [43] and TOYL [16], respectively. The predicted
mask correctly localizes the objects, but ObjectMatch fails
at estimating an accurate pose, due to translation (Fig. 3,
top) or rotation errors, (Fig. 3, bottom). Due to the small
objects size, this scenario is challenging for ObjectMatch,
while Oryon’s higher resolution allows it to retrieve a cor-
rect pose also in this case. Despite small errors in the trans-
lation component, SIFT reaches a good degree of accuracy.
On TOYL the results show that our method can handle
different light conditions between A and Q. On the other
hand, we observe that the translation component is not as
accurate as in REAL275 (see Fig. 4, bottom). Both Ob-
jectMatch and SIFT present large errors in this challenging
dataset. SIFT in particular shows large errors in translation,
which suggests that such hand-crafted features are unsuit-
able for this scenario with different light conditions.
4.7. Ablation study
We report in Tab. 3 the results obtained on REAL275 when
we remove one of the modules of our architecture.
In
rows 1, 2 we remove the feature guidance respectively from
the fusion module ϕT V and from the decoder ϕD. Both
changes result in a significant drop both in pose estimation
performance (−7.4 and −11.0 in AR respectively) and in
the segmentation (−8.0 and −12.9 in mIoU respectively).
This confirms the intuition that adding the guidance from a
general-purpose image encoder ϕG [25] can specialize the
semantic features of CLIP [35] for our task. Removing the

Anchor
Ground-truth
ObjectMatch [11]
SIFT [27]
Ours
Prompt: Light blue mug
Prompt: White tall can
Figure 3. Examples of qualitative pose results from the REAL275 [43] dataset. All the results use the segmentation mask predicted by
Oryon. We show the object model colored my mapping its 3D coordinates to the RGB space.
Anchor
Ground-truth
ObjectMatch [11]
SIFT [27]
Ours
Prompt: Green sudoku magazine
Prompt: Green plastic bottle
Figure 4. Examples of qualitative pose results from the TOYL [16] dataset. All the results use the segmentation mask predicted by Oryon.
We show the object model colored my mapping its 3D coordinates to the RGB space.
upsampling layer (row 3) results in a worse performance on
pose estimation, while the impact on the segmentation is
small (−7.1 and −1.5 in AR and mIoU respectively). This
highlights the importance of a higher-resolution feature map
to compute the matches, as only using two upsampling lay-
ers result in a feature map of size 96 × 96 instead of our
default 192 × 192. In row 4, we replace PointDSC with a
RANSAC-based algorithm from a state-of-the-art pose es-
timation method [46]. This causes an important drop in AR
(−16.2), which shows the importance of a registration algo-
rithm capable of dealing with spurious matches [1].
In Tab. 4 we report the results of two experiments on
REAL275 in which we train Oryon for a single task only
(segmentation in rows 1-2, and matching in row 3). When
training only with the segmentation loss ℓM the features
learned by Oryon led to worse pose estimation performance
than our baseline, as the AR changes by −7.4 when using
the predicted mask (row 1 vs row 5). Also the segmenta-
tion performance is lower (−2.2 mIoU): this suggests that
jointly training Oryon for the two tasks can benefit the seg-
mentation performance. In row 3 we only train Oryon with
the feature loss ℓF . In this case, the AR drops by −6.2 when
comparing with the same segmenter (row 3 vs row 4). This
confirms our intuition that jointly training a model for pose
estimation and segmentation can benefit both tasks.
In Tab. 5 we report the results of the evaluation on
REAL275 with different prompts. In rows 1-2 the object
name is replaced by “object”, in order to simulate the pos-
sible behaviour of a user describing an unknown object, for
which a name cannot be provided. In this case, the seg-
mentation completely fails, and subsequently the predicted
poses are poor (−29.2 AR and −63.5 mIoU with respect to

Table
3.
Ablation
on
the

## experiments
4.1. Experimental setup
We train our model for 20 epochs, and adopt CLIP ViT-
L/14 [35] as visual and text encoder. We use Adam [18]
with learning rate 10−4, weight decay 5·10−4, and a cosine
annealing scheduler [26] to lower the learning rate to 10−5.
Loss weights are set as λP = λN = 0.5 and λM = 1.0.
As guidance backbone, we adopt a pretrained Swin Trans-
former [25]. As augmentations we randomly apply color
jittering, horizontal flipping and vertical flipping.
We set the positive and negative margins in Eqs. (1) and
(2) as µP = 0.2 and µN = 0.9, and the excluding distance
for hardest negatives as τ = 5. At test time we set µt = 0.25
as maximum feature distance to identify a match, and limit
the match number to C = 500. The output resolution of FA,
FQ is 192 × 192, and the feature dimension is F = 32.
4.2. Datasets
We use the synthetic dataset ShapeNet6D [50] for train-
ing, as it provides several diverse objects and scenes. We
evaluate Oryon on two real-world datasets: REAL275 [43]
and Toyota-Light [16]. The standard prompt T we adopt at
test time is composed by the object name (e.g., “laptop”)
preceded by a brief description (e.g., “brown open”). For
REAL275 and TOYL, we manually annotate each object
instance with a textual prompt, while for ShapeNet6D we
rely on its metadata. For all datasets, we train and test on a
set of scene pairs of images from the original datasets.
ShapeNet6D (SN6D) [50] is a large-scale synthetic dataset
built by using ShapeNetSem [37] object models. For each
object model, ShapeNetSem includes a name and a set of
synonyms, which we use to build the textual prompt. Note
that in this dataset the prompt is only composed by the ob-
ject name, as no description is provided.
To enrich the
learned representations, during training we randomly sub-
stitute the object name in the prompt with one of the syn-
onyms provided in the metadata (e.g., possible synonyms
for “television” are “tv” and “telly”). Note that ShapeNet6D
does not contain the object instances present in the two test
datasets, although it may contain objects of similar cate-
gory. We train on 20K image pairs from SN6D.
REAL275 [43] provides a set of RGBD images in different
scenes, with a total of six object categories and three in-
stances for each category. REAL275 provides an high vari-
ety of viewpoints between the scenes, and also present mild
occlusions. We evaluate on 2K image pairs from the origi-
nal real-world test set.
Toyota-Light (TOYL) [16] focuses on pose estimation un-
der challenging light conditions, in which a single object
is present in each scene. This is relevant in our setting, as
we process pairs of images: a significant difference in light
across the two scenes poses an important challenge in the
image matching task. We evaluate on 2K image pairs.
4.3. Evaluation metrics
We evaluate all our pose results by using the metrics pro-
posed for the BOP Benchmark [17], namely AR (Average
Recall), which is the average of VSD, MSSD and MSPD.
We also report ADD(S)-0.1d, as it is typically used in 6D
pose estimation [14, 15, 24]. ADD(S)-0.1d is a recall on
the pose error: it considers a success when the error is less
than one tenth of the object diameter. Instead, VSD, MSSD
and MSPD are averages of recalls on multiple thresholds.
These latter metrics are more suitable to represent perfor-
mance in this challenging scenario. Therefore, we adopt
AR as main metric. See the Supplementary Material for an
extended discussion of the pose metrics. For all experiments
we also report the mean Intersection-Over-Union (mIoU) to
quantify the predicted mask quality [4, 52].
4.4. Comparison procedure
Our setting is fundamentally different from the ones of
current 6D pose estimation methods.
Gen6D [24] and
OnePose [42] both rely on video sequences of the novel ob-
jects, while methods for relative pose estimation [31, 51]
only work with RGB images and do not estimate the trans-
lation component. Therefore, we compare Oryon with Ob-
jectMatch [11], a state-of-the-art method designed for point
cloud registration with low overlap. This is consistent with
our scenario, as we assume that the only overlapping sec-
tion between two scenes is the one showing the query ob-
ject. ObjectMatch is based on SuperGlue [36] to estimate
the matches, and on a custom pose estimator. We run Super-
Glue on each image pair. As in our approach, the predicted
mask is used to reject all matches on the background, and
the resulting matches are passed to the final ObjectMatch
module that outputs the pose. We use the model trained

Table 1. Results on REAL275 [43]. We report in bold font and
underlined respectively the best and second best result when using
our predicted mask, and highlight our main result. The ∆score
is the difference between Oryon and the nearest competitor. Key:
Obj.Mat.: ObjectMatch, ADD: ADD(S)-0.1d.

## related_work
Model-free pose estimation of novel objects. Gen6D [24]
represents one of the first approach to tackle the pose esti-
mation of novel objects without CAD models, using only
RGB images. Gen6D implements an object-agnostic detec-
tor, a viewpoint selector, and a pose refinement step. During
evaluation, Gen6D requires the acquisition of an RGB video
sequence of the novel object, upon which a Structure-from-
Motion method (COLMAP [38]) is applied to recover the
camera pose for each frame. However, a significant limi-
tation of Gen6D is the complexity of its evaluation proce-
dure, which also requires manual cropping and orientation
of the reconstructed point cloud. Similarly, OnePose [42]
demands an RGB video sequence of the novel object and
a comparable procedure at test time. OnePose++ [13] ex-
tends OnePose to address pose estimation for low-textured
objects, but maintains the same test-time requirements as
OnePose. In contrast, Oryon eliminates the need for SfM
at test time, and only requires a textual description of the
object, which can be effortlessly provided by a user without
technical expertise.
Relative pose estimation aims to estimate the pose of an
object in a scene (query) with respect to a reference view
(anchor) [22, 31, 32, 51]. NOPE [31] trains an autoencoder
to learn a representation conditioned on the relative orien-
tation of the query image with the reference image. At test
time, a set of reference features are produced, each associ-
ated with an orientation. The orientation of the reference
feature more similar to the query one is chosen. Similarly
to Oryon, NOPE is evaluated on novel objects and does
not use object models. However, NOPE only predicts the
relative rotation and has not been tested in real-world sce-
narios. LatentFusion [32] shares a similar setting, but by
adopting RGBD scenes instead of RGB only is capable of
estimating also the translation component. A similar task is
6D pose tracking, where several model-free methods have
been recently proposed [30, 44, 45]. In this setting, the rela-
tive pose is estimated between pairs of consecutive frames.
Methods like BundleTrack [44] and BundleSDF [45] rely
on the information provided by previous frames by storing
their representation in a memory pool. Instead, our refer-
ence is a single viewpoint image. Moreover, the segmen-
tation module allows Oryon to tackle the case in which the
query and reference frames are acquired in different scenes.
See the Supplementary Material for an extended discussion
of relative pose estimation methods.
Point cloud registration. A related problem to RGBD-
based 6D pose estimation is point cloud registration. This
task is commonly addressed by using specific methods for
feature extraction [5, 6, 8, 33, 34] paired by matching and
registration, or by end-to-end optimization [48, 49]. A re-
cent advancement on this field is ObjectMatch [11], which
tackles the problem of registration of two scene fragments
captured from two 3D viewpoints with low overlap. This is
achieved by implicitly computing object matches through
the detection of objects in the scenes. Our scenario is sig-
nificantly more challenging as the only overlapping parts of
the scenes are the ones belonging to the object of interest.
Therefore, our problem could be considered as a particular
case of point cloud registration, in which the point cloud
parts to be aligned are described by the textual prompt.
3. Our approach

## conclusion
We present Oryon, an approach to tackle generalizable pose
estimation from a new perspective. Instead of relying on
a visual representation of the novel object and on complex
onboarding procedures, we use a textual description of the
object of interest, which can be provided by a non-expert
user. We show that our approach is successful in the chal-
lenging scenario in which the image pair shows different
scenes. Oryon accomplishes this by jointly addressing im-
age matching and segmentation in an open-vocabulary ap-
proach, and by leveraging on a text-visual fusion module.
We provide extensive ablation studies to assess the impor-
tance of each component and of the prompt composition.
Limitations. Oryon requires depth maps in order to run the
registration algorithm and retrieve the relative pose. This
limits the applicability of our method to contexts in which
the depth information is available. Another limitation is in
providing a description for certain objects (e.g., mechanical
components) in order to express the prompt.
Future work. Oryon could be adapted to work with RGB
images only by considering depth prediction.
Acknowledgements.
This work was supported by the
European Union’s Horizon Europe research and innova-
tion programme under grant agreement No 101058589 (AI-
PRISM), and it made use of time on the Tier 2 HPC facility
JADE2, funded by EPSRC (EP/T022205/1).