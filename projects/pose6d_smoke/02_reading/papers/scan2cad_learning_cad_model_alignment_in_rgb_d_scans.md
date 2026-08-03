# Scan2CAD: Learning CAD Model Alignment in RGB-D Scans

> 2019 · id: W2903435684 · arXiv: 1811.11187 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
We present Scan2CAD1, a novel data-driven method
that learns to align clean 3D CAD models from a shape
database to the noisy and incomplete geometry of an RGB-
D scan. For a 3D reconstruction of an indoor scene, our
method takes as input a set of CAD models, and predicts a
9DoF pose that aligns each model to the underlying scan
geometry. To tackle this problem, we create a new scan-
to-CAD alignment dataset based on 1506 ScanNet scans
with 97607 annotated keypoint pairs between 14225 CAD
models from ShapeNet and their counterpart objects in the
scans. Our method selects a set of representative keypoints
in a 3D scan for which we ﬁnd correspondences to the CAD
geometry. To this end, we design a novel 3D CNN archi-
tecture to learn a joint embedding between real and syn-
thetic objects, and thus predict a correspondence heatmaps.
Based on these correspondence heatmaps, we formulate
a variational energy minimization that aligns a given set
of CAD models to the reconstruction.
We evaluate our
approach on our newly introduced Scan2CAD benchmark
where we outperform both handcrafted feature descriptor
as well as state-of-the-art CNN based methods by 21.39%.
1The Scan2CAD dataset is publicly released along with an automated
benchmark script for testing under www.Scan2CAD.org

## introduction
Task
We address alignment between clean CAD models
and noisy, incomplete 3D scans from RGB-D fusion, as il-
lustrated in Fig. 1. Given a 3D scene S and a set of 3D CAD
models M = {mi}, the goal is to ﬁnd a 9DoF transforma-
tion Ti (3 degrees for translation, rotation, and scale each)
for every CAD model mi such that it aligns with a semanti-
cally matching object O = {oj} in the scan. One important
note is that we cannot guarantee the existence of 3D models
which exactly matches the geometry of the scan objects.
Dataset and Benchmark
In Sec. 4, we introduce the con-
struction of our Scan2CAD dataset. We propose an anno-
tation pipeline designed for use by trained annotators. An
annotator ﬁrst inspects a 3D scan and selects a model from a
CAD database that is geometrically similar to a target object
in the scan. Then, for each model, the annotator deﬁnes cor-
responding keypoint pairs between the model and the object
in the scan. From these keypoints, we compute ground truth
9DoF alignments. We annotate the entire ScanNet dataset
and use the original training, validation, and test splits to
establish our alignment benchmark.
Heatmap Prediction Network
In Sec. 5, we propose a
3D CNN taking as input a volume around a candidate key-
point in a scan and a volumetric representation of a CAD
model. The network is trained to predict a correspondence
heatmap over the CAD volume, representing the likelihood
that the input keypoint in the scan is matching with each
voxel. The heatmap prediction is formulated as a classiﬁ-
cation problem, which is easier to train than regression, and
produces sparse correspondences needed for pose optimiza-
tion.
(a) First step: Retrieval view.
(b) Second step: Alignment view.
Figure 2: Our annotation web interface is a two-step pro-
cess. (a) After the user places an anchor on the scan surface,
class-matching CAD models are displayed on the right. (b)
Then the user annotates keypoint pairs between the scan and
CAD model from which we derive the ground truth 9DoF
transformation.
Alignment Optimization
Sec. 6 describes our variational
alignment optimization. To generate candidate correspon-
dence points in the 3D scan, we detect Harris keypoints, and
predict correspondence heatmaps for each Harris keypoint
and CAD model. Using the predicted heatmaps we ﬁnd op-
timal 9DoF transformations. False alignments are pruned
via a geometric conﬁdence metric.
4. Dataset
Our Scan2CAD dataset builds upon the 3D scans from
ScanNet [7] and CAD models from ShapeNet [3]. Each
scene S contains multiple objects O = {oi}, where each ob-
ject oi is matched with a ShapeNet CAD model mi and both
share multiple keypoint pairs (correspondences) and one
transformation matrix Ti deﬁning the alignment. Note that
ShapeNet CAD models have a consistently deﬁned front
and upright orientation which induces an amodal tight ori-
ented bounding box for each scan object, see Fig. 3.
4.1. Data Annotation
The annotation is done via a web application that allows
for simple scaling and distribution of annotation jobs; see
Fig. 2. The annotation process is separated into two steps.

Figure 3: (Left) Oriented bounding boxes (OBBs) com-
puted from the instance segmentation of ScanNet [7] are
often incomplete due to missing geometry (e.g., in this case,
missing chair legs). (Right) Our OBBs are derived from the
aligned CAD models and are thus complete.
The ﬁrst step is object retrieval, where the user clicks on a
point on the 3D scan surface, implicitly determining an ob-
ject category label from the ScanNet object instance anno-
tations. We use the instance category label as query text in
the ShapeNet database to retrieve and display all matching
CAD models in a separate window as illustrated in Fig. 2a.
After selecting a CAD model the user performs alignment.
In the alignment step, the user sees two separate win-
dows in which the CAD model (left) and the scan object
(right) are shown (see Fig. 2b). Keypoint correspondences
are deﬁned by alternately clicking paired points on the CAD
model and scan object. We require users to specify at least
6 keypoint pairs to determine a robust ground truth trans-
formation. After keypoint pairs are speciﬁed, the alignment
computation is triggered by clicking a button. This align-
ment (given exact 1-to-1 correspondences) is solved with
the genetic algorithm CMA-ES [14, 13] that minimizes the
point-to-point distance over 9 parameters. In comparison
to gradient-based methods or Procrustes superimposition
method, we found this approach to perform signiﬁcantly
better in reliably returning high-quality alignments regard-
less of initialization.
The quality of these keypoint pairs and alignments was
veriﬁed in several veriﬁcation passes, with re-annotations
performed to ensure a high quality of the dataset. The veri-
ﬁcation passes were conducted by the authors of this work.
A subset of the ShapeNet CAD models have symme-
tries that play an important role in making correspondences.
Hence, we annotated all ShapeNet CAD models used in
our dataset with their rotational symmetries to prevent false
negatives in evaluations. We deﬁned 2-fold (C2), 4-fold
(C4) and inﬁnite (C∞) rotational symmetries around a
canonical axis of the object.
4.2. Dataset Statistics
The annotation process yielded 97607 keypoint pairs on
14225 (3049 unique) CAD models with their respective
scan counterpart distributed on a total of 1506. Approxi-
mately 28% out of the 3049 CAD models have a symmetry
tag (either C2, C4 or C∞).
Given the complexity of the task and to ensure high qual-
ity annotations, we employed 7 part-time annotators (in
contrast to crowd-sourcing). On average, each scene has
been edited 1.76 times throughout the re-annotation cycles.
The top 3 annotated model classes are chairs, tables and
cabinets which arises due to the nature of indoor scenes in
ScanNet. The number of objects aligned per scene ranges
from 1 to 40 with an average of 9.3. It took annotators on
average of 2.48min to align each object, where the time to
ﬁnd an appropriate CAD model dominated the time for key-
point placement. The average annotation time for an entire
scene is 20.52min.
It is interesting to note that manually placed keypoint
correspondences between scans and CAD models differ sig-
niﬁcantly from those extracted from a Harris corner detec-
tor. Here, we compare the mean distance from the anno-
tated CAD keypoint to: (1) the corresponding annotated
scan keypoint (= 3.5cm) and (2) the nearest Harris key-
point in the scan (= 12.8cm).
4.3. Benchmark
Using our annotated dataset, we designed a benchmark
to evaluate scan-to-CAD alignment methods.
A model
alignment is considered successful only if the category of
the CAD model matches that of the scan object and the pose
error is within translation, rotational, and scale bounds rel-
ative to the ground truth CAD. We do not enforce strict in-
stance matching (i.e., matching the exact CAD model of the
ground truth annotation) as ShapeNet models typically do
not identically match real-world scanned objects. Instead,
we treat CAD models of the same category as interchange-
able (according to the ShapeNetCorev2 top-level synset).
Once a CAD model is determined to be aligned correctly,
the ground truth counterpart is removed from the candidate
pool in order to prevent multiple alignments to the same
object. Alignments are fully parameterized by 9 pose pa-
rameters. A quantitative measure based on bounding box
overlap (IoU) can be readily calculated with these parame-
ters as CAD models are deﬁned on the unit box. The error
thresholds for a successful alignment are set to ϵt ≤20cm,
ϵr ≤20◦, and ϵs ≤20% for translation, rotation, and scale
respectively (for extensive error analysis please see the sup-
plemental). The rotation error calculation takes C2, C4 and
C∞rotated versions into account.
The Scan2CAD dataset and associated symmetry anno-
tations is available to the community.
For standardized
comparison of future approaches, we operate an automated
test script on a hidden test set.
5. Correspondence Prediction Network
5.1. Data Representation
Scan data is represented by its signed distance ﬁeld
(SDF) encoded in a volumetric grid and generated through

Figure 4: 3D CNN architecture of our Scan2CAD approach: we take as input SDF chunks around a given keypoint from a 3D
scan and the DF of a CAD model. These are encoded with 3D CNNs to learn a shared embedding between the synthetic and
real data; from this, we classify whether there is semantic compatibility between both inputs (top), predict a correspondence
heatmap in the CAD space (middle) and the scale difference between the inputs (bottom).
volumetric fusion [6] from the depth maps of the RGB-D re-
construction (voxel resolution = 3cm, truncation = 15cm).
For the CAD models, we compute unsigned distance ﬁelds
(DF) using the level-set generation toolkit by Batty [1].
5.2. Network Architecture
Our architecture takes as input a pair of voxel grids: A
SDF centered 

## experiments
7.1. Correspondence Prediction
To quantify the performance of correspondence heatmap
predictions, we evaluate the voxel-wise F1-score for a pre-
diction and its Gaussian-blurred target. The task is chal-
lenging and by design 2
3 test samples are false correspon-
dences, ≈99% of the target voxels are 0-valued, and only a
single 1-valued voxel out of 323 voxels exists. The F1-score
will increase only by identifying true correspondences. As
seen in Tab. 1, our best 3D CNN achieves 63.94%.
Tab. 1 additionally addressed our design choices; in par-
ticular, we evaluate the effect of using pre-training (PT), us-
ing compatibility (CP) as a proxy loss (deﬁned in Sec. 5.2),
enabling symmetry awareness (sym), and predicting scale
(scale). Here, a pre-trained network reduces overﬁtting, en-
hancing generalization capability. Optimizing for compati-
bility strongly improves heatmap prediction as it efﬁciently
detects false correspondences. While predicting scale only
slightly inﬂuences the heatmap predictions, it becomes very
effective for the later alignment stage. Additionally, incor-
porating symmetry enables signiﬁcant improvement by ex-
plicitly disambiguating symmetric keypoint matches.
7.2. Alignment
In the following, we compare our approach to other
handcrafted feature descriptors: FPFH [33], SHOT [40], Li
et al. [25] and a learned feature descriptor: 3DMatch [44]
(trained on our Scan2CAD dataset). We combine these de-
scriptors with a RANSAC outlier rejection method to obtain
pose estimations for an input set of CAD models. A detailed
description of the baselines can be found in the appendix.
As seen in Tab. 2, our best method achieves 31.68% and
outperforms all other methods by a signiﬁcant margin. We
additionally show qualitative results in Fig. 5. Compared to
Figure 5: Qualitative comparison of alignments on four different test ScanNet [7] scenes. Our approach to learning geometric
features between real and synthetic data produce much more reliable keypoint correspondences, which coupled with our
alignment optimization, produces signiﬁcantly more accurate alignments.

bath
bookshelf
cabinet
chair
display
sofa
table
trash bin
other
class avg.
avg.
FPFH (Rusu et al. [33])
0.00
1.92
0.00
10.00
0.00
5.41
2.04
1.75
2.00
2.57
4.45
SHOT (Tombari et al. [40])
0.00
1.43
1.16
7.08
0.59
3.57
1.47
0.44
0.75
1.83
3.14
Li et al. [25]
0.85
0.95
1.17
14.08
0.59
6.25
2.95
1.32
1.50
3.30
6.03
3DMatch (Zeng et al. [44])
0.00
5.67
2.86
21.25
2.41
10.91
6.98
3.62
4.65
6.48
10.29
Ours: +sym
24.30
10.61
5.97
9.49
3.90
25.26
12.34
10.74
3.58
11.80
8.772
Ours: +sym,+scale
18.99
13.61
7.24
14.73
9.76
41.05
14.04
5.26
6.29
14.55
11.48
Ours: +sym,+CP
35.90
32.35
28.64
40.48
18.85
60.00
33.11
28.42
16.89
32.74
29.42
Ours: +scale,+CP
34.18
31.76
21.82
37.02
14.75
50.53
32.31
31.05
11.59
29.45
26.75
Ours: +sym,+scale,+CP
36.20
36.40
34.00
44.26
17.89
70.63
30.66
30.11
20.60
35.64
31.68
Ours: +sym,+scale,+CP,+PT (3/3 ﬁx)
37.97
30.15
28.64
41.55
19.51
57.89
33.85
20.00
17.22
31.86
29.27
Ours: +sym,+scale,+CP,+PT (1/3 ﬁx)
34.81
36.40
29.00
40.60
23.25
66.00
37.64
24.32
22.81
34.98
31.22
Table 2: Accuracy comparison (%) on our CAD alignment benchmark. While handcrafted feature descriptors can achieve
some alignment on more featureful objects (e.g., chairs, sofas), they do not tolerate well the geometric discrepancies between
scan and CAD data – which remains difﬁcult for the learned keypoint descriptors of 3DMatch. Scan2CAD directly addresses
this problem of learning features that generalize across these domains, thus signiﬁcantly outperforming state of the art.
state-of-the-art handcrafted feature descriptors, our learned
approach powered by our Scan2CAD dataset produces con-
siderably more reliable correspondences and CAD model
alignments. Even compared to the learned descriptor ap-
proach of 3DMatch, our explicit learning across the syn-
thetic and real domains coupled with our alignment op-
timization produces notably improved CAD model align-
ment.
Fig. 6 shows the capability of our method to align in an
unconstrained real-world setting where ground truth CAD
models are not given, we instead provide a set of 400 ran-
dom CAD models from ShapeNet [3].
Figure 6: Unconstrained scenario where instead of having a
ground truth set of CAD models given, we use a set of 400
randomly selected CAD models from ShapeNetCore [3],
more closely mimicking a real-world application scenario.

## related_work
RGB-D Scanning and Reconstruction
The availability
of low-cost RGB-D sensors has led to signiﬁcant research
progress in RGB-D 3D reconstruction. A very prominent
line of research is based on volumetric fusion [6], where
depth data is integrated in a volumetric signed distance
function. Many modern real-time reconstruction methods,
such as KinectFusion [18, 29], are based on this surface
representation. In order to make the representation more
memory-efﬁcient, octree [4] or hash-based scene represen-
tations have been proposed [30, 21]. An alternative fusion
approach is based on points [22]; the reconstruction qual-
ity is slightly lower, but it has more ﬂexibility when han-
dling scene dynamics and can be adapted on-the-ﬂy for loop
closures [41]. Very recent RGB-D reconstruction frame-
works combine efﬁcient scene representations with global
pose estimation [5], and can even perform online updates
with global loop closures [8]. A closely related direction to
ours (and a possible application) is recognition of objects as
a part of a SLAM method, and using the retrieved objects
as part of a global pose graph optimization [35, 27].
3D Features for Shape Alignment and Retrieval
Geo-
metric features have a long-established history in computer
vision, such as Spin Images [20], Fast Point Feature His-
tograms (FPFH) [33], or Point-Pair Features (PPF) [11].
Based on these descriptors or variations of them, re-
searchers have developed shape retrieval and alignment
methods. For instance, Kim et al. [24] learn a shape prior in
the form of a deformable part model from input scans to ﬁnd
matches at test time; or AA2h [23] use a similar approach
to PPF, where a histogram of normal distributions of sam-
ple points is used for retrieval. Li et al. [25] propose a for-
mulation based on a hand-crafted TSDF feature descriptor
to align CAD models in real-time to RGB-D scans. While
these retrieval approaches based on hand-crafted geomet-
ric features show initial promise, they struggle to generalize
matching between the differing data characteristics of clean
CAD models and noisy, incomplete real-world data.
An alternative direction is learned geometric feature de-
scriptors. For example, Nan et al. [28] use a random deci-
sion forest to classify objects on over-segmented input ge-
ometry from high-quality scans. Shao et al. [37] introduce
a semi-automatic system to resolve segmentation ambigui-
ties, where a user ﬁrst segments a scene into semantic re-
gions, and then shape retrieval is applied. 3DMatch [44]
leverage a Siamese neural network to match keypoints in
3D scans for pose estimation. Zhou et al. [45] is of similar
nature, proposing a view consistency loss for 3D keypoint
prediction network on RGB-D image data. Inspired by such
approaches, we develop a 3D CNN-based approach target-
ing correspondences between the synthetic domain of CAD
models and the real domain of RGB-D scan data.
Other approaches retrieve and align CAD models given
single RGB [26, 19, 39, 17] or RGB-D [12, 46] images.
These methods are related, but our focus is on geomet-
ric alignment independent of RGB information, rather than
CAD-to-image.
Shape
Retrieval
Challenges
and
RGB-D
Datasets
Shape retrieval challenges have recently been organized
as part of the Eurographics 3DOR [16, 32].
Here, the
task was formulated as matching of object instances from
ScanNet [7] and SceneNN [15] to CAD models from the
ShapeNetSem dataset [3].
Evaluation only considered
binary in-category vs out-of-category (and sub-category)
match as the notion of relevance. As such, this evaluation
does not address the alignment quality between scan objects
and CAD models, which is our focus.

ScanNet [7] provides aligned CAD models for a small
subset of the annotated object instances (for only 200 ob-
jects out of the total 36000).
Moreover, the alignment
quality is low with many object category mismatches and
alignment errors, as the annotation task was performed by
crowdsourcing. The PASCAL 3D+ [43] dataset annotates
13898 objects in the PASCAL VOC images with coarse 3D
poses deﬁned against representative CAD models. Object-
Net3D [42] provides a dataset of CAD models aligned to
2D images, approximately 200K object instances in 90K
images. The IKEA objects [26] and Pix3D [39] datasets
similarly provide alignments of a small set of identiﬁable
CAD models to 2D images of the same objects in the real
world; the former has 759 images annotated with 90 mod-
els, the latter has 10069 annotated with 395 models.
No existing dataset provides ﬁne-grained object instance
alignments at the scale of our Scan2CAD dataset with
14225 CAD models (3049 unique instances) annotated to
their scan counterpart distributed on 1506 3D scans.

## conclusion
In this work, we presented Scan2CAD, which aligns a set
of CAD models to 3D scans by predicting correspondences
in form of heatmaps and then optimizes over these corre-
spondence predictions. First, we introduce a new dataset of
9DoF CAD-to-scan alignments with 97607 pairwise key-
point annotations deﬁning the alignment of 14225 objects.
Based on this new dataset, we design a 3D CNN to pre-
dict correspondence heatmaps between a CAD model and
a 3D scan. From these predicted heatmaps, we formulate
a variational energy minimization that then ﬁnds the opti-
mal 9DoF pose alignments between CAD models and the
scan, enabling effective transformation of noisy, incomplete
RGB-D scans into a clean, complete CAD model represen-
tation. This enables us to achieve signiﬁcantly more accu-
rate results than state-of-the-art approaches, and we hope
that our dataset and benchmark will inspire future work
towards bringing RGB-D scans to CAD or artist-modeled
quality.

Acknowledgements
We would like to thank the expert annotators Soh Yee
Lee, Rinu Shaji Mariam, Suzana Spasova, Emre Taha, Se-
bastian Thekkekara, and Weile Weng for their efforts in
building the Scan2CAD dataset. This work is supported
by Occipital, the ERC Starting Grant Scan2CAD (804724),
and a Google Faculty Award. We would also like to thank
the support of the TUM-IAS, funded by the German Ex-
cellence Initiative and the European Union Seventh Frame-
work Programme under grant agreement n 291763, for the
TUM-IAS Rudolf M¨oßbauer Fellowship and Hans-Fisher
Fellowship (Focus Group Visual Computing).