# Pos3R: 6D Pose Estimation for Unseen Objects Made Easy

> 2025 · id: W4413146353 · 来源: web-agent
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

Pos3R: 6D Pose Estimation for Unseen Objects Made Easy
Weijian Deng1
Dylan Campbell1
Chunyi Sun1
Jiahao Zhang1
Shubham Kanitkar2
Matthew E. Shaffer2
Stephen Gould1
1Australian National University
2RIOS Intelligent Machines
1{firstname.lastname}@anu.edu.au
2{firstname.lastname}@rios.ai
Abstract
Foundation models have significantly reduced the need
for task-specific training, while also enhancing generaliz-
ability. However, state-of-the-art 6D pose estimators ei-
ther require further training with pose supervision or ne-
glect advances obtainable from 3D foundation models. The
latter is a missed opportunity, since these models are bet-
ter equipped to predict 3D-consistent features, which are of
significant utility for the pose estimation task. To address
this gap, we propose Pos3R, a method for estimating the
6D pose of any object from a single RGB image, making
extensive use of a 3D reconstruction foundation model and
requiring no additional training. We identify template selec-
tion as a particular bottleneck for existing methods that is
significantly alleviated by the use of a 3D model, which can
more easily distinguish between template poses than a 2D
model. Despite its simplicity, Pos3R achieves competitive
performance on the Benchmark for 6D Object Pose Estima-
tion (BOP), matching or surpassing existing refinement-free
methods.
Additionally, Pos3R integrates seamlessly with
render-and-compare refinement techniques, demonstrating
adaptability for high-precision applications.
1. Introduction
Six-dimensional (6D) object pose estimation—the task
of determining the exact position and orientation of ob-
jects relative to a camera—is essential for applications in
robotics, augmented reality, and autonomous systems. Re-
liable pose estimation enables critical tasks like object ma-
nipulation, grasping, and assembly, allowing these systems
to interact effectively in complex and dynamic environ-
ments [6, 18, 29, 30, 49, 54]. Traditional approaches often
rely on learning-based methods tailored to specific objects
or categories, achieving high accuracy but struggling to gen-
eralize to new categories or unseen objects [4, 5, 7, 26, 33,
46, 47, 53–55, 59, 61]. This limitation is particularly prob-
lematic in dynamic, data-scarce environments where adapt-
ability and flexibility are crucial.
In-plane rotation
Out-of-plane rotation
MASt3R
DINOv2
Figure 1. Illustration of In-Plane and Out-of-Plane Rotations
with Correspondence Quality between DINOv2 and MASt3R.
The left diagram illustrates in-plane rotations (within the im-
age plane) and out-of-plane rotations (3D orientation changes)
from a camera view. On the right, DINOv2 [40] (top row) and
MASt3R [24] (bottom row) are compared in their handling of
these rotations. DINOv2 exhibits sparse and inconsistent corre-
spondences (shown by red lines), particularly under out-of-plane
rotations, due to its 2D feature limitations. In contrast, MASt3R
provides dense and stable correspondences across both types of
rotations, reflecting its ability to produce 3D-consistent features.
To overcome these limitations, there has been a recent
shift toward model-based approaches that aim to generalize
pose estimation to unseen objects without object-specific
training [2, 17, 41, 56]. These typically rely on a two-stage
pipeline: detecting and localizing objects within a scene,
followed by a render-and-compare process that matches de-
tected object regions to a set of template models. Building
on this concept, recent works have sought to improve gen-
eralization through large-scale model training with signifi-
cant attention to object diversity. For example, Foundation-
Pose [56] uses synthetic training data combined with a large
language model and contrastive learning, enhancing feature
alignment across varied domains for robust generalization.
The recent emergence of training-free methods offers a
promising alternative, enabling 6D pose estimation for un-
seen objects without object- or task-specific training. Foun-
dation models like DINOv2 [40] have demonstrated strong
zero-shot capabilities, capturing spatial and semantic details
through learned features [1, 58]. FoundPose [41], for in-
stance, leverages DINOv2 descriptors to bridge synthetic
This CVPR paper is the Open Access version, provided by the Computer Vision Foundation.
Except for this watermark, it is identical to the accepted version;
the final published version of the proceedings is available on IEEE Xplore.
16818

and real data, achieving competitive results on the BOP
challenge [17]. These advancements underscore the poten-
tial of training-free frameworks in 6D pose estimation.
Building on this trend, we explore the strengths of 3D
foundation models. Accurate pose estimation requires han-
dling both in-plane and out-of-plane rotations, which are
demonstrated in Fig. 1. In-plane rotations occur within the
image plane, as when an object rotates around the camera’s
line of sight, while out-of-plane rotations involve changes
in 3D orientation, such as tilting or turning, which signifi-
cantly alters appearance due to perspective shifts. Address-
ing both types of rotation is essential for robust pose es-
timation, as objects often appear from varied angles. Al-
though 2D foundation models like DINOv2 handle in-plane
rotations well through training augmentations, they lack ro-
bustness to out-of-plane rotations due to their 2D limitations
(shown in Fig. 1). In contrast, 3D foundation models, such
as MASt3R [24], are specifically designed to produce 3D-
consistent features across different viewpoints, enabling re-
liable feature alignment even with out-of-plane rotations.
Motivated by this, we propose Pos3R, a training-free
method for 6D pose estimation of unseen objects using only
RGB inputs.
Pos3R leverages the 3D foundation model
MASt3R [24] for pose estimation without requiring addi-
tional training. The core of Pos3R lies in the image match-
ing process between test crops and rendered templates from
a CAD model. While common matchers like LoFTR [50]
often struggle with synthetic-to-real matching due to do-
main gaps, MASt3R produces high-quality 2D correspon-
dences in this case. These reliable matches enable Pos3R
to establish the 2D-3D correspondences required for pose
estimation using the PnP-RANSAC algorithm.
By harnessing MASt3R’s ability to produce dense corre-
spondences robust to viewpoint and illumination changes,
we simplify template generation and achieve computational
efficiency. Unlike existing methods that render hundreds of
templates, we place camera positions to cover both in-plane
and out-of-plane rotations. Specifically, we use eight base
templates per object, positioned at the vertices of a cube
centered on the CAD model, to capture out-of-plane rota-
tions. For each base template, we generate five in-plane ro-
tational templates along the object’s principal axis, resulting
in forty templates per object. To ensure efficient and accu-
rate pose estimation, Pos3R dynamically selects the opti-
mal rotation based on matching quality. This simple yet
effective strategy maintains computational efficiency while
achieving reliable pose estimation. Tested on seven diverse
datasets in the BOP challenge, Pos3R demonstrates strong
performance as a robust, scalable option for training-free
6D pose estimation of unseen objects, adaptable to render-
and-compare refinement techniques.
In summary, our contributions are:
• While existing methods rely on 2D foundation models
for unseen object pose estimation, we present Pos3R,
a training-free method that leverages the 3D foundation
model MASt3R [24] to improve robustness.
• Leveraging MASt3R’s robust dense correspondences,
Pos3R uses only forty strategically placed templates per
object to capture in-plane and out-of-plane rotations. A
simple selection technique based on correspondence qual-
ity ensures accurate matching, achieving strong perfor-
mance on the BOP challenge.
2. Related Work
Seen Object Pose Estimation. Instance-level object pose
estimation refers to the task of estimating the poses of spe-
cific objects previously encountered during model train-
ing [5, 7, 26, 33, 46, 54, 61]. Among common approaches
are correspondence-based methods, which learn to identify
precise alignments between input data and CAD models
of the objects [7, 27, 46].
Other methods use template-
based strategies, where the model selects the most sim-
ilar pose-labeled template from a predefined set of ex-
amples [25, 35, 52].
Additionally, regression-based ap-
proaches directly predict object poses from the learned
object-specific features [12, 19, 26, 42].
These instance-specific techniques offer high precision
but typically require retraining for new object instances,
limiting their generalizability.
This limitation has led to
the exploration of category-level methods, which general-
ize within specific object categories and allow estimation
of unseen objects in known categories [4, 47, 53, 55, 59].
For example, SecondPose [4] enhances category-level 6D
pose estimation by integrating object-specific geometric
features with DINOv2 SE(3)-consistent semantic priors, ef-
fectively addressing intra-class shape variation. However,
these methods struggle outside their target categories, while
our approach enables pose estimation without relying on
such constraints, allowing broader applications.
Unseen Object Pose Estimation. To enhance pose estima-
tion flexibility, many methods aim to generalize across new
object instances without object-specific training [22, 31, 37,
39, 43]. Approaches for this task can be broadly divided
into manual reference view-based and CAD model-based
methods. (i) Reference view-based methods use multi-view
images with known poses as reference data for pose esti-
mation [14, 31, 43, 51]. For example, OnePose [14, 51] re-
constructs 3D object point clouds from posed RGB images,
establishing 2D-3D correspondences to solve 6D poses. In
a different approach, Nope [38], adopting the perspective
of generating new views, trains models to predict discrimi-
native embeddings for novel object perspectives. (ii) CAD
model-based approaches leverage 3D object model to fa-
cilitate pose estimation. Some methods match features be-
tween the CAD model and the query image [20, 28, 60],
16819

…
test segment
6-DoF Pose
forty templates
2D-2D correspondence
coordinate map
1) rendering
2) image matching
CAD model
3) PnP-RANSAC
(sec. 3.2.1)
(sec. 3.2.2)
(sec. 3.2.3)
Figure 2. Overview of the 6D Pose Estimation Process in Pos3R. (1) Template Rendering (Sec. 3.2.1): A CAD model is used to generate
a set of forty templates, each representing a unique orientation that covers both in-plane and out-of-plane rotations. This is achieved by
positioning the camera at the vertices of a cube around the object. (2) Image Matching (Sec. 3.2.2): Given a test segment from CNOS [37],
we establish 2D–2D correspondences between the test image and each rendered template. We leverage the 3D-consistent features generated
by MASt3R [24] for accurate matching and select the best template based on matching quality. Each template also includes a 3D object
coordinate map that records the corresponding 3D points. (3) Pose Fitting (Sec. 3.2.3): Using the selected correspondences, we apply the
PnP-RANSAC algorithm [23] to obtain a final pose that aligns the object with its observed position in the scene.
while others employ template matching to find the closest
initial pose, refining it further using specialized refiners for
higher accuracy [22, 34, 39]. For example, GigaPose [39]
introduces an efficient two-network system: one network
for retrieving template views (out-of-plane rotation) and an-
other for estimating the remaining degrees of freedom (in-
plane rotation and 3D translation). This separation reduces
computational costs, contrasting with methods like Mega-
Pose [22], which applies a single network to every possible
test crop-template pair. Our work builds on the CAD model-
based approach, aligning with recent BOP challenge proto-
cols [17]. We focus on a training-free pipeline for unseen
object pose estimation, a direction distinct from methods
that require extensive training of task-specific networks.
Training-Free Object Pose Estimation. Traditionally, 6D
object pose estimation is achieved by establishing 3D-to-2D
correspondences followed by a PnP algorithm [11, 13, 44,
61]. Recently, leveraging features from foundation mod-
els has gained traction, especially with models like DINO,
which offer spatial detail and semantic consistency for pose
estimation [1, 58]. FoundPose [41] uses DINOv2 descrip-
tors to bridge synthetic and real data domains, provid-
ing strong performance for symmetric objects with RGB-
only inputs. FreeZe integrates 3D point descriptors from
GeDi [45] with image features from DINOv2 [40] for RGB-
D pose estimation. By leveraging the 3D consistent feature
of MASt3R to produce dense correspondences, our work
explores training-free, unseen object pose estimation with
RGB-only inputs, enabling effective pose estimation across
diverse object categories without task-specific training.
3. 6D Pose Estimation with Pos3R
3.1. Task Definition
Given a 3D model of a query object \mathbf {Q} and an RGB im-
age \ math bf {I} \in \mathbb {R}^{H \times W \times 3} containing \mathbf {Q} , the task is to estimate
the 6D pose of \mathbf {Q} relative to the camera’s reference frame,
with known intrinsics \mathbf {K} . Specifically, we aim to determine
the 6DoF transformation \ mat hbf {T} = (\mathbf {R}, \mathbf {t}) in 3D space, where
 \ mathbf {R} \in \mathbb {R}^{3 \times 3} represents the rotation matrix and \ mathbf {t} \in \mathbb {R}^{3} de-
notes the translation vector. The segmented object region
 \ m a t hbf {I}_m = \mathbf {M} \odot \mathbf {I} is created by element-wise multiplication of
the binary segmentation mask \mathbf {M} and \mathbf {I} , isolating the visible
part of \mathbf {Q} . We note that Q may be partially occluded.
This
work
leverages
the
3D
foundation
model
MASt3R [24] to tackle model-based 6D pose estimation
for unseen objects. We introduce Pos3R, a training-free,
RGB-only method that effectively estimates 6D poses
without object- or task-specific training.
3.2. Training-Free Pipeline
Overview. Following the standard model-based pipeline
for 6D pose estimation of unseen objects [2, 39, 41], Pos3R
comprises two components: object detection and pose es-
timation. We keep each component frozen, avoiding any
object- or task-specific training.
For object detection, we use CNOS [37] to generate
segmentation masks and object identities for each target
instance, enabling localization of the target segment \mathbf {I}_m 
within the RGB image \mathbf {I} . As a default method in the BOP
challenge for segmenting unseen objects, CNOS requires
16820

only 3D models for onboarding and does not depend on ad-
ditional data or object-specific training.
For 6D pose estimation, we illustrate three steps in
Fig. 2. Specifically, given a set of templates rendered from
a textured CAD model, Pos3R utilizes the 3D foundation
model MASt3R [24] to extract features from both the target
segment \mathbf {I}_m and each template, enabling estimation of the
6D transformation \ mat hbf {T} = (\mathbf {R}, \mathbf {t}) through 2D–3D correspon-
dences using the PnP-RANSAC algorithm.
3.2.1. Template Rendering
Using a textured 3D CAD model, we render templates of
the object from various orientations. The rendering process
follows standard rasterization methods as described in [39],
with a black background and fixed lighting. The rendering
camera is configured with the same intrinsic parameters, \mathbf {K} ,
as the test camera, and the rendered templates match the
size of the test image \mathbf {I} . The object remains centered in each
template. Additionally, the 3D locations in the coordinate
space of the 3D CAD model corresponding to each pixel in
the rendered template are recorded, enabling the establish-
ment of 2D–3D correspondences.
A critical component of Pos3R is establishing correspon-
dences between pixels in the target segment and the tem-
plate. Accurate pose estimation requires addressing both in-
plane and out-of-plane rotations, as objects are frequently
viewed from diverse orientations. In-plane rotations oc-
cur within the image plane, such as when an object rotates
around the camera’s line of sight. Out-of-plane rotations,
on the other hand, involve 3D orientation changes—such as
tilting or turning—that significantly alter an object’s appear-
ance due to perspective shifts. Existing methods tackle this
challenge through extensive template libraries and special-
ized selection mechanisms. For example, MegaPose [22]
uses hundreds of templates to capture a broad range of ob-
ject poses, but maintaining such a large template library ne-
cessitates a dedicated selection network.
Template Configuration. To reduce reliance on exten-
sive template libraries, Pos3R leverages the 3D founda-
tion model MASt3R, which generates 3D-consistent fea-
tures across viewpoints. This enables Pos3R to handle out-
of-plane rotations effectively without requiring hundreds of
templates or complex selection mechanisms. We use a set
of eight base templates, denoted by \{ \
mathbf {I}_{i} \}_{i=1}^{8} , covering es-
sential orientations. They are positioned at the vertices of a
cube centered around the CAD model, effectively capturing
out-of-plane rotations.
To further address ambiguities from in-plane (axial) ro-
tations, which can impact correspondence quality, we ap-
ply controlled rotational variations to each base template.
For each template \mathbf {I}_{i} , we generate T rotations around the
camera’s principal axis, each rotation defined by an angle
 \ t
het
a _k = \frac {2\pi k}{T} , where k = 0 , \ d ots , T-1 , covering the full
 360^\circ range.
In our experiments, we set T = 5 to bal-
ance efficiency and accuracy, yielding a set of rotationally-
augmented templates \{ \ma
thbf {I}_{i,k} \}_{k=1}^{5} . In the following, we dis-
cuss the template selection process in detail.
3.2.2. Image Matching
MASt3R as an Image Matcher. Our approach builds upon
MASt3R [24], a model for joint local 3D reconstruction
and pixelwise matching between two input images \mathbf {I}_a and
 \ m athb f {I}_b \in \mathbb {R}^{H \times W \times 3} . Conceptually, MASt3R operates as a map-
ping function \mat hbf {f}(\mathbf {I}_a, \mathbf {I}_b) = \text {Dec}(\text {Enc}(\mathbf {I}_a), \text {Enc}(\mathbf {I}_b)) , where
the encoder \text {Enc}(\mathbf {I}) \rightarrow \mathbf {F} is a Siamese Vision Transformer
(ViT) that processes image \mathbf {I} into token vectors of size
 N \times 
d, with N = w \times h , yielding \ mathbf {F} \in \mathbb {R}^{N \times d} . The decoder,
 \text {Dec}(\mathbf {F}_a, \mathbf {F}_b) , employs twin ViT modules that produce pix-
elwise pointmaps \mathbf {X} , local feature maps \mathbf {D} for each image.
Using these local feature representations, correspon-
dences between images are identified via the fastNN al-
gorithm [24]. It efficiently establishes reciprocal matches
between the feature maps \mathbf {D}_a and \mathbf {D}_b by initially seeding
points across a uniform pixel grid and iteratively refining
these seeds to form high-quality mutual correspondences.
Through this process, fastNN accurately aligns key features
across images. The resulting reciprocal pixel pairs between
 \mathbf {I}_a and \mathbf {I}_b are represented as M_{ a ,b} 
= \{
 (\mathbf
 {y}_a^c, \mathbf {y}_b^c )\}_{c=1}^{|M_{a,b}|} 
,
where \
mathbf {y}_a^c and \
m a thbf {y}_b^c \in \mathbb {N}^2 denote the coordinates of matched
pixels in each respective image.
Similarity-Based Template Selection. Unlike methods
that require hundreds of templates [22, 36, 41], Pos3R sim-
plifies template selection by needing only forty templates,
significantly reducing the complexity of the selection pro-
cess. Rather than relying on a trained template selection
network, we employ a straightforward, training-free ap-
proach based on the similarity of matched correspondences.
Specifically, given a target segment \mathbf {I}_m and a set of eight
base templates \{ \
mathbf {I}_{i} \}_{i=1}^8 , each augmented with rotational
variations \{ \ma
thbf {I}_{i,k} \}_{k=1}^{5} , we identify the most similar template
for pose estimation by calculating correspondences between
the target segment and each template.
For each rotationally-augmented template \mathbf {I}_{i,k} , we obtain
reciprocal pixel pairs between \mathbf {I}_m and \mathbf {I}_{i,k} , represented as:
 M_{m, i,k}
 = \
{ (\mathbf {y
}_m
^p, \mathbf {y}_{i,k}^p )\}_{p=1}^{|M_{m, i,k}|}, where M_{m, i,k} is the set of
all reciprocal pixel pairs between \mathbf {I}_m and \mathbf {I}_{i,k} , and \
mathbf {y}_m^p and
 \
mat h bf {y}_{i,k}^p \in \mathbb {N}^2 denote the coordinates of matched pixels in the
target segment and the template variant, respectively.
For each matched pair (\
ma th
bf {y}_m^p, \mathbf {y}_{i,k}^p) in M_{m, i,k} , we retrieve
the corresponding local features from the feature maps \mathbf {D}_m 
and \mathbf {D}_{i,k} generated by MASt3R. Specifically, we obtain the
feature vector \
mathbf {f}_m^p at coordinate \
mathbf {y}_m^p in the target segment
from \mathbf {D}_m , and the feature vector \
mathbf {f}_{i,k}^p at coordinate \
mathbf {y}_{i,k}^p in
the template from \mathbf {D}_{i,k} . We compute the feature similarity
for each matched pair as: S( \
ma t h
bf { f } _
m^p ,
 \mathbf {f}_{i,k}^p) = \mathbf {f}_m^p \cdot \mathbf {f}_{i,k}^p, where \cdot de-
notes the dot product.
16821

To calculate an overall similarity score between the tar-
get segment \mathbf {I}_m and each template variant \mathbf {I}_{i,k} , we aggregate
the similarity scores across all matched pairs as follows:
 \label {eq: s
im} \tex
t
 {s
im} (
\m a t
hbf {I}_m, \mathbf {I}_{i,k}) = \sum _{p=1}^{|M_{m,i,k}|} S(\mathbf {f}_m^p, \mathbf {f}_{i,k}^p), 
(1)
where M_{m, i,k} is the set of all matched pairs between \mathbf {I}_m 
and \mathbf {I}_{i,k} . After computing \text {sim}(\mathbf {I}_m, \mathbf {I}_{i,k}) for each pair, we
select the template with the highest similarity score:
  (i_{ \text 
{op t}}
, k_{\text { opt}}) = \u
nderset {i \in \{1, \dots , 8\}, \, k \in \{1, \dots , 5\}}{\operatorname {arg\,max}} \, \text {sim}(\mathbf {I}_m, \mathbf {I}_{i,k}). 
(2)
The selected template \mathbf {I}_{i_{\text {opt}}, k_{\text {opt}}} is then used as the closest
match to the target segment for the pose estimation process.
3.2.3. Pose Fitting
After selecting the suitable template \mathbf {I}_{i_{\text {opt}}, k_{\text {opt}}} , we proceed to
estimate pose \ m athb f {T}_m = (\mathbf {R}_{m}, \mathbf {t}_{m}) , where \mathbf {R}_{m} is a 3D rotation
matrix and \mathbf {t}_{m} is a 3D translation vector that transforms
the object from model space to camera space. It relies on
a set of 2D-3D correspondences \mathc a l {C
}_ {t_{\te
xt {final}}} = \{ (\mathbf {y}_m^j, \mathbf {P}^j) \}_{j=1}^{|M|} ,
where \
m a thbf {y}_m^j \in \mathbb {R}^2 represents the coordinates of matched pix-
els in the target segment \mathbf {I}_m , \ m athbf {P}^j \in \mathbb {R}^3 denotes the corre-
sponding 3D points in model space from the selected tem-
plate, and there are |
pairs. To determine \mathbf {T}_m , we solve
the Perspective-n-Point (PnP) problem, which minimizes
the reprojection error:
  \ ope
rator
nam
e
 *{
arg\
, m in}_{\ m ath
bf {R}_{m}, \mathbf {t}_{m}} \sum _{j=1}^{|M|} \left \| \mathbf {y}_m^j - \pi (\mathbf {R}_{m} \mathbf {P}^j + \mathbf {t}_{m}) \right \|^2, 
(3)
where \pi is the projection function that maps 3D points to
2D image points according to the camera intrinsics.
To enhance robustness against outliers, we employ the
Efficient PnP (EPnP) algorithm [23] combined with a
RANSAC-based fitting strategy [11]. In this approach, PnP
is applied iteratively to random subsets of four correspon-
dences from \mathcal {C}_{t_{\text {final}}} , generating multiple pose hypotheses. For
each hypothesis, we count the number of inliers, defined as
correspondences where the reprojection error falls below a
predefined threshold \epsilon :
 \label 
{eq
: p
ose}
 \ text { i nli
er s }
 = \left | \left \{ j : \left \| \mathbf {y}_m^j - \pi (\mathbf {R}_{m} \mathbf {P}^j + \mathbf {t}_{m}) \right \| < \epsilon \right \} \right |. 
(4)
The hypothesis with the highest inlier count is selected as
the final coarse pose estimate \mathbf {T}_m .
4. Experiments
In this section, we first outline the experimental setup (Sec-
tion 4.1). We then evaluate our method’s performance in
comparison with previous approaches on the seven core
datasets of the BOP challenge [17], examining accuracy,
runtime efficiency, and effectiveness when using predicted
3D models (Section 4.2).
This evaluation highlights the
strengths and unique contributions of our approach. Lastly,
we present an ablation study to investigate the impact of
various configurations in our method (Section 4.3).
4.1. Experimental Setup
Evaluation Datasets. We evaluate our approach on the
seven core datasets of the BOP challenge [17]: LineMod
Occlusion (LM-O) [3], T-LESS [15], TUD-L [16], IC-
BIN [8], ITODD [9], HomebrewedDB (HB)[21], and YCB-
Video (YCB-V) [57]. Together, these datasets feature 132
distinct objects and 19, 048 testing instances, each in com-
plex, cluttered scenes with partial occlusions. Table 1 pro-
vides the instance count for each dataset. It is worth noting
that the unseen object pose estimation task remains chal-
lenging, with significant room for improvement.
Evaluation Metrics. We use the BOP evaluation proto-
col [17] for 6D object localization, which assesses pose ac-
curacy with three error metrics: Visible Surface Discrep-
ancy (VSD), Maximum Symmetry-Aware Surface Distance
(MSSD), and Maximum Symmetry-Aware Projection Dis-
tance (MSPD). VSD evaluates only the visible part of the
object to handle ambiguities, MSSD measures 3D surface
deviation with global symmetries, and MSPD assesses per-
ceivable deviation using object symmetries in projection. A
pose is considered correct for a metric e if e < \theta _e , with
 \theta _e as the correctness threshold. The Average Recall (AR)
for each metric e , denoted \text {AR}_e , is the mean Recall over
various \theta _e thresholds, and for VSD, multiple misalignment
tolerances \tau . The overall AR score is the average of the
three metrics: \ t ext {A R } = (\ t ext {AR}_{\text {VSD}} + \text {AR}_{\text {MSSD}} + \text {AR}_{\text {MSPD}}) / 3 .
Pose Refinement. To show Pos3R can integrate with
render-and-compare refinement techniques, we apply the
refinement method from MegaPose [22] to our result. Given
an input image and an estimated pose, the refiner predicts
the relative transformation between the initial and ground-
truth pose. The refiner is trained on a large-scale dataset
with several well-designed techniques [22]. We use the sim-
ilarity score Eq. 1 to select the top-5 templates and each test
crop-template pair gives a pose with EPnP. Then, we use the
refiner to update these five pose hypotheses.
4.2. Comparison With the State of the Art
We compare Pos3R with both training-free and training-
based methods. The training-free methods include Found-
Pose [41] and ZS6D [2], while the training-based methods
include MegaPose [22], GigaPose [39], OSOP [48], and
GenFlow [34]. For pose refinement, we apply the refiner
from MegaPose [22]. We use the publicly available code
from [39] with the default hyperparameters.
16822

#
Method
Training-Free
Refinement
Datasets (num. instances)
Mean
Time
LM-O
T-LESS
TUD-L
IC-BIN
ITODD
HB
YCB-V
1445
6423
600
1786
3041
1630
4123
Coarse Pose Estimation:
1
GigaPose [39]
✗
–
29.9
27.3
30.2
23.1
18.8
34.8
29.0
27.6
0.9
2
GenFlow [34]
✗
–
25.0
21.5
30.0
16.8
15.4
28.3
27.7
23.5
3.8
3
MegaPose [22]
✗
–
22.9
17.7
25.8
15.2
10.8
25.1
28.1
20.8
15.5
4
OSOP [48]
✗
–
31.2
–
–
–
–
49.2
33.2
–
–
5
ZS6D [2]
✓
–
29.8
21.0
–
–
–
–
32.4
–
–
6
FoundPose [41]
✓
–
39.6
33.8
46.7
23.9
20.4
50.8
45.2
37.2
1.7
7
Baseline
✓
–
30.5
23.6
43.2
21.4
15.4
42.5
49.3
32.3
0.7
7
Pos3R (Ours)
✓
–
32.3
31.5
47.3
33.1
25.1
53.7
53.9
39.5
1.4
With Pose Refinement (5 hypotheses):
8
FoundPose
✗
MegaPose
58.6
54.9
65.7
44.4
36.1
70.3
67.3
56.8
11.2
9
GigaPose
✗
MegaPose
59.9
57.0
64.5
46.7
39.7
72.2
66.3
57.9
7.3
10
GenFlow
✗
GenFlow
56.3
52.3
68.4
45.3
39.5
73.9
63.3
57.1
20.9
11
MegaPose
✗
MegaPose
56.0
50.7
68.4
41.4
33.8
70.4
62.1
54.7
47.4
12
Pos3R (ours)
✗
MegaPose
56.5
57.5
66.8
46.0
37.5
70.1
66.4
57.3
8.0
Table 1. Performance Comparison on the Seven Datasets of BOP. This table reports the Average Recall (AR) scores per dataset, the
mean AR score across all datasets, and the time required to estimate poses for all objects in an image (in seconds). The runtime data of
other methods are sourced from FoundPose [41]. The upper section lists methods for coarse pose estimation without refinement, while
the lower section presents methods that incorporate a refinement stage using multiple pose hypotheses, reporting the best refined pose.
Methods without task-specific training are marked with a green check mark (✓). In the coarse estimation category, Pos3R achieves the
highest AR scores on several datasets and demonstrates good overall generalization with a competitive mean AR score. While refinement-
based methods yield slightly higher accuracy in certain cases, Pos3R remains a robust and efficient choice for coarse pose estimation and
performs comparably to top refined methods when combined with a refinement stage. The best results in the coarse estimation section are
highlighted in green, and the best results in the refinement section are highlighted in blue. The second-best results are underlined.
Table 1 provides a detailed comparison of Pos3R
(Pos3R) with other leading 6D pose estimation methods
across the seven primary datasets in the BOP challenge.
The upper portion of the table showcases results for coarse
pose estimation without refinement, while the lower por-
tion highlights performance after incorporating a pose re-
finement step with five hypotheses. It is important to note
that pose refinement involves task-specific training on ex-
tensive pose datasets, making it non-training-free [22].
In the coarse pose estimation category, Pos3R demon-
strates superior performance over other training-free meth-
ods, including FoundPose and ZS6D, achieving the highest
mean accuracy and consistently strong results across most
datasets. For instance, Pos3R attains the highest AR (Aver-
age Recall) on the TUD-L, HB, and YCB-V datasets, along
with a mean AR of 39.5. This high performance across var-
ied datasets reflects Pos3R ’s robust generalization capabil-
ities for different object poses and scenarios. Additionally,
with a runtime of only 1.4 seconds, Pos3R stands out as an
efficient alternative to methods that either require substan-
tial computational resources or are not training-free.
Pos3R remains competitive after pose refinement us-
ing MegaPose’s refine.
Although Pos3R is not inher-
ently designed for refinement, it achieves comparable ac-
curacy to MegaPose and other refined methods across sev-
eral datasets.
These findings underscore the adaptabil-
ity of Pos3R, as it provides accurate initial pose esti-
mates and benefits from additional refinement techniques—
demonstrating its potential for scalable, practical, and flexi-
ble applications in 6D pose estimation.
Moreover, we observe that Pos3R faces challenges in
achieving competitive accuracy on the LM-O dataset, where
significant occlusion poses a major difficulty. Occlusion
disrupts the initial image matching process, often result-
ing in suboptimal pose estimates that refinement methods
struggle to correct. FoundPose [41] addresses this issue by
employing a bag-of-words strategy for template selection,
which has shown greater resilience to occlusion. To im-
prove robustness in such scenarios, incorporating the local
contrastive learning approach used by GigaPose [39] could
be a promising direction. This method is specifically de-
signed to enhance feature matching in occluded regions, and
we consider this an area for future exploration.
4.3. Component Analysis
Pose Estimation Using Predicted 3D Models. Model-
based methods for unseen object pose estimation typically
16823

Table 2. Comparison with Predicted 3D Models on LM-O [3].
Average Recall is reported. All methods use 3D models predicted
by Wonder3D [32]. Results with pose refinement, applied to the
generated 3D models, are also included for comparison.
#
Method
Single Image-to-3D
GT 3D model
Coarse
Refined
w/o refinement
1
MegaPose [22]
15.4
25.2
22.7
2
GigaPose [39]
17.6
27.2
29.4
3
Pos3R
19.5
28.5
32.3
Wonder3D
CAD model
Figure 3. Rendered Images and 3D Coordinate Maps from
Wonder3D and CAD Model. The top two rows show rendered
images and 3D coordinate maps from Wonder3D [32], while the
bottom two rows show those from the real CAD model.
require accurate, textured 3D models. To relax this require-
ment, we follow the approach in [39] and use 3D models
predicted by Wonder3D [32] from a single reference image.
The predicted 3D models use the same object coordinate
system as the ground truth (GT) frame, ensuring aligned
axes. We then assess the performance of our method us-
ing these reconstructed models in place of the high-fidelity
CAD models provided by the dataset. For a fair compari-
son, we adopt the same settings as [39], including the ref-
erence image for Wonder3D and post-processing steps, and
report results on the LM-O dataset [3]. Examples of ren-
dered views from generated 3D model are shown in Fig. 3.
As shown in Table 2, Pos3R achieves higher AR scores than
both MegaPose [22] and GigaPose [39], both before and af-
ter pose refinement, showing the effectiveness of Pos3R.
Template Selection Technique. In our experiments, we
use the similarity of correspondence matches (Eq. 1) as
the primary method for template selection. Additionally,
we consider two alternative approaches: 1) Inliers: se-
lecting templates based on the number of inliers (Eq. 4);
2) Confidence: leveraging the 3D pointmaps and confi-
dence maps provided by MASt3R for each target segment-
template pair [24], where the average confidence score is
used for selection. As illustrated in Fig. 4, template selec-
tion based on inliers proves effective, while selection based
on similarity score yields the best performance.
Table 3. Comparison of Pose Estimation Method. We evalu-
ate our proposed method (Pos3R), a variant without axial rotation
augmentation (w/o axial rotation), a variant that use face centers
as base templates, and a version that replaces the 3D foundation
model MASt3R with the 2D foundation model DINOv2. We also
replace MASt3R with another dense image matcher RoMa [10].
Pos3R achieves the highest accuracy across all datasets, highlight-
ing the importance of axial rotation handling and the advantage of
using 3D foundation models over DINOv2.
# Method
LM-O
TUD-L YCB-V
T-Less
IC-BIN ITODD
HB
1 Pos3R
32.3
47.3
53.9
31.5
33.1
25.1
53.7
2 w/o Ax. Rot.
30.5
43.2
49.3
23.6
21.4
15.4
42.5
3 Face Center
31.5
45.5
53.3
28.0
29.4
22.4
52.5
4 DINOv2 (L11)
18.5
25.0
22.5
14.2
15.6
14.1
32.1
5 DINOv2 (L9)
20.5
26.0
24.0
15.6
16.6
14.8
32.9
6 RoMa [10]
20.8
25.0
32.0
16.2
17.0
15.5
25.1
Figure 4. Comparison of Template Selection Techniques. We
compare template selection methods based on confidence, inliers,
and similarity score. The similarity score consistently achieves the
highest average recall (AR) across all datasets.
Impact of In-Plane Rotation and 3D Consistency on
Pose Estimation. Table 3 evaluates the impact of incorpo-
rating in-plane (axial) rotations and 3D-consistent features
on pose estimation performance across seven datasets. Our
method, Pos3R (row 1), achieves the highest accuracy on
all datasets by combining controlled in-plane rotations with
the 3D-consistent features provided by the MASt3R foun-
dation model. In contrast, row 2 presents results without in-
plane rotation, where only eight base templates are used to
account for out-of-plane rotations. The performance drops
considerably, highlighting the importance of addressing in-
plane rotations for improving correspondence quality and
overall pose accuracy. Rows 4 and 5 present a further vari-
ation in which we replace MASt3R with the 2D foundation
model DINOv2. Despite following the same pipeline (e.g.,
using the same 40 templates), this substitution leads to a no-
table performance decline across all datasets. Additionally,
in row 6, replacing MASt3R with the dense image matcher
RoMa [10] also results in lower accuracy. We speculate
that MASt3R’s improved performance stems from its large-
scale and 3D-aware training, which may enhance its ability
to produce features that are well-suited for pose estimation.
The above analysis highlights the importance of incorpo-
rating both in-plane rotations and 3D-consistent features to
achieve robust and accurate 6D pose estimation.
16824

YCBV
LM-O
IC-BIN
T-LESS
Figure 5. Qualitative Results of 6D Pose Estimation. Visualization of 6D object pose estimation of Pos3R across LM-O [3], T-LESS [15],
IC-BIN [8], and YCB-V [57]. Pos3R effectively handles a variety of objects and maintains accuracy even in very crowded scenes (e.g.,
IC-BIN), where predictions remain reasonably robust. Furthermore, Pos3R adapts well to texture-less objects, as demonstrated in the
T-LESS dataset. However, as marked by red crosses (✗), heavy occlusion poses a challenge, with reduced accuracy in heavily occluded
regions, illustrating both the strengths and limitations of our approach.
Qualitative Results of 6D Pose Estimation. Figure 5
shows pose estimation results of Pos3R across challeng-
ing datasets, including LM-O [3], T-LESS [15], IC-BIN [8],
and YCB-V [57]. Pos3R demonstrates robust performance
across diverse object types, sizes, and textures, handling
complex environments effectively. In cluttered scenes like
IC-BIN, Pos3R accurately localizes multiple, tightly packed
objects. It also performs well on textureless objects in T-
LESS, a challenge for vision-based methods due to mini-
mal surface features. However, Figure 5 also reveals a lim-
itation: in heavily occluded scenes, marked by red crosses,
Pos3R cannot well handle. Reduced visibility of occluded
objects results in deviations in the predicted bounding
boxes. This limitation suggests future directions for im-
provement, such as integrating occlusion-aware methods
(e.g., GigaPose [39]) or leveraging multi-view information
to enhance robustness in these challenging cases.
5. Conclusion
This work introduces Pos3R, a training-free, RGB-only
framework for 6D pose estimation of unseen objects. By
leveraging the 3D foundation model MASt3R, Pos3R gen-
erates robust, 3D-consistent features that effectively handle
both in-plane and out-of-plane rotations. Without relying
on extensive datasets or object-specific training, Pos3R es-
tablishes a strong baseline for training-free research. Using
a minimal set of templates captured from eight cube ver-
tex viewpoints with controlled rotational variations, Pos3R
achieves accurate and efficient pose estimation.
Experi-
ments on the BOP challenge demonstrate that Pos3R out-
performs other training-free methods in coarse pose estima-
tion and achieves competitive results with refined methods
when combined with MegaPose refinement. Future work
will focus on incorporating occlusion-aware techniques to
enhance robustness in occluded settings.
16825

Acknowledgments. We thank all reviewers and ACs for their
constructive comments.
This work was generously supported
through research collaboration with RIOS Intelligent Machines.
References
[1] Shir Amir, Yossi Gandelsman, Shai Bagon, and Tali Dekel.
Deep ViT features as dense visual descriptors. arXiv preprint
arXiv:2112.05814, 2(3):4, 2021. 1, 3
[2] Philipp Ausserlechner, David Haberger, Stefan Tha