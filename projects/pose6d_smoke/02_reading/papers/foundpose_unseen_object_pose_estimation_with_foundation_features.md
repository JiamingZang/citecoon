# FoundPose: Unseen Object Pose Estimation with Foundation Features

> 2024 · id: W4403842181 · arXiv: 2311.18809 · pdf: https://arxiv.org/pdf/2311.18809 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
Image-based estimation of the 6D object pose (3D rotation and 3D translation) is
an important research problem in the field of spatial AI. In robotics, for example,
the information about object poses allows a robot to act upon the objects, which
enables fully automated solutions for warehouse operation or assembly. In mixed-reality
applications, this information unlocks physical interaction with replicas of real-world
objects, such as a computer keyboard, for effective text input when fully immersed.
In this work, we address the problem of model-based 6D pose estimation of unseen
objects. We assume that 3D models of the objects are available and that budget for
onboarding the objects is limited (e.g., not sufficient for rendering a large-scale dataset
and training a neural network). This is a practical problem setup for many applications
*Work done during Evin’s internship at Meta.
arXiv:2311.18809v2  [cs.CV]  19 Jul 2024

2
Örnek et al.
Fig. 1. Bridging synthetic-to-real gap.
Patch descriptors from an intermediate layer
of DINOv2 [62] (top), a recent vision founda-
tion model, are the key enabler of FoundPose.
Thanks to the generalization capability of
these descriptors, it is possible to establish
reliable correspondences between a real query
image (left) and a synthetic template (right)
by a simple nearest-neighbor matching. The
patch descriptors are colored by the top three
components of a PCA space computed from
descriptors of all object templates. Note that
colors of the same object parts are consistent,
despite the real-to-synthetic domain gap.
since efficient object onboarding is often a key requirement and 3D object models can
be obtained from the manufacturer or readily reconstructed [57,71,90].
The very first methods for object pose estimation can, in fact, handle unseen objects as
they do not require any training, typically just a set of reference images. These methods
rely on classical techniques such as matching hand-crafted image features [14,49,69,72]
or template matching [29,55]. Later, with the rise of machine learning techniques in
computer vision, most object pose estimation methods started to rely on deep neural
networks. This shift brought a significant improvement in pose estimation accuracy [82]
but limited generalization capability as large numbers of training images and a lengthy
training process are usually necessary for every new object instance or category. As a
result, the majority of these methods focus on a small set of objects. Only recently, with
the accuracy scores of seen object pose estimation slowly saturating, the research field
started to focus again on unseen objects [56], with the first attempts achieving noticeably
lower accuracy scores while being computationally more demanding [12,34,41,75].
With their impressive generalization capabilities, foundation models [7] provide a
solid ground for solving the problem at hand. Models such as DINOv2 [15,62], CLIP [70]
or ALIGN [37] have been successfully applied on various vision tasks without any
task-specific training [24,53,58,88]. For example, CNOS [58] leverages frozen DINOv2 [15,
62] and Segment Anything [40] and outperforms Mask R-CNN [25] on the object
segmentation task. Goodwin et al. [24] show that DINO patch descriptors [11] can be
used to establish semantic correspondences between instances of the same object category.
Inspired by these success stories, we propose FoundPose, a method for model-based
pose estimation of unseen objects, which brings the power of modern foundation fea-
tures into classical computer vision techniques via careful design choices. Despite being
surprisingly simple, easy-to-interpret, and requiring no object- or task-specific training,
the method achieves state-of-the-art results on the standard BOP benchmark [34].
First, given an RGB image and an object mask from CNOS [58], we perspectively
crop the object region and rapidly retrieve a small set of similarly looking, pre-rendered
object templates. To this end, we develop an efficient retrieval approach by integrating
DINOv2 into the bag-of-words representation from 2003 [76]. This approach is 15X
faster while only slightly less accurate than the heavy render-and-compare coarse stage

FoundPose: Unseen Object Pose Estimation with Foundation Features
3
of MegaPose [41], and requires 100X less templates (several hundreds vs 90K+) than
previous approaches [60,75] (the overall memory footprint is 25X lower).
Second, we establish 2D-2D correspondences between each retrieved synthetic template
(with fixed light and black background) and the real image crop by simple one-way kNN
search of DINOv2 patch descriptors (Fig. 1). In contrast, existing methods typically train
on large-scale, heavily randomized, and task-specific datasets to bridge the synthetic-to-
real domain gap [33,41,83]. We demonstrate that patch descriptors from an intermediate
DINOv2 layer, which were shown to carry stronger positional information [2], are crucial
for achieving geometrically consistent correspondences when semantic information is
ambiguous due to object symmetries or a lack of texture. We show that the intermediate
DINOv2 descriptors are in fact the key enabler of FoundPose, yielding significantly
higher accuracy also compared to descriptors extracted with SAM [40], CLIP [70],
LoFTR [78], S2DNet [23], and dense SIFT [49]. Next, for each retrieved template, we
generate a pose hypothesis from image-to-model 2D-3D correspondences, which are
established by lifting the matched 2D patch locations in the template to 3D using
rendered depth. Finally, we further optimize the top-quality hypothesis by featuremetric
refinement, which applies the idea of the classical photometric refinement [4] to DINOv2
patch descriptors. The refinement effectively compensates for the discrepancy in the
2D-3D correspondences caused by coarse sampling of DINOv2 patches.
In summary, we make the following contributions:
1. A training-free method for model-based object pose estimation which relies on a surpris-
ing simple and easy-to-interpret DINOv2-based pipeline and achieves state-of-the-art
results on the standard BOP benchmark [34].
2. An efficient template retrieval approach which requires 100X fewer templates than
previous approaches and is robust to partial object occlusions.
3. A lightweight object representation which is fast to build and has a 25X lower memory
footprint than competitors, enabling scaling to large numbers of objects.
4. A featuremetric refinement approach which compensates for coarse patch sampling.
5. Demonstrated importance of intermediate DINOv2 descriptors for handling symmetric
and texture-less objects, also outperforming descriptors from other foundation models.
2

## method
RGB-D templates
Template-based object representation (Sec. 3.2)
DINOv2 patch
descriptors
registered in 3D
Bag-of-words
(BoW) descriptors
Patch descriptors
bi
bq
BoW
Ti
Iterative alignment
Final 6D pose
Fig. 2. FoundPose overview. During a short onboarding stage, we render RGB-D templates
showing the object in different orientations, extract DINOv2 patch descriptors [15,62] from the
RGB channels and register the descriptors in 3D using the depth channel. At inference time,
we crop the RGB query image around the object mask predicted by CNOS [58] and retrieve
a small set of most similar templates using a bag-of-words approach (with words defined by
k-means clusters of patch descriptors from all templates). For each retrieved template, a pose
hypothesis is generated by PnP-RANSAC [22,43] from 2D-3D correspondences established by
matching patch descriptors of the image crop and the template. Finally, the pose hypothesis
with the highest number of inlier correspondences is refined by featuremetric alignment.
size of templates is S×S pixels, and the objects are rendered such that the longer side
of their 2D bounding box is δS pixels long, with δ<1. At inference, we generate crops
of the query image with the same size and padding (to allow for errors of segmentation
masks around which we crop the image).
Patch descriptors registered in 3D. For each RGB-D template with an index
t ∈{1,...,n}, we split the RGB channels into m non-overlapping patches of 14×14
pixels and calculate their patch descriptors {pt,i}m
i=1. A patch descriptor is calculated
as pt,i =ϕd(p′
t,i), where p′
t,i is the raw patch descriptor extracted by DINOv2 and
ϕd:Rr 7→Rd projects the r-dimensional raw descriptor to the top d PCA components,
which are calculated from valid patch descriptors of all n templates. A patch is considered
valid if its 2D center falls inside the object mask and the PCA-based dimensionality
reduction is applied to increase efficiency. Then we represent a template t by a set
Tt={(pt,j,xj)|j∈M}, where M are indices of valid patches, xj is a 3D location (in
the coordinate space of the 3D object model) whose 2D projection is at the center of
patch j. The 3D locations, which are calculated from the depth channel of the template
and known camera intrinsics, enable establishing 2D-3D correspondences at inference.
Bag-of-words descriptors. At onboarding, we also pre-calculate bag-of-words descrip-
tors of all templates to enable efficient template retrieval at inference using the classical
bag-of-words image retrieval technique [66,76], which mimics text-retrieval systems
with the analogy of visual words. Specifically, we define visual words as the centroids
of k-means clusters of patch descriptors extracted from all templates of an object. To
calculate the bag-of-words descriptor of a template t, we assign patch descriptors from
the template representation Tt to the nearest visual words and describe the template by
a vector bt=(b1,b2,...,bk). This vector consists of weighted word frequencies defined as
bi=(ni,t/nt)log(N/ni), where ni,t is the number of occurrences of word i in template t,

FoundPose: Unseen Object Pose Estimation with Foundation Features
7
nt is the total number of words in template t, and ni is the number of occurrences of word
i in all N templates. The first term (ni,t/nt) weights words that occur often in a particu-
lar template and therefore describe the template well, while the second term (log(N/ni))
downweights words that occur often in any template. As visual words generated by clus-
tering may suffer from quantization errors, we follow [66,86] and soft-assign each patch
descriptor to several nearest words with weights defined by exp(−d2/2σ2), where d is the
Euclidean distance of the descriptor from the word and σ is a parameter of the method.
3.3
Template retrieval by bag-of-words matching
Perspective cropping. At inference, we start by cropping the image region around
a given object segmentation mask. To minimize perspective distortion and achieve a
crop that resembles a template, we generate the crop by warping the query image to
a virtual pinhole camera focused on the segmentation mask. The virtual camera is
constructed such that its optical axis passes through the center of the 2D bounding
box of the mask, the viewport size is S×S pixels, and the longer side of the warped
2D bounding box is δS pixels long.
Retrieving similar templates. To retrieve a small set of templates, we calculate the
bag-of-words descriptor of the crop (as in Sec. 3.2) and calculate its cosine similarity
(i.e., normalized scalar product) with bag-of-words descriptors of all object templates.
We select h templates with the highest cosine similarity, which provide approximate
hypotheses on the object orientation for the subsequent pose estimation stage.
This retrieval technique is efficient and robust to partial occlusions. When an object
is partially occluded, its visible part still contributes visual words describing the object.
The cosine similarity then normalizes the magnitude and focuses on the direction of
the bag-of-words descriptors and is, therefore, less sensitive to the number and more
to the type of present words. This robustness has been described in prior work [65,76]
and also in our experiments (see results on LM-O and T-LESS in Sec. 4).
Since the bag-of-words descriptor represents an image as a bag of unordered visual
words, the retrieval can be typically improved by re-ranking the results with a spatial
verification stage [65]. However, in our case, a similar verification is implicitly done
by the subsequent PnP-RANSAC (spatially consistent correspondences are expected
to yield a better pose estimate), and bags of words are not actually unordered as the
used patch descriptors from an intermediate DINOv2 layer, from which the words are
constructed, contains 2D positional information.
3.4
Pose estimation from 2D-3D correspondences
Crop-to-template patch matching. For each retrieved template t, we match patch
descriptors from the crop to the nearest descriptors from Tt (in terms of the Euclidean
distance), and establish 2D-3D correspondences Ct={(ui,xi)}m
i=1, where ui is the 2D
center of a query patch and xi is the 3D location associated with the matched patch de-
scriptor from Tt. The cyclic matching from Goodwin et al. [24] did not help in our setup.
Establishing 2D-3D correspondences by crop-to-template patch matching is a con-
siderably simpler problem than exhaustive matching against patch descriptors from all
templates, which would be necessary without the template retrieval stage. Moreover, we
demonstrate that the template-based approach can effectively handle arbitrary objects,

8
Örnek et al.
Template images
Layer 13
Layer 18
Layer 23
Fig. 3. Visualization of DINOv2 patch descriptors. Shown are top three PCA
components of patch descriptors from different layers of DINOv2 ViT-L [15], for a textured
object from YCB-V [91] (top) and a symmetric and texture-less object from T-LESS [31]
(bottom). As observed in [2] and also clearly visible in these visuals, the patch descriptors
contain gradually less positional and more semantic information when going from shallower
to deeper layers – the different coloring of object sides (red left vs yellow right) in Layer
13 gradually blends to a solid color (orange) in Layer 23. FoundPose performs the best with
descriptors from layer 18, which presumably provides the right information mix. We observed
that these descriptors produce geometrically consistent correspondences even on symmetric and
texture-less objects – when the semantic information is ambiguous (due to symmetries or a lack
of texture), the positional information prioritizes matching patches from the same object side.
including challenging objects with symmetries and without a significant texture. The
ambiguity of 2D-3D correspondences, for which such objects are notoriously known [30],
is eliminated by (1) restricting the candidate patches to only a single template and (2) us-
ing patch descriptors from an intermediate layer of DINOv2 which contain both semantic
and 2D positional information [2]. We find that the positional information is crucial
for producing geometrically consistent correspondences when the semantic information
is not discriminative due to symmetries or a lack of texture (see Fig. 3 and Sec. 4.3).
Pose fitting. An object pose (Rt,tt), defined by a 3D rotation Rt and a 3D translation
tt from the model space to the camera space, is estimated for each retrieved template
t from 2D-3D correspondences Ct by solving the Perspective-n-Point (PnP) problem.
We solve this problem by the EPnP algorithm [43] combined with the RANSAC fitting
scheme [22] for robustness. In this scheme, PnP is solved repeatedly on a randomly
sampled minimal set of 4 correspondences, and the final output is defined by the
pose hypothesis with the highest quality, which we define by the number of inlier
correspondences [22]. From the set of h p

## experiments
In this section, we compare the accuracy and speed of FoundPose with the state-of-the-art
methods evaluated on the BOP benchmark [32–34,82] and present ablation experiments.
4.1
Experimental setup
Evaluation protocol. We follow the protocol of the BOP Challenge 2019–2023 [33].
In summary, a method is evaluated on the 6D object localization problem, and the
error of an estimated pose w.r.t. the ground-truth pose is calculated by three pose-error
functions: Visible Surface Discrepancy (VSD) treats ambiguous poses as equivalent
by considering only the visible object part, Maximum Symmetry-Aware Surface Dis-
tance (MSSD) considers a set of pre-identified global object symmetries and measures
the surface deviation in 3D, and Maximum Symmetry-Aware Projection Distance
(MSPD) considers the object symmetries and measures the perceivable deviation. An
estimated pose is considered correct w.r.t. a pose-error function e, if e < θe, where
e∈{VSD,MSSD,MSPD} and θe is the threshold of correctness. The fraction of an-
notated object instances for which a correct pose is estimated is referred to as Recall.
The Average Recall w.r.t. a function e, denoted as ARe, is defined as the average of the
Recall rates calculated for multiple settings of the threshold θe and also for multiple
settings of a misalignment tolerance τ in the case of VSD. The overall accuracy of a
method is measured by the Average Recall: AR=(ARVSD+ARMSSD+ARMSPD)/3.

10
Örnek et al.
Fig. 4. Example FoundPose results on HB, LM-O, IC-BIN, TUD-L, ITODD and T-LESS
datasets, showing that our method can handle a broad range or objects, including textured,
texture-less and symmetric ones. Each example shows the query image crop with the CNOS
mask in white (top left), retrieved templates (middle row), matched patch descriptors of the
crop and the template that led to the top-quality pose estimate (bottom row), and the contour
of the ground-truth pose in red, the coarse pose in blue, and the refined pose in green (top right).
Datasets. The experiments are conducted on the seven core BOP datasets: LM-O [8],
T-LESS [31], ITODD [20], HB [39], YCB-V [91], IC-BIN [19], and TUD-L [32]. The
datasets feature 108 diverse objects ranging from texture-less and symmetric industrial
objects (ITODD, T-LESS) to typical household objects. The images show scenes whose
complexity varies from simple scenes with several isolated objects to challenging ones with
multiple object instances and a high amount of clutter and occlusion. Only 3D object
models and test images from these datasets were used for experiments with FoundPose,
not the provided synthetic nor real training images since no training is required.
Compared methods. FoundPose is compared against model-based RGB methods
evaluated on the unseen object pose estimation task of the BOP Challenge 2023 [34]:
GenFlow [54], MegaPose [41], GigaPose [59], and also against ZS6D [3] and OSOP [75].
Except for OSOP, all of these methods (including FoundPose) use the same segmentation
masks that were produced by CNOS [58] and provided to the challenge participants.
OSOP relies on a custom detector of unseen objects for LM-O, HB, and YCB-V, and
on Mask R-CNN [25] trained for specific objects for T-LESS (hence we do not include
the T-LESS result). Besides variants of FoundPose where the coarse poses are refined

FoundPose: Unseen Object Pose Estimation with Foundation Features
11
by the featuremetric refinement (Sec. 3.5), we evaluate variants with poses refined by
5 iterations of the MegaPose refiner (i.e., the last stage of the MegaPose pipeline [41]).
Implementation details. Unless stated otherwise, we use the following parameter
settings in the presented experiments. We rendered 800 templates per object with
approximately 25◦angle between depicted object orientations. We set the size of
templates and of the query image crop to 420×420 px with δ=0.6. With the patch
size of 14×14 px (for which DINOv2 is trained), we extract 30×30 patch descriptors
from each template/crop and reduce their dimensionality by projecting them to the top
256 PCA components. We use the output tokens from layer 18 of DINOv2 ViT-L/14
with registers [15] as the patch descriptors. Visual words for the bag-of-words template
retrieval are defined per object by the centroids of 2048 k-means clusters of patch
descriptors from all templates of the object. The bag-of-words descriptors are constructed
by soft-assigning each patch descriptor to 3 nearest words with σ=10. For each query
image crop, we retrieve 5 templates, and estimate the pose from 2D-3D correspondences
(established between the query image crop and the template) by PnP-RANSAC running
for up to 400 iterations with the inlier threshold set to 10 px. The featuremetric
refinement is applied to the best coarse pose and runs until convergence for up to 30
iterations, with the Barron loss [6] parameters set to α=−5 and c=0.5. By default, the
evaluated FoundPose variants use n CNOS masks per object, where n is the number
of object instances to localize (provided as input in the 6D object localization task
in BOP). The only exceptions are variants in the bottom part of Tab. 1, which use
5n CNOS masks per object. Note that all masks were loaded from files with default
CNOS masks, which were provided for BOP 2023 [34] and contain multiple masks per
object instance. The number of CNOS masks used by other methods is unknown.
4.2
Main results
Accuracy. Among methods that do not apply any refinement stage, FoundPose (with-
out the featuremetric refinement) produces significantly more accurate poses than the
competitors, achieving +10, +14, and +16 AR on the seven BOP datasets on average
compared to the coarse versions of GigaPose [59], GenFlow [54], and MegaPose [41]
(rows 1–4 in Tab. 1). The featuremetric refinement brings an extra improvement of +5
AR on average (rows 1 vs 7). At an additional computational cost, a large improvement
of +17 AR (rows 1 vs 8) can be achieved if the coarse poses from FoundPose are
refined by the iterative render-and-compare approach from MegaPose, which is trained
on 2M+ synthetic images of diverse objects and proven remarkably effective. When
initiated with coarse poses from FoundPose, the MegaPose refiner achieves +4 higher
AR score than when initiated with poses from the original coarse pose estimation stage
of MegaPose (rows 8 vs 11). Further improvements at further computational cost can
be achieved if the refinement is applied to multiple pose hypotheses and the top refined
pose is reported as the final estimate (rows 12–17). We achieve the overall best average
AR score of 59.6 AR when top 5 pose hypotheses (generated from 5 retrieved templates)
are optimized with the featuremetric refinement followed by the MegaPose refinement.
On both the single and the multi hypotheses setups, combining the two refinement
approaches achieves the best scores (rows 9 and 14), suggesting their complementarity.
This entry outperforms multi-hypotheses versions of MegaPose and GenFlow (rows
16 and 17), which are the top-performing RGB methods from the BOP Challenge
2023 [34], as well as GigaPose [59] with the MegaPose refinement (row 15).

12
Örnek et al.
#

## related_work
This paper builds on over 60 years of research in object pose estimation and on the
recent large-scale vision foundation models.
Classical methods. Estimating the 6D pose of rigid objects from a single image is one
of the first computer vision problems [72]. Early methods relied on local feature match-
ing [14,49,69] or template matching [29,55], and could rapidly onboard new objects if
provided with a set of reference images annotated with model-to-camera transformations.
With the introduction of Microsoft Kinect, the attention of the research field was steered
towards object pose estimation from RGB-D or D-only images, yielding methods based on
3D local features [85], notably successful point-pair features [21,32], and methods based
on RGB-D template matching [28,36]. The RGB-D methods produce more accurate poses
and are therefore popular in industry, but their application in open-world scenarios is lim-
ited. Besides boosting the pose estimation accuracy, the additional depth channel from

4
Örnek et al.
Kinect-like sensors enabled easy 3D object reconstruction [57], and, in turn, methods rely-
ing on 3D mesh models started to emerge. The model-based object pose estimation setup
is still popular [34,82], among both RGB and RGB-D methods, and is relevant for factory
and warehouse scenarios where CAD object models are often available. On the other hand,
the model-free setup, recently revisited in [26,79], is relevant for mixed reality applications
where the set of target objects is typically small and capturing reference images is easy.
Deep learning methods. As in other fields of computer vision, methods based on
hand-crafted features and techniques have been progressively replaced by methods
based on deep neural networks [42,45,51,87,91], which can operate on RGB or RGB-D
inputs. These methods represent the current state of the art in terms of accuracy [34,82].
However, their scalability is hindered by the requirement of a large-scale training
dataset for learning new objects. To address this issue, deep-learning methods that can
onboard new objects without any object-specific training have been proposed recently.
As examples of model-based methods, Nguyen et al. [60], Shugurov et al. (OSOP) [75],
and Thalhammer et al. [84] learn descriptors for template matching by contrastive
learning, Sundermeyer et al. [80] generate such descriptors by an augmented auto-encoder,
Pitteri et al. [67,68] predict generic 3D keypoints or local surface embeddings, and Xiao et
al. [92] directly predict the 3D object orientation. Model-free methods [26,27,48,79],
and methods relying on depth measurements [5,12,61,64] have been also proposed.
The top-performing methods on the unseen object pose estimation task of the BOP
Challenge 2023 [34] include GenFlow [54], MegaPose [41], GigaPose [59], Foundation-
Pose [89], SAM-6D [46], and PoMZ [10]. To achieve generalization to novel objects,
all except PoMZ require generating millions of task-specific training images showing
thousands of different objects. Generating such datasets requires significant effort and
opens up new types of challenges, including positioning objects in the scene [16,35],
collecting a sufficiently large set of object models, or texturing the models [89]. In
contrast, FoundPose does not require any training, uses frozen DINOv2 features, and
outperforms RGB-only GenFlow, MegaPose, and GigaPose. FoundationPose, SAM-6D,
PoMZ, as well as ZeroPose [12], are RGB-D methods and do not have RGB-only versions.
To the best of our knowledge, the only training-free methods for model-based object
pose estimation are PoMZ [10], which requires RGB-D inputs, and ZS6D [3], which
achieves significantly lower accuracy than FoundPose. ZS6D uses features from the last
ViT [18] layer for establishing correspondences, which we show inferior to our solution
(Sec. 4). Note that [3,10,12,54,59,89] are all unpublished at the time of submission.
Foundation models. A foundation model is a machine-learning model trained on broad
data by self-supervised learning that can be adapted to a wide range of downstream
tasks [7]. Foundation models initially appeared in natural language processing with
examples such as BERT [17] or GPT-3 [9]. In computer vision, foundation models
already achieve on-par or better results than supervised models [7,11,13,53,62,63,88].
A prominent example is DINOv2 [62], which is based on the Vision Transformer
architecture [18], trained in a self-distillation fashion, and has been shown to encode fine
spatial information about the object parts as well as semantic information about object
categories [2]. It has been successfully used in zero-shot setups, i.e., without any training,
for establishing semantic correspondences [2,24,47,93]. FoundPose builds on these insights.
Compared to Goodwin et al. [24] which use DINO patch descriptors to establish semantic
correspondences within an object category, FoundPose shows that DINOv2 descriptors
can be used to establish synthetic-to-real correspondences. Furthermore, [24] requires

FoundPose: Unseen Object Pose Estimation with Foundation Features
5
RGB-D inputs at test time, cannot handle symmetric objects (such objects are omitted in
their evaluation), and is compared only with custom baselines. FoundPose assumes RGB-
only test inputs, can handle symmetric objects by design, and achieves state-of-the-art
results on the standard BOP benchmark. From the already reviewed methods for pose
estimation of specific objects, PoMZ [10] and ZS6D [3] also rely on frozen DINOv1/v2.
However, the first needs RGB-D inputs, and the latter achieves noticeably lower accuracy.
3
FoundPose
In this section, we describe FoundPose, the proposed method for unseen object pose
estimation. We first provide a high-level overview of the method in Sec. 3.1 and then
focus on the key components and rationale of our design choices in Sec. 3.2–3.5.
3.1
Method overview
Problem definition. We consider the problem of estimating the 6D pose of rigid
objects from a single RGB image with known intrinsics. The objective is to estimate
the pose of all instances of target objects that are visible in the image. We assume
that the only information provided for the target objects are their 3D mesh models and
that there is only a limited budget for onboarding the objects, i.e., for preparing object
representations that can be used for online pose estimation. We constrain the onboarding
process to 5 minutes and 1 GPU, as required by the BOP Challenge 2023 [34]. We
additionally assume that segmentation masks of the target object instances, together
with per-mask object identity, are provided at inference time. In our experiments, we
obtain the masks by CNOS [58], a recent method for segmentation of unseen objects
that also requires only 3D models for onboarding the objects.
Onboarding and inference. During an offline onboarding stage, we render templates
showing 3D object models in different orientations. From each template, we extract
DINOv2 descriptors of image patches and register the descriptors in 3D, i.e., each patch
descriptor is associated with the corresponding 3D location in the object model space
(Sec. 3.2). At inference time, given a segmentation mask of an object instance, we crop the
image region around the mask, extract DINOv2 patch descriptors of the crop, and apply
a bag-of-words retrieval technique to efficiently identify a small set of templates that
show the object in orientations similar to the observation (Sec. 3.3). For each retrieved
template, we establish 2D-3D correspondences by matching patch descriptors from the
crop against patch descriptors from the template, and generate a pose hypothesis by
the PnP-RANSAC algorithm (Sec. 3.4). Finally, we refine the best pose hypothesis by
featuremetric alignment, an optimization-based algorithm inspired by photometric align-
ment that operates on features (Sec. 3.5). The pipeline of the method is shown in Fig. 2.
3.2
Template-based object representation
Template generation. Given a texture-mapped 3D object model, we render n RGB-D
templates showing the model under different orientations. The orientations are sampled
to uniformly cover the SO(3) group of 3D rotations [1], and the model is rendered using
a standard rasterization technique [74] with a black background and fixed lighting. The

6
Örnek et al.
Online inference
Offline object onboarding
Template retrieval by bag-of-words matching (Sec. 3.3)
Featuremetric pose refinement (Sec 3.5)
Pose estimation from 2D-3D correspondences (Sec. 3.4)
Query RGB image with 
known intrinsics
Instance mask from CNOS
Matching patch descriptors by NN search
Retrieved templates
3D object

## conclusion
We have proposed an RGB method for model-based pose estimation of unseen objects,
which significantly outperforms existing methods on the standard BOP benchmark.
We believe that achieving this without any object- nor task-specific training, just
with a frozen vision foundation model, is an important and non-obvious outcome.
Furthermore, we have shown that the method can be seamlessly combined with an
existing render-and-compare refinement approach to achieve RGB-only state-of-the-art
results. Our strong results are encouraging to revisit efficient classical computer vision
which is often overlooked in the modern literature.

FoundPose: Unseen Object Pose Estimation with Foundation Features
15