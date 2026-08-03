# MegaPose: 6D Pose Estimation of Novel Objects via Render &amp; Compare

> 2022 · id: W4311640782 · arXiv: 2212.06870 · pdf: https://arxiv.org/pdf/2212.06870 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
Accurate 6D object pose estimation is essential for many robotic and augmented reality applications.
Current state-of-the-art methods are learning-based [2, 3, 4, 5, 6] and require 3D models of the objects
of interest at both training and test time. These methods require hours (or days) to generate synthetic
data for each object and train the pose estimation model. They thus cannot be used in the context
of robotic applications where the objects are only known during inference (e.g. CAD models are
provided by a manufacturer or reconstructed [7]), and where rapid deployment to novel scenes and
objects is key.
The goal of this work is to estimate the 6D pose of novel objects, i.e., objects that are only available
at inference time and are not known in advance during training. This problem presents the challenge
of generalizing to the large variability in shape, texture, lighting conditions, and severe occlusions
that can be encountered in real-world applications. Some prior works [8, 9, 10, 11, 12, 13, 14]
have considered category-level pose estimation to partially address the challenge of novel objects
by developing methods that can generalize to novel object instances of a known class (e.g. mugs or
shoes). These methods however do not generalize to object instances outside of training categories.
Other methods aim at generalizing to any novel instances regardless of their category [15, 16, 17,
18, 19, 20, 21, 22]. These works present important technical limitations. They rely on non-learning
based components for generating pose hypotheses [21] (e.g. PPF [23]), for pose reﬁnement [17] (e.g.
1Inria Paris and D´epartement d’informatique de l’ENS, ´Ecole normale sup´erieure, CNRS, PSL Research
University, 75005 Paris, France.
3LIGM, ´Ecole des Ponts, Univ Gustave Eiffel, CNRS, Marne-la-vall´ee, France.
5Czech Institute of Informatics, Robotics and Cybernetics at the Czech Technical University in Prague.
†Work partially done while the author was an intern at NVIDIA.
6th Conference on Robot Learning (CoRL 2022), Auckland, New Zealand.
arXiv:2212.06870v1  [cs.CV]  13 Dec 2022

...
CAD models
Synthetic images
Region of interest
New CAD model
Output
6D Pose
Inputs
(a) Training
(b) Inference
Depth (optional)
RGB
(c) Visually guided
robotic manipulation
Figure 1: MegaPose is a 6D pose estimation approach (a) that is trained on millions of synthetic scenes with
thousands of different objects and (b) can be applied without re-training to estimate the pose of any novel object,
given a CAD model and a region of interest displaying the object. It can thus be used to rapidly deploy visually
guided robotic manipulation systems in novel scenes containing novel objects (c).
PnP [24] and ICP [25, 26]), for computing photometric errors in pixel space [15], or for estimating
the object depth [18, 16] (e.g. using only the size of a 2D detection [27]). These components however
inherently cannot beneﬁt from being trained on large amount of data to gain robustness with respect
to noise, occlusions, or object variability. Learning-based methods also have the potential to improve
as the quality and size of the datasets improve.
Pipelines for 6D pose estimation of known (not novel) objects that consist of multiple learned
stages [4, 5] have shown excellent performance on several benchmarks [2] with various illumination
conditions, textureless objects, cluttered scenes and high levels of occlusions. We take inspiration
from [4, 5] which split the problem into three parts: (i) 2D object detection, (ii) coarse pose
estimation, and (iii) iterative reﬁnement via render & compare. We aim at extending this approach to
novel objects unseen at training time. The detection of novel objects has been addressed by prior
works [17, 28, 29, 30] and is outside the scope of this paper. In this work, we focus on the coarse and
reﬁnement networks for 6D pose estimation. Extending the paradigm from [4] presents three major
challenges. First, the pose of an object depends heavily on both its visual appearance and choice of
coordinate system (deﬁned in the CAD model of the object). In existing reﬁnement networks based
on render & compare [20, 4], this information is encoded in the network weights during training,
leading to poor generalization results when tested on novel objects. Second, direct regression methods
for coarse pose estimation are trained with speciﬁc losses for symmetric objects [4], requiring that
object symmetries be known in advance. Finally, the diversity of shape and visual properties of the
objects that can be encountered in real-world applications is immense. Generalizing to novel objects
requires robustness to properties such as object symmetries, variability of object shape, and object
textures (or absence of).
Contributions. We address these challenges and propose a method for estimating the pose of any
novel object in a single RGB or RGB-D image, as illustrated in Figure 1. First, we propose a novel
approach for 6D pose reﬁnement based on render & compare which enables generalization to novel
objects. The shape and coordinate system of the novel object are provided as inputs to the network by
rendering multiple synthetic views of the object’s CAD model. Second, we propose a novel method
for coarse pose estimation which does not require knowledge of the object symmetries during training.
The coarse pose estimation is formulated as a classiﬁcation problem where we compare renderings
of random pose hypotheses with an observed image, and predict whether the pose can be corrected
by the reﬁner. Finally, we leverage the availability of large-scale 3D model datasets to generate a
highly diverse synthetic dataset consisting of 2 million photorealistic [31] images depicting over 20K
models in physically plausible conﬁgurations. The code, dataset and trained models are available on
the project page [1].
We show that our novel-object pose estimation method trained on our large-scale synthetic dataset
achieves state-of-the-art performance on ModelNet [32, 20]. We also perform an extensive evaluation
of the approach on hundreds of novel objects from all 7 core datasets of the BOP challenge [2]
and demonstrate that our approach achieves performance competitive with existing approaches that
require access to the target objects during training.
2

2

## method
In this section we present our framework for pose estimation of novel objects. Our goal is to detect
the pose TCO (the pose of object frame O expressed in camera frame C composed of 3D rotation
and 3D translation) of a novel object given an input RGB (or RGBD) image, Io, and a 3D model of
the object. Similar to DeepIM [20] and CosyPose [4], our method consists of three components (1)
object detection, (2) coarse pose estimation and (3) pose reﬁnement. Compared to these works, our
proposed method enables generalization to novel objects not seen during training, requiring novel
approaches for the coarse model, the reﬁner and the training data. Our approach can accept either
RGB or RGBD inputs, if depth is available the RGB and D images are concatenated before being
passed into the network. Detection of novel-objects in an image is an interesting problem that has
been addressed in prior work [28, 60, 17, 22, 30] but lies outside the scope of this paper. Thus for our
experiments we assume access to an object detector, but emphasize that our method can be coupled
with any object detector, including zero-shot methods such as those in [28, 60].
3

Figure 2: ⊕denotes concatenation. (a) Coarse Estimator: Given a cropped input image the coarse module
renders the object in multiple input poses {T j
CO}. The coarse network then classiﬁes which rendered image best
matches the observed image. (b) Reﬁner: Given an initial pose estimate T k
CO the reﬁner renders the objects at
the estimated pose TCO,1 := T k
CO (blue axes) along with 3 additional viewpoints {TCO,i}4
i=2 (green axes) deﬁned
such that the camera z-axis intersects the anchor point O. The reﬁner network consumes the concatenation of
the observed and rendered images and predicts an updated pose estimate T k+1
CO
.
3.1
Technical Approach
Coarse pose estimation. Given an object detection, shown in Figure 1(b), the goal of the coarse
pose estimator is to provide an initial pose estimate TCO,coarse which is sufﬁciently accurate that it can
then be further improved by the reﬁner. In order to generalize to novel-objects we propose a novel
classiﬁcation based approach that compares observed and rendered images of the object in a variety
of poses and selects the rendered image whose object pose best matches the observed object pose.
Figure 2(a) gives an overview of the coarse model. At inference time the network consumes
the observed image Io along with rendered images {Ir(T j
CO)}M
j=1 of the object in many different
poses {T j
CO}M
j=1. For each pose T j
CO the model predicts a score (Io, Ir(T j
CO)) →ξj that classiﬁes
whether the pose hypothesis is within the basin of attraction of the reﬁner. The highest scoring pose
T j∗
CO, j∗= argmaxj ξj is used as the initial pose for the reﬁnement step. Since we are performing
classiﬁcation, our method can implicitly handle object symmetries, as multiple poses can be classiﬁed
as correct.
Pose reﬁnement model. Given an input image and an estimated pose, the reﬁner predicts an updated
pose estimate. Starting from a coarse initial pose estimate TCO,coarse we can iteratively apply the
reﬁner to produce an improved pose estimate. Similar to [4, 20] our reﬁner takes as input observed
Io and rendered images Ir(T k
CO) and predicts an updated pose estimate T k+1
CO , see Figure 2 (b),
where k refers to the kth iteration of the reﬁner. Our pose update uses the same parameterization
as DeepIM [20] and CosyPose [4] which disentangles rotation and translation prediction. Crucially
this pose update ∆T depends on the choice of an anchor point O, see Appendix for more details. In
prior work [4, 20] which trains and tests on the same set of objects, the network can effectively learn
the position of the anchor point O for each object. However in order to generalize to novel objects
we must enable the network to infer the anchor point O at inference time.
In order to provide information about the anchor point to the network we always render images
Ir(T k
CO) such that the anchor point O projects to the image center. Using rendered images from
multiple distinct viewpoints {TCO,i}N
i=1 the network can infer the location of the anchor point O as
the intersection point of camera rays that pass through the image center, see Figure 2(b).
Additional information about object shape and geometry can be provided to the network by rendering
depth and surface normal channels in the rendered image Ir. We normalize both input depth (if
4

available) and rendered depth images using the currently estimated pose T k
CO to assist the network in
generalizing across object scales, see Appendix for more details.
Network architecture. Both the coarse and reﬁner networks consists of a ResNet-34 backbone
followed by spatial average pooling. The coarse model has a single fully-connected layer that
consumes the backbone feature and outputs a classiﬁcation logit. The reﬁner network has a single
fully-connected layer that consumes the backbone feature and outputs 9 values that specify the
translation and rotation for the pose update.
3.2
Training Procedure
Training data. For training, both the coarse and reﬁner models require RGB(-D)1 images with
ground-truth 6D object pose annotations, along with 3D models for these objects. In order for
our approach to generalize to novel-objects we require a large dataset containing diverse objects.
All of of our methods are trained purely on synthetic data generated using BlenderProc [31]. We
generate a dataset of 2 million images using a combination of ShapeNet [61] (abbreviated as SN)
and Google-Scanned-Objects (abbreviated as GSO) [7]. Similar to the BOP [62] synthetic data, we
randomly sampled objects from our dataset and dropped them on a plane using a physics simulator.
Materials, background textures, lighting and camera positions are randomized. Example images can
be seen in Figure 1(a) and in the Appendix. Some of our ablations also use the synthetic training
datasets provided by the BOP challenge [62]. We add data augmentation similar to CosyPose [4] to
the RGB images which was shown to be a key to successful sim-to-real transfer. We also apply data
augmentation to the depth images as explained in the appendix.
Reﬁner model. The reﬁner model is trained similarly to [4]. Given an image with an object
M at ground-truth pose T ∗
CO we generate a perturbed pose T ′
CO by applying a random translation
and rotation to T ∗
CO. Translation is sampled from a normal distribution with a standard deviations
of (0.02, 0.02, 0.05) centimeters and rotation is sampled as random Euler angles with a standard
deviation of 15 degrees in each axis. The network is trained to predict the relative transformation
between the initial and target pose. Following [4, 20] we use a loss that disentangles the prediction of
depth, x-y translation, and rotation. See the appendix for more details.
Coarse model. Given an input image Io of an object M and a pose T ′
CO the coarse model is trained
to classify whether pose T ′
CO is within the basin of attraction of the reﬁner. In other words, if the
reﬁner were started with the initial pose estimate T ′
CO would it be able to estimate the ground-truth
pose via iterative reﬁnement? Given a ground-truth pose-annotation T ∗
CO we randomly sample poses
T ′
CO by adding random translation and rotation to T ∗
CO. The positives are sampled from the same
distribution used to generate the perturbed poses the reﬁner network is trained to correct (see above),
and other poses sufﬁciently distinct to this one (see the appendix for more details) are marked as
negatives. The model is then trained with binary cross entropy loss.
4

## experiments
We evaluate our method for 6D pose estimation of novel objects using the seven challenging datasets
of the BOP [2, 62] 6D pose estimation benchmark, and the ModelNet [20] dataset. The dataset and
the standard 6D pose estimation metrics we use are detailed in Section 4.1. In all our experiments,
the objects are considered novel, i.e. they are only available during inference on a new image and
they are not used during training. In Section 4.2, we evaluate the performance of our approach
composed of coarse and reﬁnement networks. Notably, we show that (i) our method is competitive
with others that require the object models to be known in advance, and (iii) our reﬁner outperforms
current state-of-the-art on the ModelNet and YCB-V datasets. Section 4.3 validates our technical
contributions and shows the crucial importance of the training data in the success of our method.
Finally, we discuss the limitations in Section 4.4.
4.1
Dataset and metrics
We consider the seven core datasets of the BOP challenge [62, 2]: LineMod Occlusion (LM-
O) [63], T-LESS [64], TUD-L [62], IC-BIN [65], ITODD [66], HomebrewedDB (HB) [67] and
1Our method can consume either RGB or RGB-D images depending on the input modalities that are available.
5

Pose Initialization
Pose Reﬁnement
BOP Datasets

## related_work
In this section, we ﬁrst review the literature on 6D pose estimation of known rigid objects. We then
focus on the practical scenario similar to ours where the objects are not known prior to training.
6D pose estimation of known objects. Estimating the 6D pose of rigid objects is a fundamental
computer vision problem [33, 34, 35] that was ﬁrst addressed using correspondences established
with locally invariant features [35, 36, 37, 38, 23] or template matching [39, 40]. These have been
replaced by learning-based methods with convolutional neural networks that directly regress sets
of sparse [41, 42, 27, 43, 44, 45, 46] or dense [47, 48, 49, 50, 3, 44] features. All these approaches
use non-learning stages relying on PnP+Ransac [51, 24] to recover the pose from correspondences
in RGB images, or variants of the iterative closest point algorithm, ICP [25, 26], when depth is
available. The best performing methods rely on trainable reﬁnement networks [52, 20, 4, 20, 5] based
on render & compare [53, 54, 55, 20]. These methods render a single image of the object, which is
not sufﬁcient to provide complete information on the shape and coordinate system of a 3D model to
the network. This information is thus encoded in the networks weights when training, which leads to
poor generalization when tested on novel objects unseen at training. Our approach renders multiple
views of an object to provide this 3D information, making the trained network independent of these
object-speciﬁc properties.
6D pose estimation of novel objects. Other works consider a practical scenario where the objects
are not known in advance. Category-level 6D pose estimation is a popular problem [8, 9, 10, 11, 12,
13, 14] in which CAD models of test objects are not known, but the objects are assumed to belong
to a known category. These methods rely on object properties that are common within categories
(e.g. handle of a mug) to deﬁne and estimate the object pose, and thus cannot generalize to novel
categories. Our method requires the 3D model of the novel object instance to be known during
inference, but does not rely on any category-level information. Other works address a scenario similar
to ours. [56, 19, 57, 18, 16, 30] only estimate the 3D orientation of novel objects by comparing
rendered pose hypotheses with the observed image using features extracted by a network. They
rely on handcrafted [18, 16] or learning-based DeepIM [19] reﬁners to recover accurate 6D poses.
We instead propose a method that estimates the full 6D pose of the object and show our reﬁnement
network signiﬁcantly outperforms DeepIM [20] when tested on novel object instances. The closest
works to ours are OSOP [17] and ZePHyR [21]. OSOP focuses on the coarse estimation by explicitly
predicting 2D-2D correspondences between a single rendered view of the object and the observed
image, and solves for the pose using PnP or Kabsch [25] which makes inference slower and less
robust compared to directly predicting reﬁnement transforms with a network as done in our solution.
ZePHyR [21] strongly relies on the depth modality, whereas our approach can also be used in
RGB-only images. Finally, [15, 58, 22, 59] investigate using a set of real reference views of the
novel object instead of using a CAD model. These approaches have only reported results on datasets
with limited or no occlusions. Our use of a deep render & compare network trained on a large-scale
synthetic dataset displaying highly occluded object instances enables us to handle highly cluttered
scenes with high occlusions like in the LineMOD Occlusion, HomebrewedDB or T-LESS datasets.
3

## conclusion
We propose MegaPose, a method for 6D pose estimation of novel objects. Megapose can estimate
the 6D pose of novel objects given a CAD model of the object available only at test time. We
quantitatively evaluated MegaPose on hundreds of different objects depicted in cluttered scenes, and
performed ablation studies to validate our network design choices and highlight the importance of the
training data. We release our models and large-scale synthetic dataset to stimulate the development of
novel methods that are practical to use in the context of robotic manipulation where rapid deployment
to new scenes with new objects is crucial. While this work focuses on the coarse estimation and
ﬁne reﬁnement of an object pose, detecting any unknown object given only a CAD model is still a
difﬁcult problem that remains to be solved for having a complete framework for detection and pose
estimation of novel objects. Future work will address zero-shot object detection using our large-scale
synthetic dataset.
8

Acknowledgements
This work was partially supported by the HPC resources from GENCI-IDRIS (Grant 011011181R2),
the European Regional Development Fund under the project IMPACT (reg. no. CZ.02.1.01/0.0/0.0/15
003/0000468), EU Horizon Europe Programme under the project AGIMUS (No. 101070165), Louis
Vuitton ENS Chair on Artiﬁcial Intelligence, and the French government under management of
Agence Nationale de la Recherche as part of the ”Investissements d’avenir” program, reference
ANR-19-P3IA-0001 (PRAIRIE 3IA Institute).