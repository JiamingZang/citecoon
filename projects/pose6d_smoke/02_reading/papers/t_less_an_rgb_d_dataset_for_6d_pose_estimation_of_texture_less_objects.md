# T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-Less Objects

> 2017 · id: W2580726517 · arXiv: 1701.05498 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
We introduce T-LESS, a new public dataset for estimat-
ing the 6D pose, i.e. translation and rotation, of texture-less
rigid objects. The dataset features thirty industry-relevant
objects with no signiﬁcant texture and no discriminative
color or reﬂectance properties. The objects exhibit sym-
metries and mutual similarities in shape and/or size. Com-
pared to other datasets, a unique property is that some of
the objects are parts of others. The dataset includes training
and test images that were captured with three synchronized
sensors, speciﬁcally a structured-light and a time-of-ﬂight
RGB-D sensor and a high-resolution RGB camera. There
are approximately 39K training and 10K test images from
each sensor. Additionally, two types of 3D models are pro-
vided for each object, i.e. a manually created CAD model
and a semi-automatically reconstructed one. Training im-
ages depict individual objects against a black background.
Test images originate from twenty test scenes having vary-
ing complexity, which increases from simple scenes with
several isolated objects to very challenging ones with mul-
tiple instances of several objects and with a high amount of
clutter and occlusion. The images were captured from a sys-
tematically sampled view sphere around the object/scene,
and are annotated with accurate ground truth 6D poses of
all modeled objects. Initial evaluation results indicate that
the state of the art in 6D object pose estimation has ample
room for improvement, especially in difﬁcult cases with sig-
niﬁcant occlusion. The T-LESS dataset is available online
at cmp.felk.cvut.cz/t-less.

## introduction
Texture-less rigid objects are common in human environ-
ments and the need to learn, detect and accurately localize
them from images arises in a variety of applications. The
pose of a rigid object has six degrees of freedom, i.e. three
in translation and three in rotation, and its full knowledge
is often required. In robotics, for example, the 6D object
pose facilitates spatial reasoning and allows an end-effector
to act upon an object. In an augmented reality scenario, ob-
Figure 1. Examples of T-LESS test images (left) overlaid
with colored 3D object models at the ground truth 6D poses
(right). Instances of the same object have the same color.
The goal is to ﬁnd instances of the modeled objects and
estimate their 6D poses.
ject pose can be used to enhance one’s perception of reality
by augmenting objects with extra information such as hints
for assembly guidance.
The visual appearance of a texture-less object is domi-
nated by its global shape, color, reﬂectance properties, and
the conﬁguration of light sources. The lack of texture im-
plies that the object cannot be reliably recognized with tra-
ditional techniques relying on photometric local patch de-
tectors and descriptors [9, 31]. Instead, recent approaches
that can deal with texture-less objects have focused on lo-
cal 3D feature description [33, 51, 19], and semi-global or
arXiv:1701.05498v1  [cs.CV]  19 Jan 2017

global description relying primarily on intensity edges and
depth cues [20, 24, 54, 5, 14, 21, 27]. Therefore, RGB-
D data consisting of aligned color and depth images, ob-
tained with widely available Kinect-like sensors, have come
to play an important role.
In this paper, we introduce a new public dataset for 6D
pose estimation of texture-less rigid objects. An overview
of the included objects and test scenes is provided in Fig. 2.
The dataset features thirty commodity electrical parts which
have no signiﬁcant texture, discriminative color or distinc-
tive reﬂectance properties, and often bear similarities in
shape and/or size. Furthermore, a unique characteristic of
the objects is that some of them are parts of others. For ex-
ample, objects 7 and 8 are built up from object 6, object 9
is made of three copies of object 10 stacked next to each
other, whilst the center part of objects 17 and 18 is nearly
identical to object 13. Objects exhibiting similar properties
are common in industrial environments.
The dataset includes training and test images captured
with a triplet of sensors, i.e. a structured light RGB-D sen-
sor Primesense Carmine 1.09, a time-of-ﬂight RGB-D sen-
sor Microsoft Kinect v2, and an RGB camera Canon IXUS
950 IS. The sensors were time-synchronized and had sim-
ilar perspectives. All images were obtained with an auto-
matic procedure that systematically sampled images from a
view sphere, resulting in ~39K training and ~10K test im-
ages from each sensor. The training images depict objects
in isolation with a black background, while the test images
originate from twenty table-top scenes with arbitrarily ar-
ranged objects. Complexity of the test scenes varies from
those with several isolated objects and a clean background
to very challenging ones with multiple instances of several
objects and with a high amount of occlusion and clutter. Ad-
ditionally, the dataset contains two types of 3D mesh mod-
els for each object; one manually created in CAD software
and one semi-automatically reconstructed from the training
RGB-D images. All occurrences of the modeled objects
in the training and test images are annotated with accurate
ground truth 6D poses; see Fig. 1 for their qualitative and
Sec. 4.1 for their quantitative evaluation.
The dataset is intended for evaluating various ﬂavors of
the 6D object pose estimation problem [23] and other re-
lated problems, such as 2D object detection [50, 22] and
object segmentation [49, 17]. Since images from three sen-
sors are available, one may also study the importance of dif-
ferent input modalities for a given problem. Another option
is to use the training images for evaluating 3D object recon-
struction methods [44], where the provided CAD models
can serve as the ground truth.
Our objectives in designing T-LESS were to provide a
dataset of a substantial but manageable size, with a rigorous
and complete ground truth annotation that is accurate to the
level of sensor resolution, and with a signiﬁcant variability
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
Figure 2.
T-LESS includes training images and 3D mod-
els of 30 objects (top) and test images of 20 scenes (bot-
tom) – shown overlaid with colored 3D object models at
the ground truth poses. The images were captured from a
systematically sampled view sphere around an object/scene
and are annotated with accurate ground truth 6D poses of
all modeled objects.
in complexity, so that it would provide different levels of
difﬁculty and be reasonably future-proof, i.e. solvable, but
not solved by the current state-of-the-art methods. The difﬁ-
culty of the dataset for 6D object pose estimation is demon-
strated by the relatively low performance of the method by
Hodaˇn et al. [24]. This method otherwise achieves a perfor-
mance close to the state of the art on the well-established
dataset of Hinterstoisser et al. [20].

The remainder of the paper is organized as follows.
Sec. 2 reviews related datasets, Sec. 3 describes technical
details of the acquisition and post-processing of the T-LESS
dataset, Sec. 4 assesses the accuracy of the ground truth
poses and provides initial evaluation results, and Sec. 5 con-
cludes the paper.
2. Related Datasets
First we review datasets for estimating the 6D pose of
speciﬁc rigid objects, grouped by the type of provided im-
ages, then we mention a few datasets designed for simi-
lar problems. If not stated otherwise, these datasets supply
ground truth annotations in the form of 6D object poses.
2.1. RGB-D Datasets
Only a few public RGB-D datasets, from over one
hundred reported by Firman in [15], enable the evalua-
tion of 6D object pose estimation methods. Most of the
datasets reviewed in this section were captured with Mi-
crosoft Kinect v1 or Primesense Carmine 1.09, which rep-
resent the ﬁrst generation of consumer-grade RGB-D sen-
sors operating on the structured-light principle. The dataset
introduced in [17] was captured with Microsoft Kinect v2,
which is based on the time-of-ﬂight principle.
For texture-less objects, the dataset of Hinterstoisser et
al. [20] has become a standard benchmark used in most of
the recent work, e.g. [38, 4, 47, 24, 54]. It contains 15
texture-less objects represented by a color 3D mesh model.
Each object is associated with a test sequence consisting of
~1200 RGB-D images, each of which includes exactly one
instance of the object. The test sequences feature signiﬁcant
2D and 3D clutter, but only mild occlusion, and since the
objects have discriminative color, shape and/or size, their
recognition is relatively easy. In the 6D localization prob-
lem (where information about the number and identity of
objects present in the images is provided beforehand [23]),
state-of-the-art methods achieve recognition rates that ex-
ceed 95% for most of the objects. Brachmann et al. [4] pro-
vided additional ground truth poses for all modeled objects
in one of the test sequences from [20]. This extended anno-
tation introduces challenging test cases with various levels
of occlusion and allows the evaluation of multiple object lo-
calization, with each object appearing in a single instance.
Tejani et al. [47] presented a dataset with 2 texture-less
and 4 textured objects. For each object, a color 3D mesh
model is provided together with a test sequence of over 700
RGB-D images. The images show several object instances
with no to moderate occlusion, and with 2D and 3D clut-
ter. Doumanoglou et al. [14] provide a dataset with 183 test
images of 2 textured objects from [47] that appear in mul-
tiple instances in a challenging bin-picking scenario with
heavy occlusion. Furthermore, they provide color 3D mesh
models of another 6 textured objects and 170 test images
depicting the objects placed on a kitchen table.
The Challenge and Willow datasets [58], which were
collected for the 2011 ICRA Solutions in Perception Chal-
lenge, share a set of 35 textured household objects. Train-
ing data for each object is given in the form of 37 RGB-D
training images that show the object from different views,
plus a color point cloud obtained by merging the training
images. The Challenge and Willow datasets respectively
contain 176 and 353 test RGB-D images of several objects
in single instances placed on top of a turntable. The Willow
dataset also features distractor objects and object occlusion.
Similar is the TUW dataset [1] that presents 17 textured and
texture-less objects appearing in 224 test RGB-D images.
Instead of a turntable setup, im

## conclusion
This paper has presented T-LESS, a new dataset for eval-
uating 6D pose estimation of texture-less objects that can
facilitate systematic comparison of pertinent methods. The
dataset features industry-relevant objects and is character-
ized by a large number of training and test images, accu-
rate 6D ground truth poses, multiple sensing modalities, test
scenes with multiple object instances and with increasing
difﬁculty due to occlusion and clutter. Initial evaluation re-
sults using the dataset indicate that the state of the art in 6D
object pose estimation has ample room for improvement.
The T-LESS dataset is available online at:
cmp.felk.cvut.cz/t-less
Acknowledgements
This work was supported by the Technology Agency of
the Czech Republic research program TE01020415 (V3C –
Visual Computing Competence Center), CTU student grant
SGS15/155/OHK3/2T/13, and the European Commission
FP7 DARWIN Project, Grant No. 270138. The help of
Jan Pol´aˇsek and Avgousta Hatzidaki in creating the CAD
models is gratefully acknowledged.