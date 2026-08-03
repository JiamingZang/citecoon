# Recent Advances of Monocular 2D and 3D Human Pose Estimation: A Deep Learning Perspective

> 2022 · id: W3157441214 · 来源: web-agent
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

JOURNAL OF LATEX CLASS FILES„ VOL. X, NO. X, 2021
1
Recent Advances in Monocular 2D and 3D Human
Pose Estimation: A Deep Learning Perspective
Wu Liu, Member, IEEE, Qian Bao, Yu Sun, and Tao Mei, Fellow, IEEE.
Abstract—Estimation of the human pose from a monocular
camera has been an emerging research topic in the computer
vision community with many applications. Recently, beneﬁted
from the deep learning technologies, a signiﬁcant amount of
research efforts have greatly advanced the monocular human
pose estimation both in 2D and 3D areas. Although there have
been some works to summarize the different approaches, it
still remains challenging for researchers to have an in-depth
view of how these approaches work. In this paper, we provide
a comprehensive and holistic 2D-to-3D perspective to tackle
this problem. We categorize the mainstream and milestone
approaches since the year 2014 under uniﬁed frameworks. By
systematically summarizing the differences and connections be-
tween these approaches, we further analyze the solutions for
challenging cases, such as the lack of data, the inherent ambiguity
between 2D and 3D, and the complex multi-person scenarios.
We also summarize the pose representation styles, benchmarks,
evaluation metrics, and the quantitative performance of popular
approaches. Finally, we discuss the challenges and give deep
thinking of promising directions for future research. We believe
this survey will provide the readers with a deep and insightful
understanding of monocular human pose estimation.
Index Terms—Survey for human pose estimation, deep learn-
ing, 2D and 3D pose, monocular images.
I. INTRODUCTION
A. Motivation
M
ONOCULAR human pose estimation (MHPE) is a
fundamental and challenging task in the computer vision
community. It aims to predict the human pose information, such
as the spatial locations of body joints and/or the body shape
parameters, from a monocular image or video. MHPE has
been widely exploited for many computer vision tasks, such as
person re-identiﬁcation [1], [2], human parsing [3], [4], human
action recognition [5], [6], and human-computer interaction [7],
[8], etc. As MHPE does not need the complex multi-cameras
or wearable marker points, it has become a signiﬁcant part of
many real-world applications, such as virtual reality, 3D movie
making/editing, self-driving, motion and activity analysis, and
human-robot interaction.
According to the spatial dimension of the output results, the
mainstream MHPE tasks can be divided into two categories, 2D
pose estimation, and 3D pose estimation. Monocular 2D human
pose estimation, also known as 2D keypoint detection, aims
to locate the 2D coordinates of human anatomical keypoints
(body joints) from images. Considering the number of people
in a given image, the 2D human pose estimation task can
be further classiﬁed into single person and multi-person pose
estimation. Furthermore, given a video sequence, 2D pose
All authors are with JD AI Research, JD.com, Beijing, China, 100101.
Email: {liuwu1, baoqian, tmei}@jd.com, yusun@stu.hit.edu.cn
Fig. 1: The number of the published papers in mainstream
computer vision, multi-media, and computer graphics confer-
ences (CVPR, ICCV, ECCV, etc) and journals (TPAMI, TIP,
TOG, etc) from the year 2014 to 2020.
estimation can exploit temporal information to boost keypoint
prediction in a video system. Different from solely predicting
2D locations of body joints, 3D pose estimation further predicts
the depth information for more accurate spatial representation.
In this process, 2D pose estimation can be exploited as
the intermediate representation for 3D pose estimation. In
recent years, demanding for understanding the detailed pose
information of humans has driven 3D pose estimation towards
predicting not only the 3D location but also the detailed 3D
shape and body texture.
Limited by data and computational resources, early research
mainly focused on designing handcrafted features or ﬁtting the
deformable human body models with optimization algorithms.
Recently, with the increase of large-scale 2D/3D pose datasets
(e.g. COCO [9], MPII [10], Human3.6M [11], and 3DPW [12]),
deep learning technologies have signiﬁcantly boosted the
performance of human pose estimation both in accuracy
and efﬁciency. As shown in Fig. 1, from the year 2014 to
2020, the number of published papers in the mainstream
conferences (CVPR, ICCV, ECCV, etc) and journals (TPAMI,
TIP, TOG, etc) in the area of computer vision, multi-media, and
computer graphics has rapidly increased. Recent works mainly
focus on network design and optimization [13]–[22], multitask
interaction [23]–[27], body model exploration [28]–[31], etc.
Although great successes have been achieved in performance
and practice, few works have comprehensively reviewed the
representative algorithms or given insightful analyses of 2D-to-
arXiv:2104.11536v1  [cs.CV]  23 Apr 2021

JOURNAL OF LATEX CLASS FILES„ VOL. X, NO. X, 2021
2
Fig. 2: Milestones, idea or dataset breakthroughs, and the state-of-the-art methods for 2D (top) and 3D (bottom) pose estimation
from the year 2014 to 2021.
3D pose estimation. On one hand, some previous surveys [32]–
[35] reviewed traditional methods, such as body models
or handcrafted features, without recent deep-learning-based
approaches. On the other hand, recent surveys have mainly
focused on one aspect of either 2D pose estimation [36] or
3D pose estimation [37], without a comprehensive perspective
to explore the intrinsic connections between 2D and 3D. The
survey [38] describes recent works of representative 2D pose
estimation methods and a few 3D pose estimation methods up
to the year 2019. However, it does not well summarize the
relative 3D pose and shape estimation methods, and neglects the
perspective from 2D to 3D. Therefore, a more comprehensive
survey covering the recent advantage of pose estimation is of
great need in this community.
In this paper, we provide a comprehensive review of the deep
learning-based MHPE approaches from 2D to 3D in recent
years. We believe that most representative MHPE methods have
intrinsic similarities and connections. Moreover, with the rapid
development of 3D pose and shape estimation, it is necessary
to have a deeper survey on the human pose estimation from 2D
to 3D. Therefore, compared with the paper [38], our survey has
the following differences and advantages. 1) We summarize
the prevailing networks for both 2D and 3D pose estimation
in the uniﬁed frameworks. They represent the representative
paradigms. 2) We provide insightful analyses for human 3D
representation, 3D datasets, 3D shape recovery methods, as
well as the challenges and further work for 3D pose estimation.
3) Besides, we released a detailed code toolbox1 for 3D pose
data processing, which will be timely and useful for 3D pose
research. We summarize a timeline in Fig. 2, which shows
the milestones, idea or dataset breakthroughs, and the state-
of-the-art methods for 2D and 3D pose estimation from the
year 2014 to 2021. We can see that new approaches and new
datasets promote each other. 2D pose estimation has achieved
explosive development since 2016 with breakthroughs both in
1https://github.com/Arthur151/SOTA-on-monocular-3D-pose-and-shape-
estimation
ideas and datasets. Meanwhile, 3D pose estimation has also
developed rapidly in recent years.
B. Overview of Deep Learning Framework for MHPE
The human body is nonrigid and ﬂexible for high degree-
of-freedom poses, therefore, predicting human pose estimation
from a monocular camera faces many challenges, such as com-
plex or strange posture, person-object/person-person interaction
or occlusion, and crowded scenes, etc. Different camera views
and complex scenes will also introduce problems of truncation,
image blur, low resolution, and small target persons.
To address these problems, existing methods explore the
powerful representation of deep learning to mine more clues
for pose estimation. Although they are different in either global
design or detailed optimization, the network architectures of
milestone methods have internal similarities. As shown in
Fig. 3, most of the prevailing single person pose estimation
networks [14], [16]–[18], [20], [39], [40] can be regarded as
consisting of a pose encoder (also called feature extractor)
followed by a pose decoder. The former aims at extracting
high-level features through a high-to-low resolution process.
The latter estimates the target output, 2D/3D keypoint location
or 3D mesh, in a detection-based manner or a regression-based
manner. For the pose decoder, detection-based methods yield
feature maps or heatmaps, while regression-based methods
directly output the target parameters. Following the uniﬁed
frameworks, we describe details of network design for 2D and
3D pose estimation in Sections III and IV, respectively.
For multi-person scenes, to estimate the 2D or 3D pose of
each person, existing works exploit the top-down paradigm
or bottom-up paradigm. The top-down framework ﬁrst detects
the person areas and then extracts the bounding box-level
features from them. The features are used to estimate the
pose results for every single person. In contrast, the bottom-up
paradigm ﬁrst detects all target outputs and then assign them
to different people by grouping [41]–[43] or sampling [44].
As shown in Fig. 4, the representative multi-person methods

JOURNAL OF LATEX CLASS FILES„ VOL. X, NO. X, 2021
3
Fig. 3: Typical framework for single person pose estimation.
Fig. 4: Typical frameworks for multi-person pose estimation.
of the two paradigms also rely on pose-encoder-and-decoder-
based architecture with network input being either the detected
bounding box or the whole image.
Therefore, how to design an effective pose encoder and
pose decoder architecture is a common and popular topic
in pose estimation. Different from classiﬁcation, detection,
and semantic segmentation, human pose estimation needs to
deal with the subtle differences between body parts, especially
in the unavoidable truncation, crowded, and occluded cases.
To achieve this, the body structural models [45]–[47], multi-
scale feature fusion [14], [18], multistage pipelines [41], [48],
reﬁnement in a coarse-to-ﬁne manner [19], [49], multi-task
learning [23], [26], [27], etc, have been explored and designed.
We will introduce them in detail in Section III and IV.
Moreover, regarding estimating 3D poses from monocular
images, another challenge is the insufﬁcient in-the-wild 3D
training data. Because of the equipment constraints, common
3D pose datasets are often captured in restrained experimental
environments. For example, the most widely used 3D pose
dataset Human3.6M [11] contains only 15 indoor activities
performed by seven persons. Therefore, the diversities of human
poses, shapes, and scenes are extremely limited. Models solely
trained on these datasets are prone to fail on the in-the-wild
images. To address this problem, many methods take the 2D
pose as the intermediate representation or extra supervision,
and learn from in-the-wild 2D pose information. Nevertheless,
there are inherent ambiguities in this process, i.e., a single 2D
pose may correspond to multiple 3D poses and vice versa. To
solve the inherent ambiguities, we must consider how to fully
exploit the common structure prior to the human body, motion
continuity, and multi-view consistency.
In conclusion, considering the main challenges of the task
and the uniﬁed frameworks of the representative paradigms,
in this paper, we systematically analyze the deep learning-
based 2D and 3D MHPE approaches proposed since the year
2014. The rest of the paper is organized as follows. We ﬁrst
introduce the pose estimation background in Section II, which
is fundamental for understanding the MHPE task. Then in
Section III, we introduce the representative approaches for
2D pose estimation, including single person pose estimation,
multi-person pose estimation, pose estimation in videos, and
the related tasks of 2D pose estimation, respectively. Then in
Section IV, for 3D pose estimation, we detail the approaches
according to their motivations and challenges. In addition,
we introduce widely used 2D and 3D pose benchmarks in
Section V and compare their state-of-the-art methods. Finally,
in Section VI, we conclude the paper and give some insight
into future research.

JOURNAL OF LATEX CLASS FILES„ VOL. X, NO. X, 2021
4
II. BACKGROUND
A. Representations for Human Body
Various representations of the human body have been
developed to describe the complex human body pose in
different aspects. They have shown various characteristics
to handle different challenges of pose estimation. Existing
representations can be divided into two categories: 1) keypoint-
based representation; and 2) model-based representation.
1) Keypoint-based Representation : 2D or 3D coordinates
of body keypoints are the simple and intuitive representations
for the body skeleton, which have several representation forms.
2D/3D keypoint coordinates. Body keypoints can be ex-
plicitly described by the 2D/3D coordinates. As shown in Fig. 5
(a), the keypoints are connected following the inherent body
structure. The orientations of the body part can be derived
from these connected limbs.
2D/3D heatmaps. To make the coordinates more suitable
for being regressed by a convolutional neural network, many
methods represent the keypoint coordinates in a heatmap
manner. As shown in Fig. 5 (b), the Gaussian heatmap of
each keypoint has a high response value on the corresponding
2D/3D coordinates and a low response value at other positions.
Orientation maps. Some methods [41], [51] take body
keypoints’ orientation map as the auxiliary representation of
heatmaps. OpenPose [41] develops the well-known part afﬁnity
ﬁelds (PAFs) to represent the 2D orientation between limbs. As
shown in Fig. 5 (c), a PAF is a 2D vector ﬁeld that associates
two keypoints of a limb. Each pixel in the ﬁeld contains a
2D vector that points from one part of the limbs to the other.
Orinet [51] further develops it into the 3D orientation map,
which can explicitly model the limb orientations.
Hierarchical bone vectors. The 2D version of hierarchical
bone representation was proposed in the compositional human
pose (CHP) [52], which is the combination of joints and bone
vectors. Xu et al. [53] and Li et al. [50] further developed
it to 3D. As shown in Fig. 5 (d), the 3D human skeleton
is represented by a set of bone vectors. Each bone vector
is pointing from the parent keypoint to the child keypoint,
following a kinematic tree. Each parent keypoint is associated
with a local spherical coordinate system. The bone vector can
be represented by a spherical coordinates in this system.
2) Model-based Representation: Model-based representation
is developed according to the inherent structural characteristics
of the human body. It provides richer body information than
the keypoint-based description. The model-based representation
can be divided into the part-based volumetric model and the
statistical 3D human body model.
Part-based volumetric model. Part-based volumetric mod-
els are developed to address challenges in reality. For example,
in [54], the cylinder model was developed to generate the
labels of occluded parts. As shown in the blue model of Fig. 5
(e), each limb is represented as a cylinder. Each cylinder is
located by aligning the top and bottom surface centers with
the 3D keypoints of the limb. Similarly, as shown in the pink
model of Fig. 5 (e), an EllipBody model is proposed to take
the ellipsoid as the basic unit of body parts [55]. It is more
ﬂexible than a cylinder.
Detailed statistical 3D human body model. Compared
with the part-based volumetric model, the statistical 3D human
body mesh describes more detailed information including the
body pose and shape. We introduce the most widely used
skinned multi-person linear model (SMPL) [29], which is a
skeleton-driven human body model. SMPL disentangles the
shape and pose of a human body, and encodes the 3D mesh into
low-dimensional parameters. It establishes an efﬁcient mapping
M(β, θ; Φ) : R|θ|×|β| 7→R3×6890 from shape β and pose θ to
a triangulated mesh with 6,890 vertices, where Φ represents the
statistical prior of the human body. The shape parameter β ∈
R10 is the linear combination weight of 10 basic shapes. The
pose parameter θ ∈R3×23 represents the relative 3D rotation
of 23 joints in the axis-angle representation. Then a linear
regressor R ∈R6890×24 is developed to derive preselected
body joints J ∈R3×24 from 6890 vertices of human body
mesh via J = M(β, θ; Φ)R. The linear combination operation
of this regressor guarantees that joint location is differentiable
with respect to shape β and pose θ parameters.
B. 3D-to-2D Projection
3D-to-2D projection connects the 3D space to the 2D image
plane. It is important to introduce this tool to better understand
the methods that use it. 3D-to-2D projection uses a camera
model to generate 3D-2D pose pairs [50], [56], supervise 3D
poses using 2D pose annotations [20], [40], [57], or reﬁne 2D
poses via 3D pose projection [53]. The perspective camera
model and weak-perspective camera model are two kinds of
widely used camera models.
Perspective camera model. The perspective camera model
is usually used to project the points in the 3D space into 2D
pixel coordinates on the image plane. Generally, it consists
of two steps. First, we need to transform the 3D points into
the camera coordinates using the extrinsic matrix [R|t], which
describes the camera rotation and translation. Second, we need
the intrinsic matrix K to make an adaptive adjustment for
accurate projection. Therefore, the 2D projection J2d of 3D
keypoints J3d can be described as J2d = K[R|t]J3d.
Weak-perspective camera model. In most situations, the
input 2D images are un-calibrated and complete perspective
camera parameters can hardly be retrieved. Therefore, the
weak-perspective camera model is more widely used in most
existing methods for calculating the 2D projection Jwp2d of 3D
keypoints J3d by Jwp2D = sΠ(RJ3d) + t, where R ∈R3 is
the global rotation parameter, Π is an orthographic projection
operation, t ∈R2 and s ∈R represent the translation and scale
on the image plane, respectively.
III. MONOCULAR 2D POSE ESTIMATION
Monocular 2D pose estimation predicts the 2D locations
of body keypoints in images or videos. According to the
input/output, the task can be divided into single person pose
estimation and multi-person pose estimation in image-level or
video-level. Since the ﬂexibility of the human body, 2D pose
estimation has to deal with various postures, self-occlusion,
and the interaction between body and scene. Especially, in
multi-person scenes, the problems of crowd and occlusion

JOURNAL OF LATEX CLASS FILES„ VOL. X, NO. X, 2021
5
Fig. 5: Widely used human body representations: (a) 2D keypoints [41]; (b) 2D heatmap (upper) [41] and volumetric heatmap
(below) [24]); (c) orientation map PAF [41]; (d) hierarchical bone representation [50]; (e) cylinder model (blue) and ellipBody
(pink); and (f) skeleton-driven skinned multi-person linear model (SMPL) [29].
further challenge the power of algorithms. In this section, we
introduce the representative approaches according to the above
categories and summarize them in Table I. Additionally, we
also give a brief introduction to the related tasks which use
2D pose estimation, such as person re-identiﬁcation, action
recognition, human-object interaction, human parsing, etc.
A. Single Person Pose Estimation
As shown in Fig. 3, the framework of typical single person
pose estimation methods can be formulated as consisting of a
pose encoder followed by a pose decoder. The pose encoder is a
backbone to extract high-level features, while the pose decoder
yields the 2D locations of keypoints in the regression-based
manner or detection-based manner.
Most of the pose encoders are based on image classiﬁcation
networks, such as ResNet [125], with a pre-trained model on
a large-scale dataset such as ImageNet. Instead, few work
designs the task-speciﬁc pose encoders. For example, the
stacked hourglass network [14] exploits the skip connection
layer to connect the mirror features with the same resolution.
Furthermore, PoseNAS [126] exploits the Neural Architecture
Search [127] to ﬁnd that the task-driven searchable feature
extractor blocks. It directly searches a data-oriented pose
encoder with stacked searchable cells, which can provide an
optimum feature extractor for the pose speciﬁc task.
Most of the recent works focus on the design of pose decoder,
which pays more and more attention to explore the context
information and the inherent characteristics of body structure.
Toshev et al. [13] propose DeepPose, which is one of the ﬁrst
human pose estimation methods based on deep convolutional
neural networks (DCNNs). With a cascade of DCNN-based
pose predictors, DeepPose formulates the keypoint estimation
as a regression problem. It is different from previous traditional
methods like manually designed graphical models [128], [129]
and part detectors [130]–[132]. Iterative Error Feedback (IEF)
network [69] exploits a self-correcting regression model. It is a
kind of top-down feedback to progressively change the initial
keypoint predictions. Sun et al. [52] introduce the compositional
pose regression, which is body structure-aware. The method
in [24] solves the regression-based keypoint prediction along
with human action recognition in the multi-task manner.
Since the regression-based method directly maps the image
to the coordinates of body joints, it is a non-linear problem and
may fail for complex poses. Instead, the detection-based pose
decoder generates heatmaps of keypoints instead of direct
regression [45]. As the detection-based pose decoders are
widely used in many existed methods, we will introduce them
according to their design categories as following.
Structural Body Model. Along with the DCNN-based
feature representation for the whole body, graphical models
are explored to describe the structural and local parts with the
spatial relationship, as illustrated in Fig. 6 (a). Tompson et
al. [45] propose the convolutional network Part-Detector via
a hybrid DCNN architecture. They formulate the distribution
of spatial locations for body parts as an Markov Random
Field-like model, which helps to remove the anatomically
incorrect pose predictions. Similarly, Chen et al. [58] use
DCNNs to learn conditional probabilities for the presence of
body parts and their spatial relationships within image patches.
Different from those works that learn pair-wise relationship
from the predicted score maps, Chu et al. [46] ﬁrst investigate
the relationship among parts at the feature level. The proposed
end-to-end learning framework captures structural information
among body joints by the learnable geometrical transform
kernels and a bi-directional tree-structured model. Other than
relying on any assumptions about the conditional distributions
of joints, Gkioxari et al. [60] propose a chained sequence-
to-sequence model to sequentially predict each body part
based on all previously predicted body parts. Besides, to
avoid biologically implausible pose predictions, the work
in [61] proposes a structure-aware network to implicitly exploit
geometric constraint priors of the human body. It designs
discriminators to distinguish the real poses from the fake
ones by the conditional Generative Adversarial Networks
(GANs). To further learn the compositionality of human body,
Tang et al. [47] propose the deeply learned compositional
model (DLCM) that has the bottom-up/top-down inference
stages across multiple semantic levels. In the bottom-up stage,
the higher-level parts are recursively estimated from their
children, while in the top-down stage, the lower-level parts
are recursively reﬁned by their parents. Different from the
previous approaches that use fully shared features for all body
parts, Tang et al. [133] proposes to learn speciﬁc features for

JOURNAL OF LATEX CLASS FILES„ VOL. X, NO. X, 2021
6
TABLE I: Representative deep learning-based methods for monocular 2D pose estimation.
Image/
Single/
Main idea
Methods
Video
Multiple
Image
Single
Structural Body Model
• Spatial relationships of adjacent joints [45], [58], [59];
• Bi-directional tree-structured model [46];
• Chain model [60];
• GAN-based pose discriminator [61];
• Human body compositional model [47];
• Structured representation by GNN [62];
• Occlusion relational graphical model [63].
Multi-stage Pipeline
• Stacked hourglass [14] and its variants [64]–[66];
• CPM [48] with intermediate input and supervision.
Pose Reﬁnement
• Multi-model fusion [67] and Hybrid-Pose [68];
• Iterative update model [69], [70];
• Voting scheme [71];
• Coarse-to-ﬁne hierarchical network [72] and HCRN [49];
• Data-driven augmentation [73], [74].
Multi-task Learning
• Jointly 2D and 3D pose estimation [24];
• Human parsing guided [75];
• Jointly train augmentation and pose estimation [76].
Efﬁciency Improvement
• Multi-resolution and low computational cost [77];
• Binarized neural network [78];
• Hierarchical multi-scale residual architecture [78];
• Hourglass using MobileNet [79];
• Pose distillation [80].
Multiple
Top-down
• Single stage model [16], [81];
• Multi-task (Whole body pose ZoomNet [82], Mask-RCNN [23],
pose and parsing together [83], [84],and [85]);
• Multi-stage/branch fusion (CPN [86], MSPN [87], RSN [88],HRNet [18], Graph-PCNN [89]);
• Complex case (RMPE [90], CrowdPose [91], OASNet [92], ASDA [93]).
Bottom-up
• Integer linear program for joint grouping (DeepCut [15], DeeperCut [94]);
• Part Afﬁnity Fields for joint grouping (OpenPose [41], PifPaf [95],
whole body OpenPose [96], and [97]);
• Associative embedding for joint grouping [42] and HigherHRNet [98];
• Pose Partition Network [99], [100];
• Multi-task (MultiPoseNet [27] and PersonLab [26]).
Video
Single
Temporal clues
• Insert multiple frames into channel layer [101];
• Along with action recognition ( [102] and [103]);
• Optical ﬂow-based model (Thin-Slicing [104] and [105], [106]);
• Sequence model (Chained Model [60], LSTM Pose Machine [107], UniPose-LSTM [108]);
• Dynamic Kernel Distillation [109].
Multiple
Top-down
• Clip-based spatio-temporal model (Detect-and-Track [110] and [111]);
• Optical ﬂow-based FlowTrack [16] and PoseFlow [112];
• Transformer-based keypoint tracker KeyTrack [113];
• Recovering missing detection (PGPT [114] and [115]);
• Learnable similarity metric (POINet [116] and [117].
Bottom-up
• Graph partitioning-based model [118], [119];
• Temporal Flow Fields-based model [120]–[123];
• Spatio-temporal associative embedding model KE-SIE [124].
related parts. Moreover, instead of the manually deﬁned body
structure relation, they propose a data-driven approach to group
related parts based on the amount of information they shared.
Additionally, to deal with occlusion, ORGM [63] proposes
an occlusion relational graphical model to represent the self-
occlusion and object-person occlusion simultaneously, which
discriminatively encodes the interactions between human body
parts and objects.
Multi-stage Pipeline. It has been shown that multi-stage
pipeline and multi-level feature fusion (illustrated in Fig. 6 (c))
are useful for capturing the details of the human body. One of
the representative work is the stacked hourglass network [14],
as shown in Fig. 6 (b). Each hourglass network consists of a
symmetric distribution between bottom-up processing (from
high resolutions to low resolutions) and top-down processing
(from low resolutions to high resolutions). It uses a single
pipeline with skip layers to preserve spatial information at each
resolution. In conjunction with the intermediate supervision, the
whole network consecutively stacks multiple hourglass modules
together. It has been a solid baseline for its variants [64]–
[66] with various network design optimization. Among them,
Yang et al. [64] propose to insert the designed pyramid
residual modules into the hourglass network, which can handle
scale changes among human body parts. The work in [65]
designs the Hourglass Residual Units (HRUs) to increase the
receptive ﬁeld of the stacked hourglass network. Meanwhile, a
multi-context attention mechanism is exploited to enable the
representation of different granularity from local regions to

JOURNAL OF LATEX CLASS FILES„ VOL. X, NO. X, 2021
7
Fig. 6: Illustration of six widely used paradigms for 2D single person pose estimation.
global semantic consistent spaces. To exploit the structural in-
formation and multiple resolution features, the method in [134]
exploits the multi-scale supervision, multi-scare regression,
and structure-aware loss on the stacked hourglass framework.
Besides stacked hourglass, another well-known multi-stage
network Convolutional Pose Machine (CPM) [48] uses the
intermediate input and supervision to learn implicit spatial
models without an explicit graphical model. Its sequential
multi-stage convolutional architectures increasingly reﬁne the
prediction for keypoint locations.
Pose Reﬁnement. Reﬁnement for the network outputs can
improve the ﬁnal pose estimation performance. Fig. 6 (d)
shows the framework of the common coarse-to-ﬁne reﬁnement
pipeline. Ouyang et al. [67] build a multi-source deep model
to extract non-linear representation from different information
sources, including visual appearance score, appearance mixture
type and deformation. Th representations of all information
sources are fused for pose estimation. It can be viewed as the
post-processing of pose estimation results. The work in [69]
uses an iterative update module to progressively make an
incremental improvement to the pose estimation. Belagiannis
et al. [70] introduce a recurrent convolutional neural network
to iteratively improve the performance. Lifshitz et al. [71]
propose a voting scheme for optimal pose conﬁguration where
each pixel in the image votes for the optimal position of each
keypoint. Besides, there are some methods that use multi-branch
networks for pose reﬁnement. Huang et al. [72] present a coarse-
ﬁne hierarchical network consisting of multiple branches. With
multi-level supervision for the multi-resolution feature maps,
multiple branches are uniﬁed to predict the ﬁnal keypoints.
HCRN [49] is a hierarchical contextual reﬁnement network
in which keypoints of different complexities are processed
at different layers. HCRN is in a single-stage pipeline by
exploiting the contextual reﬁnement unit to transfer informative
context from easy joints to difﬁcult ones. Hybrid-Pose [68]
adopts a two-branch Stacked Hourglass Networks, a Reﬁnement
Network (RNet) for pose reﬁnement, and a Correction Network
(CNet) for pose correction. RNet reﬁnes the keypoint locations
in each hourglass stage horizontally. CNet guides the reﬁnement
and fuses the heatmaps in a hybrid manner.
Different from adding an extra network to the ahead coarse
network for end-to-end training, the works in [73] and [74]
apply a similar reﬁnement strategy to take both the RGB images
and the coarse predicted keypoints as input. Then the reﬁnement
network directly predicts a reﬁned pose by jointly reasoning the
input-output space. This kind of separate reﬁnement network
employs a data-driven augmentation for training and can be
applied to any existing method.
Multi-task Learning. As shown in Fig. 6 (e), by exploiting
complementary information from the related tasks, multi-
task learning can provide extra cues for pose estimation. For
example, Luvizon et al. [24] propose a multi-task framework
for jointly 2D/3D pose estimation and human action recognition
from video sequences. The method in [75] uses a human part
parsing learner to exploit the part segmentation information and
provide complementary features to assist pose estimation. The
adversarial data augmentation is exploited in [76] to address the
limitation of random data augmentation during network training.
It also designs a reward/penalty strategy for jointly training the
augmentation network and the target (pose estimation) network.
Improving Efﬁciency. Along with the development of
model performance, how to improve the speed of a model has
also attracted lots of attention. Fig. 6 (f) shows the commonly
used framework for improving model efﬁciency, including using
light-weight operator, network binarization, model distillation,
etc. RaﬁLGK et al. [77] propose a multi-resolution light-weight
network that explores low computational requirements. The
Binarized neural network is ﬁrst exploited in [78] to design a
light-weight network with limited computational resources.
Speciﬁcally, based on an exhaustive evaluation of various
design choices, a hierarchical, parallel, and multi-scale residual
architecture is proposed. The method in [79] investigates the
combination of MobileNets and the hourglass network to
design a light-weight architecture. In addition, the work in [80]

JOURNAL OF LATEX CLASS FILES„ VOL. X, NO. X, 2021
8
presents a pose distillation (FPD) model that trains a high-speed
pose network based on the idea of knowledge distillation.
B. Multi-person Pose Estimation
Multi-person pose estimation needs to detect and locate the
keypoints of all persons in an image, where the number of
persons is unknown in advance. According to the processing
paradigm, the representative methods can be sorted into two
categories, i.e., top-down methods and bottom-up methods. The
former is a two-stage pipeline that ﬁrstly detects all persons
in an input image, then detects keypoints of each person in
the detected bounding box. Differently, the bottom-up pipeline
predicts all keypoints at once, then assigns these keypoints to
different persons. We will introduce the representative CNN-
based methods of these two categories.
1) Top-down Methods: This kind of methods ﬁrstly detect
and crop each person in the image. Then given a cropped image
patch that only contains a single person, they use single-person
pose estimation models followed by post-processing, such as
pose Non-Maximum-Suppression (NMS) [81], to predict the
ﬁnal keypoint outputs of each person. Theoretically, the single
person methods introduced in Section III.A can be applied after
cropping the image patch. However, compared with the single
person case, multi-person scenes have to deal with truncation,
environmental occlusion, person-person occlusion, and small
targets. Therefore, the representative top-down methods not
only focus on designing networks by digging the potential of
CNN and exploring rich context information fusion or exchange,
but also pay attention to complex scenes.
Two Stage Pipeline. Papandreou et al. [81] propose one
of the ﬁrst deep learning-based two-stage top-down pipeline,
named G-RMI, which achieves the state-of-art results on the
challenging COCO 2016 keypoints task. They use the Faster
RCNN detector to detect each person, then exploit a fully
convolutional ResNet [125] to jointly predict the keypoint’s
dense heatmaps and offsets. They also introduce the keypoint-
based NMS instead of the box-level NMS to improve the
keypoint conﬁdence. Furthermore, as in Fig. 7 (a), Xiao et
al. [16] provide a simple and effective model that consists of
a ResNet backbone and three deconvolution layers to increase
the spatial resolution. It shows that a well-designed simple
top-down model can achieve surprisingly effective.
Multi-task Learning. By sharing features between related
tasks of pose estimation, multi-task learning can provide better
feature representations for pose estimation. For example, Mask-
RCNN [23] can detect person bounding boxes, then crops the
feature map of the corresponding proposal to predict human
keypoints. Since human keypoints and human semantic parts
are related and complementary, many works [83]–[85] design
multi-task networks to jointly predict the keypoints and segment
the semantic parts. Besides, ZoomNet [82] uniﬁes the human
body pose estimator, hand/face detectors, and hand/face pose
estimators into a single network. The network ﬁrst localizes
the body keypoints, then zooming in the hands/face regions to
predict those keypoints with higher resolutions. It can handle
the scale variance among different human parts. by Moreover,
to deal with the lack of the whole-body data, the COCO-
WholeBody dataset is proposed by extending the COCO dataset
with whole-body annotations.
Multi-stage or Multi-branch Fusion. Multi-stage or multi-
branch fusion strategy is developed to break the bottleneck
of a single model. The work in [17] proposes a Cascade
Pyramid Network (CPN), as shown in Fig. 7 (b), which consists
of a global network and a reﬁning network to progressively
reﬁne the keypoint prediction. It also proposes an online hard
keypoints mining (OHKM) loss to deal with hard keypoints.
CPN achieves the 1st place in the COCO 2017 keypoint
challenge. The work in [86] improves CPN by introducing the
channel shufﬂe module and the spatial channel-wise attention
residual bottleneck to boost the original model. MSPN [87],
the winner of the COCO 2018 keypoint challenge, extends
CPN in the multi-stage pipeline. It uses the global network
of CPN as each single-stage module, fuses features from
different stages by the cross-stage feature aggregation, and
supervises the whole network via the coarse-to-ﬁne loss
functions. HRNet [18], shown in Fig. 7 (c), points out that the
high-resolution representation is important for hard keypiont
detection. HRNet maintains the high-resolution representations
through the whole network, and gradually adds high-to-low
resolution sub-networks to form multi-resolution features. It
has been a solid and superior model for pose estimation and
many other computer vision tasks. Furthermore, to consider
the keypoints’ relationship and reﬁne the rough predictions,
Graph-PCNN [89] proposes a graph pose reﬁnement module
Via a model-agnostic two-stage framework. The work [88]
of the 1st place of COCO Keypoint Challenge 2019 utilizes
a multi-stage pipeline with Residual Steps Network (RSN)
modules to aggregate intra-level features. With the delicate local
representations obtained from RSN, a Pose Reﬁne Machine
(PRM) module is proposed to further balance the local/global
representations and reﬁne the output keypoints. The resulting
architecture establishes the new state of the art on the COCO
dataset and MPII dataset.
Dealing with Complex Scenes: In real-world applications,
crowded, occlusion, and truncation scenes are unavoidable. To
remove the effect of inaccurate person detection, RMPE [90]
designs a symmetric spatial transformer network to detect every
person, a parametric pose NMS to ﬁlter out the redundant pose,
and a pose-guided human proposal generator to enhance the
network capacity for multi-person pose estimation. To tackle
the problem in crowded scenes, Li et al. [91] ﬁrstly obtain
joint candidates in each cropped bounding box, then solve the
joint association problem in a graph model. They also collect
a crowded human pose estimation dataset named CrowdPose,
and deﬁne the Crowd Index to measure the crowding level of
an image. The work in [135] investigates the problem of pose
estimation in crowded and occlusion surveillance scenes. It
proposes to add an extra network branch to detect occluded
keypoints. Besides, OASNet [92] exploits the Siamese network
with an attention mechanism to remove the occlusion-aware
ambiguities and reconstruct the occlusion-free features. To
enlarge the training set for challenging cases, Bin et al. [93]
propose to augment images by combing segmented body parts
to simulate challenging examples. A generative network is
utilized to dynamically adjust the augmentation parameters and

JOURNAL OF LATEX CLASS FILES„ VOL. X, NO. X, 2021
9
Fig. 7: Three representative top-down 2D multi-person pose estimation networks: (a) Simple BaseLine [16]; (b) CPN [17]; and
(c) HRNet [18