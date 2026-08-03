# Deep Learning-Based Human Pose Estimation: A Survey

> 2020 · id: W3117784098 · arXiv: 2012.13392 · pdf: https://arxiv.org/pdf/2012.13392 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
Human pose estimation (HPE), which has been extensively studied in computer vision literature,
involves estimating the configuration of human body parts from input data captured by sensors, in
particular images and videos. HPE provides geometric and motion information about the human
body which has been applied to a wide range of applications (e.g., human-computer interaction,
motion analysis, augmented reality (AR), virtual reality (VR), healthcare, etc.). With the rapid
development of deep learning solutions in recent years, such solutions have been shown to out-
perform classical computer vision methods in various tasks including image classification [105],
semantic segmentation [142], and object detection [204]. Significant progress and remarkable per-
formance have already been made by employing deep learning techniques in HPE tasks. However,
challenges such as occlusion, insufficient training data, and depth ambiguity still pose difficulties
to be overcome. 2D HPE from images and videos with 2D pose annotations is easily achievable
and high performance has been reached for the human pose estimation of a single person using
deep learning techniques. More recently, attention has been paid to highly occluded multi-person
HPE in complex scenes. In contrast, for 3D HPE, obtaining accurate 3D pose annotations is much
more difficult than its 2D counterpart. Motion capture systems can collect 3D pose annotation in
controlled lab environments; however, they have limitations for in-the-wild environments. For
3D HPE from monocular RGB images and videos, the main challenge is depth ambiguities. In a
multi-view setting, viewpoints association is the key issue that needs to be addressed. Some works
have utilized sensors such as depth sensors, inertial measurement units (IMUs), and radio frequency
devices, but these approaches are usually not cost-effective and require special-purpose hardware.
Given the rapid progress in HPE research, this article attempts to track recent advances and
summarize their achievements in order to provide a clear picture of current research on deep
learning-based 2D and 3D HPE.
Fig. 1. Taxonomy of this survey.
1.1
Previous surveys and our contributions
There are several related surveys and reviews previously reported on HPE. Among them, [85, 163,
164, 194] focus on the general field of visual-based human motion capture including pose estimation,
tracking, and action recognition. Therefore, pose estimation is only one of the topics covered in
J. ACM, Vol. 37, No. 4, Article 111. Publication date: August 2018.

Deep Learning-Based Human Pose Estimation: A Survey
111:3
these surveys. The research works on 3D HPE before 2012 are reviewed in [67]. The body parts
parsing-based methods for single-view and multi-view HPE are reported in [141]. These surveys
published during 2001-2015 mainly focused on conventional methods without deep learning. A
survey on both traditional and deep learning-based methods related to HPE is presented in [60].
However, only a handful of deep learning-based approaches are included. The survey in [214]
covers 3D HPE methods with RGB inputs, while the survey in [169] only reviews 2D HPE methods.
Monocular HPE from the classical to recent deep learning-based methods (till 2019, less than 100
papers) is summarized in [28]. However, it only covers 2D HPE and 3D single-view HPE from
monocular cameras. 3D multi-view HPE from monocular cameras and 3D HPE from other sensors
are ignored. Also, no extensive performance comparisons or in-depth analyses are given, and the
discussion on existing challenges and future directions is relatively short.
This survey aims to address the shortcomings of the previous surveys in terms of providing a
systematic review of the recent deep learning-based solutions to 2D and 3D HPE but also covering other
aspects of HPE including the performance evaluation of (2D and 3D) HPE methods on popular datasets,
their applications, and comprehensive discussion. The key points that distinguish this survey from the
previous ones are as follows:
• A comprehensive review of recent deep learning-based 2D and 3D HPE methods (up to 2022 with
more than 260 papers) is provided by categorizing them according to 2D or 3D scenarios, single-
view or multi-view, from monocular images/videos or other sources, and learning paradigm.
• Extensive performance evaluation of 2D and 3D HPE methods. We summarize and compare
reported performances of promising methods on common datasets based on their categories.
The comparison of results provides cues for the strengths and weaknesses of different methods,
revealing the research trends and future directions of HPE.
• An overview of a wide range of HPE applications, such as surveillance, AR/VR, and healthcare.
• An thorough discussion of 2D and 3D HPE is presented in terms of key challenges in HPE
pointing to potential future research toward improving performance.
1.2
Paper organization
HPE is divided into two main categories: 2D HPE (§ 2) and 3D HPE (§ 3). Fig. 1 shows the taxonomy
of deep learning methods for HPE. According to the number of people, 2D HPE methods are
categorized into single-person and multi-person settings. For single-person methods (§ 2.1), there
are two categories: regression methods and heatmap-based methods. For multi-person methods
(§ 2.2), there are also two types of methods: top-down methods and bottom-up methods.
3D HPE methods are classified according to the input source types: monocular RGB images and
videos (§ 3.1), or other sensors (e.g., inertial measurement unit sensors, § 3.2). The majority of
these methods use monocular RGB images and videos, and they are further divided into single-
view single-person (§ 3.1.1); single-view multi-person (§ 3.1.2); and multi-view methods (§ 3.1.3).
Multi-view settings are deployed mainly for multi-person pose estimation. Hence, single-person or
multi-person is not specified in this category.
Next, depending on the 2D and 3D HPE pipelines, the datasets and evaluation metrics commonly
used are summarized followed by a comparison of results of the promising methods (§ 4). In addition,
various applications of HPE such as AR/VR are mentioned (§ 5). Finally, the paper ends by an
thorough discussion of some promising directions for future research (§ 6).
2
2D HUMAN POSE ESTIMATION
2D HPE methods estimate the 2D position or spatial location of human body keypoints from images
or videos. Traditional 2D HPE methods adopt different hand-crafted feature extraction techniques
for body parts, and these early works describe the human body as a stick figure to obtain global
J. ACM, Vol. 37, No. 4, Article 111. Publication date: August 2018.

111:4
Zheng and Wu, et al.
pose structures. Recently, deep learning-based approaches have achieved a major breakthrough in
HPE by improving the results significantly. In the following, we review deep learning-based 2D
HPE methods with respect to single-person and multi-person scenarios.
2.1
2D single-person pose estimation
2D single-person pose estimation is used to localize human body joint positions when the input is
a single-person image. If there are several people, the input image is cropped first so that there is
only one person in each cropped patch (or sub-image). This process can be achieved automatically
by an upper-body detector [161] or a full-body detector [204]. In general, there are two categories
for single-person pipelines that employ deep learning techniques: regression methods and heatmap-
based methods. Regression methods apply an end-to-end framework to learn a mapping from
the input image to the positions of body joints or parameters of human body models [233]. The
goal of heatmap-based methods is to predict approximate locations of body parts and joints [26]
[171], which are supervised by heatmaps representation [231, 254]. Heatmap-based frameworks
are now widely used in 2D HPE tasks. The general frameworks of 2D single-person HPE methods
are depicted in Fig. 2.
2D HPE 
Network
Input image
GT 2D pose
(b) Heatmap-based Methods
Deep Learning-
based Pose 
Regressor
Input image
2D pose
Keypoints (coordinates)
(a) Regression Methods
Gaussian 
kernel on 
each joint
...
GT 
heatmap
of right 
ankle
GT 
heatmap
of left 
shoulder
...
estimated 
heatmap
of right 
ankle
estimated 
heatmap
of left 
shoulder
Loss
Fig. 2. Single-person 2D HPE frameworks. (a) Regression methods directly learn a mapping (via a deep neural network)
from the original image to the kinematic body model and produce joint coordinates. (b) Given the ground-truth 2D pose,
the ground-truth heatmaps of each joint are generated by applying a Gaussian kernel to each joint’s location. Then,
heatmap-based methods utilize a model to predict the heatmap of each joint.
2.1.1
Regression methods. There are many works based on the regression framework (e.g.,
[17, 54, 116, 118, 147, 148, 153, 154, 172, 178, 190, 222, 233, 284]) to predict joint coordinates from
images as shown in Fig. 2 (a). 

## method
Regressor
(a) Skeleton-only Methods - Direct Estimation Approaches    
(b) Skeleton-only Methods - 2D to 3D Lifting Approaches    
(c) Human mesh recovery Methods (Volumetric Model)
End to End
Network
Input image
Output 3D pose
Off-the-Shelf
2D HPE 
Network
3D Pose
Network
Input image
2D pose
Output 3D pose
Fig. 4. Single-person 3D HPE frameworks. (a) Direct estimation approaches directly estimate the 3D human pose from 2D
images. (b) 2D to 3D lifting approaches leverage the predicted 2D human pose (intermediate representation) for 3D pose
estimation. (c) Human mesh recovery methods incorporate parametric body models to recover a high-quality 3D human
mesh. The 3D pose and shape parameters inferred by the 3D pose and shape network are fed into the model regressor to
reconstruct 3D human mesh. Part of the figure is from [5].
Direct estimation: As shown in Fig. 4(a), direct estimation methods infer the 3D human pose from
2D images without intermediately estimating 2D pose representation, e.g., [119, 183, 184, 222, 225].
Li and Chan [117] employed a shallow network to train the body part detector with sliding windows
and the pose coordinate regression synchronously. Sun et al. [222] proposed a structure-aware
regression approach. Instead of using a joint-based representation, they adopted a bone-based
representation with more stability. A compositional loss was defined by exploiting the 3D bone
structure with bone-based representation that encodes long-range interactions between the bones.
Pavlakos et al. [183, 184] introduced a volumetric representation to convert the highly non-linear
3D coordinate regression problem to a manageable form in a discretized space. The voxel likelihoods
for each joint in the volume were predicted by a convolutional network. Ordinal depth relations of
human joints were used to alleviate the need for accurate 3D ground truth poses.
2D to 3D lifting: Motivated by the recent success of 2D HPE, 2D to 3D lifting approaches that
infer 3D human pose from the intermediately estimated 2D human pose have become a popular 3D
HPE solution as illustrated in Fig. 4 (b). In the first stage, off-the-shelf 2D HPE models are employed
to estimate 2D pose. Then 2D to 3D lifting is used to obtain 3D pose in the second stage, e.g.,
[18, 111, 156, 168, 226, 307]. Benefiting from the excellent performance of state-of-the-art 2D pose
detectors, 2D to 3D lifting approaches generally outperform direct estimation approaches. Martinez
et al. [156] proposed a fully connected residual network to regress 3D joint locations based on the
J. ACM, Vol. 37, No. 4, Article 111. Publication date: August 2018.

111:12
Zheng and Wu, et al.
2D joint locations. Despite achieving state-of-the-art results at that time, the method could fail due
to reconstruction ambiguity of over-reliance on the 2D pose detector. Tekin et al. [226] and Zhou et
al. [307] adopted 2D heatmaps instead of 2D pose as intermediate representations for estimating 3D
pose. Wang et al. [250] developed a pairwise ranking CNN to predict the depth ranking of pairwise
human joints. Then, a coarse-to-fine pose estimator was used to regress the 3D pose from 2D joints
and the depth ranking matrix. Jahangiri and Yuille [80], Sharma et al. [215], and Li and Lee [111]
first generated multiple diverse 3D pose hypotheses then applied ranking networks to select the
best 3D pose.
Given that a human pose can be represented as a graph where the joints are the nodes and the
bones are the edges, Graph Convolutional Networks (GCNs) have been applied to the 2D-to-3D
pose lifting problem by showing promising performance [38, 42, 137, 281, 296]. Ci et al. [42] proposed
a Locally Connected Network (LCN), which leverages both a fully connected network and GCN to
encode the relationship between local joint neighborhoods. LCN can overcome the limitations of
GCN that the weight-sharing scheme harms the pose estimation model’s representation ability,
and the structure matrix lacks the flexibility to support customized node dependence. Zhao et al.
[296] also tackled the limitation of the shared weight matrix of convolution filters for all the nodes
in GCN. A Semantic-GCN was proposed to investigate the semantic information and relationship,
which is not explicitly represented in the graph. The semantic graph convolution (SemGConv)
operation is used to learn channel-wise weights for edges. Both local and global relationships
among nodes are captured since SemGConv and non-local layers are interleaved. Zhou et al. [316]
further introduced a novel modulated GCN network which consists of weight modulation and
affinity modulation. The weight modulation exploits different modulation vectors for different
nodes that disentangles the feature transformations. The affinity modulation explores additional
joint correlations beyond the defined human skeleton.
The kinematic model is an articulated body representation by connected bones and joints
with kinematic constraints, which has gained increasing attention in 3D HPE in recent years. Many
methods leverage prior knowledge based on the kinematic model such as skeletal joint connectivity
information, joint rotation properties, and fixed bone-length ratios for plausible pose estimation,
e.g., [58, 108, 160, 173, 174, 245, 263, 309]. Zhou et al. [309] embedded a kinematic model into a
network as kinematic layers to enforce the orientation and rotation constraints. Nie et al. [173] and
Lee et al. [110] employed a skeleton-LSTM network to leverage joint relations and connectivity.
Observing that human body parts have a distinct degree of freedom (DOF) based on the kinematic
structure, Wang et al. [245] and Nie et al. [174] proposed bidirectional networks to model the
kinematic and geometric dependencies of the human skeleton. Kundu et al. [108] [107] designed a
kinematic structure preservation approach by inferring local-kinematic parameters with energy-
based loss and explored 2D part segments based on the parent-relative local limb kinematic model.
Xu et al. [263] demonstrated that noise in the 2D joint is one of the key obstacles for accurate 3D
pose estimation. Hence a 2D pose correction module was employed to refine unreliable 2D joints
based on the kinematic structure. Zanfir et al. [278] introduced a kinematic latent normalizing flow
representation (a sequence of invertible transformations applied to the original distribution) with
differentiable semantic body part alignment loss functions.
3D HPE datasets are usually collected from controlled environments with selected common
motions. It is difficult to obtain accurate 3D pose annotations for in-the-wild data. Thus 3D HPE for
in-the-wild data with unusual poses and occlusions is still a challenge. To this end, a group
of 2D to 3D lifting methods estimate the 3D human pose from in-the-wild images without 3D
pose annotations such as [19, 64, 242, 271, 308]. Zhou et al. [308] proposed a weakly supervised
transfer learning method that uses 2D annotations of in-the-wild images as weak labels. A 3D pose
estimation module was connected with intermediate layers of the 2D pose estimation module. For
J. ACM, Vol. 37, No. 4, Article 111. Publication date: August 2018.

Deep Learning-Based Human Pose Estimation: A Survey
111:13
in-the-wild images, 2D pose estimation module performed a supervised 2D heatmap regression and
a 3D bone length constraint-induced loss was applied in the weakly supervised 3D pose estimation
module. Habibie et al. [64] tailored a projection loss to refine the 3D human pose without 3D
annotation. A 3D-2D projection module was designed to estimate the 2D body joint locations with
the predicted 3D pose from the earlier network layer. The projection loss was used to update the
3D human pose without requiring 3D annotations. Inspired by [50], Chen et al. [19] proposed an
unsupervised lifting network based on the closure and invariance lifting properties with a geometric
self-consistency loss for the lift-reproject-lift process. Closure means for a lifted 3D skeleton, after
random rotation and re-projection, the resulting 2D skeleton will lie within the distribution of valid
2D poses. Invariance means when changing the viewpoint of 2D projection from a 3D skeleton, the
re-lifted 3D skeleton should be the same.
Instead of estimating 3D human pose from monocular images, videos can provide temporal
information to improve accuracy and robustness of 3D HPE, e.g., [13, 34, 44, 187, 227, 248, 311, 312].
Hossain and Little [201] proposed a recurrent neural network using a Long Short-Term Memory
(LSTM) unit with shortcut connections to exploit temporal information from sequences of human
pose. Their method exploits the past events in a sequence-to-sequence network to predict temporally
consistent 3D pose. Noticing that the complementary property between spatial constraints and
temporal correlations is usually ignored by prior work, Dabral et al. [44], Cai et al. [13], and Li et al.
[127] exploited the spatial-te