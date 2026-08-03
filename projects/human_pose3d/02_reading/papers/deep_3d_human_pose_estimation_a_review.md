# Deep 3D human pose estimation: A review

> 2021 · id: W3165265377 · 来源: web-agent
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

UvA-DARE is a service provided by the library of the University of Amsterdam (https://dare.uva.nl)
UvA-DARE (Digital Academic Repository)
Deep 3D human pose estimation: A review
Wang, J.; Tan, S.; Zhen, X.; Xu, S.; Zheng, F.; He, Z.; Shao, L.
DOI
10.1016/j.cviu.2021.103225
Publication date
2021
Document Version
Final published version
Published in
Computer Vision and Image Understanding
License
CC BY-NC-ND
Link to publication
Citation for published version (APA):
Wang, J., Tan, S., Zhen, X., Xu, S., Zheng, F., He, Z., & Shao, L. (2021). Deep 3D human
pose estimation: A review. Computer Vision and Image Understanding, 210, Article 103225.
https://doi.org/10.1016/j.cviu.2021.103225
General rights
It is not permitted to download or to forward/distribute the text or part of it without the consent of the author(s)
and/or copyright holder(s), other than for strictly personal, individual use, unless the work is under an open
content license (like Creative Commons).
Disclaimer/Complaints regulations
If you believe that digital publication of certain material infringes any of your rights or (privacy) interests, please
let the Library know, stating your reasons. In case of a legitimate complaint, the Library will make the material
inaccessible and/or remove it from the website. Please Ask the Library: https://uba.uva.nl/en/contact, or a letter
to: Library of the University of Amsterdam, Secretariat, P.O. Box 19185, 1000 GD Amsterdam, The Netherlands.
You will be contacted as soon as possible.
Download date:21 Jul 2026

Computer Vision and Image Understanding 210 (2021) 103225
Contents lists available at ScienceDirect
Computer Vision and Image Understanding
journal homepage: www.elsevier.com/locate/cviu
Deep 3D human pose estimation: A review
Jinbao Wang a,1, Shujie Tan a,1, Xiantong Zhen b,e, Shuo Xu c, Feng Zheng a,∗, Zhenyu He d,
Ling Shao b
a Department of Computer Science and Engineering, Southern University of Science and Technology, 518055, China
b Inception Institute of Artificial Intelligence, Abu Dhabi, The United Arab Emirates
c Department of Electronics and Information Engineering, Anhui University, 230601, China
d Harbin Institute of Technology (Shenzhen), China
e AIM Lab, University of Amsterdam, The Netherlands
A R T I C L E
I N F O
Communicated by Nikos Paragios
MSC:
68-02
68T45
68U10
Keywords:
3D Human Pose Estimation
Deep Learning
A B S T R A C T
Three-dimensional (3D) human pose estimation involves estimating the articulated 3D joint locations of a
human body from an image or video. Due to its widespread applications in a great variety of areas, such
as human motion analysis, human–computer interaction, robots, 3D human pose estimation has recently
attracted increasing attention in the computer vision community, however, it is a challenging task due to
depth ambiguities and the lack of in-the-wild datasets. A large number of approaches, with many based on
deep learning, have been developed over the past decade, largely advancing the performance on existing
benchmarks. To guide future development, a comprehensive literature review is highly desired in this
area. However, existing surveys on 3D human pose estimation mainly focus on traditional methods and
a comprehensive review on deep learning based methods remains lacking in the literature. In this paper,
we provide a thorough review of existing deep learning based works for 3D pose estimation, summarize
the advantages and disadvantages of these methods and provide an in-depth understanding of this area.
Furthermore, we also explore the commonly-used benchmark datasets on which we conduct a comprehensive
study for comparison and analysis. Our study sheds light on the state of research development in 3D human
pose estimation and provides insights that can facilitate the future design of models and algorithms.
1. Introduction
Human pose estimation is generally regarded as the task of pre-
dicting the articulated joint locations of a human body from an image
or a sequence of images of that person. Due to its wide range of
potential applications, human pose estimation is a fundamental and
active research direction in the area of computer vision. Driven by
powerful deep learning techniques and recently collected large-scale
datasets, human pose estimation has continued making great progress,
especially on 2D images. However, the performance of 3D human
pose estimation remains barely satisfactory, which could be largely
due to the lack of sufficient 3D in-the-wild datasets. Recently, some
methods (Trumble et al., 2017; von Marcard et al., 2018) have been
proposed to solve this problem, and to a certain extent, these methods
have made some progress. However, there is still significant room for
improvement.
In this section, we will first introduce the vast number of potential
applications of 3D pose estimation to highlight the significance of
research in this topic, then discuss the main challenges, and finally
describe the scope of this survey in comparison to related work.
∗Corresponding author.
E-mail address: zhengf@sustech.edu.cn (F. Zheng).
1 These authors contributed equally to this work.
1.1. Applications
Since 3D pose representation provides additional depth information
compared with 2D pose representation, 3D human pose estimation
enables more widespread applications. To better understand the use of
3D human pose estimation, we provide a brief description of some of
its interesting real-world applications:
• Human–Computer Interaction. A robot can better serve and help
users if it can understand 3D poses, actions and emotions of peo-
ple. For example, a robot can take timely actions when it detects
the 3D pose of a person who is prone to fall. In addition, assistant
robots can better socially interact with human users, provided
they can perceive 3D human poses. Meanwhile, it is also very
useful for computer control, i.e. as input for productive software
packages. Moreover, people can play games using their poses and
gestures through Microsoft Kinect sensors (Zhang, 2012).
https://doi.org/10.1016/j.cviu.2021.103225
Received 12 August 2020; Received in revised form 13 May 2021; Accepted 14 May 2021
Available online 24 May 2021
1077-3142/© 2021 The Author(s). Published by Elsevier Inc. This is an open access article under the CC BY-NC-ND license
(http://creativecommons.org/licenses/by-nc-nd/4.0/).

J. Wang, S. Tan, X. Zhen et al.
Computer Vision and Image Understanding 210 (2021) 103225
Fig. 1. Illustration of the depth ambiguity (Li and Lee, 2019).
• Autonomous Driving. Self-driving cars are required to make deci-
sions to avoid collision with pedestrians, and thus understand-
ing a pedestrian’s pose, movement and intention is very impor-
tant (Kim et al., 2019; Du et al., 2019).
• Video Surveillance. Nowadays, video surveillance is of great sig-
nificance for public safety. In this area, 3D pose estimation tech-
niques could be used to assist the re-identification task (Su et al.,
2017; Xu et al., 2018; Zheng et al., 2019), which helps video
surveillance and enables supervisors to quickly find the targets
of interest.
• Biomechanics and Medication. Human pose and movement can
indicate the health status of humans. Thus, 3D pose estimation
techniques could be used to construct a sitting posture correc-
tion system to monitor the status of users. For exercise, the
system can be used to avoid injury by providing timely feedback
of correct movement poses to users. Moreover, pose estimation
systems are also able to assist doctors for remote diagnose and
tele-rehabilitation of patients (Airò Farulla et al., 2016).
• Sports Performance Analysis and Education. The automated extrac-
tion of 3D poses from videos can help further analysis of the
performance of athletes and provide immediate feedback for their
improvement (Hwang et al., 2017). Thus, human pose estimation
can be used to evaluate and educate people in various forms of
sports such as swimming (Zecha et al., 2018), Tai Chi (Scott et al.,
2017), soccer (Rematas et al., 2018).
• Psychology. 3D human body poses can also reveal the mental
states of people and the emotion can even be recognized from
poses (Noroozi et al., 2018). Scientists can utilize pose estimation
related techniques to quantify behavior for further research (Joo
et al., 2017). As a result, human pose estimation can be used for
psychology therapy of certain mental diseases such as children
autism (Marinoiu et al., 2018).
• Try-on and Fashion. Online shopping has become more and more
popular in recent years, especially for fashion clothes. Users can
see how they look like when wearing a certain piece of clothing
on the Internet in a virtual try-on system based on 3D pose
estimation (Pons-Moll et al., 2017; Han et al., 2018).
• Others. 3D pose estimation can also be used to assist other com-
puter vision tasks such as pose transfer (Li et al., 2019a), action
recognition (Luvizon et al., 2018), human parsing (Xia et al.,
2016), person image generation (Siarohin et al., 2018), anima-
tion (Weng et al., 2019), pose search (Ferrari et al., 2009).
1.2. Challenges
Recently, 3D human pose estimation has become an increasingly
popular research topic due to its widespread application. However, it
is far from being solved because of its unique challenges in contrast
Fig. 2. Illustration of the correspondence of people in different views (Dong et al.,
2019).
to 2D human pose estimation, in which the main challenges include
variations of body poses, complicated backgrounds, diverse clothing
appearance and occlusions.
3D human pose estimation faces further
challenges, including a lack of in-the-wild 3D datasets, depth ambigui-
ties, a huge demand for rich posture information (such as translations
and rotations), a large searching state space for each joint (representing
a discretized 3D space), etc. We will discuss the challenges of single 3D
human pose estimation from different inputs, multi-person 3D human
pose estimation and in-the-wild datasets.
(1) Different Inputs. Generally speaking, based on different consid-
erations, various types of inputs are used to estimate 3D pose and thus
the corresponding challenges are varied as well. Visual cues, such as
shadows and objects of known size, can be used to address ambiguities
in images. However, it is very difficult to directly capture such infor-
mation from images. When ignored, using 2D joints to recover a 3D
pose becomes an ill-defined problem. For instance, as shown in Fig. 1,
one 2D skeleton may correspond to many varied 3D poses. Actually, the
depth ambiguity could be considerably reduced by using temporal in-
formation, multi-view images, etc. First, for recovering 3D human pose
from a sequence of images, temporal information could be exploited
to reduce the depth ambiguity. At the same time, there are many
additional challenges such as background variation, camera movement,
fast motion, changes of clothing, illumination changes, which may
cause the shape and appearance of people that alter dramatically over
time. Second, when utilizing multi-view images, researchers face the
problem how to fuse information from multiple cameras. In fact, due
to the occlusion and inaccuracy estimation of 2D poses, this is not a
trivial problem that could be simply solved by triangularization from
estimated 2D poses, especially when there are few cameras in practical
scenes.
(2) Multiple Persons. Compared with single human pose estimation,
estimating 3D poses of multiple persons is more challenging. When
estimating multi-person from a monocular image, the additional chal-
lenge is the occlusion caused by nearby individuals. When estimating
3D poses of multiple persons from multiple views, the main challenges
include the larger state space, occlusions and cross-view ambiguities, as
shown in Fig. 2. Besides, most existing methods are based on two-stage
frameworks which suffer from problems in efficiency, while single-
stage methods (Nie et al., 2019) have been proposed to solve this
problem, they are far from mature.
(3) In-the-Wild Scenario. In addition, the lack of in-the-wild datasets
is a bottleneck for research on 3D pose estimation. For 2D human pose
estimation, it is feasible to construct large in-the-wild datasets (An-
driluka et al., 2014a; Lin et al., 2014a) by manually labeling the 2D
poses of humans in the image.
However, since 3D annotations are
generally acquired by marker-based vision systems, collecting a large-
scale in-the-wild dataset with 3D annotations is very resource-intensive.
2

J. Wang, S. Tan, X. Zhen et al.
Computer Vision and Image Understanding 210 (2021) 103225
Fig. 3.
Framework of this review.
Fig. 4. The numbers of 3D human pose estimation papers published in top conferences
(CVPR, ICCV and ECCV).
As is well-known, the most popularly used datasets such as HumanEva
and Human3.6M, are captured by motion capture systems under an
indoor environment. Thus, the algorithms trained on such datasets
inevitably confront a generalization challenge when they are used for
in-the-wild applications. To mitigate the problem, many methods have
explored, such as lifting 2D pose to 3D pose (Tome et al., 2017),
transferring knowledge (Zhou et al., 2017), utilizing weak supervision
signal (Chen et al., 2019a) and synthesizing in-the-wild images (Varol
et al., 2017). However, the in-the-wild performance of these methods
are still unsatisfactory compared with 2D pose estimation.
1.3. Scope of this survey
Previous surveys generally focus on traditional methods, such as pic-
torial models and exemplar-based approaches. Readers are encouraged
to read these review articles, in which more details have been provided.
A recent survey (Sarafianos et al., 2016) mainly focuses on the review
of work from 2008 to 2015. In that survey, the authors proposed a
rather complete taxonomy for 3D pose estimation and introduced a new
synthetic dataset as well. However, they mainly summarized classical
methods and only a few deep learning based methods were mentioned.
Furthermore, the rapid progress of deep learning in recent years has
greatly promoted the development of 3D human pose estimation. While
recent surveys do not cover these methods comprehensively or give a
summary from a specific perspective. For example, Chen et al. (2020)
merely provide a review of deep learning-based methods for monocular
human pose estimation.
Therefore, we follow the same reasonable taxonomy but instead
focus on deep learning based methods to reveal the current research
state of this field. Moreover, we observe that, in recent years, 3D human
pose estimation has gained increasing attention in the area of computer
vision community according to the numbers of published papers in top
computer vision conferences (CVPR,2 ICCV,3 and ECCV4), as shown
in Fig. 4 In addition, the representation of the 3D pose and datasets
are very important for human pose estimation. According to the types
of models, we classify the representations of poses to skeleton and
shape based approaches, as shown in Fig. 3. In recent years, many new
datasets have been proposed. We will discuss human pose modeling
and datasets in Section 2.
In summary, the framework of our review is shown in Fig. 3. We
cover deep learning based algorithms for estimating 3D human pose,
where the inputs ranging from a single image to a sequence of images,
from a single view to multiple views, and from a single person to
multiple persons. From the perspective of pose representation, the input
data can be divided into two types: skeleton and shape (contour).
Also, many parametric models are used to supplement the body shape,
such as SCAPE (Anguelov et al., 2005), SMPL (Loper et al., 2015),
and DensePose (Alp Güler et al., 2018). As for 3D pose estimation
of multiple people, the approaches can be classified into single-stage
methods and two-stage methods. The two-stage methods can be further
divided into top-down and bottom-up methods as shown in Fig. 3.
Specifically, the top-down methods detect each person first and then
locate their joints individually, whilst the bottom-up methods locate all
the body joints first and then assign them to the corresponding person.
In contrast, the one-stage methods (Nie et al., 2019) normally estimate
the locations of root position and joint displacements, simultaneously.
2. Human body modeling, datasets and evaluation metrics
2.1.
Human body modeling
Generally, the human body structure is very complex, and different
methods adopt different models based on their specific considerations.
Nevertheless, the most commonly used models are the skeleton and
shape models. Besides, a new pose estimation is a surface-based rep-
resentation called DensePose (Alp Güler et al., 2018), which is worth
mentioning due to the extension of the existing pose representation.
Next, we will introduce them in detail.
Skeleton-Based Model: First and foremost, the skeleton model is
commonly used in 2D human pose estimation (Cao et al., 2018) and
is naturally extended to 3D. The human skeleton model is treated as
a tree structure, which contains many keypoints of the human body
and connects natural adjacent joints using edges between key joints, as
shown in Fig. 5.
SMPL-Based Model: For the shape model, recent works use the
skinned multi-person linear (SMPL) model (Loper et al., 2015), as
shown in Fig. 6, to estimate 3D human body joints (Bogo et al., 2016).
The human skin is represented as a triangulated mesh with 6890
vertices, which is parameterized by shape and pose parameters. The
2 IEEE conference on Computer Vision and Pattern Recognition
3 IEEE International Conference on Computer Vision
4 European Conference on Computer Vision
3

J. Wang, S. Tan, X. Zhen et al.
Computer Vision and Image Understanding 210 (2021) 103225
Fig. 5. Human body skeleton from the MPI-INF-3DHP dataset, with the root joint 15,
O1 (blue): relative to first order and O2 (orange): relative to second order parents in
the kinematic skeleton hierarchy. (For interpretation of the references to color in this
figure legend, the reader is referred to the web version of this article.)
Fig. 6. The SMPL model (Loper et al., 2015). The white points are pre-defined
keypoints.
shape parameters are used to model the body proportions, height and
weight, while the pose parameters are used to model the determined
deformation of the body. The 3D pose positions can be estimated by
learning the shape and body parameters.
Surface-Based Model: Recent, a new model of the human body:
DensePose (Alp Güler et al., 2018) is recently proposed, considering
the fact that sparse correspondence of the image and keypoints is not
enough to capture the status of the human body. To address the issue, a
new dataset named DensePose-COCO is constructed, which establishes
the dense correspondences between image pixels and a surface-based
representation of the human body. This work further promotes the
development of human understanding in images and can be understood
as the next step in the line of works on extending the standard for
humans in 2D or 3D human estimation datasets, such as MPII Human
Pose (Andriluka et al., 2014b), Microsoft COCO (Lin et al., 2014b),
HumanEva (Sigal et al., 2009), Human3.6M (Ionescu et al., 2013).
2.2. Datasets
The 3D pose estimation datasets are often gathered by a motion
capture system. A previous review has analyzed the datasets from
2009 to 2015 (Sarafianos et al., 2016). The HumanEva dataset (Sigal
et al., 2009) and Human3.6M dataset (Ionescu et al., 2013) are still the
standard for 3D human pose estimation. Moreover, since there have
been many new datasets proposed recently, we will introduce these
dataset in detail in the following sections and sum up the main points
in Table 1.
HumanEva-I Sigal et al. (2010) contains 7 calibrated video se-
quences (4 grayscale and 3 color) that are synchronized with 3D body
poses obtained from a motion capture system. The database contains
4 subjects performing a 6 common actions, e.g. walking, jogging,
gesturing. The dataset contains training, validation and testing sets.
Human3.6M Ionescu et al. (2013) is one of the largest motion
capture datasets, which consists of 3.6 million human poses and cor-
responding images. The dataset provides accurate 3D human joint
positions and synchronized high-resolution videos acquired by a motion
capture system at 50 Hz. The dataset contains activities by 11 profes-
sional actors in 17 scenarios: discussion, smoking, taking photo, talking
on the phone, etc., from 4 different camera views.
MARCOnI (MARker-Less Motion Capture in Outdoor and Indoor
Scenes, Elhayek et al. (2016) is a comprehensive dataset that can be
used for versatile testing. The dataset is composed of 12 sequences
with different conditions, such as sensor modalities, numbers and types
of cameras, identities of actors, scene and motion complexities. All
cameras are synchronized, even the cell phone and the GoPro cameras.
This dataset provides 3D joint positions calculated by three reference
methods as follows. (1) MP: some sequences are recorded by a syn-
chronized Phasespace active-LED marker-based motion capture system
and the 3D joint locations could be captured by markers. (2) A3D:
the 2D poses of other sequences are annotated manually to calculate
ground truth 3D joint locations. (3) DMC: for sequences with enough
cameras, the dataset also provides 3D joint positions using a baseline
approach (Stoll et al., 2011).
MPI-INF-3DHP Mehta et al. (2017a) uses a commercial marker-
less motion capture system to collect data, which does not require
special suits or markers, and thus actors could wear everyday clothes
including loose clothes. There are 8 actors (4 females + 4 males),
each performing 8 action sets, each of which lasts about 1 min. The
test set consists of 2929 valid frames from 6 subjects performing 7
actions. The actions range from walking, sitting, and complex exercise
actions to dynamic actions. The number of action classes is more
than that of Human3.6M dataset. To increase the diversity of data,
each actor performs activities of both daily apparel and plain-colored
clothing sets. Moreover, the dataset increases the scope of foreground
and background augmentation by providing chroma-key masks for the
background.
Total Capture Trumble et al. (2017) is the first dataset that provides
both multi-viewpoint video (MVV), inertial measurement unit (IMU),
and skeleton annotations obtained by a commercial motion capture
system (Vicon). The dataset does not use any markers, so actors could
wear very loose clothes to increase the variation of appearance. The
XSens IMUS system (Roetenberg et al., 2009) uses 13 IMU sensors on
key body parts including head, upper/lower back, upper/lower limbs,
and feet. The dataset provides accurate background subtraction for
each pixel. It contains five kinds of actions, each of which is repeated
three times by actors. Finally, the dataset is split into several subsets
according to the subjects and action sequences, allowing for testing
both unseen subjects and seen subjects with unseen actions.
SURREAL (Synthetic hUmans foR REAL tasks, Varol et al. (2017))
is a large-scale synthetic dataset with randomly generated 3D poses,
shapes, textures, illustrations and backgrounds. The shape information
of the dataset was from the CMU motion capture (MoCap) dataset.
Next, the MoSh (Loper et al., 2014) method is explored to fit the
SMPL parameters using the raw data of the 3D MoCap markers. Then,
given the fitted parameters, the synthetic body is generated by the
SMPL model, and the real appearance image is mapped into the body
shape. Further, the texture information is obtained from 3D scans of the
subjects wearing normal clothing, largely increasing the authenticity
of the synthetic data. The background images are from a subset of
the LSUN dataset (Song and Xiao, 2015), which includes a total of
400 K images from the kitchen, living room, bedroom, and dining room.
The illumination variation uses the model of Spherical Harmonics
with 9 coefficients (Green, 2003). The SURREAL dataset is the first to
provide 3D pose annotation, part segmentation, and flow ground truth,
which can be used for multi-task training. The authors also generate
4

J. Wang, S. Tan, X. Zhen et al.
Computer Vision and Image Understanding 210 (2021) 103225
Table 1
3D human pose datasets.
Year
Dataset
No. of images
No. of subjects
Characteristics
2010
HumanEva-I
12 sequences
4 subjects, 6 actions
Indoor multi-view video, markerless motion capture
2013
Human3.6M
3.6M
11 (5 female + 6 male)
One of the largest motion capture dataset; Multiple views
2014
Shelf
4
Indoor multi-view video; multiple persons; each view suffers from heavy
occlusion
2014
Campus
3
Outdoor multi-view video; multiple persons
2016
MARCOnI
12 sequences
1 or 2 per sequence
Comprehensive dataset for versatile testing
2016
CMU Panoptic
1.5M
Up to 8 subjects
Captured in a studio with hundreds of cameras; large social interaction
2016
MPI-INF-3DHP
1.3M frames
8 (4 female + 4 male)
Indoor multi-view video, markerless motion capture, data augmentation
2017
MuCo-3DHP
MuPoTS-3D
8 (4 female + 4 male)
Build upon segmentation masks in MPI-INF-3DHP
2017
Total Capture
1.892M frames
5 (4 male + 1 female)
Indoor multi-view video, IMU, and vicon mocap
2017
SURREAL
6M frames
145
Rendered from 3D sequences of motion capture data (Human3.6M)
2017
Unite the people
8515 images
Improve SMPLify to semi-automated annotate dataset, annotate 31 segments on
the body and 91 landmarks
2018
JTA
500K frames
> 21
Massive simulated dataset, 500K frames with almost 10 million pose
2018
3DPW
60 sequences
7
The only promising 3D pose in the wild dataset; 24 train, 24 test, 12 validation
the predicted body part segmentation and depth maps for samples in
the Human 3.6M dataset. Finally, the dataset is divided according to
subjects: 115 subjects are used as the training sets and 30 of them are
used as the test set.
Unite the People Lassner et al. (2017) contains 5569 training
images and 1208 test images. This dataset is collected based on the
observations that the CNNs are often applied in isolated and separated
datasets, such as MPII (Andriluka et al., 2014a), LSP (Johnson and
Everingham, 2010), and are independent of 3D body estimation. To
unite the people of multiple human datasets, the authors improve the
SMPLify method to obtain high-quality 3D human body models, and
then manually sort these body models based on the quality. This semi-
automated approach makes annotations more efficient and enables
consistent labeling by reprojecting the body model to the original
image. The denser set of annotations that predict 31 segments on the
body and 91 landmark positions enable eliminating the ambiguity of
poses and shapes in a single view. Furthermore, a regression tree model
is proposed to predict poses or shapes, which is one to two orders of
magnitude faster than SMPLify. Finally, experiments show that using 91
landmarks the pose estimators can be trained with fewer data without
requiring gender or pose assumptions.
JTA (Joint Track Auto, Fabbri et al. (2018)) is a synthetic people
tracking dataset in urban scenarios with ground-truth annotations of
3D poses, of which 256 videos are used for training and 256 videos
are used for testing. These collected videos with varying illumination
conditions and viewpoints are from the highly photorealistic video
games Grand Theft Auto V developed by Rockstar North. The distance
from the camera varies from 0.1 to 100 m, resulting in heights of
subjects varying from 20 to 1100 pixels. By accessing the game render-
ing module, 14 body parts are automatically annotated in Andriluka
et al. (2014a, 2018). Besides that, some simulated challenges including
occlusion and self-occlusion are provided as well. Occlusion denotes that
the joint is occluded by objects or other pedestrians, while self-occlusion
denotes that the joint is occluded by the owner of the joint. Besides,
each person is assigned an identifier so that the dataset can also be
used for person re-identification research.
3DPW (3D Poses in the Wild, von Marcard et al. (2018)) is the first
dataset in the wild with accurate 3D poses for evaluation. It is created
by utilizing information from IMUs and a hand-held phone camera. A
3D pose estimation method named video inertial poser (VIP) is used to
integrate the images and IMU readings of all frames in video sequences.
The VIP has been validated on the Total Capture dataset, which has an
accuracy of 26 mm and is accurate enough to create the dataset for
image-based 3D pose estimation. For tracking single subjects, 17 IMUs
would be used, while 9–10 IMUs would be used to simultaneously track
up to 2 subjects. Then, the video and IMUs data are synchronized by
a clapping motion as in Pons-Moll et al. (2011). In total, the dataset
contains up to 18 clothing styles and actions such as walking in cities,
going up-stairs, having coffee, or taking the bus. Compared with Total
Capture, there are more subjects in a scene.
Shelf and Campus (Belagiannis et al., 2014) The shelf dataset has
annotated the body joints of four actors interacting with each other
using cameras 2, 3, and 4. Triangulation is performed using the three
camera views for deriving the 3D ground-truth. The actor 4 (Vasilis)
is occluded in most of the camera views and thus excluded from the
evaluation. The Campus dataset has annotated the body joints of the
main three actors performing different actions for the frames that are
observed from the first two cameras. The ground-truth for the third
camera view is the result of the triangulation (between cameras 1 and
2), and then projected to camera 3.
CMU Panoptic Joo et al. (2017) provides some examples with
large social interaction. It used 480 synchronized VGA cameras, 31
synchronized HD cameras (temporally aligned with VGA cameras), and
10 RGB-D sensors for motion capture. All of the 521 cameras are
calibrated by structure from the motion approach.
MuCo-3DHP (Multiperson Composited 3D Human Pose) is cre-
ated by leveraging segmentation masks provided in MPI-INF-3DHP
dataset (Mehta et al., 2017a). To collect this dataset, per-camera com-
posites with 1 to 4 subjects are first generated in the images randomly
selected from the MPI-INF-3DHP dataset, in which each camera has 16
sequences. The composited dataset covers many kinds of inter-person
overlaps and activities. Using a commercial multi-view marker-less
motion capture system, a new filmed multi-person test set named
MuPoTS-3D (Multiperson Pose Test Set in 3D) is collected as well. In
total, this dataset comprises 20 general real-world scenes (5 indoor and
15 outdoor) for up to three subjects with challenging elements such as
drastic illuminations and lens flares for outdoor settings.
In summary, for indoor 3D human pose estimation datasets, the
Human3.6m dataset is the most common one used in recent years,
although the HumanEva dataset is still frequently employed. Besides,
the MPI-INF-3DHP is also widely used, since it has more action classes
than Human3.6m and provides chroma-key masks for foreground and
background augmentation. As for the other three indoor datasets, the
CMU Panoptic dataset is created for large social interaction capture;
the MARCOnI dataset can be used for versatile testing since it contains
sequences with different conditions; the Total Capture dataset provides
MVV, IMU, and Vicon annotations in constrained environments. How-
ever, these three datasets are less used than the first two. To evaluate
the generalization ability of 3D human pose estimation algorithms,
several in-the-wild datasets have been proposed including SURREAL,
JTA, Unite the People, MuCo-3DHP, and 3DPW. The first two are
seldom used recently while the third is widely used by SMPL based
3D pose estimation methods. The fourth dataset can generally be used
for multi-person pose estimation. To some extent, the last dataset is a
promising in-the-wild dataset, since the annotations with high accuracy
of 26 mm are obtained from the Total Capture dataset.
5

J. Wang, S. Tan, X. Zhen et al.
Computer Vision and Image Understanding 210 (2021) 103225
2.3. Evaluation metrics
We list some of the most frequently used metrics below for reference
and detailed settings based on datasets.
MPJPE (Mean Per Joint Position Error): This metric is calculated
by
𝐸𝑀𝑃𝐽𝑃𝐸(𝑓, ) =
1
𝑁
𝑁
∑
𝑖=1
‖𝑃(𝑓)
𝐟,(𝑖) −𝑃(𝑓)
𝐠𝐭,(𝑖)‖2,
(1)
where 𝑓denotes a frame and denotes the corresponding skeleton.
𝑃(𝑓)
𝐟,(𝑖) is the estimated position of joint 𝑖and 𝑃(𝑓)
𝐠𝐭,(𝑖) is the corre-
sponding ground truth position. All joints are considered, 𝑁= 17.
Finally, the MPJPEs are averaged over all frames. Besides, we refer
to the resulting normalized metrics as NMPJPE. Since orientation is
left unchanged, this is a less constraining transformation than the more
commonly used procrustes alignment, to which we refer as PA-MPJPE.
PCP (Percentage of Correctly estimated Parts): The PCP metric
measures the percentage of correctly predicted parts (Ferrari et al.,
2008). As mentioned in Sarafianos et al. (2016), a body part is con-
sidered correct by the algorithm if:
‖‖𝑠𝑛−̂ 𝑠𝑛‖‖ + ‖‖𝑒𝑛−̂ 𝑒𝑛‖‖
2
≤𝛼‖‖𝑠𝑛−𝑒𝑛‖‖ ,
(2)
where 𝑠𝑛and 𝑒𝑛are the ground truth start and end location of part 𝑛,̂ 𝑠𝑛
and̂ 𝑒𝑛are the corresponding estimated locations, and 𝛼is a threshold
parameter.
PCK (Percentage of Correct Keypoints): It is first used in 2D pose
estimation (Yang and Ramanan, 2012). Mehta et al. (2017a) extend
PCK to the 3D space and calculate the area under the curve (AUC) when
varying the PCK threshold. A estimated joint is considered correct if its
distance to the corresponding ground truth is less than a threshold (e.g.,
150 mm). This metric is often used in the new MPI-INF-3DHP dataset.
The normalized version of PCK (NPCK) is used in Rhodin et al. (2018b),
Kocabas et al. (2019).
Bone Error, Bone Std, Illegal Angle: Sun et al. (2017) propose
corresponding metrics for their bone representation of the human body
because they argue that absolute joint location based metrics such
as MPJPE and PCK do not consider the pose’s internal structures.
The mean per bone position error (Bone Error) measures the relative
joint location accuracy. The bone length standard deviation (Bone
Std) measures the stability of bone length by computing the standard
deviation over a subject’s all testing samples. The percentage of illegal
joint angle (Illegal Angle) measures the feasibility of a joint’s rotation
angles (Akhter and Black, 2015).
MRPE (Mean of the Root Position Error): Moon et al. (2019)
propose this metric to evaluate the accuracy of the absolute location
of an estimated 3D human root:
𝑀𝑅𝑃𝐸= 1
𝑁
𝑁
∑
𝑖=1
‖‖‖𝐑(𝑖) −𝐑(𝑖)∗‖‖‖2 ,
(3)
where 𝐑and 𝐑(𝑖)∗are the ground truth and estimated locations of the
𝑖th sample respectively, and 𝑁is the number of testing samples.
HumanEva-I: Sigal et al. (2010) use 3D error (3D Error) metric to
evaluate performance on their HumanEva dataset. The 3D error is the
mean squared distance between coordinates of estimated and ground
truth pose.
Human3.6M: There are three main protocols for evaluating the
performance of 3D human pose estimation algorithms in terms with
MPJPE.
P1 (protocol #1, the standard protocol) uses 5 subjects (S1, S5, S6,
S7, S8) for training and 2 subjects (S9, S11) for testing.
P2 (protocol #2) differs from Protocol #1 in that it uses S11 for
testing while using 6 subjects (S1, S5, S6, S7, S8 and S9) for training.
The pose error is calculated after a similarity transformation (Procrustes
analysis) between the estimated pose and ground truth. The original
video is down-sampled to every 64th frame and evaluation is performed
on sequences from all 4 cameras and all trials. The error is averaged
over 14 joints.
P3 (protocol #3) splits the dataset in the same way as protocol
#1 (Bogo et al., 2016). However, the evaluation is only conducted on
sequences captured by the frontal camera (‘‘cam 3’’) in trial 1 and the
original video is not sub-sampled. The error is averaged over a subset
of 14 joints.
3.
3D human pose estimation based on a frame
This section will detailedly introduce 3D human pose estimation
methods which do not use temporal information, that is, only uses a
monocular image or multi-view images at a single time. Thanks to its
great advantages, e.g. suitable for indoor and outdoor use, it has been
widely studied recently.
3.1. 3D human pose estimation from a monocular image
Recovering a 3D human pose from a single image is appealing
due to the low requirement of the image, but it suffers from an ill-
defined problem that different 3D poses may correspond to the same
2D images. Besides, based on the setting, using temporal or multi-view
information to reduce the ambiguity cannot be achieved during the
recovering process. Therefore, significant research has been done and
several methods have been developed to solve these problems. In this
section, we will first introduce the methods and then illustrate some
representative works. Specifically, we will review methods from three
parts, namely directly predicting 3D poses from images, lifting from 2D
poses, and SMPL-based methods.
3.1.1. Direct 3D pose estimation
The most straightforward way to estimate 3D human poses is to
design an end-to-end network to predict the 3D coordinates of joints
for the poses. Methods that directly map input images to 3D body
joint positions can be categorized into two classes: detection-based
methods (Pavlakos et al., 2017a; Luvizon et al., 2018) and regression-
based methods (Li and Chan, 2014; Zhou et al., 2016a; Sun et al.,
2017; Tekin et al., 2017; Zhou et al., 2017; Luvizon et al., 2019).
It is worth noting that attempts have also been made to unify the
heatmap representation and joint regression (Sun et al., 2018). All of
these methods are summarized in Table 2–1).
Detection-based methods predict a likelihood heatmap for each
joint, and the joint’s location is determined by taking the maximum
likelihood of the heatmap. Pavlakos et al. (2017a) use a volume to
represent a 3D pose and then train a CNN to predict the voxel-wise like-
lihood for each joint in the volume, which greatly improves the direct
regression of joint coordinates. They adopt a coarse-to-fine prediction
scheme, which employs intermediate supervision and an iterative esti-
mation module to gradually increase the resolution of the supervision
volume. Luvizon et al. (2018) propose a multi-task framework to jointly
estimate 2D/3D poses and recognize actions, where 2D and 3D pose es-
timation is unified using volumetric heatmaps. However, such methods
rely on additional steps to convert heatmaps to joint positions, usually
by applying the argmax function, which is not differentiable. This
interfaces with the learning mechanism of neural networks. Besides, the
precision of predicted keypoints is proportional to that of the heat map
resolution, which lacks inherent spatial generalization. To achieve high
precision, the predicted heatmaps usually require a reasonable spatial
resolution, which quadratically increases the computational cost and
memory consumption.
Human pose estimation is essentially a regression problem that di-
rectly estimates the locations of joints relative to the root joint location.
Li and Chan (2014) design a simple but effective neural network with
two branches that simultaneously detect the root location and regress
the relative locations of other joints. To incorporate prior knowledge of
the geometric structure of the human body, Zhou et al. (2016a) intro-
duce a kinematic object model consisting of several joints and bones,
6

J. Wang, S. Tan, X. Zhen et al.
Computer Vision and Image Understanding 210 (2021) 103225
Table 2
Estimating 3D human pose from a single monocular image.
(1) Direct 3D pose
estimation
Highlight
Dataset
Metric
Code
Li and Chan (2014)
Network with two branches, one detects the root location and one
regresses the relative locatio