# mm-Pose: Real-Time Human Skeletal Posture Estimation Using mmWave Radars and CNNs

> 2020 · id: W2990165697 · arXiv: 1911.09592 · pdf: https://arxiv.org/pdf/1911.09592 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
W
ITH the advent in computing resources and advanced
machine learning (ML) techniques, computer vision
(CV) has emerged as an exciting ﬁeld of research to provide
Artiﬁcal Intelligence (AI) and autonomous machines with in-
formation about the visual representation of the real world [1],
[2]. Primarily using vision based sensors, such as monocular
camera, RGBD camera or IR based sensors, and applied
machine learning, CV targets several applications, including
(but not limited to) object classiﬁcation, target tracking, trafﬁc
monitoring and autonomous vehicles [3]–[7]. In the recent
years, another interesting topic that the CV community has
been exploring is the ability to estimate human skeletal pose
by identifying and detecting speciﬁc joints and/or body parts
from still/video data. This speciﬁc area of research ﬁnds
several applications, one being primarily in the health-care
industry by automating patient monitoring systems, with the
current situation of global shortage in nursing staff [8]. Such
tracking systems would also allow for effective pedestrian
Radar
mm−Wave
Radar
mm−Wave mm−Pose
Autonomous Vehicles
carrying mmWave Radar
Detection by Autonomous Vehicles
Detection by Traffic Monitoring System
Fig. 1. mm-Pose can be used in autonomous/ semi-autonomous vehicles and
trafﬁc monitoring systems for robust skeletal posture estimation of pedestrian.
monitoring for autonomous and semi-autonomous vehicles,
and aid defense forces with behavioral information of the
adversary, to trigger appropriate preventive decision making.
While vision based sensors provide a high-resolution repre-
sentation of the scene, there are a few challenges associated
with their operation. They heavily rely on external sources for
illuminating the scene and are therefore rendered ineffective in
poor lighting conditions, adverse weather conditions or when
the scene/target is occluded. These could result in irrevocable
catastrophic events similar to the one encountered at the
Uber self-driving vehicle crash incident in Arizona due to the
vision/LiDAR sensors’ inability to detect the pedestrian in time
to avoid the accident. There is therefore an imminent need for
alternate sensors to achieve the task, while overcoming the
aforementioned challenges.
Radio Frequency (RF) based sensors, such as radars, use
its own signals to illuminate the target, therefore making it
operationally robust to scene lighting and weather conditions.
However, unlike vision based sensors, radars only represent
the scene with reﬂection point clouds rather than a true-
color image representation. Radars are therefore primarily
arXiv:1911.09592v1  [eess.SP]  21 Nov 2019

2
used for target localization applications. Furthermore, object
classiﬁcation becomes non-trivial with the point cloud data
alone, and the lack of available labeled radar data-sets for this
task makes it even more challenging.
Traditionally, radar systems have been size and cost inten-
sive primarily targeted to commercial and defense applications.
However, continuing advancement in micro-electronics fab-
rication and manufacturing techniques, including Radio Fre-
quency Integrated Circuits (RFICs), have signiﬁcantly reduced
the size and cost of electronic sensors making them more
accessible to public [9]–[11]. mmWave automotive radars are
an example of this technology. They are low-power, compact
and are extremely practical to deploy. Furthermore, mmWave
radars provides us with a high resolution point cloud rep-
resentation of the scene and have therefore emerged as one
of the primary sensors in autonomous robots on a smaller
scale, to more commercial applications such as autonomous
vehicles. Higher operating bandwidths also allow mmWave
radars to roughly generate the contour of human body without
extracting facial information, thus preserving user privacy.
In this paper, we propose mm-Pose, a novel real-time
approach to estimate and track human skeleton using mmWave
radars and convolutional neural networks (CNNs). A potential
depiction of its application in trafﬁc monitoring systems and
autonomous vehicles is shown in Fig. 1. To the best of the
authors’ knowledge, this is the ﬁrst method that uses mmWave
radar reﬂection signals to estimate the real-world position of
>15 distinct joints of a human body. mm-Pose could also
ﬁnd applications in (i) privacy-protected automated patient
monitoring systems, and (ii) aiding defense forces in a hostage
situation. Radars carrying this technology on unmanned aerial
vehicles (UAVs) could scan the building and map the live
skeletal postures of the hostage and the adversary, through
the walls, which wouldn’t have been possible otherwise with
vision sensors.
The paper is organized as follows. Section II summarizes
the current skeleton tracking work carried out in the CV com-
munity and its extension to RF sensors. A concise background
theory around the two fundamental blocks of the system,
namely (i) radar signal processing chain and (ii) machine
learning and neural networks is presented in Section III.
The detailed approach, novel data representation and system
architecture are presented in Section IV, followed by the
experimental results and discussion in Section V. Finally, the
study is summarized and concluded in Section VI.
II. LITERATURE REVIEW
It is extremely critical to accurately estimate and track
human posture in several applications, as the estimated pose
is key to infer their speciﬁc behavior. Since the last decade,
scientists have been exploring various approaches to estimating
human pose. One of the early works in 2005 was Strike a Pose,
proposed by researchers at Oxford, that would detect humans
in a speciﬁc pose by identifying 10 distinct body parts/limbs
using rectangular templates from RGB images/videos [12]. A
k-poselet based keypoint detection scheme was proposed in
2016, that uses predicted torso keypoint activations to detect
multiple persons using agglomerative clustering [13]. Another
approach was to use region-based CNN (R-CNN) to learn
N masks, to detect each of the N distinct key-points to
construct the skeleton from images, using a ResNet variant
architecture [14]. In 2016, DeeperCut, an improved multi-
person pose estimation model from DeepCut was proposed
that used a bottom up approach using a ﬁne-tuned ResNet
architecture that doubled the then estimation accuracy with
a 3 orders of magnitude reduction in run-time [15], [16].
A top-down approach to pose estimation was proposed by
Google, that ﬁrst identiﬁed regions in the image containing
people using R-CNN, and then used a fully convolutional
ResNet architectiture and aggregation to obtain the keypoint
predictions, yielding a 0.649 precision on the COCO test-dev
set [17]. Another extremely popular bottom-up approach for
human pose estimation is OpenPose, proposed by researchers
at Carneigie Mellon University in 2017 [18]. OpenPose used
Part Afﬁnity Fields (PAF), a non-parametric representation of
different body parts, and then associate them to individuals
in the scene. This real-time algorithm had great results on
the MPII dataset and also won the 2016 COCO keypoints
challenge [19]. Also, the cross-platform versatility and open-
source data-sets has led to OpenPose being used as the most
popular benchmark for generating highly accurate ground truth
data-sets for training.
While the aforementioned approaches paved the way to-
wards human pose and skeleton tracking, they were limited
to 2-D estimation on account of the images/videos being
collected from monocular cameras. While monocular cameras
provide high resolution information of the azimuth and ele-
vation of the objects, extracting depth using monocular vision
sensors is extremely challenging and non-trivial. To model a 3-
D representation of the skeletal joints, HumanEva dataset was
created by researchers at the University of Toronto [20]. The
dataset was created by using 7 synchronous video cameras
(3 RGB + 4 grayscale) in a circular array, to capture the
entire scene in its ﬁeld-of-view. The human subject was made
to perform 5 different motions, and reﬂective markers were
placed on speciﬁc joint locations to track the motion and
a ViconPeak commercial motion capture system was used
to obtain the 3-D ground truth pose of the body. Another
approach to extract 3-D skeletal joint information is by using
Microsoft Kinect [21]. The Kinect consists of an RGB and
infra-red (IR) camera that allows it to capture the scene in
3-D space. It used a per-pixel classiﬁcation approach to ﬁrst
identify the human body parts, followed by joint estimation
by ﬁnding the global centroid of the probability mass for each
identiﬁed part. However the downsides of vision based sensors
for skeletal tracking are the fact that their performance is
extensively hindered in poor lighting and occlusion. Moreover,
as previously introduced, privacy concerns restrict the use of
vision based for several applications.
Studies have previously made use of micro-doppler radar
signa

## conclusion
In this paper, mm-Pose, a real-time novel skeletal pose
estimation using mmWave radars is proposed. The 3-D XYZ
radar point cloud data (upto N2 points per CPI) is ﬁrst
projected onto the XY and XZ planes, followed by an N×N×3
RGB image, with the RGB channels corresponding to the 2-D
position and intensity information of each reﬂection point. This
data representation was aimed at eliminating a voxel based
learning approach and reducing the sparsity of the input data.
A forked-CNN based deep learning architecture was trained to
estimate the X, Y, and Z locations of 25 joints and construct a
skeletal representation. 8 outlier joints were identiﬁed that did
not aid to the learning process and were subsequently removed
from our system and further analysis, as we were able to
reasonably reconstruct the skeletal pose using the remaining 17
joints. The proposed architecture offered signiﬁcant reduction
in computational complexity compared to traditional MLP
networks and offered a much lower localization error and
variance when compared to the baseline architectures. The
average localization errors of 3.2 cm in depth (X) and 2.7
cm in elevation (Z) outperforms MIT’s RF-Pose3D by ≈24%
and ≈32%, respectively. However, the localization error of
7.5 cm in azimuth (Y) was found to be greater than the
4.9 cm offered by RF-Pose3D. The end-to-end system was
veriﬁed successfully for real-time estimation, using mmWave
radars and the proposed mm-Pose architecture on ROS. The
current implementation of mmPose was developed with the
data obtained using four different motions, however, more
motions could be added by the rather expensive process of data
collection and labeling for a wide range of spatial motions for
added robustness. Finally, mm-Pose could be used for a wide
range of applications including (but not limited to) pedestrian
tracking, real-time patient monitoring systems and through-
the-wall pose estimations for military applications.