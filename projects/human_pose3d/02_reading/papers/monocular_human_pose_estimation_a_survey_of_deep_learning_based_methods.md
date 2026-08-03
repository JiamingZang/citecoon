# Monocular human pose estimation: A survey of deep learning-based methods

> 2020 · id: W3000322757 · arXiv: 2006.01423 · pdf: https://arxiv.org/pdf/2006.01423 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Vision-based monocular human pose estimation, as one of the most fundamental and challenging prob-
lems in computer vision, aims to obtain posture of the human body from input images or video se-
quences. The recent developments of deep learning techniques have been brought signiﬁcant progress
and remarkable breakthroughs in the ﬁeld of human pose estimation. This survey extensively reviews
the recent deep learning-based 2D and 3D human pose estimation methods published since 2014. This
paper summarizes the challenges, main frameworks, benchmark datasets, evaluation metrics, perfor-
mance comparison, and discusses some promising future research directions.
c⃝2020 Elsevier Ltd. All rights reserved.

## introduction
The human pose estimation (HPE) task, which has been
developed for decades, aims to obtain posture of the human
body from given sensor inputs. Vision-based approaches are
often used to provide such a solution by using cameras. In
recent years, with deep learning shows good performance
on many computer version tasks such as image classiﬁcation
(Krizhevsky et al., 2012), object detection (Ren et al., 2015),
semantic segmentation (Long et al., 2015), etc., HPE also
achieves rapid progress by employing deep learning technol-
ogy. The main developments include well-designed networks
with great estimation capability, richer datasets (Lin et al.,
2014; Joo et al., 2017; Mehta et al., 2017a) for feeding net-
works and more practical exploration of body models (Loper
et al., 2015; Kanazawa et al., 2018). Although there are some
existing reviews for HPE, however, there still lacks a survey to
summarize the most recent deep learning-based achievements.
This paper extensively reviews deep learning-based 2D/3D hu-
man pose estimation methods from monocular images or video
footage of humans. Algorithms relied on other sensors such
as depth (Shotton et al., 2012), infrared light source (Faessler
et al., 2014), radio frequency signal (Zhao et al., 2018), and
multi-view inputs (Rhodin et al., 2018b) are not included in this
survey.
∗∗Corresponding author
e-mail: chenyucheng@mail.nwpu.edu.cn (Yucheng Chen),
ytian@ccny.cuny.edu (Yingli Tian), myhe@nwpu.edu.cn (Mingyi He)
As one of the fundamental computer vision tasks, HPE is a
very important research ﬁeld and can be applied to many ap-
plications such as action/activity recognition (Li et al., 2017b;
Luvizon et al., 2018; Li et al., 2018b), action detection (Li et al.,
2017a), human tracking (Insafutdinov et al., 2017), Movies and
animation, Virtual reality, Human-computer interaction, Video
surveillance, Medical assistance, Self-driving, Sports motion
analysis, etc.
Movies and animation: The generation of various vivid dig-
ital characters is inseparable from the capture of human move-
ments. Cheap and accurate human motion capture system can
better promote the development of the digital entertainment in-
dustry.
Virtual reality: Virtual reality is a very promising technol-
ogy that can be applied in both education and entertainment.
Estimation of human posture can further clarify the relation be-
tween human and virtual reality world and enhance the interac-
tive experience.
Humancomputer interaction (HCI): HPE is very important
for computers and robots to better understand the identiﬁcation,
location, and action of people. With the posture of human (e.g.
gesture), computers and robots can execute instructions in an
easy way and be more intelligent.
Video surveillance: Video surveillance is one of the early ap-
plications to adopt HPE technology in tracking, action recogni-
tion, re-identiﬁcation people within a speciﬁc range.
Medical assistance: In the application of medical assistance,
HPE can provide physicians with quantitative human motion
information especially for rehabilitation training and physical
arXiv:2006.01423v1  [cs.CV]  2 Jun 2020

2
self occlusion
complex pose
various clothing
self-similar part
foreground occlusion
nearby person
various viewing angle
truncation
Flexible body 
conﬁguration
Diverse body 
appearance
Complex environment
Fig. 1. Typical challenges of HPE in monocular images or videos. Exam-
ple images are from Max Planck Institute for Informatics (MPII) dataset
(Andriluka et al., 2014).
therapy.
Self-driving:
Advanced self-driving has been developed
rapidly. With HPE, self-driving cars can respond more appro-
priately to pedestrians and oﬀer more comprehensive interac-
tion with traﬃc coordinators.
Sport motion analysis: Estimating players’ posture in sport
videos can further obtain the statistics of athletes’ indicators
(e.g. running distance, number of jumps). During training, HPE
can provide a quantitative analysis of action details. In physical
education, instructors can make more objective evaluations of
students with HPE.
Monocular human pose estimation has some unique charac-
teristics and challenges. As shown in Fig. 1, the challenges of
human pose estimation mainly fall in three aspects:
• Flexible body conﬁguration indicates complex interdepen-
dent joints and high degree-of-freedom limbs, which may
cause self-occlusions or rare/complex poses.
• Diverse body appearance includes diﬀerent clothing and
self-similar parts.
• Complex environment may cause foreground occlusion,
occlusion or similar parts from nearby persons, various
viewing angles, and truncation in the camera view.
The papers of human pose estimation can be categorized
in diﬀerent ways.
Based on whether to use designed hu-
man body models or not, the methods can be categorized into
generative methods (model-based) and discriminative methods
(model-free). According to from which level (high-level ab-
straction or low-level pixel evidence) to start the processing,
they can be classiﬁed into top-down methods and bottom-up
methods. More details of diﬀerent category strategies for HPE
approaches are summarized in Table 2 and described in Section
2.1.
As listed in Table 1, with the development of human pose
estimation in the past decades, several notable surveys sum-
marized the research work in this area. The surveys (Aggar-
wal and Cai, 1999; Gavrila, 1999; Poppe, 2007; Ji and Liu,
2010; Moeslund et al., 2011) reviewed the early work of hu-
man motion analysis in many aspects (e.g., detection and track-
ing, pose estimation, recognition) and described the relation be-
tween human pose estimation and other related tasks. While Hu
et al. (2004) summarized the research of human motion analy-
sis for video surveillance application, the reviews (Moeslund
and Granum, 2001; Moeslund et al., 2006) focused on the hu-
man motion capture systems. More recent surveys were mainly
focusing on relatively narrow directions, such as RGB-D-based
action recognition(Chen et al., 2013; Wang et al., 2018b), 3D
HPE (Sminchisescu, 2008; Holte et al., 2012; Saraﬁanos et al.,
2016), model-based HPE (Holte et al., 2012; Perez-Sala et al.,
2014), body parts-based HPE (Liu et al., 2015), and monocular-
based HPE (Sminchisescu, 2008; Gong et al., 2016).
Diﬀerent from existing review papers, this survey extensively
summarizes the recent milestone work of deep learning-based
human pose estimation methods, which were mainly published
from 2014. In order to provide a comprehensive summary, this
survey includes a few research work which has been discussed
in some surveys (Liu et al., 2015; Gong et al., 2016; Saraﬁanos
et al., 2016), but most of the recent advances are not been pre-
sented in any survey before.
The remainder of this paper is organized as follows. Section
2 introduces the existing review papers for human motion anal-
ysis and HPE, diﬀerent ways to category HPE methods, and the
widely used human body models. Sections 3 and 4 describe
2D HPE and 3D HPE approaches respectively. In each sec-
tion, we further describe HPE approaches for both single per-
son pose estimation and multi-person pose estimation. Since
data are a very important and fundamental element for deep
learning-based methods, the recent HPE datasets and the eval-
uation metrics are summarized in Section 5. Finally, Section
6 concludes the paper and discusses several promising future
research directions.
2. Categories of HPE Methods and Human Body Models
2.1. HPE Method Categories
This section summarizes the diﬀerent categories of deep
learning-based HPE methods based on diﬀerent characteris-
tics: 1) generative (human body model-based) and discrim-
inative (human body model-free); 2) top-down (from high-
level abstraction to low-level pixel evidence) and bottom-up
(from low-level pixel evidence to high-level abstraction); 3)
regression-based (directly mapping from input images to body
joint positions) and detection-based (generating intermediate
image patches or heatmaps of joint locations); and 4) one-stage
(end-to-end training) and multi-stage (stage-by-stage training).
Generative V.S. Discriminative: The main diﬀerence be-
tween generative and discriminative methods is whether a
method uses human body models or not. Based on the diﬀer-
ent representations of human body models, generative meth-
ods can be processed in diﬀerent ways such as prior beliefs
about the structure of the body model, geometrically projection
from diﬀerent views to 2D or 3D space, high-dimensional para-
metric space optimization in regression manners. More details
of human body model representation can be found in Section

3
Table 1. Summary of the related surveys of human motion analysis and HPE.
No.
Survey & Reference
Venue
Content
1
Human motion analysis: A review (Aggarwal and
Cai, 1999)
CVIU
A review of human motion analysis including body structure analysis,
motion tracking and action recognition.
2
The visual analysis of human movement: A survey
(Gavrila, 1999)
CVIU
A surv

## method
Backbone
Input size
Highlights
PCKh (%)
Regression-based
(Toshev and Szegedy, 2014)
AlexNet
220×220
Direct regression, multi-stage reﬁnement
-
(Carreira et al., 2016)
GoogleNet
224×224
Iterative error feedback reﬁnement from initial pose.
81.3
(Sun et al., 2017)
ResNet-50
224×224
Bone based representation as additional constraint, general for both 2D/3D HPE
86.4
(Luvizon et al., 2017)
Inception-v4+
Hourglass
256×256
Multi-stage architecture, proposed soft-argmax function to convert heatmaps
into joint locations
91.2
Detection-based
(Tompson et al., 2014)
AlexNet
320×240
Heatmap representation, multi-scale input, MRF-like Spatial-Model
79.6
(Yang et al., 2016)
VGG
112×112
Jointly learning DCNNs with deformable mixture of parts models
-
(Newell et al., 2016)
Hourglass
256×256
Proposed stacked Hourglass architecture with intermediate supervision.
90.9
(Wei et al., 2016)
CPM
368×368
Proposed Convolutional Pose Machines (CPM) with intermediate input and
supervision, learn spatial correlations among body parts
88.5
(Chu et al., 2017)
Hourglass
256×256
Multi-resolution attention maps from multi-scale features, proposed micro
hourglass residual units to increase the receptive ﬁeld
91.5
(Yang et al., 2017)
Hourglass
256×256
Proposed Pyramid Residual Module (PRM) learns ﬁlters for input features with
diﬀerent resolutions
92.0
(Chen et al., 2017)
conv-deconv
256×256
GAN, stacked conv-deconv architecture, multi-task for pose and occlusion, two
discriminators for distinguishing whether the pose is ’real’ and the conﬁdence is
strong
91.9
(Peng et al., 2018)
Hourglass
256×256
GAN, proposed augmentation network to generate data augmentations without
looking for more data
91.5
(Ke et al., 2018)
Hourglass
256×256
Improved Hourglass network with multi-scale intermediate supervision,
multi-scale feature combination, structure-aware loss and data augmentation of
joints masking
92.1
(Tang et al., 2018a)
Hourglass
256×256
Compositional model, hierarchical representation of body parts for intermediate
supervision
92.3
(Sun et al., 2019)
HRNet
256×256
high-resolution representations of features across the whole network,
multi-scale fusion.
92.3
(Tang and Wu, 2019)
Hourglass
256×256
data-driven joint grouping, proposed part-based branching network (PBN) to
learn representations speciﬁc to each part group.
92.7
vert a detection-based network to a diﬀerentiable regression-
based one. Nibali et al. (2018) designed a diﬀerentiable spatial
to numerical transform (DSNT) layer to calculate joint coordi-
nates from heatmaps, which worked well with low-resolution
heatmaps.
Prediction of joint coordinates directly from input images
with few constrains is very hard, therefore more powerful net-
works were introduced with a reﬁnement or body model struc-
ture. Carreira et al. (2016) proposed an Iterative Error Feed-
back network based on GoogleNet which recursively processes
the combination of the input image and output results. The
ﬁnal pose is improved from an initial mean pose after itera-
tions. Sun et al. (2017) proposed a structure-aware regression
approach based on a ResNet-50. Instead of using joints to repre-
sent pose, a bone-based representation is designed by involving
body structure information to achieve more stable results than
only using joint positions. The bone-based representation also
works on 3D HPE.
Networks handling multiple closely related tasks of human
body may learn diverse features to improve the prediction of
joint coordinates. Li et al. (2014) employed an AlexNet-like
multi-task framework to handle the joint coordinate predic-
tion task from full images in a regression way, and the body
part detection task from image patches obtained by a sliding-
window. Gkioxari et al. (2014a) used a R-CNN architecture to
synchronously detect person, estimate pose, and classify action.
Fan et al. (2015) proposed a dual-source deep CNNs which take
image patches and full images as inputs and output heatmap
represented joint detection results of sliding windows together
with coordinate represented joint localization results. The ﬁ-
nal estimated posture is obtained from the combination of the
two results. Luvizon et al. (2018) designed a network that can
jointly handle 2D/3D pose estimation and action recognition
from video sequences. The pose estimated in the middle of the
network can be used as a reference for action recognition.
3.1.2. Detection-based methods
Detection-based methods are developed from body part de-
tection methods. In traditional part-based HPE methods, body
parts are ﬁrst detected from image patch candidates and then
are assembled to ﬁt a human body model. The detected body
parts in early work are relatively big and generally represented
by rectangular sliding windows or patches. We refer to (Poppe,
2007; Gong et al., 2016) for a more detailed introduction. Some
early methods use neural networks as body part detectors to dis-
tinguish whether a candidate patch is a speciﬁc body part (Jain
et al., 2013), classify a candidate patch among predeﬁned tem-
plates (Chen and Yuille, 2014) or predict the conﬁdence map
belonging to multiple classes (Ramakrishna et al., 2014). Body
part detection methods are usually sensitive to complexity back-
ground and body occlusions. Therefore the independent image
patches with only local appearance may not be suﬃciently dis-
criminative for body part detection.
In order to provide more supervision information than just
joint coordinates and to facilitate the training of CNNs, more
recent work employed heatmap to indicate the ground truth

7
of the joint location (Tompson et al., 2014; Jain et al., 2014).
As shown in Fig. 4, each joint occupies a heatmap channel
with a 2D Gaussian distribution centered at the target joint lo-
cation. Moreover, Papandreou et al. (2017) proposed an im-
proved representation of the joint location, which is a combi-
nation of binary activation heatmap and corresponding oﬀset.
Since heatmap representation is more robust than coordinate
representation, most of the recent research is based on heatmap
representation.
Fig. 4. Heatmap representation of diﬀerent joints.
The neural network architecture is very important to make
better use of input information. Some approaches are mainly
based on classic networks with appropriate improvements, such
as GoogLeNet-based network with multi-scale inputs (Raﬁ
et al., 2016), ResNet-based network with deconvolutional lay-
ers Xiao et al. (2018). In terms of iterative reﬁnement, some
work designed networks in a multi-stage style to reﬁne results
from coarse prediction via end-to-end learning (Tompson et al.,
2015; Bulat and Tzimiropoulos, 2016; Newell et al., 2016; Wei
et al., 2016; Yang et al., 2017; Belagiannis and Zisserman,
2017). Such networks generally use intermediate supervision
to address vanishing gradients. Newell et al. (2016) proposed a
novel stacked hourglass architecture by using a residual module
as the component unit. Wei et al. (2016) proposed a multi-stage
prediction framework with input image for each stage. Yang
et al. (2017) designed a Pyramid Residual Module (PRMs) to
replace the residual module of the Hourglass network to en-
hance the invariance across scales of DCNNs by learning fea-
tures on various scales. Belagiannis and Zisserman (2017) com-
bined a 7 layers feedforward module with a recurrent module
to iteratively reﬁne the results. This model learns to predict
location heatmaps for both joints and body limbs. Also, they
analyzed keypoint visibility with unbalanced ground truth dis-
tribution. To keep high-resolution representations of features
across the whole network, Sun et al. (2019) proposed a novel
High-Resolution Net (HRNet) with multi-scale feature fusion.
Diﬀerent from earlier work which attempted to ﬁt detected
body parts into body models, some recent work tried to encode
human body structure information into networks.
Tompson
et al. (2014) jointly trained a network with a MRF-like spatial-
model for learning typical spatial relations between joints. Lif-
shitz et al. (2016) discretized an image into log-polar bins cen-
tered around each joint and employed a VGG-based network to
predict joint category conﬁdent for each pair-wise joints (binary
terms). With all relative conﬁdent scores, the ﬁnal heatmap for
each joint can be generated by a deconvolutional network. Yang
et al. (2016) designed a two-stage network. Stage one is a con-
volutional neural network to predict joint locations in heatmap
representation.
Stage two is a message-passing model con-
nected manually according to the human body structure to ﬁnd
optimal joint locations with a max-sum algorithm. Gkioxari
et al. (2016) proposed a convolutional Recurrent Neural Net-
work to output joint location one by one following a chain
model. The output of each step depends on both the input image
and the previously predicted output. The network can handle
both images and videos with diﬀerent connection strategy. Chu
et al. (2016) proposed to transform 