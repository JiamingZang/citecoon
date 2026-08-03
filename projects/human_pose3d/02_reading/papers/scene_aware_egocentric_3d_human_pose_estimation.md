# Scene-Aware Egocentric 3D Human Pose Estimation

> 2023 · id: W4386083079 · arXiv: 2212.11684 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Egocentric 3D human pose estimation with a single
head-mounted fisheye camera has recently attracted atten-
tion due to its numerous applications in virtual and aug-
mented reality. Existing methods still struggle in challeng-
ing poses where the human body is highly occluded or is
closely interacting with the scene. To address this issue, we
propose a scene-aware egocentric pose estimation method
that guides the prediction of the egocentric pose with scene
constraints. To this end, we propose an egocentric depth
estimation network to predict the scene depth map from a
wide-view egocentric fisheye camera while mitigating the
occlusion of the human body with a depth-inpainting net-
work.
Next, we propose a scene-aware pose estimation
network that projects the 2D image features and estimated
depth map of the scene into a voxel space and regresses
the 3D pose with a V2V network.
The voxel-based fea-
ture representation provides the direct geometric connec-
tion between 2D image features and scene geometry, and
further facilitates the V2V network to constrain the pre-
dicted pose based on the estimated scene geometry. To en-
able the training of the aforementioned networks, we also
generated a synthetic dataset, called EgoGTA, and an in-
the-wild dataset based on EgoPW, called EgoPW-Scene.
The experimental results of our new evaluation sequences
show that the predicted 3D egocentric poses are accurate
and physically plausible in terms of human-scene interac-
tion, demonstrating that our method outperforms the state-
of-the-art methods both quantitatively and qualitatively.

## introduction
Egocentric 3D human pose estimation with head- or
body-mounted cameras is extensively researched recently
because it allows capturing the person moving around in a
large space, while the traditional pose estimation methods
can only record in a fixed volume. With this advantage, the
egocentric pose estimation methods show great potential in
various applications, including the xR technologies and mo-
Image
EgoPW
Ours
Figure 1.
Previous egocentric pose estimation methods like
EgoPW predict body poses that may suffer from body floating is-
sue (the first row) or body-environment penetration issue (the sec-
ond row). Our method predicts accurate and plausible poses com-
plying with the scene constraints. The red skeletons are the ground
truth poses and the green skeletons are the predicted poses.
bile interaction applications.
In this work, we estimated the full 3D body pose from
a single head-mounted fisheye camera.
A number of
works have been proposed, including Mo2Cap2 [39], xR-
egopose [32], Global-EgoMocap [36], and EgoPW [35].
These methods have made significant progress in estimat-
ing egocentric poses. However, when taking account of the
interaction between the human body and the surrounding
environment, they still suffer from artifacts that contrast the
physics plausibility, including body-environment penetra-
tions or body floating (see the EgoPW results in Fig. 1),
which is mostly ascribed to the ambiguity caused by the
self-occluded and highly distorted human body in the ego-
centric view. This problem will render restrictions on sub-
sequent applications including action recognition, human-
object interaction recognition, and motion forecasting.
To address this issue, we propose a scene-aware pose
estimation framework that leverages the scene context to
constrain the prediction of an egocentric pose. This frame-
arXiv:2212.11684v3  [cs.CV]  25 Sep 2023

work produces accurate and physically plausible 3D human
body poses from a single egocentric image, as illustrated in
Fig. 1. Thanks to the wide-view fisheye camera mounted
on the head, the scene context can be easily obtained even
with only one egocentric image. To this end, we train an
egocentric depth estimator to predict the depth map of the
surrounding scene. In order to mitigate the occlusion caused
by the human body, we predict the depth map including the
visible human and leverage a depth-inpainting network to
recover the depth behind the human body.
Next, we combine the projected 2D pose features and
scene depth in a common voxel space and regress the 3D
body pose heatmaps with a V2V network [22]. The 3D
voxel representation projects the 2D poses and depth in-
formation from the distorted fisheye camera space to the
canonical space, and further provides direct geometric con-
nection between 2D image features and 3D scene geome-
try. This aggregation of 2D image features and 3D scene
geometry facilitates the V2V network to learn the rela-
tive position and potential interactions between the human
body joints and the surrounding environment and further
enables the prediction of plausible poses under the scene
constraints.
Since no available dataset can be used for train these net-
works, we proposed EgoGTA, a synthetic dataset based on
the motion sequences of GTA-IM [3], and EgoPW-Scene,
an in-the-wild dataset based on EgoPW [35]. Both of the
datasets contain body pose labels and scene depth map la-
bels for each egocentric frame.
To better evaluate the relationship between estimated
egocentric pose and scene geometry, we collected a new
test dataset containing ground truth joint positions in the
egocentric view. The evaluation results on the new dataset,
along with results on datasets in Wang et al. [36] and
Mo2Cap2 [39] demonstrate that our method significantly
outperforms existing methods both quantitatively and qual-
itatively. We also qualitatively evaluate our method on in-
the-wild images. The predicted 3D poses are accurate and
plausible even in challenging real-world scenes. To summa-
rize, our contributions are listed as follows:
• The first scene-aware egocentric human pose estima-
tion framework that predicts accurate and plausible
egocentric pose with the awareness of scene context;
• Synthetic and in-the-wild egocentric datasets contain-
ing egocentric pose labels and scene geometry labels;1
• A new depth estimation and inpainting networks to
predict the scene depth map behind the human body;
• By leveraging a voxel-based representation of body
pose features and scene geometry jointly, our method
1Datasets are released in our project page. Meta did not access or pro-
cess the data and is not involved in the dataset release.
outperforms the previous approaches and generates
plausible poses considering the scene context.

## method
We propose a new method for predicting accurate ego-
centric body pose by leveraging the estimated scene geom-
etry. An overview of our method is shown in Fig. 2. In
order to train the scene-aware network, we first generate a
synthetic dataset based on the GTA-IM dataset [3], called

EgoGTA, and an in-the-wild dataset based on the EgoPW
dataset [35], called EgoPW-Scene (Sec. 3.1). Next, we train
a depth estimator to estimate the geometry of the surround-
ing scene and introduce the depth-inpainting network that
estimates the depth behind the human body (Sec. 3.2). Fi-
nally, we combine 2D features and scene geometry in a
common voxel space and predict the egocentric pose with a
V2V network [22] (Sec. 3.3).
3.1. Training Dataset
Although many training datasets for egocentric pose esti-
mation [32,35,39] have been proposed, they cannot yet train
the scene-aware egocentric pose estimation network due to
the lack of scene geometry information. To solve this, we
introduce the EgoGTA dataset and EgoPW-Scene dataset
(both will be made publicly available). Both datasets con-
tain pose labels and depth maps of the scene for each ego-
centric frame, facilitating our training process. We show
examples from both datasets, as illustrated in Fig. 2.
3.1.1
EgoGTA Dataset
In order to obtain precise ground truth human pose and
scene geometry for training, we devise a new synthetic ego-
centric dataset based on GTA-IM [3], which contains var-
ious daily motions and ground truth scene geometry. For
this, we first fit the SMPL-X model on the 3D joint tra-
jectories from GTA-IM. Next, we attach a virtual fisheye
camera to the forehead of the SMPL-X model and render
the images, semantic labels, and depth map of the scene
with and without the human body. In total, we obtained
320 K frames in 101 different sequences, each with a dif-
ferent human body texture. Here, we denote the EgoGTA
dataset SG = {IG, SG, DB
G, DS
G, PG}, including synthetic
images IG and their corresponding human body segmenta-
tion maps SG, depth map with human body DB
G, depth map
of the scene without human body DS
G, and egocentric pose
labels PG.
3.1.2
EgoPW-Scene Dataset
Since we want to generalize to data captured with a real
head-mounted camera, we also extended the EgoPW [35]
training dataset. For this, we first reconstruct the scene ge-
ometry from the egocentric image sequences of the EgoPW
training dataset with a Structure-from-Motion (SfM) algo-
rithm [10]. This step provides a dense reconstruction of the
background scene. The global scale of the reconstruction
is recovered from known objects present in the sequences,
such as laptops and chairs. We further render the depth
maps of the scene in the egocentric perspective based on
the reconstructed geometry. Our EgoPW-Scene dataset con-
tains 92 K frames in total, which are distributed in 30 se-
quences performed by 5 actors. The number of frames in
the EgoPW-Scene dataset is less than EgoPW dataset since
SfM fails on some sequences. Here, we denote the EgoPW-
Scene dataset SE = {IE, DS
E, PE}, including in-the-wild
images IE and their corresponding depth map of the scene
without human body DS
E, and egocentric pose labels PE.
3.2. Scene Depth Estimator
In this section, we propose a depth estimation method
to capture the scene geometry information in the egocentric
perspective. Available depth estimation methods [7,11,16]
can only generate depth maps with the human body, but are
not able to infer the depth information behind the human,
i.e., the background scene depth. However, the area oc-
cluded by the human body, e.g. the areas of foot contact,
are crucial for generating plausible poses, as demonstrated
in Sec. 4.4. To predict the depth map of the scene behind the
human body, we adopt a two-step approach. More specifi-
cally, we first estimate the depth map including the human
body and the semantic segmentation of the human with two
separated models. Then, we use a depth inpainting network
to recover the depth behind the human body. This two-step
strategy is necessary because the human visual evidences in
the RGB images are too strong to be ignored by the depth
estimator, therefore, it is easier to train the scene depth esti-
mation as separated tasks.
We first train the depth estimator network D, which takes
as input a single egocentric image I and predicts the depth
map with human body ˆDB. The network architecture of
D is the same as Hu et al. [11]’s work. To minimize the
influence of the domain gap between synthetic and real
data, the network is initially trained on the NYU-Depth V2
dataset [23] following [11], and further fine-tuned on the
EgoGTA dataset.
Next, we train the segmentation network S for segment-
ing the human body. The network takes the egocentric im-
age I as input and predicts the segmentation mask for the
human body ˆS as output. Following Yuan et al. [44], we
use HRNet as our segmentation network. Similarly, to re-
duce the domain gap, we pretrain the network on the LIP
dataset [8] and finetune the model on the EgoGTA dataset.
We do not train network D and S on the EgoPW-Scene
dataset since it lacks the ground truth segmentation maps
and depth maps with the human body.
Finally, we propose a depth inpainting network G for
generating the final depth map of the scene without hu-
man body.
We first generate the masked depth map
ˆDM = (1 −ˆS) ⊙ˆDB, which is a Hadamard product be-
tween the background segmentation and the depth map with
human body. Then, the masked depth map ˆDM and the
segmentation mask ˆS are fed into the inpainting network G,
which predicts the final depth map ˆDS. We train the inpaint-
ing network G and finetune the depth estimation network D
on both the EgoGTA and EgoPW-Scene datasets. During

training, we penalize the differences between the predicted
depth maps and the ground truth depth of the background
scene with LS and also keep the depth map consistent in
the non-human body regions with LC. Specifically, the loss
function is defined as follows:
L = λSLS + λCLC,
with
LS =
 ˆDS
G −DS
G

2
2 +
 ˆDS
E −DS
E

2
2 ,
and
LC =
( ˆDS
G −ˆDB
G)(1 −ˆSG)

2
2
+
( ˆDS
E −ˆDB
E)(1 −ˆSE)

2
2 ,
(1)
where
ˆDS
G = G( ˆDM
G , ˆSG);
ˆDS
E = G( ˆDM
E , ˆSE);
ˆDB
G = D(IG);
ˆDB
E = D(IE);
ˆSG = S(IG);
ˆSE = S(IE),
(2)
and λS and λC are the weights of the loss terms.
3.3. Scene-aware Egocentric Pose Estimator
In this section, we introduce our scene-aware egocen-
tric pose estimator. We rely on the prior that human bodies
are mostly in contact with the scene. However, estimating
the contact explicitly from a single egocentric image is very
challenging. Therefore, we rely on a data-driven approach
by learning a model that predicts a plausible 3D pose given
the estimated scene geometry and features extracted from
the input image.
To achieve this goal, we first leverage
the EgoPW [35] body joints heatmap estimator to extract
2D body pose features F and use the scene depth estimator
from Sec. 3.2 to estimate the depth map of the scene without
human body ˆDS. Afterwards, we project the body pose fea-
tures and depth map into a 3D volumetric space considering
the fisheye camera projection model. After obtaining the
volumetric representation of human body features Vbody and
scene depth Vscene, the 3D body pose ˆP is predicted from the
volumetric representation with a V2V network [22].
Lifting the image features and depth maps to a 3D rep-
resentation allows getting more plausible results, as incon-
sistent joint predictions can be behind the volumetric scene
Vscene (pose-scene penetration) or spatially isolated from the
voxelized scene geometry (pose floating), so they can be
easily identified and adjusted by the volumetric convolu-
tional network.
3.3.1
Scene and Body Encoding as a 3D Volume
In order to create the volumetric space, we first create a
3D bounding box around the person in the egocentric cam-
era coordinate system of size L × L × L, where L de-
notes the length of the side of the bounding box in meters.
The egocentric camera is placed at the center-top of the 3D
bounding box so that the vertices of the bounding boxes are:
(±L/2, ±L/2, 0) and (±L/2, ±L/2, L) under the egocen-
tric camera coordinate system.
Next, we discretize the
bounding box by a volumetric cube V ∈RN,N,N,3. Each
voxel Vxyz ∈R3 in position (x, y, z) is filled with the coor-
dinates of its center under the egocentric camera coordinate
system (xL/N −L/2, yL/N −L/2, zL/N).
We project the 3D coordinates in V into the egocentric
image space with the fisheye camera model [27]: Vproj =
P(V ), where Vproj
∈RN,N,N,2 and P is the fisheye
camera projection function.
The volumetric representa-
tion Vbody of the human body is obtained by filling a cube
Vbody ∈RN,N,N,K by bilinear sampling from the feature
maps F with K channels using 2D coordinates in Vproj:
Vbody = F{Vproj}
(3)
where {·} denotes bilinear sampling.
Then, we project the depth map to the 3D volumetric
space. We first generate the point cloud of th

## experiments
In this section, we evaluate our method considering ex-
isting and new datasets for egocentric monocular 3D human
pose estimation. Please refer to the supplementary materials
for the implementation details.

Image
EgoPW
Ours
Image
EgoPW
Ours
xR-egopose
Mo2Cap2
xR-egopose
Mo2Cap2
Figure 3. Qualitative comparison between our method and the state-of-the-art egocentric pose estimation methods. From left to right:
input image, Mo2Cap2 result, xR-egopose result, EgoPW result and our result. The ground truth pose is shown in red. The input images
from the left part are from our test dataset, while those in the right part come from the EgoPW [35] in-the-wild test sequences (without
ground-truth poses). We also show the gt scene geometry of the in-the-studio data and scene geometry obtained by SFM method for the
in-the-wild data. For better visualizing the interaction between human body and environment, please refer to our supplementary video.
4.1. Evaluation Datasets
Evaluating human-scene interaction requires precise an-
notations for camera pose and scene geometry.
How-
ever, such information is not available in existing datasets
for egocentric human pose estimation.
To solve this is-
sue, we collected a new real-world dataset using a head-
mounted fisheye camera combined with a calibration board.
The ground truth scene geometry is obtained with SfM
method [10] from a multi-view capture system with 120
synced 4K resolution cameras and the ground truth ego-
centric camera pose is obtained by localizing a calibra-
tion board rigidly attached to the egocentric camera. This
dataset contains around 28K frames of two actors, perform-
ing various human-scene interacting motions such as sit-
ting, reading newspaper, and using a computer. This dataset
is evenly split into training and testing splits. We finetuned
the method on the training split before the evaluation. This
dataset will be made publicly available and additional de-
tails of it are shown in the supplementary materials.
Besides our new test dataset, we also evaluate our
methods in the test datasets from Wang et al. [36] and
Mo2Cap2 [39]. The real-world dataset in Mo2Cap2 [39]
contains 2.7K frames of two people captured in indoor and
outdoor scenes, and the dataset in Wang et al. [36] contains
12K frames of two people captured in the studio.
4.2. Evaluation Metrics
We measure the accuracy of the estimated body pose
with the MPJPE and PA-MPJPE. For the test dataset in
Wang et al. [36] and Mo2Cap2 [39], we evaluate PA-MPJPE
and BA-MPJPE [39] since the ground truth poses in the ego-
centric camera space are not provided. Further details of the
metrics are shown in the supplementary materials.
4.3. Comparisons on 3D Pose Estimation

## related_work
2.1. Egocentric 3D Full Body Pose Estimation
Inspired by the new applications in augmented reality
and by the limitations of traditional motion capture systems,
Rhodin et al. [26] proposed the first egocentric motion cap-
ture system based on a pair of fisheye cameras. The fol-
lowing methods proposed new architectures [4,47] and new
datasets [1,47] for stereo egocentric pose estimation. How-
ever, a stereo camera setup implies additional computation
complexity and extra energy consumption, which is critical
for low-power head-mounted devices, which are the main
target applications.
The single head-mounted fisheye camera setup was first
proposed by Xu et al. [39], who also introduced a two-
stream CNN to cope with the low resolution of regions far
from the fisheye camera, i.e., one branch for the full body
and one branch for predicting the lower body joints from
a zoom-in image. Tome et al. [32] proposed an encoder-
decoder architecture to model the high uncertainty caused
by severe occlusions present in this setup and Wang et
al. [36] leverages motion capture data to learn a human mo-
tion prior, which is applied in an optimization method to
obtain temporally stable poses for training and egocentric
pose predictor. Another challenge is the strong image dis-
tortion caused by the fisheye lens, which can be mitigated
with automatic camera calibration [46]. Other setups con-
sider the camera facing forward and try to synthesize plau-
sible human motion given only scene image evidences [18]
or partially visible body parts [13, 15]. In our work, we do
not consider this setup since in many poses only a few body
extremities are visible in the image.
Obtaining real data in the egocentric setup is a time con-
suming process, therefore, many approaches rely on syn-
thetic data for training [15,32]. A recent method has shown
that the existing gap between synthetic and real data can be
mitigated by domain adaptation techniques [35], but this ap-
proach still requires real data for the weak supervision part.
Differently, our method has a simpler training strategy, and
we reduce the gap between synthetic and real data by pro-
viding additional pseudo ground-truth scene labels for real
sequences, including indoor and outdoor scenes.
2.2. Voxel Representation for Body Pose Estimation
Volumetric voxel representations have been extensively
used with multiple view setups for the estimation of sin-
gle [31, 33] and multiple human poses [34, 37, 40, 45], and
for hand pose estimation [12, 19–21] from depth maps in-
put.
Considering a single image as input, Pavlakos et
al. [25] proposed a coarse-to-fine approach to lift from 2D

Egocentric Images
Body Pose Features
Scene Depth 
Estimator
2D Feature 
Extractor
Depth w/o Human Body
V2V Network
Sec. 3.1   Training Dataset
Sec. 3.2 Scene Depth Estimator
Sec. 3.3   Scene-aware Egocentric Pose Estimator
EgoGTA
EgoPW-Scene
Image
Depth with Body
Depth w/o Body
Egocentric Images
Depth Estimator 
with Body
Human Body 
Segmentation 
Network
Depth 
Inpainting 
Network
Image
Depth w/o Body
Depth with Human Body
Human Body Seg
Depth w/o Human Body
3D Heatmap
Predicted 3D Pose
Projected Pose Features
Scene Voxel
Human Body Seg
Figure 2. Overview of our method. We first render synthetic training dataset EgoGTA and in-the-wild training dataset EgoPW-Scene. Both
datasets contain egocentric depth maps for subsequent training process (Sec. 3.1). Next, we train an egocentric scene depth estimator that
predicts a depth map without the human body and a depth inpainting network (Sec. 3.2). Finally, we combine the 2D body pose features
and scene depth map into a common voxel space. The 3D body pose heatmaps are regressed from the voxel space with a V2V network and
the final pose prediction is obtained with soft-argmax (Sec. 3.3).
heatmaps to a voxel representation for 3D human pose es-
timation. Iskakov [14] later proposed a learnable triangu-
lation method in voxel space that can generalize to single
or multiple views, achieving less than 2 cm error in a con-
trolled multi-view data. Despite the success of voxel rep-
resentation for human and hand pose estimation from an
external camera, this technique has not yet been explored
for the interaction between human body and scene. In our
work, we show the advantages of this representation for
egocentric human pose estimation, especially when consid-
ering human-scene interaction.
2.3. Scene-aware Human Pose Estimation
In recent years, several approaches have been proposed
to predict the pose of humans considering environmental
and physical constraints from RGB [24,30,41] and inertial
measurement units (IMU) [9, 42]. Some methods assume
a simplified environment, such as a planar ground floor, to
enforce a temporal sequence that is physically consistent
with the universal law of gravity by assuming known cam-
era poses [29,30] or by tracking an object in the scene fol-
lowing a free flight trajectory [6]. Other approaches assume
that the scene is given as input, either as a 3D reconstruc-
tion [9,28] or as geometric primitives [43], whose positions
can be refined in the optimization process. Bhatnagar et
al. [2] proposed a method and dataset for human-object in-
teractions. Taking into account the interaction between hu-
mans and furniture, holistic methods are able to estimate
the position of humans and specific objects in the scene un-
der the assumption of a planar floor [5, 38, 41], or even to
estimate deformations in known objects based on human
poses [17]. Contrary to the previous work, we make no
strong assumptions about the objects and ground floor in the
scene, but instead propose a method that learns to estimate
the background scene geometry from a fisheye camera and
explores the correlation between the human body and scene
directly from egocentric data.

## conclusion
In this paper, we have proposed a new approach to es-
timate egocentric human pose under the scene constraint.
We firstly train a depth inpainting network for estimating
the depth map of the scene without human body. Next, we
combine the egocentric 2D features and scene depth map
in a volumetric space and predict the egocentric pose with
a V2V network. The experiments show that our method
outperforms all of the baseline methods both qualitatively
and quantitatively and our method can predict physically
plausible poses in terms of human-scene interaction. In fu-
ture, this method could be extended to estimate physically-
plausible egocentric motion from a temporal sequence.
Limitations. The accuracy of voxel-based pose estimation
network is constrained by the accuracy of estimated depth,
especially where the scene is occluded by the human body.
One solution is to leverage the temporal information to get
a full view of the surrounding environment.
Acknowledgments Jian Wang, Diogo Luvizon, Lingjie
Liu, and Christian Theobalt have been supported by the
ERC Consolidator Grant 4DReply (770784).