# FFB6D: A Full Flow Bidirectional Fusion Network for 6D Pose Estimation

> 2021 · id: W3179923621 · arXiv: 2103.02242 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
In this work,
we present FFB6D, a Full Flow
Bidirectional fusion network designed for 6D pose estima-
tion from a single RGBD image. Our key insight is that
appearance information in the RGB image and geometry
information from the depth image are two complementary
data sources, and it still remains unknown how to fully
leverage them. Towards this end, we propose FFB6D, which
learns to combine appearance and geometry information
for representation learning as well as output representa-
tion selection. Speciﬁcally, at the representation learning
stage, we build bidirectional fusion modules in the full
ﬂow of the two networks, where fusion is applied to each
encoding and decoding layer.
In this way, the two net-
works can leverage local and global complementary in-
formation from the other one to obtain better representa-
tions.
Moreover, at the output representation stage, we
designed a simple but effective 3D keypoints selection al-
gorithm considering the texture and geometry information
of objects, which simpliﬁes keypoint localization for pre-
cise pose estimation. Experimental results show that our
method outperforms the state-of-the-art by large margins
on several benchmarks. Code and video are available at
https://github.com/ethnhe/FFB6D.git.

## introduction
We propose a full ﬂow bidirectional fusion network to
solve the problem, as shown in Figure 2(a). The proposed
framework ﬁrst extracts the pointwise RGBD feature for
per-object 3D keypoints localization.
Then the pose pa-
rameters are recovered within a least-square ﬁtting manner.
More speciﬁcally, given an RGBD image as input, we uti-
lize a CNN to extract appearance features from the RGB
image, and a point cloud network to extract geometric fea-
tures from point clouds. During the feature extraction ﬂow
of the two networks, point-to-pixel and pixel-to-point fu-
sion modules are added into each layer as communication
bridges. In this way, the two branches can utilize the ex-
tra appearance (geometric) information from the other to
facilitate their own representation learning. The extracted
pointwise features are then fed into an instance semantic
segmentation and a 3D keypoint detection module to obtain
per-object 3D keypoints in the scene. Finally, a least-square
ﬁtting algorithm is applied to recover the 6D pose parame-
ters.
3

3.2. Full Flow Bidirectional Fusion Network
Given an aligned RGBD image, we ﬁrst lift the depth
image to a point cloud with the camera intrinsic matrix. A
CNN and a point cloud network (PCN) is then applied for
feature extraction from the RGB image and point cloud re-
spectively. When the information ﬂow through the two net-
works, point-to-pixel and pixel-to-point fusion modules are
added for bidirectional communication. In this way, each
branch can leverage local and global information from the
other to facilitate their representation learning.
Pixel-to-point fusion from image features to point
cloud features. These modules share the appearance in-
formation extracted from CNN to the PCN. One naive way
is to generate a global feature from the RGB feature map
and then concatenate it to each point cloud feature. How-
ever, since most of the pixels are background and there are
multi objects in the scene, squeezing the RGB feature map
globally would lose lots of detailed information and harm
the following pose estimation module. Instead, we intro-
duce a novel pixel to point feature fusion module. Since
the given RGBD image is well-aligned, we can use the 3D
point clouds as a bridge to connect the pixel-wise and the
point-wise features. More speciﬁcally, we lift the depth of
each pixel to its corresponding 3D point with the camera in-
trinsic matrix and get an XYZ map, aligned with the RGB
map. As shown in the left part of Figure 2(b), for each point
feature with its 3D point coordinate, we ﬁnd its Kr2p near-
est point in the XYZ map and gather their corresponding
appearance features from the RGB feature map. We then
use max polling to integrate these neighboring appearance
features following [51], and apply shared Multi-Layer Per-
ceptrons (MLPs) to squeeze it to the same channel size as
the point cloud feature:
Fr2p =MLP(
Kr2p
max
i=1 Fri),
(1)
where Fri is the ith nearest pixel of RGB feature and Fr2p
the integrated one. We then concatenate the integrated ap-
pearance feature Fr2p with the point feature Fpoint and use
shared MLP to obtained the fused point feature:
Ffusedp =MLP(Fpoint ⊕Fr2p),
(2)
where ⊕is the concatenate operation.
One thing to mention is that in the ﬂow of the appearance
feature encoding, the height and width of the RGB feature
maps get smaller when the network goes deeper. There-
fore, we need to maintain a corresponding XYZ map so that
each pixel of feature can ﬁnd its 3D coordinate. Since the
decreased size of the feature map is generated by convolu-
tion kernel scanning through the original feature map with
stride, the centers of kernels become new coordinates of the
feature maps. One simple way is to apply the same size
of kernels to calculate the mean of XYZ within it to gen-
erate a new XYZ coordinate of a pixel. The corresponding
XYZ map is then obtained by scanning through the XYZ
map with the mean kernel in the same stride as CNN. How-
ever, noise points are produced by the mean operation as
the depth changes remarkably on the boundary between the
foreground objects and the background. Instead, a better
solution is to resize the XYZ map to the same size as the
feature map within the nearest interpolation algorithm.
Point-to-pixel fusion from point cloud features to im-
age features. These modules build bridges to transfer the
geometric information obtained from the PCN to the CNN.
The procedure is shown on the right side of Figure 2(b).
Same as the pixel-to-point fusion modules, we fuse the fea-
ture densely rather than naively concatenating the global
point feature to each pixel. Speciﬁcally, for each pixel of
feature with its XYZ coordinate, we ﬁnd its Kp2r nearest
points from the point cloud and gather the corresponding
point features. We squeeze the point features to the same
channel size as the RGB feature and then use max pooling
to integrate them. The integrated point feature is then con-
catenated to the corresponding color feature and mapped by
a shared MLP to generate the fused one:
Fp2r =
Kp2r
max
j=1 (MLP(Fpj)),
(3)
Ffusedr =MLP(Frgb ⊕Fp2r),
(4)
where Fpj denotes the jth nearest point features, Fp2r the
integrated point features and ⊕the concatenate operation.
Dense RGBD feature embedding With the proposed
full ﬂow fusion network, we obtain dense appearance em-
beddings from the CNN branch and dense geometry fea-
tures from the PCN branch. We then ﬁnd the correspon-
dence between them by projecting each point to the image
plane with the camera intrinsic matrix. According to the
correspondence, we obtain pairs of appearance and geome-
try features and concatenate them together to form the ex-
tracted dense RGBD feature. These features are then fed
into an instance semantic segmentation module and a 3D
keypoint detection module for object pose estimation in the
next step.
3.3. 3D Keypoint-based 6D Pose Estimation
Recently, the PVN3D [17] work by He et al. opens up
new opportunities for using 3D keypoints to estimate object
pose. In this work, we follow their 3D keypoint formulation
but further improve the 3D keypoint selection algorithm to
fully leverage the texture and geometry information of ob-
jects. Speciﬁcally, we ﬁrst detect the per-object selected 3D
keypoints in the scene and then utilize a least-squares ﬁtting
algorithm to recover the pose parameters.
Per-object 3D keypoint detection So far we have ob-
tained the dense RGBD embeddings.
We then follow
4

PVN3D [17] and obtain the per-object 3D keypoints by
adding an instance semantic segmentation module to distin-
guish different object instances and a keypoint voting mod-
ule to recover 3D keypoints. The instance semantic segmen-
tation module consists of a semantic segmentation module
and a center point voting module, where the former one pre-
dicts per-point semantic labels, and the latter one learns the
per-point offset to object centers for distinguishment of dif-
ferent instances. For each object instance, the keypoint vot-
ing module learns the point-wise offsets to the selected key-
points that vote for 3D keypoint within a MeanShift [10]
clustering manners.
Keypoint selection Previous works [47, 17] select key-
points from the target object surface using the Farthest Point
Sampling (FPS) algorithm. Speciﬁcally, they maintain a
keypoint set initialized by a random point on the object sur-
face and iteratively add other points that are farthest to those
within the set until N points are obtained. In this way, the
selected keypoints spread on the object surface and stabilize
the following pose estimation procedure [47, 17]. However,
since the algorithm only takes the Euclidean distance into
account, the selected points may appear in non-salient re-
gions, such as ﬂat planes without distinctive texture. These
points are hard to detect and the accuracy of the estimated
pose decrease. To fully leverage the texture and geometry
information of objects, we propose a simple but effective
3D keypoint selection algorithm, named SIFT-FPS. Speciﬁ-
cally, we use the SIFT [40] algorithm to detect 2D keypoints
that are distinctive in texture images and then lift them to
3D. The FPS algorithm is then applied for the selection of
top N keypoints among them. In this way, the selected key-
points not only distribute evenly on the object surface but
are also distinctive in texture and easy to detect.
Least-Squares Fitting Given the selected 3D keypoints
in the object coordinates system {pi}N
i=1, and the corre-
sponding 3D keypoints in the camera coordinated system
{p∗
i }N
i=1. The Least-Squares Fitting [1] algorithm calculate
the pose parameters R and T by minimizing the squared
loss:
Llsf =
N
X
i=1
||p∗
i −(R · pi + T)||2.
(5)

## method
DPOD
[71]
Hu et
al.[25]
HybridPose
[56]
PVN3D
[17]
Our
FFB6D
ADD-0.1d
47.3
43.3
47.5
63.2
66.2
Table 4: Quantitative evaluation of 6D pose (ADD-0.1d) on
the Occlusion-LineMOD dataset.
Fusion Stage
Pose Result
FE
FD
DF
ADD-S
ADD(S)
91.9
87.5
√
96.2
92.2
√
93.0
89.2
√
94.0
90.8
√
√
96.6
92.7
√
√
√
96.4
92.6
Table 5: Effect of fusion stages on the YCB-Video dataset.
FE: fusion during encoding; FD: fusion during decoding;
DF: Dense Fusion on the two ﬁnal feature maps.
the supplementary material.
Robustness towards occlusion. We follow [66, 17] to
report the ADD-S less than 2cm accuracy under the growth
of occlusion level on the YCB-Video dataset. As is shown
in Figure 3, previous methods degrade as the occlusion in-
crease. In contrast, FFB6D didn’t suffer from a drop in
performance. We think our full ﬂow bidirectional fusion
mechanism makes full use of the texture and geometry in-
formation in the captured data and enables our approach to
locate 3D keypoints even in highly occluded scenes.
Evaluation on the LineMOD dataset & Occlusion
LineMOD dataset. The proposed FFB6D outperforms the
state-of-the-art on the LineMOD dataset, presented in Table
10. We also evaluate FFB6D on the Occlusion LineMOD
dataset, shown in Table 11. In the table, our FFB6D with-
out iterative reﬁnement advances state-of-the-art by 4.7%,
further conﬁrming its robustness towards occlusion.
4.5. Ablation Study
In this subsection, we present extensive ablation studies
on our design choices and discuss their effect.
Effect of full ﬂow bidirectional fusion.
To validate
that building fusion modules between the two modality net-
works in full ﬂow help, we ablate fusion stages in Table
5. Compared to the mechanism without fusion, adding fu-
sion modules either on the encoding stage, decoding stage,
Fusion Direction
Pose Result
P2R
R2P
ADD-S
ADD(S)
91.9
87.5
√
95.8
91.1
√
94.5
90.6
√
√
96.6
92.7
Table 6:
Effect of fusion direction on the YCB-Video
dataset. P2R means fusion from point cloud embeddings
to RGB embeddings, and R2P means fusion from RGB em-
beddings to point cloud embeddings.
or on the ﬁnal feature maps, can all boost the performance.
Among the three stages, fusion on the encoding stages ob-
tained the highest improvement. We think that’s because the
extracted local texture and geometric information are shared
through the fusion bridge on the early encoding stage, and
more global features are shared when the network goes
deeper. Also, adding fusion modules in full ﬂow of the
network, saying that on both the encoding and decoding
stages, obtains the highest performance. While adding ex-
tra DenseFusion behind full ﬂow fusion obtains no perfor-
mance gain as the two embeddings have been fully fused.
We also ablate the fusion direction in Table 6 to vali-
date the help of bidirectional fusion. Compared with no
fusion, both the fusion from RGB to point cloud and the in-
verse way facilitate better representation learning. Combin-
ing the two obtains the best results. On the one hand, from
the view of PCN, we think the rich textures information
obtained from high-resolution RGB images helps semantic
recognition. In addition, the high-resolution RGB features
provide rich information for blind regions of depth sensors
caused by reﬂective surfaces. It serves as a completion to
point cloud and improve pose accuracy, as shown in Fig-
ure 4(a). On the one hand, geometry information extracted
from point cloud helps the RGB branch by distinguishing
foreground objects from the background that are in similar
colors. Moreover, the shape size information extracted from
point clouds helps divide objects with a similar appearance
but in a different size, as is shown in Figure 4(b).
Effect of representation learning frameworks. We ex-
plore the effect of different representation learning frame-
works for the two modalities of data in this part. The result
is presented in Table 7. We ﬁnd that neither concatenating
the XYZ map as extra information to CNN (CNN-R⊕D)
nor adding RGB values as extra inputs to the PCN (PCN-
R⊕D) achieves satisfactory performance. Using two CNNs
7

Large clamp
Extra-large clamp
RGB Scene
Our FFB6D
PVN3D
(b)  Effect of point-to-pixel fusion from point cloud features to image features. The point-to-pixel fusion provides 
geometry information from PCN on point cloud to help distinguish objects with similar appearance during appearance 
representation learning.
PVN3D
Our FFB6D
Depth
RGB
(a)  Effect of pixel-to-point fusion from image features to point cloud 
features. The pixel-to-point fusion provides a vision of reflective surfaces 
from CNN on RGB to ease point cloud reasoning.
Figure 4: Effect of full ﬂow bidirectional fusion, compared to PVN3D [17] with DenseFusion architecture.
CNN-R⊕D
PCN-R⊕D
CNN-R+CNN-D
ADD-S
91.0
90.9
92.1
ADD(S)
85.5
81.2
87.2
PCN-R+PCN-D
CNN-R+3DC-D
CNN-R+PCN-D
ADD-S
91.8
94.9
96.6
ADD(S)
84.3
90.1
92.7
Table 7: Effect of representation learning framework on the
two modalities of data. CNN: 2D Convolution Neural Net-
work; PCN: point cloud network; 3DC: 3D ConvNet; R:
RGB images; D: XYZ maps for CNN, point clouds for PCN
and voxelized point clouds for 3D ConvNet.
FPS4
S-F4
FPS8
S-F8
FPS12
S-F12
KP err. (cm)
1.3
1.2
1.4
1.2
1.6
1.3
ADD-S
95.9
96.4
96.1
96.6
96.0
96.5
ADD(S)
92.0
92.4
92.3
92.7
92.0
92.6
Table 8: Effect of keypoint selection algorithm. S-F means
the proposed SIFT-FPS algorithm.
(CNN-R+CNN-D) or two PCNs (PCN-R+PCN-D) with full
ﬂow bidirectional fusion modules get better but are still far
from satisfactory. In contrast, applying CNN on the RGB
image and PCN on the point cloud (CNN-R+PCN-D) gets
the best performance. We think that the grid-like image data
is discrete, on which the regular convolution kernel ﬁts bet-
ter than continuous PCN. While the geometric information
residing in the depth map is deﬁned in a continuous vector
space, and thus PCNs can learn better representation.
Effect of 3D keypoints selection algorithm. In Table
8, we study the effect of different keypoint selection algo-
rithms. Compared with FPS that only considers the mutual
distance between keypoints, our SIFT-FPS algorithm taking
both object texture and geometry information into account
is easier to locate. Therefore, the predicted keypoint error
is smaller and the estimated poses are more accurate.
Effect of the downsample strategy of the assisting
XYZ map. The size of RGB feature maps are shrunk by
Run-time (ms/frame)
Parameters
NF
PE
All
PVN3D[17]
39.2M
170
20
190
Our FFB6D
33.8M
57
18
75
Table 9: Model parameters and run-time breakdown on the
LineMOD dataset. NF: Network Forward; PE: Pose Esti-
mation. Our FFB6D with fewer parameters is 2.5x faster.
stridden convolution kernels. To maintain the correspond-
ing XYZ maps, we ﬁrst scale it down with the same size of
mean kernels and got 96.3 ADD-S AUC. However, simply
resize the XYZ map with the nearest interpolation got 96.6.
We ﬁnd the average operation produces noise points on the
boundary and decrease the performance.
Model parameters and time efﬁciency. In Table 9, we
report the parameters and run-time breakdown of FFB6D.
Compared to PVN3D [17], which obtained the fused RGBD
feature by dense fusion modules [66] in the ﬁnal layers, our
full ﬂow bidirectional fusion network achieve better perfor-
mance with fewer parameters and is 2.5 times faster.

## experiments
4.1. Benchmark Datasets
We evaluate our method on three benchmark datasets.
YCB-Video [5] contains 92 RGBD videos that capture
scenes of 21 selected YCB objects. We followed previous
works [69, 66, 17] to split the training and testing set. Syn-
thetic images were also taken for training as in [69] and the
hole completion algorithm [30] is applied for hole ﬁlling for
depth images as in [17].
Figure 3: Performance of different approaches under in-
creasing levels of occlusion on the YCB-Video dataset.
LineMOD [19] is a dataset with 13 videos of 13 low-
textured objects. The texture-less objects, cluttered scenes,
and varying lighting make this dataset challenge.
We
split the training and testing set following previous works
[69, 47] and generate synthesis images for training follow-
ing [47, 17].
Occlusion LINEMOD [3] was selected and annotated
from the LineMOD datasets. Each scene in this dataset con-
sists of multi annotated objects, which are heavily occluded.
The heavily occluded objects make this dataset challenge.
4.2. Evaluation Metrics
We use the average distance metrics ADD and ADD-S
for evaluation. For asymmetric objects, the ADD metric
calculate the point-pair average distance between objects
vertexes transformed by the predicted and the ground truth
pose, deﬁned as follows:
ADD = 1
m
X
v∈O
||(Rv + T) −(R∗v + T ∗)||.
(6)
where v denotes a vertex in object O, R, T the predicted
pose and R∗, T ∗the ground truth. For symmetric objects,
the ADD-S based on the closest point distance is applied:
ADD-S = 1
m
X
v1∈O
min
v2∈O ||(Rv1 + T) −(R∗v2 + T ∗)||.
(7)
In the YCB-Video dataset, we follows previous methods
[69, 66, 17] and report the area under the accuracy-threshold
curve obtained by varying the distance threshold (ADD-
S and ADD(S) AUC). In the LineMOD and Occlusion
LineMOD datasets, we report the accuracy of distance less
than 10% of the objects diameter (ADD-0.1d) as in [20, 47].
5

PoseCNN [69]
PointFusion [70]
DCF [35]
DF (per-pixel) [66]
PVN3D [17]
Our FFB6D
Object
ADDS
ADD(S)
ADDS
ADD(S)
ADDs
ADD(S)
ADDS
ADD(S)
ADDS
ADD(S)
ADDS
ADD(S)
002 master chef can
83.9
50.2
90.9
-
90.9
74.6
95.3
70.7
96.0
80.5
96.3
80.6
003 cracker box
76.9
53.1
80.5
-
87.1
79.3
92.5
86.9
96.1
94.8
96.3
94.6
004 sugar box
84.2
68.4
90.4
-
94.3
84.2
95.1
90.8
97.4
96.3
97.6
96.6
005 tomato soup can
81.0
66.2
91.9
-
90.5
79.8
93.8
84.7
96.2
88.5
95.6
89.6
006 mustard bottle
90.4
81.0
88.5
-
90.6
83.5
95.8
90.9
97.5
96.2
97.8
97.0
007 tuna ﬁsh can
88.0
70.7
93.8
-
91.7
73.8
95.7
79.6
96.0
89.3
96.8
88.9
008 pudding box
79.1
62.7
87.5
-
89.3
84.1
94.3
89.3
97.1
95.7
97.1
94.6
009 gelatin box
87.2
75.2
95.0
-
92.9
89.5
97.2
95.8
97.7
96.1
98.1
96.9
010 potted meat can
78.5
59.5
86.4
-
83.2
74.6
89.3
79.6
93.3
88.6
94.7
88.1
011 banana
86.0
72.3
84.7
-
84.8
71.0
90.0
76.7
96.6
93.7
97.2
94.9
019 pitcher base
77.0
53.3
85.5
-
89.5
80.3
93.6
87.1
97.4
96.5
97.6
96.9
021 bleach cleanser
71.6
50.3
81.0
-
88.4
79.8
94.4
87.5
96.0
93.2
96.8
94.8
024 bowl
69.6
69.6
75.7
75.7
80.3
80.3
86.0
86.0
90.2
90.2
96.3
96.3
025 mug
78.2
58.5
94.2
-
90.7
76.6
95.3
83.8
97.6
95.4
97.3
94.2
035 power drill
72.7
55.3
71.5
-
87.4
78.4
92.1
83.7
96.7
95.1
97.2
95.9
036 wood block
64.3
64.3
68.1
68.1
84.2
84.2
89.5
89.5
90.4
90.4
92.6
92.6
037 scissors
56.9
35.8
76.7
-
84.2
70.3
90.1
77.4
96.7
92.7
97.7
95.7
040 large marker
71.7
58.3
87.9
-
89.5
81.0
95.1
89.1
96.7
91.8
96.6
89.1
051 large clamp
50.2
50.2
65.9
65.9
63.6
63.6
71.5
71.5
93.6
93.6
96.8
96.8
052 extra large clamp
44.1
44.1
60.4
60.4
64.4
64.4
70.2
70.2
88.4
88.4
96.0
96.0
061 foam brick
88.0
88.0
91.8
91.8
83.1
83.1
92.2
92.2
96.8
96.8
97.3
97.3
ALL
75.8
59.9
83.9
-
85.7
77.9
91.2
82.9
95.5
91.8
96.6
92.7
Table 1: Quantitative evaluation of 6D Pose without iterative reﬁnement on the YCB-Video Dataset. The ADD-S [69] and
ADD(S) [20] AUC are reported. Symmetric objects are in bold. DF (per-pixel) means DenseFusion (per-pixel).
PCNN+ICP
DF(iter.)
MoreFusion
PVN3D+ICP
FFB6D+ICP
ADD-S
93.0
93.2
95.7
96.1
97.0
ADD(S)
85.4
86.1
91.0
92.3
93.1
Table 2: Quantitative evaluation of 6D Pose with itera-
tive reﬁnement on the YCB-Video Dataset (ADD-S [69]
and ADD(S) AUC [20]). Baselines: PoseCNN+ICP [69],
DF(iter.) [66], MoreFusion [65], PVN3D+ICP [17].
4.3. Training and Implementation
Network architecture. We apply ImageNet [11] pre-
trained ResNet34 [16] as encoder of RGB images, followed
by a PSPNet [72] as decoder. For point cloud feature extrac-
tion, we randomly sample 12288 points from depth images
following [17] and applied RandLA-Net [24] for represen-
tation learning. In each encoding and decoding layers of the
two networks, max pooling and shared MLPs are applied to
build bidirectional fusion modules. After the process of the
full ﬂow bidirectional fusion network, each point has a fea-
ture fi ∈RC of C dimension. These dense RGBD features
are then fed into the instance semantic segmentation and the
keypoint offset learning modules consist of shared MLPs.
Optimization regularization.
The semantic segmen-
tation branch is supervised by Focal Loss [37]. The cen-
ter point voting and 3D keypoints voting modules are opti-
mized by L1 loss as in [17]. To jointly optimize the three
tasks, a multi-task loss with the weighted sum of them is
applied following [17].
SIFT-FPS keypoint selection algorithm. We put the
target object at the center of a sphere and sample viewpoints
of the camera on the sphere equidistantly. RGBD images
with camera poses are obtained by render engines. We then
detect 2D keypoints from RGB images with SIFT. These 2D
keypoints are lifted to 3D and transformed back to the object
coordinates system. Finally, an FPS algorithm is applied to
select N target keypoints out of them.
4.4. Evaluation on Three Benchmark Datasets.
We evaluate the proposed models on the YCB-Video, the
LineMOD, and the Occlusion LineMOD datasets.
Evaluation on the YCB-Video dataset. Table 1 shows
the quantitative evaluation results of the proposed FFB6D
on the YCB-Video dataset. We compare it with other sin-
gle view methods without iterative reﬁnement. FFB6D ad-
vances state-of-the-art results by 1.1% on the ADD-S met-
ric and 1.0% on the ADD(S) metric. Equipped with ex-
tra iterative reﬁnement, our approach also achieves the best
performance, demonstrated in Table 2. Note that the pro-
posed FFB6D without any iterative reﬁnement even outper-
forms state-of-the-arts that require time-consuming post-
reﬁnement procedures. Qualitative results are reported in
6

RGB
RGB-D
PoseCNN
DeepIM
[69, 33]
PVNet[47]
CDPN[34]
DPOD[71]
Point-
Fusion[70]
Dense-
Fusion[66]
G2L-
Net[7]
PVN3D[17]
Our
FFB6D
MEAN
88.6
86.3
89.9
95.2
73.7
94.3
98.7
99.4
99.7
Table 3: Quantitative evaluation of 6D pose on the LineMOD dataset (ADD-0.1d [20] metrics).

## related_work
2.1. Pose Estimation with RGB Data
This line of works can be divided into three classes,
holistic approaches, dense correspondence exploring, and
2D-keypoint-based. Holistic approaches [27, 15, 18, 69,
33, 64, 59, 61, 45, 55] directly output pose parameters from
RGB images. The non-linearity of rotation space limit the
generalization of these approaches. Instead, dense corre-
spondence approaches [36, 60, 14, 2, 42, 28, 12, 34, 67,
6, 6, 21, 4] ﬁnd the correspondence between image pixels
and mesh vertexes and recover poses within Perspective-n-
Point (PnP) manners. Though robust to occlusion, the large
output space limits the prediction accuracy. Instead, 2D-
keypoint-based [54, 53, 43, 29, 44, 47, 73, 38] detect 2D
keypoints of objects to build the 2D-3D correspondence for
pose estimation. However, the loss of geometry informa-
tion due to perspective projections limit the performance of
these RGB only methods.
2.2. Pose Estimation with Point Clouds
The development of depth sensors and point cloud rep-
resentation learning techniques [50, 51] motivates sev-
eral point clouds only approaches. These approaches ei-
ther utilize 3D ConvNets [57, 58] or point cloud network
[75, 48, 24] for feature extraction and 3D bounding box pre-
diction. However, sparsity and non-texture of point cloud
limit the performance of these approaches. Besides, objects
with reﬂective surfaces can not be captured by depth sen-
sors. Therefore, taking RGB images into account is neces-
sary.
2.3. Pose Estimation with RGB-D Data
Traditional methods utilize hand-coded [19, 20, 52] tem-
plates or features optimized by surrogate objectives [2,
62, 68, 28] from RGBD data and perform correspondence
grouping and hypothesis veriﬁcation. Recent data-driven
approaches propose initial pose from RGB images and re-
ﬁne it with point cloud using ICP [69] or MCN [32] algo-
rithms. However, they are time-consuming and are not end-
to-end optimizable. Instead, works like [31, 35] add appear-
ance information from CNN on RGB images as comple-
mentary information for geometry reasoning on bird-eye-
view (BEV) images of point clouds. But they neglect the
help of geometry information for RGB representation learn-
ing, the BEV ignore the pitch and roll of object pose, and
the regular 2D CNN is not good at contiguous geometry rea-
soning either. Instead, [70, 49, 66, 74, 65] extract features
2

3D Keypoints 
Detection
3D-keypoint-based 6D Pose Estimation
Instance Semantic 
Segmentation
Least-Squares 
Fitting
Full Flow Bidirectional Fusion Network
Convolution 
Layer
Point Cloud
Network Layer
Fusion from 
Point to RGB
Fusion from 
RGB to Point
RGB
Features
Point Cloud
Features
(a) The pipeline of FFB6D. A CNN and a point cloud network is utilized for representation learning of RGB image and point cloud respectively. In ﬂow
of the two networks, bidirectional fusion modules are added as communicate bridges. The extracted per-point features are then fed into an instance semantic
segmentation and a 3D keypoint voting modules to obtain per-object 3D keypoints. Finally, the pose is recovered within a least-squares ﬁtting algorithm.
XYZ Map
(b1) Pixel-to-point Fusion From RGB Feature to Point Cloud Feature
Point Cloud
RGB Feature Map
Point Cloud Feature
Kr2p nearest neighbours
Max pooling & Shared MLP
Shared MLP
Fused feature
XYZ Map
Point Cloud
RGB Feature Map
Point Cloud Feature
(b2) Point-to-pixel Fusion from Point Cloud Feature to RGB Feature
Kp2r nearest neighbours
Shared MLP & Max pooling
Shared MLP
Fused feature
(b) Dense bidirectional fusion modules. (1) The pixel-to-point fusion modules fuse RGB features to point cloud features. For each point, we ﬁnd its Kr2p
nearest neighbors in the XYZ map and gather their corresponding appearance features from the RGB feature map. These features are then processed by max
pooling and a shared MLP to obtain the most signiﬁcant appearance features. Finally, a shared MLP fuses the concatenation of the appearance and geometry
features to obtain the fused point features. (2) The point-to-pixel fusion modules similarly obtain fused pixel features as the pixel-to-point fusion.
Figure 2: Overview of FFB6D.
from RGB images and point clouds using CNN and point
cloud network individually and then fused them for pose es-
timation. Such approaches are more effective and efﬁcient.
Nevertheless, since the appearance and geometry features
are extracted separately, the two networks are not able to
communicate and share information, and thus limit the ex-
pression ability of the learned representation. In this work,
we add bidirectional fusion modules in the full network ﬂow
as communication bridges between the two networks. As-
sisted by supplementary information from another branch,
better representation of appearance and geometry features
are obtained for pose estimation.
3. Proposed Method
Given an RGBD image, the task of object 6D pose es-
timation aims to predict a transformation matrix that trans-
forms the object from its coordinate system to the camera
coordinate system. Such transformation consists of a rota-
tion matrix R ∈SO(3) and a translation matrix T ∈R3.
To tackle the problem, pose estimation algorithms should
fully explore the texture and the geometric information of
both the scene and the target object.

## conclusion
We propose a novel full ﬂow bidirectional fusion net-
work for representation learning from a single RGBD im-
age, which extract rich appearance and geometry informa-
tion in the scene for pose estimation. Besides, we intro-
duce a simple but effective SIFT-FPS keypoint selection al-
gorithms that leverage texture and geometry information of
objects to simplify keypoint localization for precise pose
estimation.
Our approach outperforms all previous ap-
proaches in several benchmark datasets by remarkable mar-
gins. Moreover, we believe the proposed full ﬂow bidirec-
tional fusion network can generalize to more applications
built on RGBD images, such as 3D object detection, 3D in-
stance semantic segmentation and salient object detection
etc. and expect to see more future research along this line.
8

A. Appendix
A.1. Details of the Network Architecture
Figure 5 shows the detailed architecture of the proposed
FFB6D. We applied ImageNet [11] pre-trained ResNet34
[16] and PSPNet [72] as encoder and decoder of the input
RGB image. Meanwhile, a RandLA-Net [24] is applied
for point cloud representation learning.
On each encod-
ing and decoding layer of the two networks, point-to-pixel
and pixel-to-point fusion modules are added for informa-
tion communication. Finally, the extracted dense appear-
ance and geometry features are concatenated and fed into
the semantic segmentation, center point voting, and 3D key-
points voting modules for pose estimation. Details of each
part are as follows:
Network Input. The input of the convolution neural net-
work (CNN) branch is a full scene image with a size of
H×W ×3, where H is the height of the RGB image, W the
width, and 3 the three channels of color information (RGB).
For the point cloud learning branch, the input is a randomly
subsampled point cloud from the scene depth image, with
a size of N × Cin, where N set to 12288 is the number
of sampled points, and Cin the input coordinate, color and
normal information of each point (x-y-z-R-G-B-nx-ny-nz).
Encoding Layers. We utilize ResNet34 [16] as the en-
coder of RGB images, which consists of ﬁve convolution
layers to reduce the size of feature maps and increase the
number of feature channels. The Pyramid Pooling Modules
(PPM) from PSPNet [72] is also applied in the last encoding
layer. Meanwhile, in the point cloud network branch, after
being processed by a fully connected layer, the point fea-
tures are fed into four encoding layers of RandLA-Net [24]
for feature encoding, each of which consists of a local fea-
ture aggregation module and a random sampling operation
designed in the work [24].
Decoding Layers.
In the decoding stage, three up-
sampling modules and a ﬁnal convolution layer from PSP-
Net are used for appearance feature decoding. Meanwhile,
in the point cloud network branch, four decoding layers
from RandLA-Net are utilized as point cloud features de-
coders, which consists of the random sampling operations
and local feature aggregation modules designed in [24].
Bidirectional Fusion Modules. On each encoding and
decoding stage, point-to-pixel and pixel-to-point fusion
modules (Section 3.2) are added for bidirectional informa-
tion communication. For each pixel-to-point fusion mod-
ule, we set Kr2p = 16 and aggregate 16 nearest pixel
of appearance features through a max-pooling and a sin-
gle layer shared MLP, MLP[cr, cp], where cr denotes the
channel size of RGB features and cp the channel size of cor-
responding point features. The aggregated pixels of appear-
ance features are then concatenated with the corresponding
point features and map by a shared MLP, MLP[2 ∗cp, cp]
to generate each fused point feature. Meanwhile, we set
Kp2r = 1 and get the fused appearance features similarly
in each point-to-pixel fusion module.
Prediction Headers. Three headers are added after the
extracted dense RGBD features to predict the semantic la-
bel, center point offset as wel as the 3D keytpoints offsets
of each point. These headers consists of shared MLPs, de-
noted as MLP[cr+cp, c1, c2, ..., ck], where cr and cp repre-
sent the channel size of extracted appearance and geometry
features respectively, and ci the output channel size of the
i-th layer in the MLP. Speciﬁcally, the semantic segmenta-
tion module consists of MLP[cr + cp, 128, 128, 128, ncls],
the center offset learning module comprises MLP[cr +
cp, 128, 128, 128, 3], and the 3D keypoints offset module is
composed of MLP[cr + cp, 128, 128, 128, nkps ∗3], where
ncls denotes number of object classes and nkps means the
number of keypoints of each object.
Pose Estimation Modules. Given the predicted seman-
tic label and center point offset of each point in the scene,
a MeanShift [10] clustering algorithm is applied to distin-
guish different object instances with the same semantic.
Then, for each instance, each point within it votes for its
3D keypoint with the MeanShift [10] algorithm. Finally,
a least-squares ﬁtting algorithm is applied to recover the
object pose parameters according to the detected 3D key-
points.
A.2. Implementation:
Different Representation
Learning Frameworks
In this section, we demonstrate the implementation de-
tails of different representation learning frameworks in Ta-
ble 4.
To implement CNN-R⊕D, we lift each pixel in
the depth image to its corresponding 3D point to get the
XYZ map as well as the normal map. We then concate-
nate them with the RGB map and feed it into a ResNet34-
PSPNet encoding-decoding network for feature extraction
of each point (pixel). For PCN-R⊕D, we append RGB val-
ues of each point to its 3D coordinate as well as its nor-
mal vector and then utilize the RandLA-Net for representa-
tion learning. The CNN-R+CNN-D utilizes two ResNet34-
PSPNet networks for feature extraction from the RGB im-
age and the XYZ and normal maps respectively. Bidirec-
tional fusion modules (Section 3.2) are added to each en-
coding and decoding layer. For PCN-R+PCN-D, we lever-
age one RandLA-Net to extract features from the RGB
value of each point and another one for representation learn-
ing of the 3D coordinate and the normal vector of each
point. In the network ﬂow, bidirectional fusion modules
are added to each layer for information communication
as well.
To implement CNN-R+3DC-D, we replace the
RandLA-Net with a 3D convolution neural network. In the
encoding stages, the voxel size decreases from 323 to 43
(323 →163 →83 →43) and the feature dimensions in-
9

(H//4, W//4, 64)
Input RGB Image 
(H, W, 3)
Input Point Cloud
(N, Cin)
(H//8,W//8,1024)
(H//8, W//8, 512)
(H//8, W//8, 128)
(H, W, 64)
(H//4, W//4, 256)
(H//2, W//2, 64)
(N//4, 64)
(N//256, 512)
(N//64, 256)
(N//16, 128)
(N//4, 64)
(N//64, 256)
(N//16, 128)
P2RF
R2PF
ConvL
LFA & RS
P2RF
R2PF
ConvL
LFA & RS
P2RF
R2PF
ConvL & PPM
LFA & RS
P2RF
R2PF
PUP
US & MLP
P2RF
R2PF
PUP
US & MLP
P2RF
R2PF
PUP
US & MLP
(H//4, W//4, 64)
(N, 8)
P2RF
R2PF
ConvL
LFA & RS
(H, W, 64)
(N, 64)
ConvL
FC
US & MLP
Conv2D
Gathering & Concatenation
(N, 128)
3D Keypoint Offsets
Center Point Offsets
Semantic Labels
Voting & Clustering
Instance Semantic Segmentation
Voting & Clustering
Per-Object 3D Keypoints
Least-Squares Fitting
6D Pose Parameters
MLPs
MLPs
MLPs
Figure 5: The detailed architecture of our FFB6D. For the convolution neural network (CNN) branch on the RGB image,
we utilize ResNet34 [16] and PSPNet [72] as encoder and decoder. ConvL: Convolution Layers of ResNet34, PPM: Pyramid
Pooling Modules of PSPNet, PUP: PSPNet Up-sampling, Conv2D: 2D convolution layer. For the point cloud network
(PCN) branch on the point cloud, we apply RandLA-Net [24] for feature extraction. FC: Fully Connected layer, LFA: Local
Feature Aggregation, RS: Random Sampling, MLP: shared Multi-Layer Perceptron, US: Up-sampling. In the ﬂow of the two
networks, point-to-pixel fusion modules, P2RF, and pixel-to-point fusion modules, R2PF consists of max pooling and shared
MLPs are added. The extracted features from the two networks are then concatenated and fed into the following semantic
segmentation, center point voting and 3D keypoints voting modules [17] composed by shared MLPs. A clustering algorithm
is then applied to distinguish different instances with the same semantic labels and points on the same instance vote for their
target keypoints. With detected 3D keypoints, a least-squares ﬁtting algorithm is applied to recover the pose parameters.
10

RGB
RGB-D
PoseCNN
DeepIM
[69, 33]
PVNet[47]
CDPN[34]
DPOD[71]
Point-
Fusion[70]
Dense-
Fusion[66]
G2L-
Net[7]
PVN3D[17]
Our
FFB6D
ape
77.0
43.6
64.4
87.7
70.4
92.3
96.8
97.3
98.4
benchvise
97.5
99.9
97.8
98.5
80.7
93.2
96.1
99.7
100.0
camera
93.5
86.9
91.7
96.1
60.8
94.4
98.2
99.6
99.9
can
96.5
95.5
95.9
99.7
61.1
93.1
98.0
99.5
99.8
cat
82.1
79.3
83.8
94.7
79.1
96.5
99.2
99.8
99.9
driller
95.0
96.4
96.2
98.8
47.3
87.0
99.8
99.3
100.0
duck
77.7
52.6
66.8
86.3
63.0
92.3
97.7
98.2
98.4
eggbox
97.1
99.2
99.7
99.9
99.9
99.8
100.0
99.8
100.0
glue
99.4
95.7
99.6
96.8
99.3
100.0
100.0
100.0
100.0
holepuncher
52.