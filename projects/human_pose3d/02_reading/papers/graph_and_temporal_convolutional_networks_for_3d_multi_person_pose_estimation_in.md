# Graph and Temporal Convolutional Networks for 3D Multi-person Pose Estimation in Monocular Videos

> 2021 · id: W3177383258 · arXiv: 2012.11806 · pdf: https://ojs.aaai.org/index.php/AAAI/article/download/16202/16009 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Despite the recent progress, 3D multi-person pose estimation
from monocular videos is still challenging due to the com-
monly encountered problem of missing information caused
by occlusion, partially out-of-frame target persons, and inac-
curate person detection. To tackle this problem, we propose
a novel framework integrating graph convolutional networks
(GCNs) and temporal convolutional networks (TCNs) to ro-
bustly estimate camera-centric multi-person 3D poses that
does not require camera parameters. In particular, we intro-
duce a human-joint GCN, which unlike the existing GCN,
is based on a directed graph that employs the 2D pose es-
timator’s conﬁdence scores to improve the pose estimation
results. We also introduce a human-bone GCN, which mod-
els the bone connections and provides more information be-
yond human joints. The two GCNs work together to esti-
mate the spatial frame-wise 3D poses, and can make use of
both visible joint and bone information in the target frame
to estimate the occluded or missing human-part information.
To further reﬁne the 3D pose estimation, we use our tempo-
ral convolutional networks (TCNs) to enforce the temporal
and human-dynamics constraints. We use a joint-TCN to es-
timate person-centric 3D poses across frames, and propose a
velocity-TCN to estimate the speed of 3D joints to ensure the
consistency of the 3D pose estimation in consecutive frames.
Finally, to estimate the 3D human poses for multiple per-
sons, we propose a root-TCN that estimates camera-centric
3D poses without requiring camera parameters. Quantitative
and qualitative evaluations demonstrate the effectiveness of
the proposed method. Our code and models are available at
https://github.com/3dpose/GnTCN.

## introduction
Signiﬁcant progress has been made in 3D human pose es-
timation in recent years, e.g. (Sun et al. 2019a; Pavllo et al.
2019; Cheng et al. 2019, 2020). In general, existing methods
can be classiﬁed as either top-down or bottom-up. Top-down
approaches use human detection to obtain the bounding box
of each person, and then perform pose estimation for every
person. Bottom-up approaches are human-detection free and
can estimate the poses of all persons simultaneously. Top-
down approaches generally demonstrate more superior per-
formance in pose estimation accuracy, and are suitable for
Copyright © 2021, Association for the Advancement of Artiﬁcial
Intelligence (www.aaai.org). All rights reserved.
Frame 140
Frame 150
Frame 180
Person-centric

## method
The overview of our framework is shown as Fig. 2. Having
obtained the 2D poses from the 2D pose estimator, the poses
are normalized so that they are centered at the root point,
which is at the hip of human body. Each pose is then fed
into our joint- and bone-GCNs to obtain its 3D full pose,
despite the input 2D pose might be incomplete. Finally, a 3D
full pose sequence is fed into the joint-TCN, root-TCN, and
velocity-TCN to obtain the camera-centric 3D human poses
that have smooth motion and comply with natural human
dynamics.
Joint-GCN and Bone GCN
Existing top-down methods are erroneous when the target
human bounding box is incorrect, due to missing informa-
tion (occlusion, partially out-of-frame, blur, etc.). To address
this common problem, we introduce joint-GCN and bone-
GCN that can correct the 3D poses from inaccurate 2D pose
estimator. These GCNs work on a frame-by-frame basis.
Following the structure of the human body, we assign the
coordinates (xi, yi) of the human joints from the 2D pose
estimator to each vertex of our graph, and establish connec-
tions between each pair of the joints. Unlike most GCNs,
which are based on an undirected graph, we propose a GCN
based on a directed graph. The directed graph allows us to
propagate information more from high-conﬁdent joints to
low-conﬁdent ones, and thus reduces the risk of propagat-
ing erroneous information (e.g., occluded joints or missing
joints) in the graph. In other words, the low-conﬁdent joints
contribute less to the message propagation than the high-
conﬁdent ones. Details of the directed graph are available in
the supplementary material.
The joint-GCN uses the 2D joints as the vertices and the
conﬁdence scores of the 2D joints as the edge weights, while
the bone-GCN uses the conﬁdence scores of part afﬁnity
ﬁeld (Cao et al. 2017)) as the edge weights. The features
produced by the two GCNs are concatenated together and
fed to a Multi Layer Perceptron to obtain the person-centric
3D pose estimation.
In GCNs, the message is propagated according to adja-
cent matrix, which indicates the edge between each pair of
vertices. The adjacency matrix is formed by the following
rule:
Ai,j =
max(Hi)e−order(i,j)(i̸ = j)
max(Hi)(i = j)
,
(1)
where H is the heatmap from the 2D pose estimator.
order(i, j) stands for the number of the order of neighbor-
ing vertices, which means the number of hops required to
reach vertex j from vertex i. This formation of adjacency
imposes more weight for close vertices and less for distance
vertices.
The forward propagation of each GCN layer can be ex-
pressed as:
hi = σ(F(hi−1)Wi),
(2)
where F is the feature transformation function, and W is
the learnable parameter of layer i. To learn a network with
strong generalization ability, we follow the idea of Graph

SAGE (Hamilton, Ying, and Leskovec 2017) to learn a gen-
eralizable aggregator, which is formulated as:
F(hi) = eAhi ⊕hi,
(3)
where hi is the output of layer i in the GCN and ⊕stands for
the concatenation operation. eA is the normalized adjacency
matrix. Since our method is based on a directed graph, which
uses a non-symmetric adjacency matrix, the normalization is
eAi,j = Ai,j
Dj instead of eAi,j =
Ai,j
√
DiDj in (Kipf and Welling
2016), Di and Dj are the indegree of vertices i and j, re-
spectively. This normalization ensures that the indegree of
each vertex sums to 1, which prevents numerical instability.
Our joint-GCN considers only human-joints and does not
include the information of bones, which can be critical for
the cases when the joints are missing due to occlusion or
other reasons. To exploit the bone information, we created
a bone-GCN. First, we construct the incidence matrix In of
shape [#bones, #joints] to represent the bone connections,
where each row represents an edge and the columns repre-
sent vertices. For each bone, the parent joint is assigned with
−1 and the child joint is assigned with 1. Second, the inci-
dence matrix In is multiplied with the joint matrix J to ob-
tain the bone matrix B, which will be further fed into our
bone-GCN.
In joint matrix J, each row stands for the 2D coordi-
nate (x, y) of a joint. Unlike our joint-GCN, where the ad-
jacency matrix is drawn from the joint heatmap produced
by 2D pose estimator, our human-bone GCN utilizes the
conﬁdence scores from the part afﬁnity ﬁeld, following the
method of (Cao et al. 2017), as the adjacency. Finally, the
outputs from our human-joint GCN and human-bone GCN
are concatenated together and fed into an MLP (Multi-layer
Perceptron). The loss function we use is the L2 loss between
GCN 3D joints output PGCN and 3D ground-truth skeleton
eP, which is LGCN = || eP −PGCN||2
2.
In the training stage, to obtain sufﬁcient variation and to
increase the robustness of our GCNs, we use not only the
results from our 2D pose estimator, but also augmented data
from our ground-truths. Each joint is assigned with a random
conﬁdence score and random noise.
Root-TCN
In most of the videos, the projection can be modelled as
weak perspective:
" x
y
1
#
= 1/Z
"f
0
cx
0
f
cy
0
0
1
# " X
Y
Z
#
,
(4)
where x and y are the image coordinates, X, Y and Z are
the camera coordinates. f, cx, cy stands for the focal length
and camera centers, respectively. Thus we have:
X = Z
f (x −cx)
Y = Z
f (y −cy).
(5)
By assuming (cx, cy) as the image center, which is appli-
cable for most cameras, the only parameters we need to esti-
mate is depth Z and focal length f. To be more practical, we
jointly estimate Z/f, instead of estimating them separately.
This enables our method to be able to take wild videos that
the camera parameters are unknown.
According to the weak perspective assumption, the scale
of a person in a frame indicates the depth in the camera coor-
dinates. Hence, we propose a network, root Temporal Con-
volutional Network (root-TCN), to estimate the Z/f from
2D pose sequences. We ﬁrst normalize each 2D pose by scal-
ing the average joint-to-pelvis distance to 1, using a scale
factor s. Then we concatenate the normalized pose p, scale
factor s, as well as the person’s center in the frame as c, and
feed a list of such concatenated features in a local tempo-
ral window into the TCN for depth estimation in the camera
coordinates.
As directly learning Z/f is not easy to converge, we trans-
form this regression problem into a classiﬁcation problem.
For each video, we divide the depth into N discrete ranges,
set to 60 in our experiments, and our root-TCN outputs a
vector with length N as {x1, ..., xN}, where xi indicates the
probability that Z/f is within the ith discrete range. Then,
we apply Soft-argmax to this vector to get the ﬁnal continu-
ous estimation of the depth as:
[Z
f ]t = Soft-argmax ( fR( pt−n:t+n, ct−n:t+n, st−n:t+n)),
(6)
where t is the time stamp, and n is half of the temporal win-
dow’s size. This improves the training stability and reduces
the risk of large errors.
The loss function for the depth estimation is deﬁned as
the mean squared error between the ground truth and pre-
dictions, expressed as LRoot = ( Z
f −
ˆ
Z
ˆ
f )2, where Z/f is
the predicted value, and ˆZ/ ˆf denotes the ground truth. Ac-
cording to Eq.(5), we can calculate the coordinates for the
person’s center as P t
D.
Joint-TCN and Velocity-TCN
To increase the accuracy of the 3D poses across the input
video, we impose temporal constraints, by employing a tem-
poral convolutional network (TCN) (Cheng et al. 2020) that
takes a sequence of consecutive 3D poses as input. We call
this TCN a joint-TCN, which is trained using various 3D
poses and their augmentation, and hence capture human dy-
namics. The joint-TCN outputs the person-centric 3D pose,
PD. The TCN utilizes temporal information to interpolate
the poses of occluded frames with temporal information.
However, when persons get close and occlude each other,
there may be fewer visible joints belonging to a person
and more distracting joints from other persons. To resolve
the problem, in addition to the joint-TCN, we propose
a velocity-based estimation network, velocity-TCN, which
takes the 3D joints and their velocities as input, and predicts
the velocity of all joints as:
V t = (vt
x, vt
y, vt
z) = TCNv(pt−n:t−1, V t−n:t−1),
(7)
where p stands for the 2D pose and V t denotes the veloc-
ity at time t. TCNv is the velocity-TCN. The velocity here

is proportional to 1/f according to Eq. (5). We normalize
the velocity both in training and testing. With estimated V t,
we can obtain the coordinate P t
S = P t−1 + V t, where P t
S
and P t−1 are estimated coordinates at time t and t −1. The
calculation of P t−1 is discussed later in Eq.(8).
The joint-TCN predicts the joints by interpolating the past
and future poses, while our velocity-TCN predicts the future
poses using motion cues. Both of them are able to handle
the occlusion frames, but the joint-TCN focuses on the con-
nection between past and future frames regardless of the tra-
jectory, while the velocity-TCN focuses on the motion pre-
diction, which ca

## experiments
MuPoTS-3D is a 3D multi-person testing set with both in-
door and outdoor scenes (Mehta et al. 2018). The ground-
truth 3D pose of each person in the video is obtained from
multi-view markerless capture, which is suitable for evalu-
ating 3D multi-person pose estimation performance in both
person-centric and camera-centric coordinates. Unlike pre-
vious methods (Moon, Chang, and Lee 2019) using the train-
ing set (MuCo-3DHP) to train their models and then do eval-
uation on MuPoTS-3D, we use MuPoTS-3D for testing only
without ﬁne-tuning.
3DPW is an outdoor multi-person dataset for 3D human
pose reconstruction (von Marcard et al. 2018). Following
previous methods (Kanazawa et al. 2019; Sun et al. 2019b),
we use 3DPW for testing only without any ﬁne-tuning. The
ground-truth of 3DPW is SMPL 3D mesh model (Loper
et al. 2015), where the deﬁnition of joints differs from the
one commonly used in 3D human pose estimation (skeleton-
based) like Human3.6M (Tripathi et al. 2020), so it is un-
fair to evaluate skeleton-based methods on it even after joint
adaption or scaling. To perform a fair comparison, we select
an occlusion subset from the 3DPW test set (please refer
to the supplementary material for details). And the perfor-
mance change of a method between the full test set and the
subset indicates how well the method can handle the missing
information problem caused by occlusions.

## related_work
3D human pose estimation in video
Recent 3D hu-
man pose estimation methods utilize temporal informa-
tion via recurrent neural network (RNN) (Hossain and Lit-
tle 2018; Lee, Lee, and Lee 2018; Chiu et al. 2019) or
TCN (Pavllo et al. 2019; Cheng et al. 2019; Sun et al. 2019b;
Cheng et al. 2020) improve the temporal consistency and
show promising results on single-person video datasets such
as HumanEva-I, Human3.6M, and MPI-INF-3DHP (Sigal,
Balan, and Black 2010; Ionescu et al. 2014; Mehta et al.
2017a), but they still suffer from the inter-person occlu-
sion issue when applying to multi-person videos. Although
a few works take occlusion into account (Ghiasi et al. 2014;
Charles et al. 2016; Belagiannis and Zisserman 2017; Cheng
et al. 2019, 2020), in a top-down framework, it is difﬁcult to
reliably estimation 3D multi-person human poses in videos
due to erroneous detection and occlusions. Moreover, none
of these method estimate camera-centric 3D human poses.
Monocular 3D human pose estimation
Earlier ap-
proaches that tackle camera-centric 3D human pose from
monocular camera require camera parameters as input or
assume ﬁxed camera pose to project the 2D posture into
camera-centric coordinate (Mehta et al. 2017b, 2019; Pavllo
et al. 2019). As a result, these methods are inapplicable for
wild videos where camera parameters are not available. Re-
moving the requirement of camera parameters has drawn re-
searcher’s attention recently. Moon et al. (Moon, Chang, and
Lee 2019) ﬁrst propose to learn a correction factor for a per-
son’s root depth estimation from a single image. Several re-
cent works (Li et al. 2020; Lin and Lee 2020; Zhen et al.
2020) show improved performance compared with (Moon,
Chang, and Lee 2019). Li et al. (Li et al. 2020) develop
an integrated method for detection, person-centric pose, and
depth estimation from a single image. Lin et al. (Lin and
Lee 2020) propose to formulate the depth regression as a
bin index estimation problem. Zhen et al. (Zhen et al. 2020)
propose to estimate 2.5D representation of body parts ﬁrst
and then reconstruct 3D human pose. Unlike their approach,
our method is video-based where temporal information is
utilized by TCN on top of GCN output, which leads to im-
proved 3D pose estimation.

Joints TCN
𝑁𝑜𝑑𝑒= 𝐽
𝑎𝑑𝑗= max (𝐻)
Person-centric 
3D Pose: 𝑃1
2
Root depth:
𝑍𝑓
⁄
Velocity: 
𝑉2
TCNs
Human
Joint-GCN 
MLP
Human 
Bone-GCN 
𝑁𝑜𝑑𝑒= 𝐼𝑛𝑐: 𝐽
𝑎𝑑𝑗= PAF (𝑝?, 𝑝A)
Inc
Inc-1
Root TCN
Velocity TCN
GCNs
Input Frame
Result sequence
Time
…
…
…
Time
PAF
Heatmap’s
confidence
Persons
…
…
𝑃B
2 = 𝑃2CD + 𝑉2
𝑃2CD
𝑃2
𝑃2FD
𝑃2 = 𝑤2𝑃1
2 +
1 −𝑤2 𝑃B
2
Persons
Persons
Figure 2: The framework of our approach. The 2D poses and part afﬁnity ﬁeld for each bounding box are fed into our joint- and
bone-GCNs to obtain the full 3D poses (left). After obtaining all poses in the video, they are grouped by IDs which is provided
by pose tracker, and fed into the the joint-, root- and velocity-TCN to obtain the camera-centric 3D pose estimation (right).
GCN for pose estimation
Graph convolutional network
(GCN) has been applied to 2D or 3D human pose estimation
in recent years (Zhao et al. 2019; Cai et al. 2019; Ci et al.
2019; Qiu et al. 2020). Zhao et al. (Zhao et al. 2019) pro-
pose a graph neural network architecture to capture local and
global node relationship and apply the proposed GCN for
single-person 3D pose estimation from image. Ci et al (Ci
et al. 2019) explore different network structures by compar-
ing fully connected network and GCN and develop a locally
connected network to improve the representation capabil-
ity for single-person 3D human pose estimation from image
as well. Cai et al. (Cai et al. 2019) construct an undirected
graph to model the spatial-temporal dependencies between
different joints for single-person 3D pose estimation from
video data. Qiu et al. (Qiu et al. 2020) develop a dynamic
GCN framework for multi-person 2D pose estimation from
a image. Our method is different from all these methods in
terms of we propose to use directed graph to incorporate
heatmap and part afﬁnity ﬁeld conﬁdence in graph construc-
tion, which brings the beneﬁt of overcoming the limitation
of human detection on top-down pose estimation methods.

## conclusion
We propose a new framework to unify GCNs and TCNs
for camera-centric 3D multi-person pose estimation. The
proposed method successfully handles missing information
due to occlusion, out-of-frame, inaccurate detections, etc.,
in videos and produces continuous pose sequences. Experi-
ments on different datasets validate the effectiveness of our
framework as well as our individual modules.

Acknowledgements
This research/project is supported by the National Research
Foundation, Singapore under its Strategic Capability Re-
search Centres Funding Initiative. Any opinions, ﬁndings
and conclusions or recommendations expressed in this ma-
terial are those of the author(s) and do not reﬂect the views
of National Research Foundation, Singapore.