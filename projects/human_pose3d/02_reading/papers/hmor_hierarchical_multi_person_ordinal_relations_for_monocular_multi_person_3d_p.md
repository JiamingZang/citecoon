# HMOR: Hierarchical Multi-person Ordinal Relations for Monocular Multi-person 3D Pose Estimation

> 2020 · id: W3116592456 · arXiv: 2008.00206 · pdf: https://arxiv.org/pdf/2008.00206 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
Estimating 3D human poses from a monocular RGB camera is fundamental
and challenging. It has found applications in robotics [13,72], activity recogni-
tion [50,32], human-object interaction detection [15,51,28,29], and content cre-
ation for graphics [4,1]. With deep neural networks [57,19,43,44] and large scale
publicly available datasets [56,21,3,31,23,36,38,33], signiﬁcant improvement has
been achieved in the ﬁeld of 3D pose estimation. Most of the works [47,35,59,71,17,60,64,9]
focus on estimating the single-person pose. Recently, some methods [52,38,67,66,53,48,39]
⋆Denotes equal contribution.
⋆⋆Cewu Lu is the corresponding author. He is the member of Qing Yuan Research
Institute and MoE Key Lab of Artiﬁcial Intelligence, AI Institute, Shanghai Jiao
Tong University, China.
arXiv:2008.00206v2  [cs.CV]  10 Aug 2020

2
Li et al.
start to deal with multi-person cases. However, recovering absolute 3D poses in
the camera-centered coordinate system is quite a challenge. Since multi-person
activities take place in cluttered scenes, inherent depth ambiguity and occlusions
make it still diﬃcult to estimate the absolute position of multiple instances.
Recently, top-down approaches [52,53,39] achieve noticeable improvements in
estimating multi-person 3D poses. These approaches ﬁrst perform human detec-
tion and estimate the 3D pose of each person by a single-person pose estimator.
However, the pose estimator is applied to each bounding box separately, which
raises the doubt that the top-down models are not able to understand multi-
person relationships and handle complex scenes. Without a broad view of the
input scenario, it is challenging to get rid of inherent depth ambiguity and oc-
clusion problems. In this paper, the relationship among multiple persons is fully
considered to address this limitation of top-down approaches.
We propose a novel form of supervision for 3D pose estimation - Hierarchical
Multi-person Ordinal Relations (HMOR). HMOR explicitly encodes the inter-
action information as ordinal relations, supervising the networks to output 3D
poses in the correct order. Diﬀerent from previous works [46,61,54] that only
use relative depth information, HMOR considers both depths and angles rela-
tions and expresses the ordinal information hierarchically, i.e., instance →part
→joint, which makes up for the lack of a global perspective of the top-down
approaches.
Further, we propose an integrated top-down model to learn this knowledge
by encoding it into the learning process. The integrated model can be end-to-end
trained with back-propagation and performs human detection, pose estimation,
and human-depth estimation simultaneously. Since metric depth from a single
image is fundamentally ambiguous, estimating absolute 3D pose suﬀers from
inaccurate human-depth estimation. To improve the accuracy, we take a coarse-
to-ﬁne approach to estimate human depth: i) initializes a global depth map, and
ii) ﬁnetunes the human depths by estimating the correction residual.
We evaluate our method on two multi-person [38,23] and one single-person 3D
pose datasets [21]. Our method signiﬁcantly outperforms previous multi-person
3D pose estimation methods [52,38,67,37,26,39] by 12.3 PCKabs improvement on
the MuPoTS-3D [38] dataset, and 20.5 mm improvement on CMU Panoptic [23]
dataset, with lower computation complexity and fewer model parameters. Com-
pared to state-of-the-art single-person methods [17,59,60,68], our method does
not need ground-truth bounding-box in the inference phase and still achieves
comparable performance. Additionally, our proposed method is compatible with
2D pose annotations, which allows the 2D-3D mixed training strategy.
The contributions of this paper can be summarized as follows:
• We propose HMOR, a novel form of supervision, to explicitly leverage the
relationship among multiple persons for pose estimation. HMOR divides
human relations into three levels: instance, part and joint. This hierarchical
manner ensures both the global consistency and the ﬁne-grained accuracy of
the predicted results.

Hierarchical Multi-Person Ordinal Relations for 3D Pose Estimation
3
• An integrated end-to-end top-down model is proposed for multi-person 3D
pose estimation from a monocular RGB input. We design a coarse-to-ﬁne
architecture to improve the accuracy of human-depth estimation. Our model
jointly performs human detection, human-depth estimation, and 2D/3D pose
estimation.
2

## method
We propose a novel representation, Hierarchical Multi-person Ordinal Relation
(HMOR), to explicitly leverage ordinal relations among multiple persons and im-
prove the performance of 3D pose estimation. Compared with previous works [46,61,54]
that use ordinal relation in 3D pose estimation, HMOR extends this idea in
three dimensions: i) single-person to multi-persons, ii) joint level to hierarchi-
cal instance-part-joint levels, iii) depth relations to angle relations. Further, we
develop an integrated model to aggregate HMOR into the end-to-end training
process. In this section, we ﬁrst describe the uniﬁed representation of the abso-
lute multi-person 3D pose recovery under the top-down framework (§3.1). Then
we detail the encoding and training schemes of the proposed HMOR (§3.2).
Finally, the integrated model with a coarse-to-ﬁne depth estimation design is
elaborated (§3.3).
3.1
Representation
Our task is to recover multiple absolute 3D human poses P = {Pabs
m }N
m=1 in the
camera-centered coordinate system, where N denotes the number of persons in
the input RGB image. We assume that there are J joints in a single 3D pose
skeleton. The mth absolute 3D pose can be formulated as:
Pabs
m = {km,j : (xabs
m,j, yabs
m,j, zabs
m,j)T}J
j=1,
(1)

Hierarchical Multi-Person Ordinal Relations for 3D Pose Estimation
5
Input Image
Body Skeleton
θ
Part-level
Joint-level
Instance-level
HMOR
(a)
(b)
(c)
(d)
(e)
S = 14
start joint
end joint
tm,s
Fig. 1. Illustration of the proposed HMOR. (a) Deﬁnition of skeletal parts. (b) Monoc-
ular input image. (c-e) Hierarchical Multi-person Ordinal Relations. HMOR supervises
the ordinal relations among multiple persons
where km,j is the jth joint position of the mth absolute pose.
Human bounding boxes { ˆBm}N
m=1, root-relative 3D poses {ˆPrel
m }N
m=1, and
absolute depth of the root-joint {ˆzabs
m,R}N
m=1 are needed to estimate the absolute
3D poses. We term root-joint’s absolute depth as human depth, corresponding
to the pelvis bone position (the Rth joint of the body skeleton). We use ˆ to
denote the predicted values. The mth human bounding box ˆBm and root-relative
3D pose ˆPrel
m are formulated as:
ˆBm = (ˆutop
m , ˆvtop
m , ˆwm, ˆhm)T,
(2)
ˆPrel
m = {(ˆum,j, ˆvm,j, ˆzrel
m,j)T}J
j=1,
(3)
where ˆum,j and ˆvm,j represent pixel coordinates of the estimated body joint with
respect to the bounding box. ˆzrel
m,j denotes the estimated depth of joint j relative
to the root-joint. ˆutop
m , ˆvtop
m , ˆwm, and ˆhm are the top left corner coordinates,
the width, and the height of the predicted bounding box, respectively. With
the intrinsic matrix M, the ﬁnal absolute 3D pose ˆPabs
m
can be obtained via
back-projection, where each joint is calculated by:


ˆxabs
m,j
ˆyabs
m,j
ˆzabs
m,j

= (ˆzrel
m,j + ˆzabs
m,R)M−1


ˆum,j + ˆutop
m
ˆvm,j + ˆvtop
m
1

.
(4)
3.2
Hierarchical Multi-person Ordinal Relations
Our initial goal is to leverage multi-person interaction relations to improve the
performance of 3D pose estimation. Traditional top-down methods [52,53,39] lack
a global perspective because they estimate single human poses in each bound-
ing box separately. Therefore, they are vulnerable to truncation, self-occlusions,
and inter-person occlusions. Here, we develop a novel form of supervision named
Hierarchical Multi-person Ordinal Relations (HMOR) to model human relations

6
Li et al.
explicitly. Basically, given an image of human activities, we divide the relation-
ship into three levels: i) instance-level depth relations, ii) part-level angle rela-
tions, iii) joint-level depth relations. In each level, HMOR formulates pair-wise
ordinal relations and punishes the wrong-order pairs. In the following, we detail
our HMOR formulations that reﬂect interpretable relations of human activities.
Instance-Level Depth Relations. In a given camera view, for two persons
(p1, p2), we denote the instance depth-relation function as Rins(p1, p2; n⊥), tak-
ing the value:
• +1, if p1 is closer than p2 in the n⊥direction,
• −1, if p2 is closer than p1 in the n⊥direction,
• 0, if the depths of two person are equal,
where n⊥is the camera normal vector. We deﬁne the position of a person as the
arithmetic mean of its body joints, i.e. pm = 1
J
PJ
j ˆkm,j. The ordinal error of a
pair of instances is denoted as:
err ins(ˆp1, ˆp2) = log(1 + max(0, Rins(ˆp1, ˆp2; n⊥) ∗[(ˆp1 −ˆp2) · n⊥])).
(5)
This diﬀerentiable instance ranking expression will punish the wrong-order in-
stance pairs and ignore the correct results. For example, if p1 is closer than p2,
and the prediction relation is correct, i.e., (ˆp1 −ˆp2) · n⊥< 0, the multiplication
result will be smaller than 0 and ignored by the maximum operation.
Supervising the instance-level depth relations is to help the network build a
global understanding of the input scenario. Ablative study in §4.4 reveals that
the accuracy of human-depth estimation beneﬁts a lot from instance-level depth
relations.
Part-Level Angle Relations. As shown in Fig. 1(a), we divide the body
skeleton into S = 14 parts according to the kinematically connected joints.
Each part t is a vector deﬁned by start-joint kstart and end-joint kend, i.e.,
t = kend −kstart. Since body-parts are a set of 3D vectors with direction and
length values, we can not directly compare their depths. Here, we utilize a unique
attribute of body-part – direction, and compare their angle relations. To simplify
the ordinal relation of angles, we ﬁrst project the body-part vector tm,s onto the
camera plane:
tn⊥
m,s = tm,s −(tm,s · n⊥)n⊥,
(6)
where m is the person index, and s is the body-part index. In a given cam-
era view, for a pair of body parts (tm1,s1, tm2,s2), we denote the angle-relation
function as Rarg(tm1,s1, tm2,s2; n⊥), taking the value:
• +1, if Arg(tn⊥
m1,s1) < Arg(tn⊥
m2,s2),
• −1, if Arg(tn⊥
m1,s1) > Arg(tn⊥
m2,s2),
• 0, if Arg(tn⊥
m1,s1) = Arg(tn⊥
m2,s2),

Hierarchical Multi-Person Ordinal Relations for 3D Pose Estimation
7
where Arg(tn⊥) computes the principal value of the argument of the projection
vector. The ordinal error of a pair of body-parts is:
err part(ˆtm1,s1,ˆtm2,s2) = [Rarg(ˆtm1,s1,ˆtm2,s2; n⊥)∗[(ˆtm1,s1 ×ˆtm2,s2)·n⊥]]+. (7)
With the cross-product operation ×, we supervise the direction of the angle
between a pair of body-parts. If the angle between ˆtm1,s1 and ˆtm2,s2 is in the
correct direction, the projection of the cross-product (ˆtm1,s1 × ˆtm2,s2) · n⊥will
have an opposite sign of Rarg(·). Therefore, the negative multiplication results
will be ignored by the [·]+ operation.
Another intuitive way is to express body-parts as particles and supervise
their depth relations, using the average position of its two endpoints. To compare
vector and particle representations, we conduct ablative experiments and ﬁnd
out vector is superior to particle representation. We suspect this is because the
depth relations have been fully utilized in the other two levels, supervising depths
of body-part is redundant. More experimental details are reported in §4.4.
Joint-Level Depth Relations. The deﬁnition of body joint depth-relation
function Rjt(km1,s1, km2,s2; n⊥) is similar to Rins:
• +1, if km1,s1 is closer than km2,s2 in the n⊥direction,
• −1, if km2,s2 is closer than km1,s1 in the n⊥direction,
• 0, if the depths of two joints are equal.
The ordinal error of a pair of joints is denoted as:
err jt(ˆkm1,s1, ˆkm2,s2) = log(1+[Rjt(ˆkm1,s1, ˆkm2,s2; n⊥]+∗[(ˆkm1,s1−ˆkm2,s2)·n⊥])).
(8)
Denoting the set of estimated persons, body-parts, and joints pairs as Iins, Ipart,
and Ijt, respectively, the HMOR loss is computed as follows:
LHMOR =
1
|Iins|
X
ˆp1,ˆp2
err ins +
1
|Ipart|
X
ˆt1,ˆt2
err part +
1
|Ijt|
X
ˆk1,ˆk2
err jt
(9)
Augmented Training Scheme. As mentioned before, HMOR computes the
ordinal relations with respect to a vector n⊥. Initially, this vector is set as the
camera normal vector. However, we notice that annotations from 3D human pose
datasets (Human3.6M, MuPoTS-3D, and CMU Panoptic) are mostly captured
in an laboratory environment, limited to the ﬁxed viewing angle. To alleviate
camera restrictions, we sample virtual views to improve the generalization ability.
In the training phase, we generate a virtual view vector nv by rotating the
camera normal vector n⊥randomly. We adapt the uniform sphere sampling
strategy from Marsaglia et al. [34]:
nv = (
p
1 −u2 cos θ,
p
1 −u2 sin θ, u)T,
(10)
where θ ∼U[0, 2π) and u ∼U[0, 1]. In this way, HMOR can calculate the
ordinal relations with respect to an arbitrary viewing angle. The eﬀectiveness of
the sampled view is validated in §4.4.

8
Li et al.
PoseHead
Root-relative
3D Poses
RoI Features
RoIAlign
RoIAlign
RoIAlign
Initial Depth Map
locate depths of root-joints
Initial Depths
Absolute
3D Poses
Final
Depth
C
DepthHead
DetHead
Bounding
Boxes
Backbone
C Concat
Sum
LHMOR
Linit
Ldet
Lpose
Lreﬁne
Loss
Linit
LHMOR
Lreﬁne
Lpose
Ldet
∆ˆz
Fig. 2. Architecture of the integrated model. The ResNet-50 based backbone network
extract RoI features and initial depth map. PoseHead and Det

## experiments
In this section, we ﬁrst introduce the datasets employed for quantitative eval-
uation and elaborate implementation details. Then we report our results and
compare the proposed method with state-of-the-art methods. Finally, ablation
experiments are conducted to evaluate our contributions and show how each
choice contributes to our state-of-the-art performance.
4.1
Datasets
MuCo-3DHP and MuPoTS-3D:
MuCo-3DHP is a multi-person compos-
ited 3D human pose training dataset. MuPoTS-3D is the real-world scenes test
set. Following [38,39], 400K composited frames are utilized for training.
CMU Panoptic.
CMU Panoptic [23] is a multi-person 3D pose dataset cap-
tured in an indoor dome with multiple cameras. Here we follow the evaluation
protocol of [66,67].
3DPW.
3D Poses in the Wild (3DPW) [33] is a recent challenging dataset,
captured mostly in outdoor conditions. It contains 60 video sequences (24 train,
24 test, and 12 validation).
Human3.6M.
Human3.6M [21] is an indoor benchmark for single-person 3D
pose estimation. A total of 11 professional actors (6 male, 5 female) perform 15
activities in a laboratory environment.

Hierarchical Multi-Person Ordinal Relations for 3D Pose Estimation
11
Table 1. Quantitative comparisons with state-of-the-art methods on the MuPoTS-3D
dataset. “-” shows the results that are not available

## related_work
Multi-person 2D Pose Estimation.
Most of the multi-person 2D pose es-
timation methods can be divided into two categories: bottom-up and top-down
approaches. Bottom-up approaches localize the body joints and group them into
diﬀerent persons. Traditional top-down approaches ﬁrst detect human bounding
boxes in the image and then estimate single-person 2D poses separately.
Representative works [5,42,25,22] of the bottom-up approaches are reviewed.
Cao et al. [5] propose part aﬃnity ﬁelds (PAFs) to model human bones. Complete
skeletons are assembled by detected joints with PAFs. Newell et al. [42] introduce
a pixel-wise tag to assign joints to a speciﬁc person. Kocabas et al. [25] assign
joints to detected persons by a pose residual network.
Top-down approaches [16,18,10,63,62,27,58] achieve impressive accuracy in
multi-person 2D pose estimation. Mask R-CNN [18] is an end-to-end model to
estimate multiple human poses but still process multiple persons separately. Fang
et al. [16] propose a two-stage framework (RMPE) to reduce the eﬀect of the
inaccurate human detector. Sun et al. [58] propose the HRNet that maintains
high-resolution representations through the whole process.
Single-person 3D Pose Estimation.
There are two approaches to the prob-
lem of single-person 3D pose estimation from monocular RGB: single-stage and
two-stage approaches.
Single-stage approaches [47,36,59,24,60] directly locate 3D human joints from
the input image. For example, Pavlakos et al. [47] propose a coarse-to-ﬁne ap-
proach to estimate a 3D heatmap for pose estimation. Kanazawa et al. [24] re-
cover 3D pose and body mesh by minimizing the reprojection loss. Sun et al. [60]
operate an integral operation as soft-argmax to obtain 3D pose coordinates in a
diﬀerentiable manner.
Two-stage approaches [2,45,65,7,40,35,71,17,64] ﬁrst estimate 2D pose or uti-
lize the oﬀ-the-shelf accurate 2D pose estimator, and then lift them to the 3D
space. Martinez et al. [35] propose a simple baseline to regress 3D pose from 2D
coordinates directly. Moreno-Noguer [40] obtains more precise pose estimation
by the distance matrix representation. Yang et al. [64] utilize a multi-source
discriminator to generate anthropometrically valid poses.
Multi-person 3D Pose Estimation.
A few works explore the problem of
multi-person 3D pose estimation from a monocular RGB. Rogez et al. [52,53]
propose LCR-Net and LCR-Net++. They locate human bounding boxes and

4
Li et al.
classify those boxes into a set of K anchor-poses. A regression module is pro-
posed to reﬁne the anchor-pose to the ﬁnal prediction. Instead of using a learning-
based manner, they obtain the human depth by minimizing the distance between
the projected 3D pose and the estimated 2D pose. Mehta et al. [38] propose
a bottom-up method. Their proposed occlusion-robust pose-map (ORPM) en-
ables full body pose inference even under strong partial occlusions. Zanﬁr et
al. [67] propose MubyNet, a bottom-up model. MubyNet integrates a limb scor-
ing model and formulates the person grouping problem as an integer program.
Moon et al. [39] propose a top-down two-stage model. They utilize the oﬀ-the-
shelf human detection model and then perform single-person 3D pose estimation
and root-joint localization. Those top-down approaches are not able to utilize
multi-person relations since they estimate individual 3D pose separately. The
bottom-up approaches are still suﬀering from limited accuracy. Our method
combines the advantages of both approaches and boosts multi-person absolute
3D pose estimation by leveraging the multi-person relations in the integrated
end-to-end top-down model.
Ordinal Relations.
In the context of computer vision, several works learn
ordinal apparent depth [73,8] or reﬂectance [41,69] relationship as weak supervi-
sion. They motivated by the fact that ordinal relations are easier for humans to
annotate. In the case of single-person 3D pose estimation, [46,54,55] use depth
relations of body joints to generate 3D pose from 2D pose.
3

## conclusion
In this paper, we proposed a novel form of supervision - HMOR, to learn multi-
person 3D poses from a monocular RGB image. HMOR supervises the multi-
person ordinal relations in a hierarchical manner, which captures ﬁne-grained
semantics and maintains global consistency at the same time. To end-to-end
learn the ordinal relations, we further proposed an integrated model with a
coarse-to-ﬁne depth-estimation architecture. We demonstrate the eﬀectiveness of
our proposed method on standard benchmarks. The proposed method surpasses
state-of-the-art multi-person 3D pose estimation methods, with lower computa-
tion complexity and fewer model parameters. We believe the idea of leveraging
multi-person relations can be further explored to improve 3D pose estimation,
e.g., exploit the relations via network design.
Acknowledgements. This work is supported in part by the National Key R&D
Program of China, No. 2017YFA0700800, National Natural Science Foundation
of China under Grants 61772332Shanghai Qi Zhi Institute.

Hierarchical Multi-Person Ordinal Relations for 3D Pose Estimation
15