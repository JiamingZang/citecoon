# BodyNet: Volumetric Inference of 3D Human Body Shapes

> 2018 · id: W2797515701 · arXiv: 1804.04875 · pdf: https://arxiv.org/pdf/1804.04875 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
Parsing people in visual data is central to many applications including mixed-
reality interfaces, animation, video editing and human action recognition. To-
wards this goal, human 2D pose estimation has been signiﬁcantly advanced by
recent eﬀorts [1–4]. Such methods aim to recover 2D locations of body joints and
provide a simpliﬁed geometric representation of the human body. There has also
been signiﬁcant progress in 3D human pose estimation [5–8]. Many applications,
however, such as virtual clothes try-on, video editing and re-enactment require
accurate estimation of both 3D human pose and shape.
3D human shape estimation has been mostly studied in controlled settings
using speciﬁc sensors including multi-view capture [9], motion capture mark-
ers [10], inertial sensors [11], and 3D scanners [12]. In uncontrolled single-view
settings 3D human shape estimation, however, has received little attention so
far. The challenges include the lack of large-scale training data, the high dimen-
sionality of the output space, and the choice of suitable representations for 3D
∗´Ecole normale sup´erieure, Inria, CNRS, PSL Research University, Paris, France
†Univ. Grenoble Alpes, Inria, CNRS, INPG, LJK, Grenoble, France
‡Currently at Argo AI, USA. This work was performed while EY was at Adobe.
arXiv:1804.04875v3  [cs.CV]  18 Aug 2018

2
Varol, Ceylan, Russell, Yang, Yumer, Laptev, Schmid
Fig. 1: Our BodyNet predicts a volumetric 3D human body shape and 3D body
parts from a single image. We show the input image, the predicted human voxels,
and the predicted part voxels.
human shape. Bogo et al. [13] present the ﬁrst automatic method to ﬁt a de-
formable body model to an image but rely on accurate 2D pose estimation and
introduce hand-designed constraints enforcing elbows and knees to bend natu-
rally. Other recent methods [14–16] employ deformable human body models such
as SMPL [17] and regress model parameters with CNNs [18, 19]. In this work,
we compare to such approaches and show advantages.
The optimal choice of 3D representation for neural networks remains an open
problem. Recent work explores voxel [20–23], octree [24–27], point cloud [28–30],
and surface [31] representations for modeling generic 3D objects. In the case of
human bodies, the common approach has been to regress parameters of pre-
deﬁned human shape models [14–16]. However, the mapping between the 3D
shape and parameters of deformable body models is highly nonlinear and is cur-
rently diﬃcult to learn. Moreover, regression to a single set of parameters cannot
represent multiple hypotheses and can be problematic in ambigous situations.
Notably, skeleton regression methods for 2D human pose estimation, e.g., [32],
have recently been overtaken by heatmap based methods [1, 2] enabling repre-
sentation of multiple hypotheses.
In this work we propose and investigate a volumetric representation for body
shape estimation as illustrated in Fig. 1. Our network, called BodyNet, generates
likelihoods on the 3D occupancy grid of a person. To eﬃciently train our network,
we propose to regularize BodyNet with a set of auxiliary losses. Besides the main
volumetric 3D loss, BodyNet includes a multi-view re-projection loss and multi-
task losses. The multi-view re-projection loss, being eﬃciently approximated on
voxel space (see Sec. 3.2), increases the importance of the boundary voxels. The
multi-task losses are based on the additional intermediate network supervision
in terms of 2D pose, 2D body part segmentation, and 3D pose. The overall
architecture of BodyNet is illustrated in Fig. 2.
To evaluate our method, we ﬁt the SMPL model [13] to the BodyNet output
and measure single-view 3D human shape estimation performance in the re-
cent SURREAL [33] and Unite the People [34] datasets. The proposed BodyNet
approach demonstrates state-of-the-art performance and improves accuracy of
recent methods. We show signiﬁcant improvements provided by the end-to-end
training and auxiliary losses of BodyNet. Furthermore, our method enables vol-
umetric body-part segmentation. BodyNet is fully-diﬀerentiable and could be

BodyNet: Volumetric Inference of 3D Human Body Shapes
3
2D pose loss
2D segmentation loss
3D pose loss
Volumetric loss
volumetric
shape
SMPL
ﬁt
z
x
y
Ls + L2D
j
+ L3D
j
+ Lv + LF V
p
+ LSV
p
Ls
Lv
LF V
p
L2D
j
LSV
p
L3D
j
Re-projection loss
end-to-end
optimization
Re-projection loss
Fig. 2: BodyNet: End-to-end trainable network for 3D human body shape esti-
mation. The input RGB image is ﬁrst passed through subnetworks for 2D pose
estimation and 2D body part segmentation. These predictions, combined with
the RGB features, are fed to another network predicting 3D pose. All subnet-
works are combined to a ﬁnal network to infer volumetric shape. The 2D pose,
2D segmentation and 3D pose networks are ﬁrst pre-trained and then ﬁne-tuned
jointly for the task of volumetric shape estimation using multi-view re-projection
losses. We ﬁt the SMPL model to volumetric predictions for the purpose of eval-
uation.
used as a subnetwork in future application-oriented methods targeting e.g., vir-
tual cloth change or re-enactment.
In summary, this work makes several contributions. First, we address single-
view 3D human shape estimation and propose a volumetric representation for
this task. Second, we investigate several network architectures and propose an
end-to-end trainable network BodyNet combining a multi-view re-projection loss
together with intermediate network supervision in terms of 2D pose, 2D body
part segmentation, and 3D pose. Third, we outperform previous regression-based
methods and demonstrate state-of-the art performance on two datasets for hu-
man shape estimation. In addition, our network is fully diﬀerentiable and can
provide volumetric body-part segmentation.
2

## experiments
This section presents the evaluation of BodyNet. We ﬁrst describe evaluation
datasets (Sec. 4.1) and other methods used for comparison in this paper (Sec. 4.2).
We then evaluate contributions of additional inputs (Sec. 4.3) and losses (Sec. 4.4).
Next, we report performance on the UP dataset (Sec. 4.5). Finally, we demon-
strate results for 3D body part segmentation (Sec. 4.6).
4.1
Datasets and evaluation measures
SURREAL dataset [33] is a large-scale synthetic dataset for 3D human body
shapes with ground truth labels for segmentation, 2D/3D pose, and SMPL body
parameters. Given its scale and rich ground truth, we use SURREAL in this work
for training and testing. Previous work demonstrating successful use of synthetic
images of people for training visual models include [62–64]. Given the SMPL
shape and pose parameters, we compute the ground truth 3D mesh. We use the
standard train split [33]. For testing, we use the middle frame of the middle
clip of each test sequence, which makes a total of 507 images. We observed that
testing on the full test set of 12, 528 images yield similar results. To evaluate the
quality of our shape predictions for diﬃcult cases, we deﬁne two subsets with
extreme body shapes, similar to what is done for example in optical ﬂow [65].
We compute the surface distance between the average shape (β = 0) given the
ground truth pose and the true shape. We take the 10th (s10) and 20th (s20)
percentile of this distance distribution that represent the meshes with extreme
body shapes.
Unite the People dataset (UP) [34] is a recent collection of multiple datasets
(e.g., MPII [56], LSP [66]) providing additional annotations for each image. The
annotations include 2D pose with 91 keypoints, 31 body part segments, and 3D
SMPL models. The ground truth is acquired in a semi-automatic way and is
therefore imprecise. We evaluate our 3D body shape estimations on this dataset.
We report errors on two diﬀerent subsets of the test set where 2D segmentations

10
Varol, Ceylan, Russell, Yang, Yumer, Laptev, Schmid
as well as pseudo 3D ground truth are available. We use notation T1 for images
from the LSP subset [34], and T2 for images used by [14].
3D shape evaluation. We evaluate body shape estimation with diﬀerent mea-
sures. Given the ground truth and our predicted volumetric representation, we
measure the intersection over union directly on the voxel grid, i.e., voxel IOU.
We further assess the quality of the projected silhouette to enable comparison
with [14,16,34]. We report the intersection over union (silhouette IOU), F1-score
computed for foreground pixels, and global accuracy (ratio of correctly predicted
foreground and background pixels). We evaluate the quality of the ﬁtted SMPL
model by measuring the average error in millimeters between the correspond-
ing vertices in the ﬁt and ground truth mesh (surface error). We also report
the average error between the corresponding 91 landmarks deﬁned for the UP
dataset [34]. We assume the depth of the root joint and the focal length to be
known to transform the volumetric representation into a metric space.
4.2
Alternative methods
We demonstrate advantages of BodyNet by comparing it to alternative methods.
BodyNet makes use of 2D/3D pose estimation and 2D segmentation. We deﬁne
alternative methods in terms of the same components combined diﬀerently.
SMPLify++. Lassner et al. [34] extended SMPLify [13] with an additional
term on 2D silhouette. Here, we extend it further to enable a fair comparison
with BodyNet. We use the code from [13] and implement a ﬁtting objective with
additional terms on 2D silhouette and 3D pose besides 2D pose (see Appendix D).
As shown in Tab. 2, results of SMPLify++ remain inferior to BodyNet despite
both of them using 2D/3D pose and segmentation inputs (see Fig. 3).
Shape parameter regression. To validate our volumetric representation, we
also implement a regression method by replacing the 3D shape estimation net-
work in Fig. 2 by another subnetwork directly regressing the 10-dim. shape
parameter vector β using L2 loss. The network architecture corresponds to the
encoder part of the hourglass followed by 3 additional fully connected layers (see
Input
BodyNet
Ground 
truth
Input
Shape 
parameter 
regression
SMPLify++
BodyNet
Ground 
truth
Shape 
parameter 
regression
SMPLify++
Fig. 3: SMPL ﬁt on BodyNet predictions compared with other methods. While
shape parameter regression and the ﬁtting only to BodyNet inputs (SM-
PLify++) produce shapes close to average, BodyNet learns how the true shape
observed in the image deviates from the average deformable shape model. Exam-
ples taken from the test subset s10 of SURREAL dataset with extreme shapes.

BodyNet: Volumetric Inference of 3D Human Body Shapes
11
Table 1: Performance on the SURREAL dataset using alternative combinations
of intermediate representations at the input.
voxel IOU (%)
SMPL surface error (mm)
2D pose
47.7
80.9
RGB
51.8
79.1
Segm
54.6
79.1
3D pose
56.3
74.5
Segm + 3D pose
56.4
74.0
RGB + 2D pose + Segm + 3D pose
58.1
73.6
      input             2D           3D pose    3D voxels    SMPL   Ground
     image      predictions   prediction  prediction       ﬁt          truth                          
      input                   2D            3D pose     3D voxels         SMPL        Ground
     image            predictions    prediction    prediction            ﬁt             truth             
Fig. 4: Our predicted 2D pose, segmentation, 3D pose, 3D volumetric shape, and
SMPL model alignments. Our 3D shape predictions are consistent with pose
and segmentation, suggesting that the shape network relies on the intermediate
representations. When one of the auxiliary tasks fails (2D pose on the right), 3D
shape can still be recovered with the help of the other cues.
Appendix B for details). We recover the pose parameters θ from our 3D pose
prediction (initial attempts to regress θ together with β gave worse results).
Tab. 2 demonstrates inferior performance of the β regression network that often
produces average body shapes (see Fig. 3). In contrast, BodyNet results in better
SMPL ﬁtting due to the accurate volumetric representation.
4.3
Eﬀect of additional inputs
We ﬁrst motivate our proposed architecture by evaluating performance of 3D
shape estimation in the SURREAL dataset using alternative inputs (see Tab. 1).
When only using one input, 3D pose network, which is already trained with
additional 2D pose and segmentation inputs, performs best. We observe im-
provements as more cues, speciﬁcally 3D cues are added. We also note that
intermediate representations in terms of 3D pose and 2D segmentation outper-
form RGB. Adding RGB to the intermediate representations further improves
shape results on SURREAL. Fig. 4 illustrates intermediate predictions as well
as the ﬁnal 3D shape output. Based on results in Tab. 1, we choose to use all
intermediate representations as parts of our full network that we call BodyNet.
4.4
Eﬀect of re-projection error and end-to-end multi-task training
We evaluate contributions provided by additional supervision from Sec. 3.2-3.3.
Eﬀect of re-projection losses. Tab. 2 (lines 4-10) provides results when the
shape network is trained with and without re-projection losses (see also Fig. 5).

12
Varol, Ceylan, Russell, Yang, Yumer, Laptev, Schmid
Table 2: Volumetric prediction on SURREAL with diﬀerent versions of our model
compared to alternative methods. Note that lines 2-10 use same modalities (i.e.,
2D/3D pose, 2D segmentation). The evaluation is made on the SMPL model
ﬁt to our voxel outputs. The average SMPL surface error decreases with the
addition of the proposed components.
full
s20
s10
1.
Tung et al. [15]
(using GT 2D pose and segmentation) 74.5
-
-
Alternative methods:
2.
SMPLify++ (θ, β optimized)
75.3
79.7
86.1
3.
Shape parameter regression (β regressed, θ ﬁxed)
74.3
82.1
88.7
BodyNet:
4.
Voxels network
73.6
81.1
86.3
5.
Voxels network with [FV] silhouette re-projection
69.9
76.3
81.3
6.
Voxels network with [FV+SV] silhouette re-projection
68.2
74.4
79.3
7.
End-to-end without intermediate tasks [FV]
72.7
78.9
83.2
8.
End-to-end without intermediate tasks [FV+SV]
70.5
76.9
81.3
9.
End-to-end with intermediate tasks [FV]
67.7
74.7
81.0
10.
End-to-end with intermediate tasks [FV+SV]
65.8
72.2
76.6
The voxels network without any additional loss already outperforms the base-
lines described in Sec. 4.2. When trained with re-projection losses, we observe
increasing performance both with single-view constraints, i.e., front view (FV),
and multi-view, i.e., front and side views (FV+SV). The multi-view re-projection
loss puts more importance on the body surface resulting in a better SMPL ﬁt.
Eﬀect of intermediate losses. Tab. 2 (lines 7-10) presents experimental eval-
uation of the proposed intermediate supervision. Here, we ﬁrst compare the end-
to-end network ﬁne-tuned jointly with auxiliary tasks (lines 9-10) to the networks
trained independently from the ﬁxed rep

## related_work
3D human body shape. While the problem of localizing 3D body joints has
been well-explored in the past [5–8,35–38], 3D human shape estimation from a
single image has received limited attention and remains a challenging problem.
Earlier work [39,40] proposed to optimize pose and shape parameters of the 3D
deformable body model SCAPE [41]. More recent methods use the SMPL [17]
body model that again represents the 3D shape as a function of pose and shape
parameters. Given such a model and an input image, Bogo et al. [13] present
the optimization method SMPLify estimating model parameters from a ﬁt to
2D joint locations. Lassner et al. [34] extend this approach by incorporating sil-
houette information as additional guidance and improves the optimization per-

4
Varol, Ceylan, Russell, Yang, Yumer, Laptev, Schmid
formance by densely sampled 2D points. Huang et al. [42] extend SMPLify for
multi-view video sequences with temporal priors. Similar temporal constraints
have been used in [43]. Rhodin et al. [44] use a sum-of-Gaussians volumetric
representation together with contour-based reﬁnement and successfully demon-
strate human shape recovery from multi-view videos with optimization tech-
niques. Even though such methods show compelling results, inherently they are
limited by the quality of the 2D detections they use and depend on priors both
on pose and shape parameters to regularize the highly complex and costly opti-
mization process.
Deep neural networks provide an alternative approach that can be expected
to learn appropriate priors automatically from the data. Dibra et al. [45] present
one of the ﬁrst approaches in this direction and train a CNN to estimate the 3D
shape parameters from silhouettes, but assume a frontal input view. More recent
approaches [14–16] train neural networks to predict the SMPL body parameters
from an input image. Tan et al. [14] design an encoder-decoder architecture that
is trained on silhouette prediction and indirectly regresses model parameters at
the bottleneck layer. Tung et al. [15] operate on two consecutive video frames
and learn parameters by integrating re-projection loss on the optical ﬂow, sil-
houettes and 2D joints. Similarly, Kanazawa et al. [16] predict parameters with
re-projection loss on the 2D joints and introduce an adversary whose goal is to
distinguish unrealistic human body shapes.
Even though parameters of deformable body models provide a low-dimensional
embedding of the 3D shape, predicting such parameters with a network requires
learning a highly non-linear mapping. In our work we opt for an alternative vol-
umetric representation that has shown to be eﬀective for generic 3D objects [21]
and faces [46]. The approach of [21] operates on low-resolution grayscale images
for a few rigid object categories such as chairs and tables. We argue that human
bodies are more challenging due to signiﬁcant non-rigid deformations. To accom-
modate for such deformation, we use segmentation and 3D pose as proxy to 3D
shape in addition to 2D pose [46]. Conditioning our 3D shape estimation on a
given 3D pose, the network focuses on the more complicated problem of shape
deformation. Furthermore, we regularize our voxel predictions with additional
re-projection loss, perform end-to-end multi-task training with intermediate su-
pervision and obtain volumetric body part segmentation.
Others have studied predicting 2.5D projections of human bodies. DenseReg [47]
and DensePose [48] estimate image-to-surface correspondences, while [33] out-
puts quantized depth maps for SMPL bodies. Diﬀerently from these methods,
our approach generates a full 3D body reconstruction.
Multi-task neural networks. Multi-task networks are well-studied. A com-
mon approach is to output multiple related tasks at the very end of the neural
network architecture. Another, more recently explored alternative is to stack
multiple subnetworks and provide guidance with intermediate supervision. Here,
we only cover related works that employ the latter approach. Guiding CNNs
with relevant cues has shown improvements for a number of tasks. For example,
2D facial landmarks have shown useful guidance for 3D face reconstruction [46]
and similarly optical ﬂow for action recognition [49]. However, these methods
do not perform joint training. Recent work of [50] jointly learns 2D/3D pose

BodyNet: Volumetric Inference of 3D Human Body Shapes
5
together with action recognition. Similarly, [51] trains for 3D pose with inter-
mediate tasks of 2D pose and segmentation. With this motivation, we make use
of 2D pose, 2D human body part segmentation, and 3D pose, that provide cues
for 3D human shape estimation. Unlike [51], 3D pose becomes an auxiliary task
for our ﬁnal 3D shape task. In our experiments, we show that training with a
joint loss on all these tasks increases the performance of all our subnetworks (see
Appendix C.1).
3
BodyNet
BodyNet predicts 3D human body shape from a single image and is composed
of four subnetworks trained ﬁrst independently, then jointly to predict 2D pose,
2D body part segmentation, 3D pose, and 3D shape (see Fig. 2). Here, we ﬁrst
discuss the details of the volumetric representation for body shape (Sec. 3.1).
Then, we describe the multi-view re-projection loss (Sec. 3.2) and the multi-task
training with the intermediate representations (Sec. 3.3). Finally, we formulate
our model ﬁtting procedure (Sec. 3.4).
3.1
Volumetric inference for 3D human shape
For 3D human body shape, we propose to use a voxel-based representation. Our
shape estimation subnetwork outputs the 3D shape represented as an occupancy
map deﬁned on a ﬁxed resolution voxel grid. Speciﬁcally, given a 3D body, we
deﬁne a 3D voxel grid roughly centered at the root joint, (i.e., the hip joint) where
each voxel inside the body is marked as occupied. We voxelize the ground truth
meshes (i.e., SMPL) into a ﬁxed resolution grid using binvox [52,53]. We assume
orthographic projection and rescale the volume such that the xy-plane is aligned
with the 2D segmentation mask to ensure spatial correspondence with the input
image. After scaling, the body is centered on the z-axis and the remaining areas
are padded with zeros.
Our network minimizes the binary cross-entropy loss after applying the sig-
moid function on the network output similar to [46]:
Lv =
W
X
x=1
H
X
y=1
D
X
z=1
Vxyz log ˆVxyz + (1 −Vxyz) log(1 −ˆVxyz),
(1)
where Vxyz and ˆVxyz denote the ground truth value and the predicted sigmoid
output for a voxel, respectively. Width (W), height (H) and depth (D) are 128
in our experiments. We observe that this resolution captures suﬃcient details.
The loss Lv is used to perform foreground-background segmentation of the
voxel grid. We further extend this formulation to perform 3D body part seg-
mentation with a multi-class cross-entropy loss. We deﬁne 6 parts (head, torso,
left/right leg, left/right arm) and learn 7-class classiﬁcation including the back-
ground. The weights for this network are initialized by the shape network by
copying the output layer weights for each class. This simple extension allows the
network to directly infer 3D body parts without going through the costly SMPL
model ﬁtting.

6
Varol, Ceylan, Russell, Yang, Yumer, Laptev, Schmid
3.2
Multi-view re-projection loss on the silhouette
Due to the complex articulation of the human body, one major challenge in
inferring the volumetric body shape is to ensure high conﬁdence predictions
across the whole body. We often observe that the conﬁdences on the limbs away
from the body center tend to be lower (see Fig. 5). To address this problem, we
employ additional 2D re-projection losses that increase the importance of the
boundary voxels. Similar losses have been employed for rigid objects by [54,55]
in the absence of 3D labels and by [21] as additional regularization. In our case,
we show that the multi-view re-projection term is critical, particularly to obtain
good quality reconstruction of body limbs. Assuming orthographic projection,
the front view projection, ˆSF V , is obtained by projecting the volumetric grid to
the image with the max operator along the z-axis [54]. Similarly, we deﬁne ˆSSV
as the max along the x-axis:
ˆSF V (x, y) = max
z
ˆVxyz
and
ˆSSV (y, z) = max
x
ˆVxyz.
(2)
The true silhouette, SF V , is deﬁned by the ground truth 2D body part segmen-
tation provided by the datasets. We obtain the ground truth side view silhouette
from the voxel representation that we computed from the ground truth 3D mesh:
SSV (y, z) = maxx Vxyz. We note that our voxels remain slightly larger than the
original mesh due to the voxelization step that marks every voxel that intersects
with a face as occupied. We deﬁne a binary cross-entropy loss per view as follows:
LF V
p
=
W
X
x=1
H
X
y=1
S(x, y) log ˆSF V (x, y) + (1 −S(x, y)) log(1 −ˆSF V (x, y)),
(3)
LSV
p
=
H
X
y=1
D
X
z=1
S(y, z) log ˆSSV (y, z) + (1 −S(y, z)) log(1 −ˆSSV (y, z)).
(4)
We train the shape estimation network initially with Lv. 

## conclusion
We have presented BodyNet, a fully automatic end-to-end multi-task network
architecture that predicts the 3D human body shape from a single image. We
have shown that joint training with intermediate tasks signiﬁcantly improves the
results. We have also demonstrated that the volumetric regression together with
a multi-view re-projection loss is eﬀective for representing human bodies. More-
over, with this ﬂexible representation, our framework allows us to extend our
approach to demonstrate impressive results on 3D body part segmentation from
a single image. We believe that BodyNet can provide a trainable building block
for future methods that make use of 3D body information, such as virtual cloth-
change. Furthermore, we believe exploring the limits of using only intermediate
representations is an interesting research direction for 3D tasks where acquiring
training data is impractical. Another future direction is to study the 3D body
shape under clothing. Volumetric representation can potentially capture such
additional geometry if training data is provided.
Acknowledgements. This work was supported in part by Adobe Research,
ERC grants ACTIVIA and ALLEGRO, the MSR-Inria joint lab, the Alexander
von Humbolt Foundation, the Louis Vuitton ENS Chair on Artiﬁcial Intelligence,
DGA project DRAAF, an Amazon academic research award, and an Intel gift.

BodyNet: Volumetric Inference of 3D Human Body Shapes
15