# End-to-End Human Pose and Mesh Reconstruction with Transformers

> 2021 · id: W3175199633 · arXiv: 2012.09760 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
We present a new method, called MEsh TRansfOrmer
(METRO), to reconstruct 3D human pose and mesh ver-
tices from a single image. Our method uses a transformer
encoder to jointly model vertex-vertex and vertex-joint in-
teractions, and outputs 3D joint coordinates and mesh ver-
tices simultaneously. Compared to existing techniques that
regress pose and shape parameters, METRO does not rely
on any parametric mesh models like SMPL, thus it can be
easily extended to other objects such as hands.
We fur-
ther relax the mesh topology and allow the transformer
self-attention mechanism to freely attend between any two
vertices, making it possible to learn non-local relationships
among mesh vertices and joints. With the proposed masked
vertex modeling, our method is more robust and effective
in handling challenging situations like partial occlusions.
METRO generates new state-of-the-art results for human
mesh reconstruction on the public Human3.6M and 3DPW
datasets. Moreover, we demonstrate the generalizability of
METRO to 3D hand reconstruction in the wild, outperform-
ing existing state-of-the-art methods on FreiHAND dataset.
Code and pre-trained models are available at https:
//github.com/microsoft/MeshTransformer.

## introduction
3D human pose and mesh reconstruction from a single
image has attracted a lot of attention because it has many
applications including virtual reality, sports motion analy-
sis, neurodegenerative condition diagnosis, etc. It is a chal-
lenging problem due to complex articulated motion and oc-
clusions.
Recent work in this area can be roughly divided into two
categories. Methods in the ﬁrst category use a parametric
model like SMPL [29] and learn to predict shape and pose
coefﬁcients [14, 26, 39, 22, 24, 34, 44, 23]. Great success
has been achieved with this approach. The strong prior en-
coded in the parametric model increases its robustness to
environment variations. A drawback of this approach is that
the pose and shape spaces are constrained by the limited ex-
(a)
(b)
(c)
Figure 1:
METRO learns non-local interactions among
body joints and mesh vertices for human mesh reconstruc-
tion. Given an input image in (a), METRO predicts hu-
man mesh by taking non-local interactions into consider-
ation. (b) illustrates the attentions between the occluded
wrist joint and the mesh vertices where brighter color indi-
cates stronger attention. (c) is the reconstructed mesh.
emplars that are used to construct the parametric model. To
overcome this limitation, methods in the second category do
not use any parametric models [25, 8, 32]. These methods
either use a graph convolutional neural network to model
neighborhood vertex-vertex interactions [25, 8], or use 1D
heatmap to regress vertex coordinates [32]. One limitation
with these approaches is that they are not efﬁcient in mod-
eling non-local vertex-vertex interactions.
Researchers have shown that there are strong correla-
tions between non-local vertices which may belong to dif-
ferent parts of the body (e.g.
hand and foot) [55].
In
computer graphics and robotics, inverse kinematics tech-
niques [2] have been developed to estimate the internal joint
positions of an articulated ﬁgure given the position of an
end effector such as a hand tip. We believe that learning
the correlations among body joints and mesh vertices in-
cluding both short range and long range ones is valuable for
handling challenging poses and occlusions in body shape
reconstruction. In this paper, we propose a simple yet effec-
tive framework to model global vertex-vertex interactions.
The main ingredient of our framework is a transformer.
Recent studies show that transformer [53] signiﬁcantly
improves the performance on various tasks in natural lan-
1
arXiv:2012.09760v3  [cs.CV]  15 Jun 2021

guage processing [4, 9, 40, 41]. The success is mainly at-
tributed to the self-attention mechanism of a transformer,
which is particularly effective in modeling the dependen-
cies (or interactions) without regard to their distance in both
inputs and outputs. Given the dependencies, transformer is
able to soft-search the relevant tokens and performs predic-
tion based on the important features [4, 53].
In this work, we propose METRO, a multi-layer Trans-
former encoder with progressive dimensionality reduction,
to reconstruct 3D body joints and mesh vertices from a
given input image, simultaneously. We design the Masked
Vertex Modeling objective with a transformer encoder ar-
chitecture to enhance the interactions among joints and ver-
tices. As shown in Figure 1, METRO learns to discover both
short- and long-range interactions among body joints and
mesh vertices, which helps to better reconstruct the 3D hu-
man body shape with large pose variations and occlusions.
Experimental results on multiple public datasets demon-
strate that METRO is effective in learning vertex-vertex and
vertex-joint interactions, and consequently outperforms the
prior works on human mesh reconstruction by a large mar-
gin. To the best of our knowledge, METRO is the ﬁrst ap-
proach that leverages a transformer encoder architecture to
jointly learn 3D human pose and mesh reconstruction from
a single input image. Moreover, METRO is a general frame-
work which can be easily applied to predict a different 3D
mesh, for example, to reconstruct a 3D hand from an input
image.
In summary, we make the following contributions.
• We introduce a new transformer-based method, named
METRO, for 3D human pose and mesh reconstruction
from a single image.
• We design the Masked Vertex Modeling objective
with a multi-layer transformer encoder to model both
vertex-vertex and vertex-joint interactions for better re-
construction.
• METRO achieves new state-of-the-art performance on
the large-scale benchmark Human3.6M and the chal-
lenging 3DPW dataset.
• METRO is a versatile framework that can be easily re-
alized to predict a different type of 3D mesh, such as
3D hand as demonstrated in the experiments. METRO
achieves the ﬁrst place on FreiHAND leaderboard at
the time of paper submission.

## method
Figure 2 is an overview of our proposed framework. It
takes an image of size 224 × 224 as input, and predicts a
set of body joints J and mesh vertices V . The proposed
framework consists of two modules: Convolutional Neural
Network, and Multi-Layer Transformer Encoder. First, we
use a CNN to extract an image feature vector from an input
image. Next, Multi-Layer Transformer Encoder takes as
input the feature vector and outputs the 3D coordinates of
the body joint and mesh vertex in parallel. We describe each
module in details as below.
3.1. Convolutional Neural Network
In the ﬁrst module of our framework, we employ a
Convolutional Neural Network (CNN) for feature extrac-
tion. The CNN is pre-trained on ImageNet classiﬁcation
task [45]. Speciﬁcally, we extract a feature vector X from
the last hidden layer. The extracted feature vector X is typ-
ically of dimension 2048. We input the feature vector X to
the transformer for the regression task.
With this generic design, it allows an end-to-end train-
ing for human pose and mesh reconstruction. Moreover,
transformer can easily beneﬁt from large-scale pre-trained
CNNs, such as HRNets [56]. In our experiments, we con-
duct analysis on the input features, and discover that high-
resolution image features are beneﬁcial for transformer to
regress 3D coordinates of body joints and mesh vertices.
3.2. Multi-Layer Transformer Encoder with Pro-
gressive Dimensionality Reduction
Since we need to output 3D coordinates, we cannot
directly apply the existing transformer encoder architec-
ture [10, 6] because they use a constant dimensionality
of the hidden embeddings for all the transformer layers.
Inspired by [18] which performs dimentionality reduction
gradually with multiple blocks, we design a new architec-
ture with a progressive dimensionality reduction scheme.
As shown in Figure 2 right, we use linear projections to
reduce the dimensionality of the hidden embedding after
each encoder layer. By adding multiple encoder layers, the
model is viewed as performing self-attentions and dimen-
sionality reduction in an alternating manner. The ﬁnal out-
put vectors of our transformer encoder are the 3D coordi-
nates of the joints and mesh vertices.
As illustrated in Figure 2 left, the input to the transformer
encoder are the body joint and mesh vertex queries. In the
same spirit as positional encoding [53, 25, 13], we use a
3

template human mesh to preserve the positional information
of each query in the input sequence. To be speciﬁc, we
concatenate the image feature vector X ∈R2048×1 with
the 3D coordinates (xi, yi, zi) of every body joint i. This
forms a set of joint queries QJ = {qJ
1 , qJ
2 , . . . , qJ
n}, where
qJ
i ∈R2051×1. Similarly, we conduct the same positional
encoding for every mesh vertex j, and form a set of vertex
queries QV = {qV
1 , qV
2 , . . . , qV
m}, where qV
j ∈R2051×1.
3.3. Masked Vertex Modeling
Prior works [9, 49] use the Masked Language Model-
ing (MLM) to learn the linguistic properties of a training
corpus. However, MLM aims to recover the inputs, which
cannot be directly applied to our regression task.
To fully activate the bi-directional attentions in our trans-
former encoder, we design a Masked Vertex Modeling
(MVM) for our regression task. We mask some percentages
of the input queries at random. Different from recovering
the masked inputs like MLM [9], we instead ask the trans-
former to regress all the joints and vertices.
In order to predict an output corresponding to a missing
query, the model will have to resort to other relevant queries.
This is in spirit similar to simulating occlusions where par-
tial body parts are invisible. As a result, MVM enforces
transformer to regress 3D coordinates by taking other rel-
evant vertices and joints into consideration, without regard
to their distances and mesh topology. This facilitates both
short- and long-range interactions among joints and vertices
for better human body modeling.
3.4. Training
To train the transformer encoder, we apply loss functions
on top of the transformer outputs, and minimize the errors
between predictions and ground truths.
Given a dataset
D = {Ii, ¯V i
3D, ¯Ji
3D, ¯Ji
2D}T
i=1, where T is the total num-
ber of training images. I ∈Rw×h×3 denotes an RGB im-
age.
¯V3D ∈RM×3 denotes the ground truth 3D coordi-
nates of the mesh vertices and M is the number of vertices.
¯J3D ∈RK×3 denotes the ground truth 3D coordinates of
the body joints and K is the number of joints of a person.
Similarly, ¯J2D ∈RK×2 denotes the ground truth 2D coor-
dinates of the body joints.
Let V3D denote the output vertex locations, and J3D is
the output joint locations, we use L1 loss to minimize the
errors between predictions and ground truths:
LV = 1
M
M
X
i=1
V3D −¯V3D

1 ,
(1)
LJ = 1
K
K
X
i=1
J3D −¯J3D

1 .
(2)
It is worth noting that, the 3D joints can also be cal-
culated from the predicted mesh. Following the common
practice in literature [8, 22, 25, 24], we use a pre-deﬁned
regression matrix G ∈RK×M, and obtain the regressed 3D
joints by Jreg
3D = GV3D. Similar to prior works, we use L1
loss to optimize Jreg
3D :
Lreg
J
= 1
K
K
X
i=1
Jreg
3D −¯J3D

1 .
(3)
2D re-projection has been commonly used to enhance
the image-mesh alignment [22, 25, 24]. Also, it helps visu-
alize the reconstruction in an image. Inspired by the prior
works, we project the 3D joints to 2D space using the esti-
mated camera parameters, and minimize the errors between
the 2D projections and 2D ground truths:
Lproj
J
= 1
K
K
X
i=1
J2D −¯J2D

1 ,
(4)
where the camera parameters are learned by using a linear
layer on top of the outputs of the transformer encoder.
To perform large-scale training, it is highly desirable to
leverage both 2D and 3D training datasets for better gen-
eralization. As explored in literature [34, 22, 25, 24, 23,
8, 32], we use a mix-training strategy that leverages differ-
ent training datasets, with or without the paired image-mesh
annotations. Our overall objective is written as:
L = α × (LV + LJ + Lreg
J ) + β × Lproj
J
,
(5)
where α and β are binary ﬂags for each training sample,
indicating the availability of 3D and 2D ground truths, re-
spectively.
3.5. Implementation Details
Our method is able to process arbitrary sizes of mesh.
However, due to memory constraints of current hardware,
our transformer processes a coarse mesh: (1) We use a
coarse template mesh (431 vertices) for positional encod-
ing, and transformer outputs a coarse mesh; (2) We use
learnable Multi-Layer Perceptrons (MLPs) to upsample the
coarse mesh to the original mesh (6890 vertices for SMPL
human mesh topology); (3) The transformer and MLPs are
trained end-to-end; Please note that the coarse mesh is ob-
tained by sub-sampling twice to 431 vertices with a sam-
pling algorithm [42]. As discussed in the literature [25],
the implementation of learning a coarse mesh followed by
upsampling is helpful to reduce computation. It also helps
avoid redundancy in original mesh (due to spatial locality
of vertices), which makes training more efﬁcient.

## experiments
We ﬁrst show that our method outperforms the previous
state-of-the-art human mesh reconstruction methods on Hu-
man3.6M and 3DPW datasets. Then, we provide ablation
4

3DPW
Human3.6M

## related_work
Human Mesh Reconstruction (HMR): HMR is a task of
reconstructing 3D human body shape, which is an active
research topic in recent years. While pioneer works have
demonstrated impressive reconstruction using various sen-
sors, such as depth sensors [33, 48] or inertial measurement
units [20, 54], researchers are exploring to use a monocular
camera setting that is more efﬁcient and convenient. How-
ever, HMR from a single image is difﬁcult due to complex
pose variations, occlusions, and limited 3D training data.
Prior studies propose to adopt the pre-trained parametric
human models, i.e., SMPL [29], STAR [35], MANO [43],
and estimate the pose and shape coefﬁcients of the para-
metric model for HMR. Since it is challenging to regress
the pose and shape coefﬁcients directly from an input im-
age, recent works further propose to leverage various human
body priors such as human skeletons [26, 39] or segmenta-
tion maps [34], and explore different optimization strate-
gies [24, 22, 51, 14] and temporal information [23] to im-
prove reconstruction.
On the other hand, instead of adopting a parametric hu-
man model, researchers have also proposed approaches to
directly regress 3D human body shape from an input image.
For example, researchers have explored to represent human
body using a 3D mesh [25, 8], a volumetric space [52], or an
occupancy ﬁeld [46, 47]. Each of the prior works addresses
a speciﬁc output representation for their target application.
Among the literature, the relevant study is GraphCMR [25],
which aims to regress 3D mesh vertices using graph convo-
lutional neural networks (GCNNs). Moreover, recent pro-
posed Pose2Mesh [8] is a cascaded model using GCNNs.
Pose2Mesh reconstructs human mesh based on the given
human pose representations.
While GCNN-based methods [8, 25] are designed to
model neighborhood vertex-vertex interactions based on a
pre-speciﬁed mesh topology, it is less efﬁcient in model-
ing longer range interactions. In contrast, METRO models
global interactions among joints and mesh vertices with-
out being limited by any mesh topology. In addition, our
method learns with self-attention mechanism, which is dif-
ferent from prior studies [8, 25].
Attentions and Transformers: Recent studies [36, 28, 53]
have shown that attention mechanisms improve the per-
formance on various language tasks. Their key insight is
to learn the attentions to soft-search relevant inputs that
are important for predicting an output [4].
Vaswani et
al. [53] further propose a transformer architecture based
solely on attention mechanisms. Transformer is highly par-
allelized using multi-head self-attention for efﬁcient train-
ing and inference, and leads to superior performance in
language modeling at scale, as explored in BERT [9] and
GPT [40, 41, 5].
Inspired by the recent success in neural language ﬁeld,
there is a growing interest in exploring the use of trans-
former architecture for various vision tasks, such as learn-
ing the pixel distributions for image generation [7, 37] and
classiﬁcation [7, 10], or to simplify object detection as a set
2

Figure 2: Overview of the proposed framework. Given an input image, we extract an image feature vector using a convo-
lutional neural network (CNN). We perform position encoding by adding a template human mesh to the image feature vector
by concatenating the image feature with the 3D coordinates (xi, yi, zi) of every body joint i, and 3D coordinates (xj, yj, zj)
of every vertex j. Given a set of joint queries and vertex queries, we perform self-attentions through multiple layers of a
transformer encoder, and regress the 3D coordinates of body joints and mesh vertices in parallel. We use a progressive di-
mensionality reduction architecture (right) to gradually reduce the hidden embedding dimensions from layer to layer. Each
token in the ﬁnal layer outputs 3D coordinates of a joint or mesh vertex. Each encoder block has 4 layers and 4 attention
heads. H denotes the dimension of an image feature vector.
prediction problem [6]. However, 3D human reconstruction
has not been explored along this direction.
In this study, we present a multi-layer transformer archi-
tecture with progressive dimensionality reduction to regress
the 3D coordinates of the joints and vertices.

## conclusion
We present a simple yet effective mesh transformer
framework to reconstruct human pose and mesh from a sin-
gle input image. We propose the Masked Vertex Model-
ing objective to learn non-local interactions among body
joints and mesh vertices. Experimental results show that,
our method advances the state-of-the-art performance on
3DPW, Human3.6M, and FreiHAND datasets.
A detailed analysis reveals that the performance im-
provements are mainly attributed to the input-dependent
non-local interactions learned in METRO, which enables
1According to the ofﬁcial FreiHAND leaderboard in November 2020:
https://competitions.codalab.org/competitions/21238
8

predictions based on important joints and vertices, regard-
less of the mesh topology. We further demonstrate the gen-
eralization capability of the proposed approach to 3D hand
reconstruction.