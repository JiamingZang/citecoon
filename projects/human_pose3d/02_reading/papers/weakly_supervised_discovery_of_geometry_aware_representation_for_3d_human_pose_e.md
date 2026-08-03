# Weakly-Supervised Discovery of Geometry-Aware Representation for 3D Human Pose Estimation

> 2019 · id: W2969450957 · arXiv: 1903.08839 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Recent studies have shown remarkable advances in 3D
human pose estimation from monocular images, with the
help of large-scale in-door 3D datasets and sophisticated
network architectures. However, the generalizability to dif-
ferent environments remains an elusive goal.
In this work, we propose a geometry-aware 3D repre-
sentation for the human pose to address this limitation by
using multiple views in a simple auto-encoder model at the
training stage and only 2D keypoint information as super-
vision. A view synthesis framework is proposed to learn the
shared 3D representation between viewpoints with synthe-
sizing the human pose from one viewpoint to the other one.
Instead of performing a direct transfer in the raw image-
level, we propose a skeleton-based encoder-decoder mech-
anism to distil only pose-related representation in the latent
space. A learning-based representation consistency con-
straint is further introduced to facilitate the robustness of
latent 3D representation. Since the learnt representation
encodes 3D geometry information, mapping it to 3D pose
will be much easier than conventional frameworks that use
an image or 2D coordinates as the input of 3D pose esti-
mator. We demonstrate our approach on the task of 3D hu-
man pose estimation. Comprehensive experiments on three
popular benchmarks show that our model can signiﬁcantly
improve the performance of state-of-the-art methods with
simply injecting the representation as a robust 3D prior.

## introduction
3D human pose estimation refers to estimating 3D loca-
tions of body parts given an image or a video. This task is
an active research topic in the computer vision community
∗Xipeng Chen and Kwan-Yee Lin have contributed equally and assert
joint ﬁrst authorship. The work was done during the internship at Sense-
Time Research.
φ
G
Pre-­trained  
2D  Human  Pose  Estimator  
Shallow    
Network
Pre-­trained
Encoder
2D
Skeleton
Learnt
Representation
Input  Image  
3D  Pose  
Prediction
Input  Image  
Deep    Network
3D  Pose  Prediction
Input  Image  
Deep    Network
3D  Pose  Prediction
2D  Keypointheat  maps  
(a)  
(b)	  
(c)	  
Figure 1: Motivation. Most state-of-the-arts usually directly learn
the 3D poses from monocular images (as shown in (a)), or ﬁrst es-
timate 2D poses and then lift 2D poses to 3D poses (as shown
in (b)). Both categories require sophisticated deep network ar-
chitectures and abundant annotated training samples. Instead, we
consider learning a geometry representation from multi-view in-
formation with only 2D annotations as supervision. The learnt
representation could map to 3D pose with a shallow network and
less annotated training samples, as shown in (c).
for serving as a key step for many applications, e.g., action
recognition, human-computer interaction, and autonomous
driving.
Signiﬁcant advances in particular datasets have
been achieved in recent years due to the abundant anno-
tations and sophisticated designed deep neural networks.
However, since precise 3D annotation requires large efforts,
and usually subjects to speciﬁc conditions in practice, like
motions, environments, and appearances, etc., the bottle-
neck of generalizability still exists.
Weakly-supervised learning provides an alternative
paradigm for learning robust geometry representation with-
out requiring extensive precise 3D annotation. Most of ap-
proaches [45, 29, 27, 41, 15] leverage knowledge transfor-
mation to learn the robustness by training 3D annotations
with abundant 2D annotations in-the-wild.
These meth-
ods face the difﬁculties of large domain shift between con-
strained lab environment for 3D annotations and uncon-
strained in-the-wild environment for 2D annotations. Some
approaches try to represent body shape through multiple
1
arXiv:1903.08839v2  [cs.CV]  27 Mar 2019

view images acquired by synchronized cameras with the
usage of view-consistency property [29], pre-deﬁned para-
metric 3D model ﬁtting [3, 25, 10], or by sequence with
the usage of time-independent features [13]. Nevertheless,
ﬁtting a pre-deﬁned 3D model or exploiting limited multi-
view information in a particular dataset can hardly capture
all subtle poses of the human body.
The emergence of approaches for novel view synthesis,
e.g., [8, 34], provides an appealing and succinct solution
for capturing geometry representation with multi-view in-
formation. However, despite the success of this ﬁeld on
many generic objects, like chairs, cars, and planes, it is non-
trivial to utilize existing frameworks to learn geometry rep-
resentation for the human body, since the human body is
articulated and much more deformable than rigid objects.
The objective of this paper is to devise a simple yet effec-
tive framework that learns a 3D geometry-aware structure
representation of human pose with only accessible 2D an-
notation as supervision. In particular, we use an encoder-
decoder to generate a novel view pose from a given view
pose. The latent code of the encoder-decoder is regarded as
the desired geometry representation. Instead of generating
the novel view pose on image-level [13, 2], we propose the
use of the 2D skeleton map as a compact medium. Con-
cretely, we ﬁrst map the source and target images into 2D
skeleton maps, then an encoder-decoder is trained to syn-
thesis target skeleton from source skeleton.
Introducing the 2D skeleton as the source/target space of
the encoder-decoder is beneﬁcial for learning a robust ge-
ometry representation. Firstly, 2D skeleton could be easily
obtained from an image with the usage of well-studied 2D
human pose estimator [21, 5, 16], which is accurate and ro-
bust under diverse poses, appearances and environment con-
ditions. This advantage could guarantee body pose and ge-
ometry information are faithfully kept. Secondly, skeleton
representation avoids the variances among datasets, which
could be leveraged to cover pose changes as much as possi-
ble by training existing datasets together and augment sam-
ples on continuous views. Thirdly, the representation in the
latent space could be simply distilled to only pose-related
information without consideration of disentangling shape
with appearance and other unessential nature of encoding
geometry information.
However, the premise of obtaining a robust geometry
representation under an encoder-decoder framework is the
accurate generation of the target view. While, there is no
theoretical assurance for generating the correct one, since
the conventional view synthesis losses (e.g., reconstruction
loss and adversarial loss) do not facilitate semantic infor-
mation. To address the problem, we introduce a representa-
tion consistency loss in latent space to constrain the process
without requiring any other auxiliary information.
We summarize our contributions as follows:
1) We propose a novel weakly-supervised encoder-decoder
framework to learn the geometry-aware 3D representa-
tion for the human pose with multi-view data and only
existing 2D annotation as supervision. To distil the rep-
resentation from unessential factors, and meanwhile in-
crease the training space, a skeleton-based view synthe-
sis is introduced. Our approach allows the substantial
3D pose estimator to generalize well in different condi-
tions.
2) To ensure the robustness of the desired representation,
a representation consistency loss is introduced to con-
strain the learning process of latent space. In contrast to
conventional weakly-supervised methods which require
auxiliary information, our framework is more ﬂexible
and easier to train and implement.
3) A comprehensive quantitative and qualitative evaluation
on public 3D human pose estimation datasets shows the
signiﬁcant improvements of our model applied on state-
of-the-art methods, which demonstrates the effective-
ness of learnt 3D geometry representation to pose es-
timation task.

## experiments
Datasets. We evaluate our approach both quantitatively
and qualitatively on popular human pose estimation bench-
marks: Human3.6M [11], MPI-INF-3DHP [19], and MPII
Human Pose [1].
Human3.6M is the largest dataset for
3D human pose estimation, which consists of 3.6 million
poses and corresponding video frames featuring 11 actors
performing 15 daily activities from 4 camera views. MPI-
INF-3DHP is a recently proposed 3D benchmark consists
of both constrained indoor and complex outdoor scenes.
MPII Human Pose dataset is a challenging benchmark for
estimating in-the-wild 2D human pose. Following previous
methods [41, 7, 24, 18], we adopt this dataset for evaluating
the cross-domain generalization qualitatively.
Evaluation Protocols. For Human3.6M dataset, we fol-
low the standard protocol, i.e.,Protocol#1, to use all 4 cam-
era views in subjects 1, 5, 6, 7 and 8 for training, and same
all 4 camera views in 9 and 11 for testing. In some works,
the predictions are further aligned with the ground-truth via
a rigid transformation [41, 7], which is referred as Proto-
col#2. To further validate the robustness of different models
to new subjects and views, we follow [7] to use subjects 1,
5, 6, 7 and 8 in 3 camera views for training, while 9 and
11 in the other camera view for testing. This protocol is re-
ferred as Protocol#3. The evaluation metric is the Mean Per
Joint Position Error (MPJPE), measured in millimeters.
Implementation Details.
For
‘image-skeleton map-
ping’ module, we adopt a state-of-the-art 2D pose estima-
tor [21] to perform 2D pose detection. We adopt the net-
work architecture on the U-Net as the backbone of our
generator(·, ·). The skip connections are removed to en-
sure all information can be encoded into the latent codes.
For model acceleration, we also halve the feature channels
and modify the input and output to 15-channel 64×64. The
regression module is a two-layer fully-connected network
of dimensions 1024 and 48, which is referred to as Regres-
sion#1. To further validate the ﬂexibility and complemen-
tarity of our proposed framework to other approaches, we
also try to use state-of-the-art 3D pose estimators [18, 31]
as the regression components. The learnt representation G,
behaves as a 3D structure prior, is injected into their frame-
200
107.9
96.2
94.5
93.3
91.9
88.5
83.4
80.2
207.5
127.3
122.7
122
121.5
121.6
117.6
115.3
114.7
70
90
110
130
150
170
190
210
49	  	  	  
(0.1%S1)
496	  	  	  	  	  	  	  	  
(1%	  S1)
2.5k	  	  	  	  	  	  	  	  
(5%	  S1)
5k	  	  	  	  	  	  	  	  
(10%	  S1)
25k	  	  	  	  	  	  	  	  
(50%	  S1)
49k	  	  	  	  	  	  
(S1)
129k	  	  	  	  	  	  	  	  	  	  
(	  S1	  +	  S5)
179k	  	  	  	  	  	  	  	  	  
(	  S1	  +	  S5	  +	  
S6)
312k	  	  	  	  	  	  	  	  	  	  
(all	  )
MPJPE	  in	  mm
OursShallow
Baseline#1
(a)
153.2
82.1
71.3
70.7
68.6
68
62.5
60.3
58.2
135.7
85.3
82.9
82.4
81.6
81.5
78.9
77
76.1
50
70
90
110
130
150
170
49	  	  	  
(0.1%S1)
496	  	  	  	  	  	  	  	  
(1%	  S1)
2.5k	  	  	  	  	  	  	  	  
(5%	  S1)
5k	  	  	  	  	  	  	  	  
(10%	  S1)
25k	  	  	  	  	  	  	  	  
(50%	  S1)
49k	  	  	  	  	  	  
(S1)
129k	  	  	  	  	  	  	  	  	  	  
(	  S1	  +	  S5)
179k	  	  	  	  	  	  	  	  	  
(	  S1	  +	  S5	  +	  
S6)
312k	  	  	  	  	  	  	  	  	  	  
(all	  )
PMPJPE	  in	  mm
OursShallow
Baseline#1
(b)
Figure 5: Evaluation on the Human3.6M using different number
of training data. (a) presents the results under MPJPE metric. (b)
presents the results under PMPJPE metric.
works. These two conﬁgurations are referred to as Regres-
sion#2 and Regression#3 respectively. Note that, in order
to evaluate the robustness and ﬂexibility of the proposed ge-
ometry representation in a straightforward manner, we only
forward the geometry representation G to fully connection
layers to match the feature dimension of baselines, and then
directly do element-wise sum with baselines, instead of de-
signing sophisticated feature fusion mechanism to poten-
tially better fuse the representation with original features.
All the experiments are conducted on Titan X GPUs. Please
refer to the supplemental materials for architecture details.
Results on Human3.6M. We ﬁrstly validate the effec-
tiveness of learnt representation G to 3D human pose esti-
mation task, on the condition of using different amount of
3D annotated samples (under Protocol#1) to train the re-
gression module. We adopt Regression#1 as the regressor
with only G as the input. The conﬁguration is referred as
OursShallow. Since only 2D annotation is utilized to learn
G, we also list the performances of directly regressing 3D
pose coordinates from 2D detections with the same regres-
sor, which is referred to Baseline#1. Figure 5 shows the
results. The phenomenon is consistent on both MPJPE and
PMPJPE metrics. Given only about 500 annotated train-
ing samples, our method achieves 17.98% relative improve-
ments than Baseline#1 on MPJPE, and 3.90% on PMPJPE.
The margin becomes larger when more annotated samples
are used for training. Our general improvements over dif-
ferent setting demonstrate the robustness of the learnt rep-
resentation to different amount of 3D training samples. We
also perform above experiments on Regression#2 and Re-
gression#3 to further verify the effectiveness of the learnt
representation to strong baselines (For space saving, the de-
tail results are shown in the supplementary material). Under
fewer amount of training samples, our proposed represen-
tation could help improve the performance of baselines to
comparable results with the one trained on a larger amount
of samples by themselves.
We then evaluate the models under all three protocols to
demonstrate the effectiveness and ﬂexibility of learnt rep-
resentation G as a robust 3D prior to different 3D human
pose estimation methods. Table 1 reports the comparison
with current state-of-the-arts. We draw two key observa-
6

Protocol #1
Direction
Discuss
Eat
Greet
Phone
Photo
Pose
Purchase
Sit
SitDown
Smoke
Wait
WalkDog
Walk
WalkT.
Avg.
Zhou et al. (ICCV’17) [45]
54.8
60.7
58.2
71.4
62.0
65.5
53.8
55.6
75.2
111.6
64.1
66.0
51.4
63.2
55.3
64.9
Martinez et al. (ICCV’17) [18]
51.8
56.2
58.1
59.0
69.5
78.4
55.2
58.1
74.0
94.6
62.3
59.1
65.1
49.5
52.4
62.9
Fang et al. (AAAI’18) [7]
50.1
54.3
57.0
57.1
66.6
73.3
53.4
55.7
72.8
88.6
60.3
57.7
62.7
47.5
50.6
60.4
Sun et al. (ICCV’17) [30]
52.8
54.8
54.2
54.3
61.8
67.2
53.1
53.6
71.7
86.7
61.5
53.4
61.6
47.1
53.4
59.1
Yang et al. (CVPR’18) [41]
51.5
58.9
50.4
57.0
62.1
65.4
49.8
52.7
69.2
85.2
57.4
58.4
43.6
60.1
47.7
58.6
Pavlakos et al. (CVPR’18) [23]
48.5
54.4
54.4
52.0
59.4
65.3
49.9
52.9
65.8
71.1
56.6
52.9
60.9
44.7
47.8
56.2
Wang et al. (IJCAI’18) [36]
49.2
55.5
53.6
53.4
63.8
67.7
50.2
51.9
70.3
81.5
57.7
51.5
58.6
44.6
47.2
57.8
Trumble et al. (ECCV’18) [33]
41.7
43.2
52.9
70.0
64.9
83.0
57.3
63.5
61.0
95.0
70.0
62.3
66.2
53.7
52.4
62.5
Park et al. (BMVC’18) [22]
49.4
54.3
51.6
55.0
61.0
73.3
53.7
50.0
68.5
88.7
58.6
56.8
57.8
46.2
48.6
58.6
Sun et al. (ECCV’18) [31]
46.5
48.1
49.9
51.1
47.3
43.2
45.9
57.0
77.6
47.9
54.9
46.9
37.1
49.8
41.2
49.8
Ours + Regression#1 (2 fc layers)
63.9
73.7
70.9
76.1
82.6
69.5
75.1
96.1
120.6
75.4
96.8
78.7
69.1
83.5
72.2
80.2
Ours + Regression#2 ( [18])
45.9
53.5
50.1
53.2
61.5
72.8
50.7
49.4
68.4
82.1
58.6
53.9
57.6
41.1
46.0
56.9
Ours + Regression#3 ( [31])
41.1
44.2
44.9
45.9
46.5
39.3
41.6
54.8
73.2
46.2
48.7
42.1
35.8
46.6
38.5
46.3
Protocol #2
Direction
Discuss
Eat
Greet
Phone
Photo
Pose
Purchase
Sit
SitDown
Smoke
Wait
WalkDog
Walk
WalkT.
Avg.
Moreno-Noguer (CVPR’17) [20]
66.1
61.7
84.5
73.7
65.2
67.2
60.9
67.3
103.5
74.6
92.6
69.6
71.5
78.0
73.2
74.0
Zhou et al. (Arxiv’17) [47]
47.9
48.8
52.7
55.0
56.8
65.5
49.0
45.5
60.8
81.1
53.7
51.6
54.8
50.4
55.9
55.3
Sun et al. (ICCV’17) [30]
42.1
44.3
45.0
45.4
51.5
53.0
43.2
41.3
59.3
73.3
51.0
44.0
48.0
38.3
44.8
48.3
Martinez et al. (ICCV’17) [18]
39.5
43.2
46.4
47.0
51.0
56.0
41.4
40.6
56.5
69.4
49.2
45.0
49.5
38.0
43.1
47.7
Fang et al. (AAAI’18) [7]
38.2
41.7
43.7
44.9
48.5
55.3
40.2
38.2
54.5
64.4
47.2
44.3
47.3
36.7
41.7
45.7
Sun et al. (ECCV’18) [31]
40.9
41.4
45.0
45.2
42.1
37.6
41.1
52.0
71.4
42.5
47.4
41.6
32.0
42.6
36.9
44.1
Yang et al. (CVPR’18) [41]
26.9
30.9
36.3
39.9
43.9
47.4
28.8
29.4
36.9
58.4
41.5
30.5
29.5
42.5
32.2
37.7
Ours + Regression#1 (2 fc layers)
47.0
51.8
53.3
55.3
59.7
48.4
51.7
72.1
90.6
56.6
65.4
55.1
50.2
59.4
53.9
58.2
Ours + Regression#2 ([18])
36.5
41.0
40.9
43.9
45.6
53.8
38.5
37.3
53.0
65.2
44.6
40.9
44.3
32.0
38.4
44.1
Ours + Regression#3 ([31])
36.9
39.3
40.5
41.2
42.0
34.9
38.0
51.2
67.5
42.1
42.5
37.5
30.6
40.2
34.2
41.6
Protocol #3
Direction
Discuss
Eat
Greet
Phone
Photo
Pose
Purchase
Sit
SitDown
Smoke
Wait
WalkDog
Walk
WalkT.
Avg.
Pavlakos et al. (CVPR’17) [24]
79.2
85.2
78.3
89.9
86.3
87.9
75.8
81.8
106.4


## related_work
Geometry-Aware Representations. To capture the in-
trinsic structure of objects, existing studies [40, 34, 13, 44]
typically disentangle visual content into multiple predeﬁned
factors like camera viewpoints, appearance and motion.
Some works [39, 43] leverage the correspondence among
intra-object instance category to encode the structure rep-
resentation. [43] discovery landmark structure as an inter-
mediate representation for image autoencoding with several
constraints. Other approaches utilize multiple views to ei-
ther directly learn the geometry representation [32, 42, 9]
with object reconstruction, or take advantage of view syn-
thesis [26] to learn the structure with shared latent repre-
sentation between views. For example, [26] learn 3D hand
pose representation by synthesizing depth maps under dif-
ferent views. [13] conditionally generate an image of the ob-
ject from another one, where the generated image differs by
acquisition time or viewpoint, to encourage representation
distilled to object landmarks. These methods mainly focus
on structure representation of generic objects or hand/face
pose. Whereas, the human body is articulated and much
more deformable. How to capture the geometry represen-
tation of the human body with fewer data and simpler con-
straints is still an open question.
3D Human Pose Estimation. Most of the existing stud-
ies for 3D human pose estimation beneﬁt from the availabil-
ity of large-scale datasets and sophisticated deep-net archi-
tectures. These methods could be roughly categorized into
fully-supervised and weakly-supervised manners.
2

iv
ivˆ
jv
jvˆ
φ
µ
ν
⊗
⊗
i
G
ij
G
j
G~
ji
G~
Image-Skeleton Mapping
View Synthesis
Representation Consistency Constraint
Pre-trained 2D Human Pose Estimator 
(Hourglass Architecture)
Encoder
Decoder
Latent 
Representation
Loss
Backward
Forward
𝐼"
𝐼#
𝝍
𝑅"→#
𝑅#→"
Figure 2:
The framework of learning a geometry representation for 3D human pose in a weakly-supervised manner. There are three
main components. (a)Image-skeleton mapping module is used to obtain 2D skeleton maps from raw images. (b)View synthesis module is
in a position to learn the geometry representation in latent space by generating skeleton map under viewpoint j from skeleton map under
viewpoint i. (c) Since there is no explicit constrain to facilitate the representation to be semantic, a representation consistency constrain
mechanism is proposed to further reﬁne the representation.
A vast amount of fully-supervised 3D pose estimation
methods via monocular image exist in the literature [18,
20, 4, 36]. Despite the performance these methods achieve,
modeling 3D mapping from a given dataset limits their gen-
eralizability due to the constrained lab environment, limited
motion and inter-dataset variation.1
Several works focus on weakly-supervised learning to
increase the diversity of samples and meanwhile restrain the
usage of labeled 3d annotated data. For example, synthesize
training data by deforming a human template model with
known 3D ground truth [35], or generating various fore-
ground/background [19]. [45] proposes to transform knowl-
edge from 2D pose to 3d pose estimation network with
re-projection constraint to 2D results.
A converse strat-
egy is employed in [41] to distil 3D pose structure to un-
constrained domain under an adversarial learning frame-
work. [25] proposes to learn the parameters of the statistical
model SMPL [17] to obtain 3D mesh from image with an
end-to-end network, and regresses 3d coordinates from the
mesh. Other approaches [29, 46] exploit views consistency
with the usage of multiple viewpoints of the same person.
Nevertheless, these methods still rely on a large quantity of
3D training samples or auxiliary annotations, like silhou-
ettes [6] and depth [46] to initialize or constrain the models.
In contrast to above approaches, our framework aims at
discovering a robust geometry-aware 3D representation of
human pose in latent space, with only 2D annotation in
hand. This allows us to train the subsequent monocular 3D
pose estimation network with much less labeled 3D data.
1Inter-dataset variation refers to bias among different datasets on view-
points, environments, the deﬁnition of 3D key points, etc.
Recently, a concurrent work is published in the community
with similar spirits. In contrast to [28] that can only han-
dle one particular dataset due to the dependency of appear-
ance and inter-frame information during the training pro-
cess, our framework tries to break the gap of inter-dataset
variation, which permits more practical usages. Moreover,
our framework is complementary to previous 3D pose esti-
mation works, and can use current approaches as the base-
line with the injection of learnt representation as a 3D struc-
ture prior.
3. Weakly-Supervised Geometry Representa-
tion
Recall that our goal is to learn a geometry-aware 3D rep-
resentation G for the human pose, which is expected to be
robust to diverse pose changes and can be learnt with less
effort than conventional weakly-supervised methods. To-
ward this end, we propose to discover the geometry relation
between paired images(Ii
t, Ij
t ), which are acquired from
synchronized and calibrated cameras, with the only exist-
ing 2D coordinate annotation used for supervision, where i
and j denote different viewpoints, t denotes acquiring time.
The proposed approach is depicted in Figure 2. The frame-
work includes three components: an image-skeleton map-
ping component, a skeleton-based view synthesis compo-
nent, and a representation consistency constraint compo-
nent. The desired representation is encoded in the bottle-
neck of the encoder-decoder on the view synthesis compo-
nent. In the inference phase, the learnt representation will
be obtained by forwarding a single image through the ﬁrst
two components, as illustrated in Figure 1(c). We will detail
each component in the remainder of this section.
3

3.1. Image-skeleton mapping
It is habitual to directly feed forward the raw image
to the network to learn geometry representation [13, 34].
However, under the setting of multiple-view with encoder-
decoder framework, we demonstrate that utilizing only 2D
skeleton information is sufﬁcient and better than raw im-
ages to learn the representation, as shown in the Sec 4.
Consequently, given a pair of raw images (Ii
t, Ij
t ) with the
size of W × H under different viewpoints of camera i and
camera j respectively, a pre-trained 2D human pose es-
timator2 is ﬁrstly applied to obtain two stacks of K key
point heatmaps Ci
t, and Cj
t . Then, the corresponding 2D
skeleton maps, regarded as a person tree-structured kine-
matic graph, are constructed from the heatmaps with 8 pix-
els width. Consequently, we are given the binary skeleton
maps pair (Si
t, Sj
t ), where S(·)
t
∈{0, 1}(K−1)×W ×H.
Intuitively, we could sample (i, j) randomly from ex-
isting cameras.
However, such a sampling strategy will
lead to two problems in practice. Firstly, the ﬁnite samples
limit the diversity of the training set. Secondly, the nonuni-
form distribution3 of viewpoints will increase the difﬁculty
of network learning.
To solve the above problems, it is
straightforward to utilize virtual cameras-based data aug-
mentation. While, conventional methods can only achieve
in-plane rotations due to image-level inputs [13, 28]. In-
stead, we draw on virtual cameras applied in [7] to increase
training pairs on a torus4. Different from [7] that gener-
ate new 2D coordinates-3D coordinates pairs, we randomly
sample 2D skeleton pairs. Thus, we could obtain inﬁnite
training pairs and calculate their relative rotation matrix in
theory. This augmentation strategy facilitates our model to
be robust to different camera conﬁgurations.
3.2. Geometry representation via view synthesis
Assume that we are given a training set T
=
{(Si
t, Sj
t , Ri→j)}NT
t=1 containing pairs of two views of pro-
jection of same 3D skeleton (Si
t, Sj
t ) and relative rotation
matrix Ri→j from coordinate system of camera i to j, af-
ter image-skeleton mapping step. We now turn to discover
the geometry representation G. A straightforward way for
learning representation in unsupervised/weakly-supervised
manner is to utilize autoencoding mechanism reconstruct-
ing input image. Then, the latent codes of the auto-encoder
could be regarded as the features that encode compact in-
formation of the input [43, 14]. While, such a represen-
tation neither contains geometry structure information nor
provides more useful information for 3D pose estimation
than 2D coordinates, as demonstrated in Figure 6.
2We follow previous works [45, 20, 18] to train the 2D estimator on
MPII dataset.
3For example, in Human3.6M dataset [11], four cameras are approxi-
mately located at four corners of a rectangle.
4Please refer to the supplemental materials for detail operation.
SG
DG
GT
Figure 3:
An illustration of the effectiveness of representation
consistency constraint. Compared with only applying the 

## conclusion
We have presented a weakly-supervised method of learn-
ing a geometry-aware representation for 3D human pose es-
8

timation. Our method is novel in that we take a radically dif-
ferent approach to learn the geometry representation under
multi-view setting. Speciﬁcally, we leverage view synthesis
to distill shared representation in the latent space with only
the usage of 2D annotation and simple representation con-
sistency constraint, which provides a new aspect to learn
the representation with fewer annotation efforts and sim-
pler network architecture. Meanwhile, we bridge different
3D human pose datasets by introducing a skeleton-based
encoder-decoder. Experimental results validate the effec-
tiveness and ﬂexibility of the proposed framework on 3D
human pose estimation task.