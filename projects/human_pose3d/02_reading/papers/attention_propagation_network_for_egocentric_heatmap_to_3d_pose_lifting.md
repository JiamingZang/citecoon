# Attention-Propagation Network for Egocentric Heatmap to 3D Pose Lifting

> 2024 · id: W4402754008 · arXiv: 2402.18330 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
We present EgoTAP, a heatmap-to-3D pose lifting
method for highly accurate stereo egocentric 3D pose esti-
mation. Severe self-occlusion and out-of-view limbs in ego-
centric camera views make accurate pose estimation a chal-
lenging problem. To address the challenge, prior methods
employ joint heatmaps-probabilistic 2D representations of
the body pose, but heatmap-to-3D pose conversion still re-
mains an inaccurate process. We propose a novel heatmap-
to-3D lifting method composed of the Grid ViT Encoder and
the Propagation Network. The Grid ViT Encoder summa-
rizes joint heatmaps into effective feature embedding using
self-attention. Then, the Propagation Network estimates the
3D pose by utilizing skeletal information to better estimate
the position of obscure joints.
Our method significantly
outperforms the previous state-of-the-art qualitatively and
quantitatively demonstrated by a 23.9% reduction of er-
ror in an MPJPE metric. Our source code is available in
GitHub 1.

## introduction
Overall Architecture. Fig. 4 illustrates the comprehensive
architecture of EgoTAP. It comprises two essential compo-
nents: the Grid ViT Heatmap Encoder and the Propaga-
tion Network. The Grid ViT Heatmap Encoder takes joint
heatmaps as input and generates effective feature embed-
dings for each joint. The Propagation Network processes
these embeddings with awareness of the skeletal structure to
estimate the 3D pose accurately. Notably, the per-joint fea-
ture embedding is propagated through a skeletal hierarchy,
represented as a tree structure with a root representing the
head. In Fig. 4, a simplified skeleton is depicted, showcas-
ing the propagation from the head to the hand, highlighted
in red. The feature propagation utilizes the PU (Propaga-
tion Unit in Fig. 5), which calculates joint states based on
the parent joint’s states along with other self-joint features.
The hidden states of the last PU layer are concatenated with
the joint features from the Grid ViT encoder and linearly
projected to estimate the 3D pose of each joint.
Input and Output. Our method utilizes a pre-trained and
frozen heatmap estimator that takes stereo RGB images I ∈
R2×256×256×3 and estimates stereo heatmaps for NJ joints
HJ ∈R2NJ×64×64 and NL limbs HL ∈R2NL×2×64×64.
EgoTAP takes the heatmaps and reconstructs the 3D pose
P ∈RN′
J×3 of N ′
J joints relative to the user’s root defined
in the dataset. Note that the number of estimation targets
N ′
J can differ from the number of joints with heatmap NJ
depending on the dataset.
Loss.
We use the Euclidean distance and the cosine
similarity-based loss between the ground-truth pose and the
estimated pose to train the Attention-Propagation network.
The loss formulation is in the supplementary material.
Heatmaps. Two types of heatmaps for joints and limbs
are used.
We follow the standard definition of joint
heatmap [21] where pixel values represent the probability
that the joint is in that 2D coordinate. The limb heatmaps
have two channels and are used to get relational features
between two joints for the Propagation Network in Sec. 3.3.
We use a limb heatmap suggested by Kang et al. [8], repre-
senting 3D information along with limb visibility as a line
connecting joints. From the next section, we denote two
types of heatmaps: joint heatmaps and limb heatmaps. We
use a pre-trained ResNet-18 [5] based U-Net [18] architec-
ture with a shared weight for two input image encoders and
shared decoder, suggested by Akada et al. [2] for heatmap
estimation.
3.2. Grid ViT Heatmap Encoder
Our encoder, described in Fig. 4, combines all joint
heatmaps into a large single grid image. The grid is split
into patches, linearly projected to make the input embed-
3

Grid ViT Encoder
Propagation
Network
Propagation Network
FR,i
PU
PU
PU
PU
Zero Init
o
i
j
k
FR,j
FR,k
FJ,i
FJ,j
FJ,k
FP,i
FP,j
FP,k
Project
Pi
FP,i
…
i
j
o
k
3D Pose
Per Joint
HJ,2i
HJ,2i-1
HJ,2i+1
HJ,2i+2
Heatmap
Estimator
…
…
…
HL,2i-1
Stereo Pair
FR,i
Flatten
MLP
Transformer Encoder
…
…
…
…
…
…
…
HJ,2i-1HJ,2iHJ,2i+1
…
…
…
p
…
…
z’ from
HJ,2i-1 Patches
z’ from
HJ,2i Patches
FJ,i
…
FJ,i
Joint
Heatmaps
Limb
Heatmaps
HL,2i
Patches
HL,2i
HL,2i-1
Norm
3x
Norm
Multi-Head
Attention
MLP
Stereo Pair
MLP
MLP
MLP
Flatten
…
p
p
p
p
p
p
Figure 4. Overall network architecture of EgoTAP. EgoTAP takes heatmaps from pre-trained heatmap estimators taking stereo input images
and lifts the heatmaps to the 3D pose with the Grid ViT Encoder, Propagation Network, and finally, a projection layer.
ding, and fed to a transformer [22] encoder architecture
with multi-head attention. The transformer encoding pro-
cess preserves the correspondence between a patch and the
input feature embedding in the output. The output feature
embeddings corresponding to individual input patches are
concatenated and re-encoded to form a feature embedding
vector for the heatmap.
Unlike the CNN encoder, where the communication oc-
curs within the nearby pixels of different heatmaps, the
Grid ViT Heatmap Encoder allows communication between
heatmap patches that are far spatially. This allows features
to be shared without downsampling, minimizing the loss of
information. The efficiency of the encoder is demonstrated
by the precisely reconstructed heatmaps from the embed-
dings in Fig. 3 and Table 3, and improved pose estimation
accuracy.
To formulate the process, let {HJ,i ∈R64×64|i =
1, 2, . . . , 2NJ} be sets of 2 × NJ stereo joint heatmaps.
Heatmaps are arranged into a single grid image. The im-
age is subsequently split to total 4 × 4 × 2NJ patches
{Xi ∈R16×16|i = 1, 2, . . . , 32NJ} where 16 patches cor-
responds to a heatmap. X16(i−1)+1 to X16i corresponds to
i-th heatmap for simplicity.
Each patch Xi is then projected to an input embed-
ding space R1024 with a learnable projection matrix Wz ∈
R1024×256.
Additionally, learnable positional encodings
pi ∈R1024 are added, resulting in the transformer input
embedding zi. The projected embedding with positional en-
coding for each patch is:
zi = Wz · Flatten(Xi) + pi
(1)
z = [z1, z2, . . . , z32NJ] is encoded by three ViT trans-
former encoder [3] layers with multi-head attention to out-
put z′ = [z′
1, z′
2, . . . , z′
32NJ].
For the j-th heatmap, the
corresponding output embeddings from 16 patches are con-
catenated to Zj and then re-encoded to smaller dimensional
feature embedding kj through multiple fully connected lay-
ers denoted as EK. The process is formulated as follows:
z′ = TransformerEncoder(z)
(2)
Zj = [z′
16(j−1)+1, z′
16(j−1)+2, . . . , z′
16j]
(3)
kj = EK(Zj)
(4)
A joint feature FJ,i ∈R256 that corresponds to a spe-
cific joint is obtained by concatenating the stereo heatmap
features. Let’s say 2i −1 and 2i-th heatmap correspond to
i-th joint.
FJ,i = [k2i−1, k2i], for 1 ≤i ≤NJ
(5)
3.3. Propagation Network
Propagation Process. The Propagation Network estimates
the joint positions using their parent joints’ positions and
the relationships between the joints. The Propagation Net-
work is inspired by the stereo setup’s capability to estimate
3D pose without the help of other joints and the general
trend of higher visibility on joints closer to the camera in
the egocentric setup. Sec. 4.3.2 shows that the Propagation
Network effectively takes advantage of accurate estimation
of the parent joint with a Propagation Potential and Propa-
gation Effect metric.
The Propagation Network comprises a relational feature
encoder and the 2-layered PU that handles the propaga-
tion process. The relational feature encoder takes the es-
timated limb heatmaps to output the relational feature be-
tween joints.
The PU handles the propagation process,
which takes the parent states, relational and joint features
4

of the child joint as input and generates the child joint’s
states. The states of joints are propagated through the tree
hierarchy from the head directly attached to the camera to
the extremities. During propagation, the reflection of the
parent joint information is flexibly determined based on the
certainty of the parent and child joint features by the PU.
We leverage the limb heatmaps with 3D information em-
bedded with a trigonometric function of camera view an-
gle [8] to provide information about the connection between
the parent and child joint. An encoder with fully connected
layers ER encodes limb heatmaps HL,i ∈R2×64×64 into a
limb feature. Stereo limb features are concatenated to form
relational feature FR. Let’s say HL,2i−1 and HL,2i corre-
sponds to a limb that connects the i-th joint and its parent.
The process is:
FR,i = [EL(HL,2i−1), EL(HL,2i)], for 1 ≤i ≤NL (6)
The Propagation Network consists of two layers of
the Propagation Unit, described later.
For a tree hi-
erarchy where parent(i) denotes a parent joint’s index,
and PropagationNet((H, C), R, J) denotes the Propagation
Network, which takes hidden and cell states for two PU lay-
ers H = [h1, h2], C = [c1, c2], relational feature R and
joint feature J, the hidden and cell state for i-th joint Hi, Ci
is computed as follows:
Si = (Hi, Ci)
(7)
H0 =⃗0, C0 =⃗0
(8)
Si = PropagationNet(Sparent(i), FJ,i, FR,i), for 1 ≤i ≤NJ
(9)
The root joint head is indexed 0 and initialized with zero
vector, as it is not visible from an egocentric view and,
thus, does not have features. The i-th Propagated Feature
FP,i ∈R256 is a hidden state from the second layer of the
Propagation Network h2,i.
The output of the Propagation Network FP,i and trans-
former output joint features FJ,i for each joint are con-
catenated and projected to estimate the 3D position of each
joint.
Propagation Unit. We devise a Propagation Unit inspired
by the LSTM cell for the above propagation process. Fig. 5
shows the internal structure of the Propagation Unit. The
Propagation Unit weights the parent’s hidden state and the
relational feature with the joint feature. The joint heatmap
from stereo views can be sufficient for precise 3D estima-
tion, and this weighting limits the role of the predictive es-
timation for obscure joints.
To fo

## method
UnrealEgo [2]
EgoCap [17]
Heatmap Encoder
CNN
63.53 (47.76)
70.77 (52.91)
Channel ViT
61.62 (47.05)
83.39 (56.29)
Grid ViT
49.03 (41.03)
63.97 (53.17)
Propagation Network
Grid ViT + RF
48.12 (40.79)
63.09 (52.60)
Grid ViT + LSTM
49.43 (41.31)
60.16 (49.18)
Grid ViT + LSTM RF Alter
44.97 (38.99)
62.60 (50.78)
Grid ViT + LSTM RF Concat
44.77 (38.91)
58.35 (47.06)
Ours (Grid ViT + PU)
41.06 (35.39)
55.38 (45.24)
Table 2. Ablation results of our method for two main components
on two datasets. The metric is MPJPE, and in the bracket is PA-
MPJPE. The bold text for metrics indicates the best results.
Heatmap Reconstruction Error
10−4/Pixel
Zeros
5.45
CNN Encoder
4.84
Grid ViT Heatmap Encoder
1.68
Table 3. Reconstruction mean square error of the heatmaps from
the features encoded with a different frozen encoder architecture,
experimented in the UnrealEgo [2] dataset.
realEgo [2], utilizing a CNN. “Channel ViT” showcases the
outcomes with a typical encoder with ViT, where heatmaps
are concatenated along the channel axis before being split
into patches, resulting in feature embeddings that do not
align with the heatmaps.
Simply adopting transform-
ers [22] yields minimal improvement, i.e., a 3% reduction
in MPJPE, compared to the CNN-based lifting for the Un-
realEgo [2] baseline and dataset. However, this approach
significantly degrades performance on EgoCap [17]. This
observation underscores the importance of addressing the
correspondence between feature embedding and heatmaps
in the pose estimation process.
Heatmap Reconstruction: We conducted experiments
to evaluate the heatmap encoder’s efficiency in encoding
heatmap features. A simple decoder is appended to our en-
coder and baseline encoders to achieve this. The decoder is
trained to reconstruct the estimated heatmaps from the fea-
ture embedding. Table 3 presents the reconstruction error
of the heatmap in the test set. The “Zeros” row provides
the error for a zero-only output for comparison. The re-
sults demonstrate that the Grid ViT Heatmap Encoder ef-
fectively extracts heatmap features, evidenced by the recon-
structed fine details of the heatmap in Fig. 3. In contrast,
the heatmaps were not recoverable from features encoded
by CNN, highlighting its inefficiency.
7

(a) UnrealEgo [2]
(b) UnrealEgo (Camera-relative)
(c) EgoCap [17]
Figure 7. Hexagonal-grid density plot of the Propagation Potential
and the Propagation Effect(mm) in our evaluation datasets. The
dark line shows linear regression results.
4.3.2
Propagation Network
Pose Estimation: We investigate if including relational fea-
tures alone can significantly enhance accuracy through “+
RF” when incorporated with our Grid ViT encoder. The re-
lational features are concatenated to the joint features for
the final projection layer without the involvement of a prop-
agation network. This approach demonstrates marginal im-
pact or even degrades the estimation accuracy. Addition-
ally, we analyze the effect of the Propagation Network with
LSTM [7]. In the case of “+ LSTM,” only joint features are
utilized in the propagation, yielding a marginal effect.
Additional experiments investigate the impact of the
Propagation Network without PU, denoted as “+ LSTM RF
Alter” and “+ LSTM RF Concat.” Relational and joint fea-
tures are alternately taken in the former, and the propagation
feature is output in the joint feature step. The latter takes
both as a concatenated vector. Both methods demonstrate
improvements, with the latter achieving an 8.7% and 8.8%
reduction in MPJPE for two datasets compared to the Grid
ViT Heatmap Encoder-only approach. The final model, in-
corporating PU, maximizes the potential of the Propagation
Network, showcasing a 16.3% and 13.4% improvement in
MPJPE for the two datasets. This highlights the significance
of balancing the role of predictive estimation using parent
joints and direct estimation using self-joint features.
Propagation Potential and Effect: The Propagation
Network leverages more evident parent joint features to im-
prove the child joint’s pose estimation. The hexagonal-grid
density plot in Fig. 7 illustrates its impact quantitatively.
The x-axis represents the Propagation Potential (PP). PP
approximates the upper bound of the improvement using the
parent’s feature, with a difference between the parent and
child joint’s pose estimation error. On the y-axis, the Prop-
agation Effect (PE) is the improvement of the child joint’s
pose error by the Propagation Network. Using ∆to denote
the pose estimation error, subscripts to denote joints, and
superscripts to denote the model (NP without propagation,
P with propagation), we define these metrics as follows.
PP = ∆NP
child −∆NP
parent
(14)
PE = ∆NP
child −∆P
child
(15)
For all datasets, linear regression reveals a positive rela-
tionship between PP and PE with a p-value of the null
hypothesis < 10−3, indicating that the Propagation Net-
work is more effective when the parent joint has a more
precise estimation, aligning with expectations. The aver-
age PP and PE were 16.97 and 8.50 for the UnrealEgo
dataset [2] and 4.32 and 9.39 for the EgoCap [17] dataset.
The UnrealEgo [2] dataset exhibits higher potential due to
the cameras closer to the head, unlike cameras around 20cm
away from the head in the EgoCap dataset [17].
The effect is more pronounced for the UnrealEgo [2]
dataset when the 3D pose is estimated in camera-relative
coordinates. This eliminates the global offset (pelvis pose)
bias from per-joint improvement. Fig. 7 (b), exhibits trends
where PE is similar to PP or close to zero. When the PE
is similar to PP, the child joint’s pose error is improved
close to the parent joint’s error. The effect of the Propa-
gation Network is near the upper bound (PP). The propa-
gation cannot improve the child joint’s pose error in some
cases, possibly due to the occlusion of limbs. Such cases
exhibit near zero PE. 66.07% of PE and 75.62% of PP
in the samples are positive, and 54.16% of samples lie in
the first quadrant. The average positive PE is 10.75, while
the average negative PE is only −0.51, demonstrating that
many joints significantly benefit from the propagation.

## experiments
4.1. Experiment Setup
4.1.1
Datasets
Overview. We used two datasets: UnrealEgo [2] and Ego-
Cap [17] for the 3D pose estimation in the stereo egocentric
camera setup. We conducted the within-dataset evaluation
using each dataset’s train and test set split since the egocen-
tric datasets have significantly different setups and resulting
views.
UnrealEgo. The UnrealEgo [2] is a synthetic dataset con-
taining 450k frames with 17 characters. The dataset covers
a variety of environments and motions that are challenging
to capture in a real-world setup. There are a total of 16 joints
to estimate. The dataset defines the target local 3D pose in
5

a pelvis-relative coordinate system, as opposed to the cam-
era coordinate system in most datasets, and has a head pose
to estimate. The pelvis and head do not have correspond-
ing heatmaps and features. We added a learnable matrix for
linear projection, taking all the final features FJ and FP
to estimate offset for all joints and head pose. We found
that this simple change effectively deals with different pose
definitions.
EgoCap. The EgoCap [17] dataset is captured with ego-
centric cameras attached at the end of the stick on the hel-
met. It comprises 35k frames for training from six subjects
and 1k for testing from one subject with 3D pose annota-
tion. Evaluation with this dataset showcases applicability in
a real-world textured image. There are a total of 17 joints to
estimate.
4.1.2
Baselines
We experiment with three baseline stereo egocentric pose
estimation methods: EgoGlass [32], UnrealEgo [2], and
Ego3DPose [8].
We use the official UnrealEgo [2] and
Ego3DPose [8] implementations. EgoGlass [32] implemen-
tation is taken from the latter as no official source code is
provided. For the UnrealEgo [2] and Ego3DPose [8], we
changed the embedding and pose decoder dimension, which
gives higher estimation accuracy than their original setups.
The change does not impact the EgoGlass [32], possibly due
to the joint training of the heatmap and pose estimator.
4.1.3
Metrics
The MPJPE and PA-MPJPE metrics are used. The MPJPE
is a mean per joint position error in a 3D Euclidian distance.
PA-MPJPE applies Procrustes analysis before computing
the MPJPE to calculate transform-invariant positional error.
4.2. Overall Performance
4.2.1
Qualatative Results
Fig. 6 presents a qualitative comparison between our
method and previous approaches on the UnrealEgo and
EgoCap datasets. A more detailed qualitative comparison is
available in the supplementary video. Our method demon-
strates a significant improvement over baseline methods.
4.2.2
Evaluation on UnrealEgo
The second column of Table 1 presents the quantitative
evaluation results on UnrealEgo [2] using MPJPE and PA-
MPJPE metrics. Our method demonstrates superior per-
formance compared to state-of-the-art methods, achieving
a 23.9% reduction in MPJPE and a 17.7% decrease in PA-
MPJPE. These improvements extend across all 31 activity
categories detailed in the supplementary material, covering
Figure 6. Qualitative comparison of EgoTAP with state-of-the-art
stereo egocentric pose estimation methods. The blue is the ground
truth, and the red is the estimated pose.
a range of movements from common actions like sitting and
standing to less frequent crawling and crouching and more
complex motion categories, including sports.
Noteworthy improvements are observed across various
categories, with the most substantial enhancement in the
“Crouching-Forward” category, boasting a 31.3% reduc-
tion in MPJPE. Conversely, the smallest improvement is
noted in the “Crawling” activity, with an 8.8% decrease
in MPJPE. It’s important to acknowledge that while our
method relies on visual cues, the effectiveness varies based
on the visibility of body parts. For instance, in activities
like “Crouching-Forward,” where many body parts are par-
tially visible, our method excels in improving accuracy. On
the other hand, in activities like “Crawling,” where visible
6

## related_work
2.1. Egocentric Pose Estimation
Egocentric pose estimation can be classified into two main
categories. The first category focuses on estimating the pose
of other people within the camera’s field of view, as in Ng
et al.[15] while the second category estimates the pose of
the user self [11]. Our work belongs to the second category,
especially with a downward-oriented egocentric camera.
EgoCap [17] showcased its potential using stereo cam-
eras on a helmet-mounted stick.
Mo2Cap2 [27] and
xR-EgoPose [20] have introduced single-camera methods,
which handle occlusion.
The former proposes a two-
branched heatmap, one for the lower body with a magnified
view. The latter adds a heatmap reconstructor to preserve
the probabilistic information of heatmaps. Recent methods
utilize an external camera view to make a weakly labeled
large-scale dataset [24] and a scene depth estimation model
2

to estimate 3D pose with volumetric heatmaps [25]. These
methods, however, require additional external cameras or
depth datasets from specific views.
Recently, a stereo egocentric setup has gained atten-
tion for a wide-view stereo perspective. EgoGlass [32] in-
troduces an unobtrusive eyeglass-mounted stereo camera
setup, minimizing obtrusiveness. It incorporates an addi-
tional segmentation branch on the heatmap estimator mod-
ule to improve the awareness of body parts and pixel cor-
respondence. UnrealEgo [2] introduces a publicly available
synthetic large-scale dataset based on the EgoGlass setup
and proposes to share weights and merge features across
the stereo view in the heatmap estimator. Ego3DPose [8]
suggests making an independent estimate of the 3D orienta-
tion of each limb, using the concatenated orientation vector
for the final decoder. We observed two problems in these
prior works, i.e., information loss in feature embedding and
data-dependant estimation of obscure joints, and propose
two corresponding techniques to address the problems.
2.2. 3D Human Pose Estimation with Transformer
The transformer-based architecture has been explored for
the 3D pose estimation task.
Epipolar Transformers [6]
utilizes attention to match features along the epipolar line
from the stereo view. Most methods focused on using trans-
formers for 2D to 3D pose lifting spatially and temporally.
PoseFormer [34] is the first transformer-based 2D-to-3D
pose lifting method consisting of spatial and temporal trans-
former networks. MixSTE [31] and PoseFormerV2 [33] im-
proved it with the per joint temporal characteristics and fre-
quency domain feature. Unlike prior works, we exploit the
transformer to effectively embed heatmap information for
accurate heatmap-to-3D pose lifting.
2.3. Skeletal Network Models
Multiple works utilize skeletal hierarchy for vision tasks.
For instance, Liu et al. [13] uses spatio-temporal LSTM to
iterate through all joints for action recognition. Most recent
efforts utilize a graph-based model to represent skeletal hi-
erarchy. The Graph Convolutional Networks [10] is widely
utilized for activity recognition [4] while ST-GCN [28]
models a dynamic skeletal graph in a spatiotemporal man-
ner. The graph-based models are adapted for the pose esti-
mation [28–30], using dynamic skeletal graphs with action-
specific edges or adopting adaptive ST-GCN [28, 29].
Our work is the first to leverage skeletal information in
the ego-centric setup.
Specifically, we address the chal-
lenge of obscure features, particularly for body extremities,
which impact the pose estimation of all body parts. Intro-
ducing a skeleton-aware uni-directional Propagation Net-
work model, we leverage clear visual cues from camera-
proximate joints to estimate the pose of body parts with ob-
scure visual features.

## conclusion
In this study, we introduce a novel heatmap-to-3D lifting
method tailored for the stereo egocentric setup, employ-
ing a transformer for efficient feature embedding and an
attention-driven Propagation Network focused on evident
features.
We demonstrate effective heatmap feature ex-
traction through the Grid ViT Heatmap Encoder, employ-
ing patch-wise communication with self-attention to pre-
serve correspondence between the heatmap and the feature
embedding. The Propagation Network utilizes visual cues
from the proximate parent joint, leveraging joint relational
information to predictively estimate less visible child joint
poses. Our experiments highlight significant advancements
over state-of-the-art stereo egocentric pose estimation meth-
ods, underscoring the efficacy of our proposed approach.
8

Acknowledgement
This work was supported by the National Research Foun-
dation of Korea(NRF) grant funded by the Korea govern-
ment(MIST) (No. 2022R1A2C3008495). This work was
supported by the National Research Foundation of Ko-
rea(NRF) grant funded by the Korea government(MSIT)
(No.RS-2023-00218601).
9

Attention-Propagation Network for Egocentric Heatmap to 3D Pose Lifting
Supplementary Material
A. Overview
The supplementary material contains the following:
• Dataset Processing
• Implementation
• Training
• Experiment
• Example Figure
• Limitations and Future Works
B. Dataset Processing
We explain the details of the train and test dataset we used
in this section. Our method requires a 2D and 3D pose an-
notation and stereo input images. The 2D annotation is nec-
essary for generating the heatmaps.
B.1. UnrealEgo
We utilize the full dataset, including metadata files and pre-
processed pickles. The public Ego3DPose [8] code loads
metadata and pickles. Their code adds 2D and 3D pose data
in the camera coordinate system and their limb heatmap rep-
resentation in the pickle files. Our method uses these final
pickles.
B.2. EgoCap
We used publicly available 2D pose annotation on the train
set. Additionally, we got the full ground truth 3D pose for
the train set of the EgoCap [17] dataset from the authors. In
the fisheye views of the dataset, images are projected only in
the circular area due to strong distortion. Thus, the original
images contain areas that do not have real views. Following
the Kang et al. [8], we cropped the image horizontally into a
square area centered at the x-axis focal center (fx) provided
in the dataset calibration data. We resized the images to 256
by 256 images to fit our model.
The dataset has a train set and 2D and 3D validation sets.
The 3D validation set contains a ground truth 3D pose and
is used for testing. The 2D validation dataset provides the
2D annotation for the images in the 3D validation sets from
a subject labeled 7. The 3D pose is converted from a mm
to a cm unit to scale the pose loss in accordance with the
UnrealEgo dataset.
C. Implementation
C.1. Grid ViT Heatmap Encoder
The 64 × 64 sized heatmaps are put into one image with
resolution 384 × 384. The image comprises 36 areas as
Norm
3x
Norm
Multi-Head
Attention
MLP
Figure 8. The ViT encoder architecture.
a 6 × 6 grid. The number of joint heatmaps is 30 for the
UnrealEgo [2] dataset and 34 for the EgoCap [17]. The
heatmaps fill in the grid in order. The areas that do not
correspond to any heatmap are masked in the ViT encoder
module and don’t impact the output.
We adopt the ViT encoder [3] architecture. Our imple-
mentation adopts the public Transformers [26] module ViT-
Model class for the PyTorch [16]. We removed the [CLS]
token since we are not using the module for a classification
task. Doing so improves pose estimation accuracy empir-
ically. The module follows the standard ViT [3] encoder
architecture shown in Fig. 8 that takes the input embedding
z and outputs feature embedding z′.
The ViT encoder takes embeddings of size 1024 per each
of 32NJ patches, z = [z1, z2, . . . , z32NJ]. The multi-head
attention layer has 8 heads. The intermediate layer size of
the MLP is 4096. The Grid ViT Heatmap Encoder uses
three ViT encoder layers. It outputs a total 16384 size of the
embedding vector from 16 patches for each heatmap. The
embedding vector is then compressed with MLP denoted
EK in the paper. The MLP has ReLU [1] non-linearity for
the intermediate layers. The MLP’s hidden sizes of the first
two layers are 2048 and 512, and the last layer outputs a
final embedding of size 128.
C.2. Propagation Network
In an extension of the typical LSTM [7], the Propagation
Unit’s relational features, joint features, hidden and cell
states, and gate outputs all have the same size. We chose
256 for the size.
1

C.2.1
Limb Heatmap Encoder
The limb heatmap encoder ER extracts relational features.
The encoder consists of three layers with the same structure
as the final MLP layers of the Grid ViT Heatmap Encoder,
with only an input size difference. The input two-channeled
limb heatmap [8] has 2 × 64 × 64 size. The encoder takes
it after flattening it. The encoder consists of three fully con-
nected layers, the first two layers with 2048 and 512 output
size, with the ReLU [1] activation, and the final layer out-
puts the embedding with a size 128.
C.2.2
Second Layer of the PU
The second layer of PU does not take distinct relational and
joint features. It takes the parent joint’s second layer cell
and hidden state with the first layer’s hidden state of the
joint. Since hidden states from different layers are used in
this section, let’s denote the n-th layer hidden states of i-th
joint hn,i. The additional forget gate in the second layer gi
controls the parent joint’s second PU layer’s hidden state,
resulting in the modified hidden state h′
2,i. This is formu-
lated as follows:
gi = σ(Wg · h1,i + bg)
(16)
h′
2,i = gi ⊙h2,parent(i)
(17)
The modified parent hidden state and the joint’s first
layer hidden state are input for the inner LSTM [7].
C.2.3
Internal LSTM of the PU
We explain the formulation of the LSTM [7] inside the PU
in more detail here.
Formulation of typical LSTM. The LSTM is formulated
as follows, where hi−1 denotes the hidden state of the pre-
vious step, ci−1 denotes the cell state of the previous step,
and xi denotes the input. Here, W and b denote weights
and biases for each gate. The symbol ⊙represents element-
wise multiplication, and the + sign represents element-wise
addition. tanh and σ denote the hyperbolic tangent and sig-
moid activation.
fi = σ(Wf · [hi−1, xi] + bf)
(18)
ii = σ(Wi · [hi−1, xi] + bi)
(19)
oi = σ(Wo · [hi−1, xi] + bo)
(20)
˜ci = tanh(Wc · [hi−1, xi] + bc)
(21)
ci = fi ⊙ci−1 + ii ⊙˜ci
(22)
hi = oi ⊙tanh(ci)
(23)
The fi, ii, and oi are forget, input, and output gates. ˜ci de-
notes the candidate cell value. hi and ci are the final hidden
and cell state for step i.
Formulation of internal LSTM. Unlike the LSTM taking
the cell and hidden state, the internal LSTM of the first PU
layer takes three states in addition to input joint features.
The three states are the modified parent’s hidden state h′
i,
the modified relational feature of the joint r′
i, and the cell
state of the parent cparent(i). The input is joint features FJ,i.
This section explains the first and second layers together;
thus, we denote the n-th layer of i-th joint with a n, i sub-
script, as in hn,i for the hidden state. In the computation
of the forget, input, and output gates and the candidate
cell value, a concatenated vector of the modified parent’s
hidden state and relational features, and the joint features
[h′
1,i, r′
1,i, FJ,i] replaces [hi−1, xi].
f1,i = σ(W1,f · [h′
1,i, r′
i, FJ,i] + b1,f)
(24)
i1,i = σ(W1,i · [h′
1,i, r′
i, FJ,i] + b1,i)
(25)
o1,i = σ(W1,o · [h′
1,i, r′
i, FJ,i] + b1,o)
(26)
˜c1,i = tanh(W1,c · [h′
1,i, r′
i, FJ,i] + b1,c)
(27)
For the second layer, the modified second layer parent
hidden state h′
2,i from the Sec. C.2.2 takes the place of
hi−1. The previous layer’s hidden state h1,i replaces input
xi, analogous to the standard multi-layered LSTM.
f2,i = σ(W2,f · [h′
2,i, h1,i] + b2,f)
(28)
i2,i = σ(W2,i · [h′
2,i, h1,i] + b2,i)
(29)
o2,i = σ(W2,o · [h′
2,i, h1,i] + b2,o)
(30)
˜c2,i = tanh(W2,c · [h′
2,i, h1,i] + b2,c)
(31)
The Propagation Unit takes features from the parent
joint, not the previous index. In the computation of the final
cell and hidden state, both layers of PU take cn,parent(i) in-
stead of ci−1 in the formula. The hidden state is computed
in the same way.
cn,i = fn,i ⊙cn,parent(i) + in,i ⊙˜cn,i
(32)
hn,i = on,i ⊙tanh(cn,i)
(33)
D. Training
D.1. Hardware Setup
We trained and tested our method on a server with NVIDIA
RTX A6000 GPU and AMD EPYC 7313 16-Core Processor
CPU.
2

D.2. Heatmap Estimator
The heatmap estimator is trained using UnrealEgo [2] code
and their scripts for the UnrealEgo dataset.
The default
configuration utilizes Adam [9] optimizer with a learning
rate 10−3. They train the network for 10 epochs, the later
5 epochs with linear decay, with batch size 16. For the
EgoCap dataset, we trained the heatmap est