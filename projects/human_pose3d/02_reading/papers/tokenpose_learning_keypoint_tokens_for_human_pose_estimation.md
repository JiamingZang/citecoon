# TokenPose: Learning Keypoint Tokens for Human Pose Estimation

> 2021 · id: W3203925315 · arXiv: 2104.03516 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Human pose estimation deeply relies on visual clues and
anatomical constraints between parts to locate keypoints.
Most existing CNN-based methods do well in visual repre-
sentation, however, lacking in the ability to explicitly learn
the constraint relationships between keypoints. In this pa-
per, we propose a novel approach based on Token repre-
sentation for human Pose estimation (TokenPose). In de-
tail, each keypoint is explicitly embedded as a token to
simultaneously learn constraint relationships and appear-
ance cues from images. Extensive experiments show that
the small and large TokenPose models are on par with
state-of-the-art CNN-based counterparts while being more
lightweight. Speciﬁcally, our TokenPose-S and TokenPose-L
achieve 72.5 AP and 75.8 AP on COCO validation dataset
respectively, with signiﬁcant reduction in parameters (↓
80.6% ; ↓56.8%) and GFLOPs (↓75.3%; ↓24.7%). Code
is publicly available1.

## introduction
2D human pose estimation aims to localize human
anatomical keypoints which deeply relies on both visual cue
and keypoints constraint relationships. It is a fundamental
task in computer vision, which has attracted extensive at-
tention from academia and industry.
Over the past decade, deep convolutional neural net-
works have achieved impressive performances on human
pose estimation due to their powerful capacity in visual rep-
resentation and recognition [8, 29, 22, 21, 38, 12, 37, 24].
Since heatmap representation has become the standard la-
*This work was done when Yanjie and Sen Yang were interns at
MEGVII Tech.
†Corresponding author.
1https://github.com/leeyegy/TokenPose
…
Token-
Pose
no. ey.(l) ey.(r) ea.(l) ea.(r) sh.(l) sh.(r) el.(l) el.(r) wr.(l) wr.(r) hi.(l) hi.(r) kn.(l) kn.(r) an.(r)
ankle(l)
CONSTRAINT cue 
VISUAL cue – attention map between the ankle(l) token and visual tokens.
Layer #1
Layer #2
Layer #3
Layer #N-2 Layer #N-1 Layer #N
Figure 1. The process of predicting the location of the left ankle.
For visual cue learning, the proposed TokenPose focuses on the
global context in the ﬁrst few layers, and then gradually converges
to some local regions as the network goes deeper. In the last few
layers, TokenPose has considered hip and knee in turn which are
close to the target keypoint, and ﬁnally localizes the position of
the left ankle. For constraint cue learning, TokenPose shows that
localizing the left ankle mostly relies on the left knee and right
ankle, corresponding to adjacency constraint and symmetric con-
straint respectively.
bel representation to encode the positions of keypoints,
most existing models tend to use fully convolutional lay-
ers to maintain the 2D-structure of feature maps until the
network output. Nevertheless, there are usually no concrete
variables abstracted by such CNN models to directly rep-
resent the keypoint entities, which limits the ability of the
model to explicitly capture constraint relationships between
parts.
Recently, Transformer [35] and its variants that origi-
nated from natural language processing (NLP) have merged
as new choices for various vision tasks. Its ability of model-
ing global dependencies is more powerful than CNN, which
points out a promising way to efﬁciently capture relation-
arXiv:2104.03516v3  [cs.CV]  13 Aug 2021

ships between visual entities/elements. And in the ﬁeld of
NLP, all language elements such as words or characters are
usually symbolized by embeddings or token vectors with
ﬁxed dimensions, so as to better measure their similarities
in a vector space, like the way of word2vec [20].
We borrow such a concept of “token” and present a
novel token-based representation for human pose estima-
tion, namely TokenPose. Speciﬁcally, we conduct two dif-
ferent types of tokenizations: keypoint tokens and visual
tokens. Visual tokens are yielded by uniformly splitting an
image into patches and mapping the ﬂattened patches into
embeddings with ﬁxed dimensions. Meanwhile, keypoint
tokens are randomly initialized embeddings, each of which
represents a speciﬁc type of keypoint (e.g., left knee, left
ankle, right eye, etc.). The resulting keypoint tokens can
learn both visual clues and constraint relations from inter-
actions with visual tokens and the other keypoint tokens re-
spectively. An example of how the proposed model predicts
the location of left ankle is shown in Figure 1. The positions
of keypoints are ﬁnally estimated over the token-based rep-
resentation outputted by our network. The architecture of
TokenPose is illustrated in Figure 2.
It is worth noting that TokenPose learns the statistic con-
straint relationships between keypoints from large amounts
of data.
Such information is encoded into keypoint to-
kens that can record their relationships by vector similari-
ties. During inference, TokenPose associates keypoint to-
kens with those visual tokens whose corresponding patches
possibly contain the target keypoints. By visualizing the
attentions, we can observe how they interact and how the
model exploits cues to localize keypoints.
The contributions are summarized as follows:
• We propose to use token to represent each keypoint en-
tity. In this way, visual cue and constraint cue learning
are explicitly incorporated into a uniﬁed framework.
• Both hybrid and pure Transformer-based architectures
are explored in this work. As far as we know, proposed
TokenPose-T is the ﬁrst pure Transformer-based model
for 2D human pose estimation.
• We conduct experiments over two widely-used bench-
mark datasets: COCO keypoint detection dataset [19]
and MPII Human Pose dataset [1].
TokenPose
achieves competitive state-of-the-art performance with
much fewer parameters and computation cost com-
pared with existing CNN-based counterparts.

## method
Token fusion
AP
#Params
TokenPose-S

73.5
6.2M
TokenPose-S
!
72.6
6.7M
TokenPose-L+/D12

75.3
35.8M
TokenPose-L+/D12
!
75.5
38.2M
Table 6. The effects of keypoint token fusion for different models.
The input image size is 256 × 192.
MLP head to obtain the ﬁnal heatmaps.
We report the results of TokenPose-S and TokenPose-
L+/D12 with and without keypoint token fusion in Table 6.
For TokenPose-L+/D12, using keypoint token fusion im-
proves the result by 0.2 AP. However, for small variant like
TokenPose-S, it causes performance degradation instead.
For TokenPose-Large with keypoint token fusion, we
ﬁnd the lower Transformer layers provide more meaningful
evidence than the higher layers to understand the interaction
process. We attribute this to the token fusion, which enables
the ﬁnal keypoint representation to directly exploit the in-
formation from the early layers. And such a phenomenon
does not appear in the TokenPose-Small model without to-
ken fusion, in which the attention interactions progressively
show clear and meaningful attention process. We will fur-
ther describe it in Sec. 4.5.
Note that keypoint token fusion is only used in
TokenPose-L given its very deep and complex structure.
Position embedding.
Keypoint localization is a position-
sensitive vision task. To illustrate the effect of position em-
bedding, we conduct experiments based on TokenPose-S-v1
with different position embedding types (i.e., no position
embedding, 2D sine and learnable position embedding). As
Table 7 shown, employing position embedding signiﬁcantly
Position embedding
#Params
GFLOPs
AP
AR

6.62M
2.07
67.0
73.4
Learnable
6.67M
2.23
71.4
77.1
2D sine
6.67M
2.23
72.5
78.0
Table 7. Results for various positional encoding strategies for
TokenPose-S-v1. The input image size is 256 × 192.
improves the performance by 5.5 AP at most. In particular,
2D sine position embedding performs better than learnable
position embedding, which is as expected since the 2D spa-
tial information is required for predicting heatmaps.
Scaling.
Model scaling is a widely-used method to boost
model performance, including width-wise [35, 10] scaling
and depth-wise scaling [3, 26]. As shown in Table 3, both
increasing depth and width help improve the results.
4.5. Visualization
To illustrate how the proposed TokenPose explicitly uti-
lizes visual cue and constraint cue between parts to localize
keypoints, we visualize the details during inference. We
observe that a single model has similar behaviors for most
common examples.
We randomly choose some samples
from the COCO validation set and visualize the details in
Figure 3 and Figure 5.
Appearance cue.
We visualize the attention maps be-
tween keypoint tokens and visual tokens of different Trans-
former layers in Figure 3. The attention maps are formed
based on the attention scores between keypoint tokens and
visual tokens. Note, we reshape the 1D sequence of atten-
tion scores according to their original space positions for
the visualization.

Layer #1 to Layer #12
knee (l)
knee (r)
ankle (l)
knee (l)
knee (r)
ankle (l)
Figure 3. Visualization of the attention maps between keypoint tokens (e.g., nose, elbow(l), and elbow(r), etc.) and visual tokens in different
layers of TokenPose-S, which consists of 12 Transformer layers. Note that we transform all visual token into its corresponding patch areas
in the image. Redder color areas mean that the given type of keypoint has higher attention at these patches/visual tokens. The examples
shown above and below are non-occluded and occluded cases, respectively.
nose
eye(l)
eye(r)
ear(l)
ear(r)
sho.(l)
sho.(r)
elb.(l)
elb.(r)
wri.(l)
wri.(r)
hip(l)
hip(r)
kne.(l)
kne.(r)
ank.(l)
ank.(r)
nose
eye(l)
eye(r)
ear(l)
ear(r)
sho.(l)
sho.(r)
elb.(l)
elb.(r)
wri.(l)
wri.(r)
hip(l)
hip(r)
kne.(l)
kne.(r)
ank.(l)
ank.(r)
0.000
0.025
0.050
0.075
0.100
0.125
0.150
0.175
similarity
Figure 4. The inner product matrix of the learned keypoint tokens.
We take the keypoint tokens that are fed into the ﬁrst Transformer
layer, compute their inner product matrix, scale them by
√
d, and
use softmax to normalize them at columns. Thus each row can
represent the learned prior constraint relationships for a given type
of keypoint with other ones.
We choose two images for comparisons in Figure 3. As
we can see, with the layer depth increasing, what the key-
point tokens capture is gradually from the whole body ap-
Keypoint
Constraint
Top-1
Top-2
left shoulder
left elbow (0.026)
right shoulder (0.012)
left hip
right hip (0.037)
left knee (0.037)
right ankle
right knee (0.023)
left ankle (0.014)
nose
left eye (0.016)
right eye (0.016)
right wrist
right elbow (0.012)
left wrist (0.011)
Table 8. Top-2 constraints with regard to some keypoints for a ran-
domly chosen sample. The values in parentheses represent the at-
tention scores obtained from the ﬁnal self-attention layer.
pearance cues to more precise local part cues. In the ﬁrst
few layers, multiple crowded persons may simultaneously
give appearance cues as interference, but the model can pro-
gressively attend to the target person. In the subsequent lay-
ers, different types of keypoint tokens attend to their adja-
cent keypoints and the joints with high conﬁdence evidence.
When inferring the occluded keypoints, the model be-
haves differently. As shown in Figure 3, we notice that the
occluded left-ankle keypoint token pays higher attention to
its symmetric joint (i.e., right-ankle) to obtain more clues.
Keypoint constraints cue.
The attention maps of key-
point tokens in the 2nd, 4th, 6th, 8th, 10th and 12th self-

Layer #2
Layer #4
Layer #8
Layer #6
Layer #10
Layer #12
Figure 5. The attention interactions between keypoint tokens in the 2nd, 4th, 6th, 8th, 10th and 12th Transformer layers of TokenPose-S.
attention layers are visualized in Figure 5. In the ﬁrst few
layers, each keypoint pays attention to almost all other ones
to construct global context. As the network goes deeper,
each keypoint tends to mostly rely on several parts to yield
the ﬁnal prediction.
Speciﬁcally, we show top-2 constraints of some typical
keypoints based on the ﬁnal self-attention layer in Table 8.
In particular, we observe that the top-2 constraints tends to
be the adjacent and symmetric constraint of the target key-
point, which also conforms to the human visual system. For
instance, predicting the right wrist mostly focuses on the
constraints from the right elbow and left wrist, correspond-
ing to its adjacent and symmetric constraints respectively.
Keypoint tokens learn prior knowledge from data.
In proposed TokenPose, the input [keypoint] tokens
which are taken as input to the ﬁrst Transformer layer are
totally learnable parameters. Such knowledge is related to
the bias from the whole training dataset but independent of
any speciﬁc image. During inference it will be exploited
to facilitate the model to decode visual information from a
concrete image and further make predictions.
We point out that such [keypoint] tokens act like
object queries in DETR [5], in which each query slot ﬁ-
nally has learned prior preference from data to specialize
on certain areas and box sizes. In our settings, the input
[keypoint] tokens learn statistical relevance between
keypoints from the dataset, serving as prior knowledge.
To show what information is encoded in these input key-
point tokens, we calculate the inner product matrix of them.
After being scaled and normalized, the matrix is visualized
in Figure 4. We can see that one tends to be highly similar to
its symmetric keypoints or adjacent keypoints. For instance,
left hip is mostly related to right hip and left shoulder with
similarity score 0.104 and 0.054 respectively. Such ﬁnding
conforms to our common sense and reveals what the model
learns. We also notice there is a work [31] which analyzes
the statistic distributions between joints by computing the
mutual information from MPII dataset annotation. In con-
trast, our model is able to automatically learn prior knowl-
edge from training data and explicitly encode it in the input
[keypoint] tokens.

## experiments
4.1. Model Variants
We provide both hybrid and pure Transformer-based
variants for TokenPose.
For hybrid architecture, convo-
lutional neural networks with various depths are used for
image feature extracting.
The conﬁguration details are
presented in Table 1.
Note, TokenPose-T* is the pure
Transformer-based variant.
TokenPose-S*, TokenPose-B
and TokenPose-L* adopt stem-net2, HRNet-W32 [29] and
HRNet-W48 [29] as backbone, respectively.
In this paper, brief notation is used for convenience.
For example, TokenPose-L/D24 means the “Large” vari-
ant with 24 Transformer layers. Unless noted otherwise,
TokenPose-S and TokenPose-L are used as the abbrevia-
tions for TokenPose-Small-v2 and TokenPose-Large/D24.
4.2. COCO Keypoint Detection
Dataset.
The COCO dataset [19] consists of more than
200, 000 images and 250, 000 person instances which are
labeled with 17 keypoints. The COCO dataset is divided
into train/val/test-dev sets, which contains 57k, 5k and 20k
images respectively. All the experiments reported in this
paper are trained only on the train2017 set. The methods
are evaluated on the val2017 set and test-dev2017 set.
Evaluation metric.
We adopt standard average precision
(AP) as our evaluation metric on the COCO dataset. AP
is calculated based on Object Keypoint Similarity (OKS):
OKS =
P
i exp(−ˆd2
i /2s2k2
i )σ(vi>0)
P
i σ(vi>0)
, where ˆdi is the Eu-
clidean distance between the i-th predicted keypoint coor-
dinate and the corresponding groundtruth, vi is the visibil-
ity ﬂag of the keypoint, s is the object scale, and ki is a
keypoint-speciﬁc constant.
Baseline settings.
For model training, we use the Adam
optimizer. For HRNet [29] and SimpleBaseline [38], we
simply follow the original settings in their paper.
Implementation details.
In this paper, we follow the two-
stage top-down human pose estimation paradigm similar to
[29, 7, 38, 25]. In the paradigm, the single person instance
is ﬁrstly detected by a person detector, and then keypoints
are predicted. We adopt the widely-used person detectors
provided by SimpleBaseline [38] on the validation set and
test-dev set. To alleviate the quantisation error, the well-
designed coordinate decoding strategy [42] is adopted.
For our work, the base learning rate is set as 1e-3, and
is dropped to 1e-4 and 1e-5 at the 200th and 260th epochs,
2It’s widely used to quickly downsample the feature map into 1/4 input
resolution, consisting of a very shallow convolutional structure[29, 8].

## related_work
2.1. Human Pose Estimation
Deep convolutional neural networks have been applied
to human pose estimation which greatly boost the model
performance [32, 13, 29, 38, 22, 17, 21, 4, 7].
Recent heatmap-based methods tend to improve per-
formance by stacking deeper network architecture. Hour-
glass [22] stacks blocks to enhance the heatmap estimation
quality. SimpleBaseline [38] designs a simple architecture
by stacking transposed convolution layers and achieves im-
pressive performances. HRNet [29] proposes to maintain
high-resolution representation through the whole process in
order to provide spatially precise heatmap estimation. How-
ever, it is still hard for convolutional neural networks to cap-
ture and model constraint relationships between keypoints,
which are important for human pose estimation.
2.2. Vision Transformer
Transformer [35] adopts encoder-decoder architecture
based on self-attention and feed-forward network, which
achieves great success in NLP. Recently, Transformer-based
models [11, 34, 5, 14, 44, 45, 9, 39, 6, 36, 41, 28] have also
shown enormous potential in various vision tasks.
Detection.
DETR [5] proposes a Transformer based ar-
chitecture to handle object detection end-to-end, effec-
tively eliminating the need for many hand-designed com-
ponents. Deformable DETR [45] then proposes to make
attention modules only attend to a small set of key sam-
pling points around a reference, achieving better perfor-
mance than DETR. UP-DETR [9] unsupervisedly pre-train
DETR by design randomly cropped patches.
Classiﬁcation.
ViT [11] proposes a pure Transformer
model with patch embedding representation, which is pre-
trained on large amounts of data and then ﬁne-tuned on Im-
ageNet dataset. DeiT [34] introduces a distillation token to
ViT to learn knowledge from a teacher network, to avoid
the pre-training on a large dataset. Tokens2Token [41] pro-
gressively encodes image into tokens and models the local
structure information to reduce the sequence length.
Human Pose Estimation.
Recent several works [27, 15,
18, 39, 43, 28] introduce Transformer for human pose esti-
mation. PoseFormer [43] introduces Transformer for 3D
pose estimation, based on 2D pose sequences in video
frames. TransPose [39] tends to utilize attention layers built
in Transformer to reveal the long-range dependencies of the
predicted keypoints. However, TransPose lacks the ability
to directly model the constraint relationships between key-
points. In this work, we propose to explicitly represent key-
points as token embeddings. And then both visual clues
and constraint relations are simultaneously learned through
self-attention interactions.

Linear Projection
…
…
Transformer Layer
.
.
.
MLP head
…
Transformer Layer
CNN
Feature
map
…
Layer #1 
                   Layer #N
Visual attention evolution to predict the obscured left ankle
keypoint token
visual token
sum
feature patch
position embedding
2D reshape
nose eye(l) ankle(r)
…
LayerNorm
Multi-Head
Attention
LayerNorm
Feed-Forward
…
Figure 2. Schematic illustration of the proposed TokenPose. The feature maps extracted by CNN backbone are uniformly split into patches
and ﬂattened to 1D vectors. Visual tokens are yielded by adopting a linear projection to embed the ﬂattened vectors. In addition, keypoint
tokens are initialized randomly to represent each speciﬁc type of keypoint. Then, the 1D sequence of visual tokens and keypoint tokens are
taken as input to Transformer encoder. Both appearance cues and anatomical constraint cues are captured through self-attention interactions
in each Transformer layer. Finally, the keypoint tokens outputted by the last Transformer layer are used to predict the keypoints heatmaps
via an MLP head.

## conclusion
In this paper, we propose a novel token-based presen-
tation for human pose estimation, namely TokenPose. In
particular, we split the image into patches to yield visual
tokens and represent keypoint entities into token embed-
dings. This way, the proposed TokenPose is able to ex-
plicitly capture appearance cues and constraint cues by the
self-attention interaction. We show that a low-capacity pure
Transformer architecture without any pre-training can also
work well. Besides, the hybrid architectures achieve com-
petitive results compared to the state-of-the-art CNN-based
methods at a much lower computational cost.
Acknowledgments
This paper is supported in part by the National Key R&D
Plan of the Ministry of Science and Technology (Project
No. 2020AAA0104400), and in part by the National Key
Research and Development Program of China under Grant
2018YFB1800204, the National Natural Science Founda-
tion of China under Grant 61771273, the R&D Program of
Shenzhen under Grant JCYJ20180508152204044, and in
part by the National Natural Science Foundation of China
under Grant 61773117.