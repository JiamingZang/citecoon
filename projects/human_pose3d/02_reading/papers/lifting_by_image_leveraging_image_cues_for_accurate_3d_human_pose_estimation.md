# Lifting by Image – Leveraging Image Cues for Accurate 3D Human Pose Estimation

> 2024 · id: W4393156217 · arXiv: 2312.15636 · pdf: https://ojs.aaai.org/index.php/AAAI/article/download/28596/29159 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
The “lifting from 2D pose” method has been the dominant
approach to 3D Human Pose Estimation (3DHPE) due to
the powerful visual analysis ability of 2D pose estimators.
Widely known, there exists a depth ambiguity problem when
estimating solely from 2D pose, where one 2D pose can be
mapped to multiple 3D poses. Intuitively, the rich semantic
and texture information in images can contribute to a more ac-
curate “lifting” procedure. Yet, existing research encounters
two primary challenges. Firstly, the distribution of image data
in 3D motion capture datasets is too narrow because of the
laboratorial environment, which leads to poor generalization
ability of methods trained with image information. Secondly,
effective strategies for leveraging image information are lack-
ing. In this paper, we give new insight into the cause of poor
generalization problems and the effectiveness of image fea-
tures. Based on that, we propose an advanced framework.
Specifically, the framework consists of two stages. First, we
enable the keypoints to query and select the beneficial fea-
tures from all image patches. To reduce the keypoints at-
tention to inconsequential background features, we design a
novel Pose-guided Transformer Layer, which adaptively lim-
its the updates to unimportant image patches. Then, through
a designed Adaptive Feature Selection Module, we prune
less significant image patches from the feature map. In the
second stage, we allow the keypoints to further emphasize
the retained critical image features. This progressive learn-
ing approach prevents further training on insignificant image
features. Experimental results show that our model achieves
state-of-the-art performance on both the Human3.6M dataset
and the MPI-INF-3DHP dataset.

## introduction
Monocular 3D Human Pose Estimation (3DHPE) aims to
estimate the relative 3D coordinates of human joints from
an image. It is a fundamental computer vision task related
to a wide range of applications, including human motion
forecasting (Ding and Yin 2022; Liu et al. 2020), human ac-
tion recognition (Dang, Yang, and Yin 2020), human-centric
generation (Cao et al. 2023b, 2022, 2023a) and so on.
In recent years, 3D human pose estimation has been dom-
inated by the “lifting” technique (Martinez et al. 2017). This
*Corresponding author
Copyright © 2024, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.
Background Overfitting
Assistance Image Feature
2D Pose
Image
Key Image Feature
3D Pose
query
regress
2D Pose
Image
3D Pose
Figure 1: The main idea of this paper is to design a frame-
work that enables 2D poses to regress 3D poses by querying
information from the image. The framework is specifically
designed based on two key insights: First, excessive atten-
tion to dataset-biased background information leads to poor
generalization ability. Second, it is not only the image fea-
tures corresponding to the keypoints that are helpful for the
task, but also the associated body structural positions of the
keypoints that can provide valuable assistance.
approach consists of two stages. First, utilize off-the-shelf
2D pose estimators (Sun et al. 2019; Newell, Yang, and
Deng 2016; Dang et al. 2022) to estimate the 2D pose from
the image and then regress the 3D pose from the obtained
2D human pose. Compared to direct estimation, this cas-
caded approach has the following advantages: 2D estima-
tor is trained on more diverse and extensive 2D human pose
datasets, which enables stronger visual perception and gen-
eralization ability (Martinez et al. 2017). Besides, the “lift-
ing” can be trained with infinite 2D-3D pairs by setting dif-
ferent camera views (Xu et al. 2021). Nevertheless, estimat-
ing 3D pose from 2D pose introduces the depth ambiguity
problem, one 2D pose can be mapped to multiple 3D poses.
Intuitively, rich texture and semantic information in im-
ages can assist in regressing a more accurate 3D pose from
2D pose. There has been some exploration in this direction.
For example, Nie, Wei, and Zhu; Xu et al. segment im-
arXiv:2312.15636v1  [cs.CV]  25 Dec 2023

Poor Generalization Ability
Figure 2: Cross-dataset evaluation between straightforward
image-based model and our approach. With continuous
training on Human3.6m, the former’s accuracy on 3DHP de-
creased, highlighting poor generalization ability.
age patches around keypoint locations to aid in generating
the 3D pose. Likewise, Zhao et al.; Liu et al. introduced a
method of superimposing extracted image features around
keypoints’ position onto 2D keypoints to offer complemen-
tary information to the network. Yet despite the considerable
progress, there exist some issues that need to be addressed.
Firstly, because 3D human motion datasets were primarily
captured in constrained laboratory environments, the distri-
bution of image data is limited. Consequently, methods that
are trained with image information tend to suffer from poor
generalization ability, as shown in Fig 2. Additionally, effec-
tive strategies for leveraging image information are lacking.
This paper gives a novel insight into the cause of weak
generalization ability and the specific effectiveness of image
information in predicting the 3D pose. Based on that, we
propose a novel framework that estimates 3D human pose
from 2D pose by leveraging effective image cues, as shown
in Fig 1.
To begin, we utilize the attention mechanism (Vaswani
et al. 2017) to study the response of human keypoints to the
image features. By analyzing the attention maps, we derived
two noteworthy insights: 1. In general, for all keypoints, the
attention maps exhibit a high-proportion, wide-range, and
indiscriminate emphasis on irrelevant background informa-
tion outside the human body. This may shed light on the
weak generalization abilities of image-based methods, as
they overly focus on dataset-biased information. 2. For a
specific keypoint, its required image features are not con-
fined solely to its own location in the image. Instead, the
required positions also encompass body structure positions
that provide depth information for that keypoint. For in-
stance, the features of elbow can be instrumental in esti-
mating the depth of wrist keypoint. This underscores the
constraints of previous methods that exclusively concatenate
localized image patches or features around keypoints (Nie,
Wei, and Zhu 2017; Zhao et al. 2019; Liu et al. 2019).
Based on these understandings, we propose a novel 3D
pose estimation framework. The key concept is to allow the
keypoints to adaptively focus on critical image features. To
give an overview, the progressive learning framework con-
sists of two stages. We enable the keypoints to query and
select the beneficial features from all image patches in the
first stage, called “Broad Query”. Then we prune the irrel-
evant image features (mostly background features). At last,
we allow the keypoints to further explore information from
these critical image features to obtain an accurate 3D pose,
called “Focused Exploration”.
Specifically, in Stage 1, we introduce a Pose-guided
Transformer Layer, which effectively reduces the keypoints’
attention to the background. It leverages the pose-to-image
attention matrix to allow the image features to reversely
query and aggregate the keypoints features. Through our de-
sign, the more crucial image features can extract more rele-
vant information from the keypoint, while less important im-
age features, like background patches, receive comparatively
less information. Then we proposed an Adaptive Feature Se-
lection Module, which aims to rank and prune the less im-
portant image features by the attention mechanism. In Stage
2, the keypoints are allowed to refocus on critical human
image features by several Transformer Layers. Through this
cascaded approach, the keypoints are empowered to dynam-
ically explore critical features broadly and simultaneously
prevent over-training to the background features.
We demonstrate quantitative results by conducting our
method on standard 3D human pose benchmarks. Experi-
mental results show that our method outperforms state-of-
the-art performance on Human3.6M (Ionescu et al. 2013)
and MPI-INF-3DHP (Mehta et al. 2017). Mention that our
method not only significantly improves the accuracy of
single-frame 3D pose estimation but also outperforms even
the accuracy of 3D pose estimation networks based on tem-
poral information. Our contribution can be summarized as
follows:
• We propose two novel insights about 3DHPE methods
involving image information. For one thing, overly fo-
cusing on dataset-biased background leads to poor gen-
eralization ability. For another, valuable image patches
for estimating specific keypoint’s 3D coordinates are not
confined to its exact image location; they extend to areas
with structurally related positions.
• We propose a 3DHPE framework leveraging effective
image features in two stages: broad query followed by
focused exploration. It not only enables keypoints to de-
termine all the necessary image features but also pre-
vents excessive training on the background, thus improv-
ing generalization.
• We propose a novel Pose-guided Transformer Layer that
effectively improves the keypoints’ ability to significant
features. Besides, we propose an Adaptive Feature Se-
lection Module, which adaptively stops irrelevant image
features from further training.

## method
In this section, we provide a detailed description of the pro-
posed framework, as illustrated in Fig 5. Given a 2D pose
J2d ∈RN×2, our method aims to reconstruct the 3D pose
J3d ∈RN×3 by effectively leveraging the information from
a cropped image I ∈Rh×w×3, where N is the number of
keypoints, h, w is the input image size. To accomplish this,
we proposed a Progressive Training framework. It consists
of two stages. In Stage 1, we allow the keypoints to query
beneficial information from all image features under coarse
pose supervision until convergence. Subsequently, to coun-
teract the detriment of background features’ excessive train-
ing on generalization, we employ an Adaptive Feature Se-
lection Module to prune less crucial image features. Then,
in Stage 2, keypoints exclusively query the preserved image
features to generate a refined pose.
Stage 1: Broad Query
The image I is fed into a 2d-pose-estimation-pretrained im-
age encoder, resulting in features FI ∈RH×W ×d, which
are flattened into tokens TI ∈RHW ×d with sequence length
HW and dimension d. Similarly, the 2D pose J2d is trans-
formed to pose tokens TP ∈RN×d by linear projection.
Subsequently, the image tokens and keypoint tokens are
fed into three consecutive Transformer Layers. Situated in
the middle is the specially crafted Pose-guided Transformer
Layer, intended to selectively enhance image tokens while
diminishing the keypoints’ focus on irrelevant image tokens.
The resulting keypoint tokens are then projected linearly to
generate the coarse 3D pose denoted as J3d1 ∈RJ×3.
Transformer Layer consists of three consecutive mod-
ules, including Multi-head Self-Attention (MSA), Multi-
head Cross-Attention (MCA), and Feed Forward Network
(FFN). Multi-head Attention can be formulated as:
A(Q, K, V ) = softmax(Q · K⊺
√
d
) · V
(1)
In MSA, keypoints tokens are linearly mapped to Queries
Q ∈RN×d, Keys K ∈RN×d, and Values V ∈RN×d.
Similarly, in MCA, keypoints tokens are linearly mapped to
Queries Q ∈RN×d, Image tokens are linearly mapped into
Keys K ∈RHW ×d, and Values V ∈RHW ×d.
Pose-guided Transformer Layer. We designed a Pose-
guided dual attention structure that effectively reduces the
keypoints’ attention to the background. It leverages the pose-
to-image attention matrix to allow the image features to re-
versely query and aggregate the keypoints features. By our
design, the more crucial image patches can obtain more in-
formation from the keypoint features.
Specifically, the novel attention mechanism produces two
outputs: keypoint tokens ˆ
TJ and enhanced image tokens ˆTI.
The update of image tokens will be influenced by the update
of keypoint tokens through Attention Map (A). The formu-
lations are as follows:
A = softmax(Q · K⊺
√
d
)
ˆ
TJ = A · VI + TJ
ˆTI = A⊺· VJ + TI
(2)

Pruned feature
Enhanced Feature
mean
Pruning by 
proportion
aggregate
𝐽!
𝐽"
𝐽#
I# I" I! I$ I%
Retained Patch
Figure 6: Details of Adaptive Feature Selection Module.
Similarly, keypoints tokens are linearly mapped to Queries
Q ∈RN×d, Image tokens are linearly mapped into Keys
K ∈RHW ×d. VI ∈RHW ×d and VJ ∈RN×d are Values
linearly mapped from images tokens and keypoints tokens,
respectively. The attention map A ∈RN×HW represents
the weighting that keypoint tokens assign to image tokens,
and it is normalized using the softmax function. Similarly,
the transposed attention map A⊺∈RHW ×N represents the
weights image tokens assign to keypoint tokens. Because of
the normalization before transposition, intuitively, the im-
age tokens deemed more important receive a greater overall
weight from the keypoint tokens. This signifies that image
tokens with higher significance can gather more information
from keypoint tokens, while on the contrary, less significant
image tokens (typically background tokens) collect limited
information.
We choose to replace only the second Transformer Layer
with the proposed Pose-guided Transformer Layer for the
following reasons: before guiding the update of image fea-
tures, the keypoint tokens require an initial perception of the
image features (by the first transformer layer) to evaluate
their significance. The final layer cannot be replaced, as the
update of image tokens lacks direct supervision in Stage 1.
Adaptive Feature Selection Module
To prevent less significant image tokens from further train-
ing and aggregating critical image tokens. We propose an
Adaptive Feature Selection Module, details shown in Fig.
6. We leverage the attention map from the last transformer
layer to rank the importance of image tokens. For simplicity,
we aggregate the attention weights of all keypoints on image
features and set a retention rate, denoted as r ( 0 < r < 1).
The top r × HW image tokens with the highest weights are
retained.
Stage 2: Focused Exploration
In this stage, we allow the keypoint tokens to further dig
information from selected critical image tokens and generate
a refined 3D pose J3d2 ∈RJ×3. Specifically, we freeze the
weights trained in the former stage and feed the keypoints
tokens and selected image tokens into a new Transformer
Block consisting of several Transformer Layers. Then, the
output keypoint tokens will be projected to a refined 3D pose
J3d2by Linear Projection.
Loss Function
Our model is traind with Mean Squared Error (MSE) loss.
L =
J
X
i=1
∥Yi −ˆYi∥2
(3)
where Yi and ˆYi represent the predicted and ground-truth
3D pose of joint i, respectively.

## experiments
Datasets and Evaluation Metrics
We evaluate our method on two widely-used datasets for
3DHPE: Human3.6M (Ionescu et al. 2013) and MPI-INF-
3DHP (Mehta et al. 2017).
Human3.6M (H3.6M) is the largest and most representative
benchmark for 3DHPE. Following Martinez et al., we use
subject S1, S5, S6, S7 and S8 for training, and S9, S11 for
testing. We down-sampled the original frame rate from 50
fps to 5 fps for faster training. The Mean Per Joint Position
Error (MPJPE) is computed under two protocols: Protocol
1 computes MPJPE between ground truth and the estimated
3D poses after aligning their root (pelvis) keypoints; Pro-
tocol 2 is the MPJPE after aligning the estimated 3D pose
with the ground truth using translation, rotation, and scale
(P-MPJPE).
MPI-INF-3DHP (3DHP) provides monocular videos of six
subjects acting in three different scenes, including indoors
and outdoors. This dataset is often used to evaluate the gen-
eralization performance of different models. Following the
convention, we directly apply our model trained on H36M
dataset to this dataset without re-training. We report results
using three metrics, Mean Per Joint Position Error (MPJPE),
Percentage of Correctly estimated Keypoints (PCK) with a
threshold of 150 mm, and Area Under the Curve (AUC) a
range of PCK thresholds.
Implementation Details
We take HRNet-w32 as our backbone with input size 256 ×
192, which is pretrained on MS COCO 2017 dataset (Lin
et al. 2014), provided by Sun et al.. The retention r is set to
0.3. The number of Transformer layers in Stage 2 is set to 3.
For a fair comparison, following previous work (Pavllo et al.
2019; Martinez et al. 2017), we obtain 2D pose detections
cascaded pyramid network (CPN) (Chen et al. 2018) and
stacked hourglass network (SH) (Newell, Yang, and Deng
2016). We take the ground-truth bounding boxes. Our model
is implemented in Pytorch and optimized via Adam. All ex-
periments are conducted on two NVIDIA RTX 3090 GPUs.
The initial learning rate is set to 0.001 with a shrink factor of
0.9 per 4 epochs with 128 batch size. We first train the ini-
tial interaction stage and image encoder for 20 epochs and
freeze them to train the remaining modules.
Comparison with the State-of-the-art Methods
Results on Human3.6M. The proposed method is com-
pared with the state-of-the-art methods on Human3.6M. The
result and comparison of our model with SH detected, CPN

## related_work
In the past few years, there has been extensive research
on deep-learning-based algorithms for monocular 3D hu-
man pose estimation. Methods that directly regress 3D pose
from the image are popular in the early stages (Li and Chan

H3.6M Dataset
3DHP Dataset
Figure 3: Examples of attention map visualization of all key-
points on image features in two datasets.
Traindata
Testdata
MPJPE↓
Background Attention
H3.6M
H3.6M
30.4
73%
H3.6M
3DHP
74.2
75%
Table 1: Comparison of different test datasets and observed
issues of excessive attention to the background and poor
generalization.
2015). However, these approaches suffered from limited per-
formance due to their reliance on training and testing within
the constraints of 3D Motion Capture data (Xu et al. 2021).
To address this limitation, the “lifting” method emerged as
the dominant approach, offering better solutions to the prob-
lem.
“Lifting” Based 3D Human Pose Estimation
“Lifting” based approaches leverage off-the-shelf 2D hu-
man pose estimators trained on large and more diverse 2D
datasets. By adopting this, the process of 3D human pose
estimation is simplified to lifting the 2D pose to 3D pose
without image participation. Martinez et al. first proposed a
fully connected residual network in this approach. To han-
dle the issue of depth ambiguity in the lifting process, some
methods have leveraged temporal information (Pavllo et al.
2019; Chen et al. 2021) or proposed models with multiple
hypotheses (Li and Lee 2019; Li et al. 2022).
Fusion Approach
Apart from the two mainstream approaches, there exist some
methods that combine 2D pose with image information. De-
spite the remarkable attempts made by these methods, they
still exhibit certain limitations. For example, some methods
did not leverage off-the-shelf 2D pose estimators to gener-
ate 2D pose (Zhao et al. 2019; Liu et al. 2019). These ap-
proaches not only add a burden to the network but also fail to
leverage the benefits of 2D estimators mentioned before. Be-
sides, some methods employ rudimentary approaches to in-
tegrate image information. For instance, Nie, Wei, and Zhu;
Xu et al. segment image patches around keypoint positions
to assist in generating the 3D pose. Similarly, Zhao et al.; Liu
et al. overlay image features extracted from keypoints posi-
tion onto 2D keypoints. Nevertheless, the insight proposed
in the next section proves this local concatenation approach
might be ineffective. Moreover, Zhou et al.; Gong et al. uti-
lize 2D keypoints heatmap on image to provide extra infor-
knee
ankle
wrist
knee
ankle
wrist
elbow
elbow
Figure 4: Visualization examples of heatmaps depicting the
attention of specific keypoints.
mation. Indeed, heatmap only contains limited information,
which may not be sufficient to accurately regress 3D poses.
Insight of Image effect to 3DHPE
In this section, we study the roles and limitations of image
features in estimating 3D pose using attention mechanisms.
The keypoint-to-image attention map represents which im-
age patches offer beneficial information for estimating the
3D coordinates of that keypoint.
Background Overfitting
Given the task of estimating the relative coordinates of hu-
man keypoints, the presence of non-contact backgrounds in
3D capture-environment datasets can be considered a form
of dataset-biased noise. When we visualized the average at-
tention maps of keypoints on the images, we observed a
wide-ranging and indiscriminate focus on background fea-
tures, as shown in Fig 3. This reveals the model’s overfit-
ting to background information, which could be a potential
cause of the poor generalization of image-based models. We
further quantified the proportion of attention on background
features, and found very high proportions on both datasets
(73%, 75%), as shown in Table 1.
Structural Assistance
Logically speaking, for a specific human body keypoint in
the image, the keypoint’s own image features can only pro-
vide its 2D coordinates in the image. However, the relative
depth coordinate with respect to the pelvis point requires
prior knowledge derived from combining other human struc-
ture features. We present attention of specific keypoints on
image features, examples shown in Fig 4. Not surprisingly,
our findings lead to the conclusion: not only the image fea-
tures of keypoint’s locations are required. It will extend to
body structure positions that provide depth information for
that keypoint. For instance, the knee keypoint gives attention
to ankles, the wrist keypoint gives attention to elbows and
shoulders, and the elbow keypoint gives attention to shoul-
ders. Hence, previous methods have been mistaken in their

…
Feature map
2D Pose
Linear Projection
…
k
v
…
q
Transformer Layer
Pose-guided Transformer  Layer
Transformer Layer
…
…
Image
Encoder
Linear Projection
Multihead
Cross-Attention
FFN
Multihead
Self-Attention
Adaptive Feature Selection
Transformer Layer
×N
…
Refined Pose
FFN
Multihead
Self-Attention
FFN
Softmax
𝐽!
𝐽"
𝐽#
I#
I"
I!
I$
I%
Attention 
map
Transpose
Normalized
Pretrained
Weight sharing
Linear Projection
Stage 1: Broad Query
Stage 2: Focused Exploration
Coarse Pose
Keypoint token
Image token
Enhanced 
Image token
flatten
dot product
element-wise 
addition
gradient 
truncation
Figure 5: The overview of the proposed network.
assumption that only concatenating image patches or fea-
tures around keypoints is sufficient.

## conclusion
This paper gives a new insight into the cause of poor gener-
alization problems and the effectiveness of image features.
Based on that, we propose an advanced 3DHPE framework
that leverages effective image cues and improves the gener-
alization ability. It comprises two stages: the first involves
a broad query for valuable image features, and the second
stage focuses on critical features. To accomplish this, we
proposed a novel Pose-guided Transformer Layer to reduce
the keypoints’ attention to background and an Adaptive Fea-
ture Selection Module to prune less significant image fea-
tures. Extensive experiments show that our method achieves
state-of-the-art performance on two widely used benchmark
datasets and shows great generalization ability. We hope our
exploration can provide insights for future 3DHPE research.

Acknowledgement
This work was supported partly by the National Natural Sci-
ence Foundation of China (Grant No. 62173045), and the
Natural Science Foundation of Hainan Province (Grant No.
622RC675).