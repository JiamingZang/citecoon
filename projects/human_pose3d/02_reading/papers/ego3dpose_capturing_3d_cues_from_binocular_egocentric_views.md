# Ego3DPose: Capturing 3D Cues from Binocular Egocentric Views

> 2023 · id: W4386993482 · arXiv: 2309.11962 · pdf: https://arxiv.org/pdf/2309.11962 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
We present Ego3DPose, a highly accurate binocular egocentric 3D
pose reconstruction system. The binocular egocentric setup of-
fers practicality and usefulness in various applications, however,
it remains largely under-explored. It has been suffering from low
pose estimation accuracy due to viewing distortion, severe self-
occlusion, and limited field-of-view of the joints in egocentric 2D
images. Here, we notice that two important 3D cues, stereo corre-
spondences, and perspective, contained in the egocentric binocular
input are neglected. Current methods heavily rely on 2D image
features, implicitly learning 3D information, which introduces bi-
ases towards commonly observed motions and leads to low overall
accuracy. We observe that they not only fail in challenging occlu-
sion cases but also in estimating visible joint positions. To address
these challenges, we propose two novel approaches. First, we de-
sign a two-path network architecture with a path that estimates
pose per limb independently with its binocular heatmaps. Without
full-body information provided, it alleviates bias toward trained
full-body distribution. Second, we leverage the egocentric view
of body limbs, which exhibits strong perspective variance (e.g., a
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
SA Conference Papers’23, December 12–15, 2023, Sydney, NSW, Australia
© 2023 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 979-8-4007-0315-7/23/12...$15.00
https://doi.org/10.1145/3610548.3618147
significantly large-size hand when it is close to the camera). We
propose a new perspective-aware representation using trigonome-
try, enabling the network to estimate the 3D orientation of limbs.
Finally, we develop an end-to-end pose reconstruction network
that synergizes both techniques. Our comprehensive evaluations
demonstrate that Ego3DPose outperforms state-of-the-art models
by a pose estimation error (i.e., MPJPE) reduction of 23.1% in the
UnrealEgo dataset. Our qualitative results highlight the superiority
of our approach across a range of scenarios and challenges.
CCS CONCEPTS
• Computing methodologies;
KEYWORDS
Egocentric, 3D Human Pose Estimation, Stereo vision, Heatmap
ACM Reference Format:
Taeho Kang, Kyungjin Lee, Jinrui Zhang, and Youngki Lee. 2023. Ego3DPose:
Capturing 3D Cues from Binocular Egocentric Views. In SIGGRAPH Asia
2023 Conference Papers (SA Conference Papers’23), December 12–15, 2023,
Sydney, NSW, Australia. ACM, New York, NY, USA, 12 pages. https://doi.
org/10.1145/3610548.3618147
1

## introduction
Egocentric 3D pose reconstruction brings mobility to many promis-
ing applications such as avatar creation [Ahuja et al. 2019], mo-
tion analysis for sports [Hwang et al. 2017; Zecha et al. 2018] and
health [Maskeli¯unas et al. 2023], and action recognition [Núñez-
Marcos et al. 2022]. Among various egocentric setups [Akada et al.
arXiv:2309.11962v1  [cs.CV]  21 Sep 2023

SA Conference Papers’23, December 12–15, 2023, Sydney, NSW, Australia
Taeho Kang, Kyungjin Lee, Jinrui Zhang, and Youngki Lee
2022; Rhodin et al. 2016; Tome et al. 2019; Xu et al. 2019; Zhao
et al. 2021], several works show the potential of eyeglasses with
two binocular fisheye cameras [Zhao et al. 2021]. EgoCap [Rhodin
et al. 2016] is one of the first works that adopt a helmet form factor
with two mounted cameras on an extended stick. EgoGlass [Zhao
et al. 2021] extends the work to handle self-occlusion cases and
UnrealEgo [Akada et al. 2022] improves the accuracy by using a
large-scale synthetic dataset. The accuracy, however, is still far
below third-person-view-based 3D pose estimation techniques.
In this work, we introduce Ego3DPose, a novel system designed
with an in-depth investigation of the binocular and egocentric set-
ting. Compared to third-person view inputs, egocentric binocular
inputs inherently possess limited information due to their restricted
field of view and significant self-occlusions among the joints. Yet,
existing approaches have not fully harnessed the potential of these
already constrained inputs. Figure 2 shows that existing work fails
on joints clearly shown in the inputs with sufficient visual informa-
tion. This is especially common in motions with highly independent
body parts movement, as shown in the supplementary video, where
full-body information barely helps estimate specific limb poses.
We identify two under-utilized sources of 3D information within
binocular and egocentric inputs: stereo correspondence and per-
spective. While these concepts are not new, we recognize that
effectively incorporating them into the network poses unique chal-
lenges and requires careful investigation.
Utilizing stereo correspondence is challenging in two-fold. Firstly,
existing methods in the domain of multiview pose estimation are
highly data-dependent. Their performance lacks generalizability
across poses or camera configurations [Bartol et al. 2022]. This is, in
particular, challenging in egocentric settings with limited datasets.
Secondly, 2D-to-3D lifting is inevitably ambiguous unless both 2D
joint positions and camera calibrations are accurate [Chen and
Ramanan 2017a]. Existing methods focus on optimizing these two
components working sequentially. In contrast to third-person-view
settings, where high accuracy can be achieved in 2D pose estima-
tion, egocentric settings suffer from severe self-occlusions and view
distortions propagating the error to stereo correspondence estima-
tion. Thus, tighter cooperation between stereo information and the
2D pose estimation holds greater significance.
The amplified perspective effect of fisheye camera views, espe-
cially with the egocentric setup, has been considered a challenge
rather than an opportunity. Specifically, the body parts further from
the camera, such as the lower limb, are shown much smaller than
the real size, while the shoulder and upper limb views are enlarged.
Existing methods aim to tackle the challenge with reprojection
methods to alleviate the distortion effect [Liu et al. 2023; Xu et al.
2019; Zhang et al. 2021] or extensively train the network with large
synthetic datasets [Akada et al. 2022; Tome et al. 2019]. However,
these approaches overlook the potential value of the perspective
factor, which can be useful information conveying the intensity of
the 3D effect.
We propose two novel modules integrated into an end-to-end
neural network to tackle the challenges. First, we devise a Stereo
Matcher network to serve two purposes: it i) focuses on learning
the stereo correspondences independent from the full body pose
distribution in the dataset and ii) generates an explicit estimation
of the 3D orientation instead of directly lifting 2D joint estimations
Figure 2: Heatmaps (Left) and 3D pose reprojected to the in-
put images (Right) of [Akada et al. 2022] and our methon.
The heatmap estimation indicates that the visual informa-
tion is sufficient.
to a final 3D position. Specifically, the Stereo Matcher takes the
optical features of each limb independently and predicts per-limb
3D orientation with a weight-shared network. Using the stereo
correspondences and perspective effect, the architecture enables
the module to focus on estimating the limb’s orientation without
reliance on the full-body information. Furthermore, the estimated
3D orientation is concatenated with the 2D heatmaps’ encoded
pose features to assist lifting from 2D to 3D poses.
We also introduce the Perspective Embedding Heatmap repre-
sentation. This representation allows the 2D module to explicitly
extract and utilize the 3D information of the perspective effect. We
represent the perspective effect as the 3D orientation of the limbs.
For example, as shown in Figure 2, the width of the top part of the
upper limb is much larger than the width of the lower part. This vi-
sual cue suggests that the upper limb is nearly perpendicular to the
camera view plane. Then, we devise a trigonometry-based represen-
tation of the 3D orientation. It allows embedding the information
into a 2D heatmap to utilize its uncertainty estimates.
Our end-to-end network, which integrates the two modules,
significantly outperforms prior works. The quantitative measure
of 3D pose estimation errors, MPJPE, was improved by 27.0% and
23.1% compared to the baselines EgoGlass and UnrealEgo in the
UnrealEgo dataset, respectively. Furthermore, the qualitative results
shown in Figure 1 and Figure 10 also show noticeable improvements
made by our work.
In summary, the main contributions are as follows:
• We identify two under-utilized but crucial 3D information
from binocular egocentric 3D pose estimation.
• We propose a novel two-path network that separates stereo
correspondence features from the pose features.
• We introduce a new heatmap representation for 3D orienta-
tion to leverage the perspective effect of egocentric views.
• We achieve significant quantitative and qualitative perfor-
mance improvements compared to state-of-the-art methods.

Ego3DPose: Capturing 3D Cues from Binocular Egocentric Views
SA Conference Papers’23, December 12–15, 2023, Sydney, NSW, Australia
Figure 3: Our system consists of three stages: optical feature extraction, two-path feature aggregation, and 3D pose reconstruction
with our two key components Perspective Embedding Heatmap and Stereo Matcher (marked with asterisk*). The optical feature
extraction uses two feature extractors to generate Joint Position Heatmaps and Perspective Embedding Heatmaps. The Stereo
Matcher network estimates the 3D orientation and learns the stereo correspondence from Perspective Embedding Heatmaps.
The heatmap encoder takes all heatmaps as input and generates pose features. Finally, the 3D pose decoder takes the estimated
3D orientations and pose features to predict the 3D pose of the joints. The heatmap reconstructor is only used during training.
2

## method
To better highlight the effectiveness of our two modules, incorporat-
ing stereo correspondence and perspective effect, we integrate them
into widely used network architecture instead of redesigning the
end-to-end network. In this paper, the overall structure follows the
encoder-decoder network, where the 2D encoder module extracts
optical features from the binocular RGB inputs as 2D Joint Position
Heatmaps. The heatmaps are encoded into pose features and then
reconstructed into 3D joint positions with the final 3D decoder.
Here, we add the Perspective Embedding Heatmap generator in
the optical feature extraction stage and the Stereo Matcher to the
encoder stage.
Figure 3 shows the overall network architecture. The input to the
network is two 256×256 RGB images. The first stage, optical feature
extraction, generates two sets of heatmaps: i) Perspective Embed-
ding Heatmaps based on our novel perspective-aware trigonometry
representation and ii) Joint Position Heatmaps similar to previous
work. Then, the two-path feature aggregation network separately
learns i) the stereo correspondences from the binocular Perspec-
tive Embedding Heatmaps and ii) pose feature vector extracted
from both Joint Position Heatmaps and Perspective Embedding
Heatmaps. Our network then concatenates the two disentangled
information and generates the final 3D local pose. The local pose
is defined as a 3D position of the joints relative to the pelvis in
the camera coordinate system. A total of 15 joints are estimated,
including the neck, shoulders, elbows, hands, thighs, calves, ankles,
and feet. We additionally train a heatmap reconstruction network
which is not used for inferences.
3.1
Optical Feature Extraction
This module consists of two networks. Both networks take the
binocular RGB images as input but generate different types of
optical features: Perspective Embedding Heatmap and Joint Position
Heatmap. They use the same network architecture, a U-Net-based
model with a ResNet18 backbone. We first explain the Joint Position
Heatmap as a preliminary for our Perspective Embedding Heatmap.
3.1.1
Joint Position Heatmap. This heatmap is a single-channel
image where each pixel represents the confidence level associated
with the 2D positions of the joints. It has been widely adopted in
various pose estimation methods [Bulat and Tzimiropoulos 2016]
due to its effectiveness in capturing uncertainty. To align with
previous work, we adopt a per-joint heatmap generation approach
Egocentric Camera
View Plane
Limb-view 
Angle
𝜽𝒍
Input
Limb
Joint
Depth
𝒄𝒐𝒔𝜽/ sin𝜽
Figure 4: Visualization of a joint position, limb position, limb
depth, and our Perspective Embedding Heatmap, where the
color represents the cosine and sine of the limb-view angle
shown in the right image.
instead of using a single heatmap for all joints. A total of 30 Joint
Position Heatmaps are generated, corresponding to the left and
right input images for the 15 individual joints.
3.1.2
Perspective Embedding Heatmap. The goal of the Perspective
Embedding Heatmap is to incorporate the probabilistic informa-
tion of the perspective effect instead of the joint positions. The
perspective effect can be expressed in different ways, such as the
depth of each joint or the angle between two adjacent joints (Refer
to Figure 4 for the visualization of different types and the specific
explanations in Section 5.3.3). We, in particular, choose the 3D ori-
entation of the limbs defined with the angle between the viewing
plane and the limb (referred to as limb-view angle hereinafter). The
limb-view angle 𝜃𝑙for each limb is computed as follows:
𝜃𝑙= atan
©­­
«
𝑧𝑙
√︃
𝑥2
𝑙+ 𝑦2
𝑙
ª®®
¬
, where 𝑥𝑙,𝑦𝑙,𝑧𝑙denotes the relative 3D coordinate of the child
joint from the parent joint in the camera coordinate space, and
𝑥𝑦-plane is equivalent to the viewing plane of the camera. Limbs
are represented as a line rather than a pair of two adjacent joints
to enhance the robustness against occlusions because the visible
parts of the limbs can be utilized.
Embedding the 3D orientation to a heatmap is not straightfor-
ward. We have to embed three pieces of information, 2D position,
confidence, and 3D orientation, into a 2D heatmap. One approach
would be simply multiplying the magnitude of the orientation with
the probabilistic output of the heatmap. However, this can confuse
the network as the pixel values contain information on confidence
and orientation. Adding the perspective value as a separate channel
is also not an effective representation because the confidence only
refers to the 2D joint position estimation rather than the orientation
value.
We use the trigonometric function to embed the limb-view angle
into a 2D heatmap. This choice is motivated by the properties of
trigonometric functions. The sine and cosine value represents an
angle with the constant norm as a vector. Our heatmap has two
channels where the pixel values are scaled sine and cosine of the

Ego3DPose: Capturing 3D Cues from Binocular Egocentric Views
SA Conference Papers’23, December 12–15, 2023, Sydney, NSW, Australia
Figure
5:
Our
Stereo
Matcher module without
Perspective
Embedding
Heatmap.
Figure
6:
Pose
feature
extraction module from
prior work.
limb-view angle. The scale of the resulting vector represents the
confidence of the orientation estimation, while the ratio of the sine
and cosine values can represent the estimated limb-view angle. In
this way, using the trigonometric functions ensures that the per-
spective information is entangled with the probability. Perspective
Embedding Heatmaps are defined for 14 limbs, excluding the head-
neck connection from the 15 limbs connecting the tree hierarchy
of 16 joints. Each set of Perspective Embedding Heatmaps contains
four channels of heatmaps for sine and cosine for the left and right
camera views.
3.2
Two-Path Feature Aggregation
The key to our high 3D pose estimation accuracy is devising the
network to learn the stereo feature matching separately from the
distribution of full body pose in the train dataset. Thus, we replace
the one-path encoder stage of the previous work, which generates
pose features from the 2D joint heatmaps, with a two-path feature
aggregation stage. We first explain the heatmap encoder network,
which extends prior work estimating the pose features. Then, we
provide the details of the new Stereo Matcher network.
3.2.1
Heatmap Encoder. The input of the heatmap encoder is all the
heatmaps generated from the previous optical feature extraction
stage. The heatmaps include the joint positions of all 15 joints
and Perspective Embedding Heatmaps for 14 limbs concatenated
channel-wise. The heatmap encoder is a CNN-based network, and
the output is a 1D vector of 20 pose feature values.
3.2.2
Stereo Matcher Network. Our Stereo Matcher network is a
simple yet effective technique that takes the Perspective Embedding
Heatmaps of each limb as input and estimates the 3D orientation of
the limb individually. The 3D orientation of each limb is defined as,
o𝑙=
[x𝑙, y𝑙, z𝑙]
√︃
x2
𝑙+ y2
𝑙+ z2
𝑙
, a normalized vector of relative positions of a child from the parent
joint in a camera’s coordinate system. 𝑥𝑙,𝑦𝑙,𝑧𝑙denotes the relative
3D coordinate.
The Stereo Matcher takes the heatmaps of one limb at a time and
estimates the specific limb’s 3D orientation. Figure 5 visualizes the
architecture, with the conventional heatmaps used for comparison.
Each limb is represented as a pair of position heatmaps of adja-
cent joints (i.e., a total of 4 channel heatmaps, including binocular
Joint Position Heatmaps of two joints). The per-limb process is
clearly distinguished from the heatmap encoder shown in Figure 6,
which takes heatmaps for all body parts to encode them as one
pose feature vector. The estimation without full body information
is possible due to stereo information and 3D cues from Perspec-
tive Embedding Heatmaps, and it prevents reliance on the trained
full-body pose distribution. The process does not require specific
knowledge of the limb. Thus, the weights of the Stereo Matcher
network are shared across different limbs for sample efficiency.
Our final network architecture integrates the Stereo Matcher with
the Perspective Embedding Heatmaps shown in Figure 3. The four
channels input is the Perspective Embedding Heatmaps of one
limb, including two channels each for sine and cosine. Our em-
pirical analysis shows that the integration has a synergistic effect
(Table 3).
3.3
3D Pose Reconstruction
In the final 3D pose reconstruction stage, the 3D pose decoder takes
the concatenated vector of the pose features and the 3D orientations
of 14 limbs as input. Then, it estimates the final 3D local pose of
the 15 joints. The 3D pose decoder consists of only fully connected
layers. Lastly, our system adopts a heatmap reconstructor of [Tome
et al. 2019], only used during training, to constrain the heatmap
encoder to preserve the uncertainty represented in the heatmap.
4
TRAINING DETAILS
Our Ego3DPose is trained and eval

## experiments
5.1
Experimental Setup
5.1.1
Datasets. We evaluate using the publicly available synthetic
UnrealEgo dataset, generated through the Unreal Engine 1. More-
over, we further demonstrate the practicality of our approach on
1https://www.unrealengine.com
the real-world EgoCap [Rhodin et al. 2016] dataset. For our ablation
studies, we primarily use the UnrealEgo dataset. It provides much
more comprehensive coverage of diverse human movements across
various settings, which surpasses that of the EgoCap dataset. Fur-
thermore, it encompasses more challenging scenarios, particularly
those involving severe self-occlusions. Lastly, the dataset covers
various ethnicities and body types with 17 different characters.
5.1.2
Baselines. We evaluate our system against two state-of-the-
art binocular egocentric pose estimation models, e.g., UnrealEgo [Akada
et al. 2022] and EgoGlass [Zhao et al. 2021], using the evaluation
dataset. The results for baselines are based on the provided open-
source code [Akada et al. 2022]. The evaluation of UnrealEgo on
its dataset is based on the open-source weight. To perform a com-
prehensive evaluation of EgoGlass, we re-implement its full archi-
tecture as the body part branch, which is the primary factor for
its achieved performance, is not included in the source code. As
the body part branch requires segmentation data not provided in
the original UnrealEgo dataset, we utilize the pseudo-segmentation
technique proposed by EgoGlass.
5.1.3
Metrics. We utilize the widely recognized Mean Per Joint
Position Error (MPJPE) as a metric, which calculates the mean
Euclidean distance between the predicted joint positions and their
respective ground truth counterparts. Additionally, we also employ
Procrustes Analysis MPJPE (PA-MPJPE) as a complementary met-
ric to further quantify the precision and robustness of our approach.
5.2
Overall Performance
5.2.1
Evaluations on UnrealEgo Dataset. Table 1 shows the 3D pose
estimation errors of our proposed system, Ego3DPose, and two
baseline systems, e.g., EgoGlass and UnrealEgo, on the UnrealEgo
dataset. Tests are conducted for all motion categories within the
dataset, and our proposed system achieves the best results across
the board. Specifically, our system demonstrates an MPJPE of only
60.82, which represents a significant improvement of 23.1% and
27.0% over the UnrealEgo and EgoGlass systems, respectively, in
terms of overall error. Our system outperforms previous methods
in every motion category by a large margin, including challenging
motions such as “Crawling" and “Sitting on the Ground".
5.2.2
Evaluations on EgoCap Dataset. As the motions are not cate-
gorized in this dataset, we present the overall results. It should be
noted that the EgoCap dataset solely provides 3D pose annotations
for its test dataset publicly. We sorted the 3D test dataset by subject
and frame, trained on 80% of it, and reserved the remaining 20% for
testing purposes. Table 2 shows that our method performs well in
different settings and in real-world images. Our approach manifests
augmentations, as evidenced by 11.2% improvements in MPJPE and
12.6% enhancements in PA-MPJPE, in contrast to EgoGlass. More-
over, it outperforms UnrealEgo with 8.0% advances in MPJPE and
16.5% progressions in PA-MPJPE. It is worth noting that the Ego-
Cap dataset exhibits fewer instances of severe self-occlusions and a
narrower range of pose variations in comparison to the UnrealEgo
dataset. The baseline methods achieve modest accuracy resulting
in smaller performance improvement compared to the evaluation
on the UnrealEgo dataset.

Ego3DPose: Capturing 3D Cues from Binocular Egocentric Views
SA Conference Papers’23, December 12–15, 2023, Sydney, NSW, Australia
Table 1: Comparison with the state-of-the-art binocular egocentric pose reconstruction methods on the UnrealEgo dataset.
MPJPE(PA-MPJPE)

## related_work
2.1
Binocular Egocentric 3D Pose
Reconstruction
There has been a growing body of literature on egocentric 3D
pose reconstruction, particularly in the context of binocular vision-
based setups. Compared to monocular vision-based methods that
use a single fisheye camera with a downward view towards the
full body [Tome et al. 2019; Xu et al. 2019], binocular vision allows
higher accuracy in 3D pose reconstruction via the wider view with
cameras in proximity to the face [Zhao et al. 2021].
EgoCap [Rhodin et al. 2016] proposes the first binocular fisheye
camera-based prototype mounted on a helmet or a headset. It uses
the prototype to collect data with 8 subjects in a multiview studio
setting for ground truth collection and creates a CNN-based 3D pose
estimation model. EgoGlass [Zhao et al. 2021] proposes compact
binocular cameras embedded in eyeglasses, which are located in
closer proximity to the user’s body. However, the compact form
factor leads to frequent self-occlusions of partial body parts and
limited body coverage for various motions. The work partially
solves the problem by explicitly forcing the model to learn the body
part information with a segmentation mask. UnrealEgo [Akada
et al. 2022] recently released a large-scale synthesized dataset for
a similar setup of EgoGlass. It also shows that the overall 3D pose
estimation accuracy improves when the weights between the two
heatmap estimators, one for each of the binocular inputs, are shared.
Despite the potential of such a setting, the overall pose estimation
accuracy still requires significant improvement.
2.2
Methods for Robust 3D Pose Estimation
3D pose estimation solutions can be broadly divided into two types:
i) direct estimation of 3D poses [Li and Chan 2014; Li et al. 2015;
Pavlakos et al. 2017, 2018; Sun et al. 2017; Tekin et al. 2016a] and
ii) leveraging robust 2D pose estimation as an intermediate result
and lifting to 3D [Chen and Ramanan 2017b; Jain et al. 2013; Li
and Lee 2019; Moreno-Noguer 2016; Qiu et al. 2019; Tekin et al.
2016b; Tompson et al. 2015; Zhou et al. 2022]. The 2D-to-3D lifting
method is known to be more robust in various situations [Chen and
Ramanan 2017b]. In particular, maintaining the probabilistic nature
of the intermediate 2D representation significantly contributed to
performance improvement. Earlier work designed the 2D module
to make multiple pose predictions to have ambiguity [Jahangiri and
Yuille 2017; Moreno-Noguer 2017]. Currently, a common approach
is representing the estimated 2D joint positions as heatmaps [Tekin
et al. 2016b; Tompson et al. 2015; Zhou et al. 2022]. This approach
has been explored in various ways. In particular, using skeletal
heatmaps instead of joint heatmaps was proposed to handle oc-
clusion cases in multi-person 3D pose estimation [Chen and Yang
2018]. The key idea was to represent the skeleton as lines connect-
ing two joints. We also leverage the line-based representation but
propose a trigonometry-based embedding showing better perfor-
mance. The heatmap-based approach was explored in multiview
settings as well [Iskakov et al. 2019; Qiu et al. 2019; Zhang et al.
2020]. We also leverage the heatmap representations but propose
methods to integrate 3D supervision from earlier stages of the 2D
module.

SA Conference Papers’23, December 12–15, 2023, Sydney, NSW, Australia
Taeho Kang, Kyungjin Lee, Jinrui Zhang, and Youngki Lee
Most of the recent works in egocentric 3D pose reconstruction
also adopt the heatmap-based two-staged approach. 𝑥R-EgoPose
[Tome et al. 2019], which relies on a monocular fisheye view, uses a
2D module to generate Joint Position Heatmaps. Then, the 3D pose
estimator module uses the heatmap as input to estimate the 3D pose.
EgoGlass [Zhao et al. 2021] uses the same architecture but extends
the 2D module to take binocular inputs and estimates an additional
segmentation mask for body part information. UnrealEgo [Akada
et al. 2022] improves this work by using a weight-shared encoder
of the U-Net for the two binocular inputs and a shared decoder for
the heatmap estimation. Compared to the large body of work in
third-person view 3D pose estimation, more studies are required to
increase the robustness in egocentric settings.
3

## conclusion
In this work, we push the performance boundaries of binocular
egocentric 3D pose estimation. We identify an under-utilized source
of information in the highly constrained setting, stereo correspon-
dence, and perspective effect. Our system incorporates a novel
Stereo Matcher network, which learns to estimate 3D pose inde-
pendently using stereo correspondence of heatmaps for each limb.
It alleviates the issue of overfitting to the distribution of full body
pose in the dataset. Furthermore, we introduce a 3D Perspective
Embedding Heatmap representation. It effectively captures the per-
spective effect while preserving the inherent uncertainty associated
with heatmaps. Our system demonstrates substantial improvements
compared to state-of-the-art methods. It is noteworthy that our
proposed techniques are not exclusively tailored to the egocentric
and binocular setting. They hold potential for direct application
in various contexts of human 3D pose reconstruction, providing
benefits beyond the scope of this work.
ACKNOWLEDGEMENT
This work was supported by the National Research Foundation
of Korea(NRF) grant funded by the Korea government(MIST) (No.
2022R1A2C3008495, No. RS-2023-00218601).