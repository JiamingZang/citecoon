# UnrealEgo: A New Dataset for Robust Egocentric 3D Human Motion Capture

> 2022 · id: W4312926163 · arXiv: 2208.01633 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
Egocentric 3D human pose estimation has been actively researched recently
[59,69,42,63,62,71,68,72]. Compared to cumbersome motion capture systems that
require a fixed recording volume, the egocentric setup is more suitable to capture
daily human activities in unconstrained environments. Example applications in-
clude XR technologies [34] and motion analysis for sport and health [55].
Several setup types were proposed for egocentric 3D human pose estimation.
Some methods work on mobile devices such as a cap [69], a helmet [59] or a head-
mounted display [63,62] equipped with a camera to capture egocentric views of a
user’s whole body. Although these methods show promising results, their setups
are still not satisfactory for daily use; the cameras are mounted far from the user’s
body, which is inconvenient and restrictive. The recently introduced EgoGlass
approach [72] tackles this issue by an eyeglasses-based setup with two cameras
attached to the glasses frame. Their setup imposes fewer restrictions on users’
3 https://4dqv.mpi-inf.mpg.de/UnrealEgo/
arXiv:2208.01633v1  [cs.CV]  2 Aug 2022

2
H. Akada et al.
(a)
Proposed concept of glasses
equipped with two fisheye cameras
(b)
Human model
wearing the glasses
(c)
Egocentric
fisheye views
Fig. 1: Overview of the proposed UnrealEgo setup.
activities. We envision that with the recent development of smaller cameras [2]
and smart glasses [4,6], the eyeglasses-based setup can be a de facto standard to
capture daily human activities in various situations.
Along with that, there is a lack of datasets that would account for this
new and advanced capture setting and that would allow developing algorithmic
frameworks involving it. Furthermore, existing egocentric datasets are limited
in several ways and cannot be easily re-purposed for 3D human pose estimation
with the compact eyeglasses-based setup. First, the existing datasets do not
contain complex human motions (such as breakdance and backflip) that are seen
in daily human activities [59,69,63,72]. Second, the available egocentric datasets
do not faithfully model the 3D environment [69,63]. Next, the existing stereo-
based datasets [59,72] do not contain in-the-wild images. All in all, we note that
there is no large-scale stereo-based dataset currently available. Consequently, a
lack of a comprehensive and versatile egocentric dataset is a severely limiting
factor in the development of methods for egocentric 3D perception.
To alleviate the issues mentioned above, we present UnrealEgo, i.e., a new
large-scale naturalistic and synthetic dataset for egocentric 3D human pose esti-
mation. UnrealEgo is based on an advanced concept of an eyeglasses-based setup
with two fisheye cameras symmetrically attached to the glasses frame. Fisheye
cameras are getting more and more compact; they can capture a wider range of
views than normal cameras which is beneficial for egocentric human pose esti-
mation [59]. We use Unreal Engine [10] to synthetically design the eyeglasses as
shown in Fig. 1-(a). We then attach the eyeglasses to realistic 3D human models
(RenderPeople) [7] and capture in-the-wild stereo views in various 3D environ-
ments as shown in Fig. 1-(b), (c). Note that we prioritize the motion diversity
in UnrealEgo. Fig. 2 shows examples of 3D human models in diverse poses from
UnrealEgo. In total, UnrealEgo contains 450k in-the-wild stereo views (900k im-
ages in total) with the largest variety of motions among the existing egocentric
datasets. UnrealEgo allows developing new methods that account for temporal

UnrealEgo: A New Dataset for Robust Egocentric 3D Human MoCap
3
Fig. 2: Samples of characters and poses from UnrealEgo. We use 17 high-quality
3D RenderPeople models [7]. Also, we utilize Mixamo motions [5] and modify
them to diversify our motion data. Please refer to our video for better visu-
alizations and our supplementary asset list for characteristics of each human
model.
changes of surrounding 3D environments (see Sec. 3) and evaluating the current
state-of-the-art methods in highly challenging scenarios (see Sec. 5).
Furthermore, we propose a new benchmark approach that achieves state-of-
the-art accuracy on UnrealEgo. At the core of our method is a heatmap-based
2D keypoint estimation module. It accepts stereo inputs and passes them to
two weight-sharing encoders that produce feature maps in the latent space. The
obtained feature maps are concatenated along with the channel dimensions and
processed by a decoder that estimates 2D keypoint heatmaps (see Fig. 5). In ex-
tensive experiments, we observe that this simple but effective architecture brings
significant improvements compared with existing methods [63,72] qualitatively
and quantitatively by 13.5% on MPJPE and 14.65% on PA-MPJPE metrics.
In summary, the primary contributions of this work are as follows:
– UnrealEgo, i.e., a new large-scale naturalistic dataset for egocentric 3D hu-
man motion capture.
– A new approach for 3D human pose estimation achieving state-of-the-art
accuracy on the new benchmark dataset.
UnrealEgo is the first to provide 1) naturalistic in-the-wild stereo images with
the largest variety of motions and 2) sequences with realistically and accurately-
modeled changes of the surrounding 3D environments. This allows a more thor-
ough evaluation of existing and upcoming methods for egocentric 3D vision,
including the temporal component and global 3D poses.

4
H. Akada et al.
2

## method
Settings
MPJPE (σ)
PA-MPJPE (σ)
xR-EgoPose
Monocular
112.86 (1.16) / 123.15 (2.05)
88.71 (0.98) / 96.56 (1.27)
EgoGlass
Stereo
91.44 (0.84) / 107.70 (1.88)
70.21 (0.90) / 84.22 (0.99)
Ours
Stereo
79.06 (0.25) / 87.31 (0.57)
59.95 (0.74) / 64.65 (0.93)
apply the mean squared error (mse) between the ground-truth heatmaps HLeft
and HRight and the estimated 2D heatmaps bHLeft and bHRight:
  L _ {\text {2D }} = \t e xt {mse}(\m athbf {H}_{\text {Left}}, \widehat {\mathbf {H}}_{\text {Left}}) + \text {mse}(\mathbf {H}_{\text {Right}}, \widehat {\mathbf {H}}_{\text {Right}}). 
(1)
4.2
3D Module
Following previous work [72], we adopt the same multi-branch autoencoder for
our 3D module. Given the heatmaps bHLeft and bHRight predicted by the 2D
module as inputs, the 3D module firstly encodes them to get embedding features.
These features are used in two decoder branches. The first branch is a 3D pose
branch, which outputs the final 3D pose ˆP ∈R16×3. Here, the number of output
3D joints is 16 as the head position is included. The second branch is a heatmap
branch, which tries to reconstruct the predicted 2D heatmaps eHLeft and eHRight
so that the network can capture the uncertainty of the heatmaps.
Similar to [72], the overall loss function for the 3D module is as follows:
L3D = λpose(mpjpe(P, ˆP) + λcoscos(P, ˆP)) +
λhm(mse( bHLeft, eHLeft) + mse( bHRight, eHRight)),
(2)

UnrealEgo: A New Dataset for Robust Egocentric 3D Human MoCap
11
Table 4: Quantitative evaluation on the general motions of UnrealEgo (MPJPE).

## experiments
We present results on the UnrealEgo test sequence. Table 3 quantitatively eval-
uates our approach and competing methods with and without ImageNet pre-
training for the encoder. Overall, our method outperforms the previous best-
performing method [72], across all metrics for both experiments with and without
ImageNet. Specifically, our method with the pre-trained encoder shows signifi-
cant improvement by 13.5% on MPJPE and 14.65% on PA-MPJPE compared

UnrealEgo: A New Dataset for Robust Egocentric 3D Human MoCap
13
Stereo inputs
EgoGlass [72]
Ours
Fig. 7: Qualitative results for failure
cases on UnrealEgo.
Right view
End-to-end
Separate
GT
Fig. 8: Heatmap estimation results with
two different training strategies.
Table 6: Ablation study for the back-
bone of the 2D heatmap module.
Backbones MPJPE (σ)
PA-MPJPE (σ)
ResNet18
79.06 (0.25) 59.95 (0.74)
ResNet34
80.50 (0.78)
60.04 (0.60)
ResNet50
80.07 (0.45)
60.08 (0.63)
ResNet101
80.15 (0.06)
60.57 (0.79)
Table 7: Ablation study for the weight
sharing in the 2D heatmap module.
Backbones
MPJPE (σ)
PA-MPJPE (σ)
weight sharing
79.06 (0.25) 59.95 (0.74)
no weight sharing 83.54 (1.30)
62.29 (0.45)
to EgoGlass [72]. All methods, including ours, benefit from the ImageNet pre-
training; the performance of our approach is boosted by 9.4% on MPJPE and
7.2% on PA-MPJPE.
We also break down the test sequence into 30 motion types as shown in
Table 4 for general motions and Table 5 for sports motions. Both tables indicate
that our method achieves significant superiority for all motion types. See Fig. 6
for the qualitative results. Even with the occlusions and complex poses in various
environments, our method estimates the 3D poses much better than EgoGlass.
It is also worth analyzing failure cases. According to Table 4, bending motions
(such as sitting on the ground or crouching) are reconstructed with comparably
low accuracy. This is because the lower body parts are occluded by the upper
body, especially when people crouch down as shown in Fig. 7. Even with the
stereo inputs, these methods still can not perform well on some motions that are
occasionally seen in daily human activities.
5.4
Ablation Study
We first ablate different encoder backbone architectures for our 2D module in
Table 6. All variants generate the heatmap with the same resolution and the 3D
module shares the same architecture. The experiment suggests that all of the
models yield similar results but at a higher computational cost for a larger back-
bone. For example, the difference between ResNet18 and Resnet50 is only 0.2%
on PA-MPJPE. This result is also observed in the previous work [62], showing
that a larger backbone does not necessarily lead to performance improvements.

14
H. Akada et al.
Table 8: Ablation study on the training strategy.
Backbones
MPJPE (σ)
PA-MPJPE (σ)
Separate training
79.06 (0.25) 59.95 (0.74)
End-to-end training 80.67 (0.58)
61.72 (0.55)
Next, we show the effect of weight sharing in the encoder backbone of our 2D
keypoint estimation module in Table 7. The weight-sharing backbone performs
better than the encoder without weight sharing by 5.4% on MPJPE and 3.8% on
PA-MPJPE. One possible reason for this result is that the weight-sharing back-
bone can see more views during training, leading to a better feature extractor.
Therefore, we use the weight-sharing strategy for all experiments.
Lastly, we conduct the experiment with different training strategies, i.e.,
separate training and end-to-end training for our 2D keypoint estimation and
3D estimation module, as shown in Table 8. The result indicates that the separate
training yields slightly better performance than the end-to-end training by 2.0%
on MPJPE and 2.9% on PA-MPJPE. We also visualize the heatmaps predicted
by our network with the different training strategies in Fig. 8. It is interesting to
note that separate training leads to relatively accurate heatmap estimation while
the network trained in an end-to-end manner tries to capture the whole body.
Although this visual result can change depending on the hyper-parameters, we
follow the same hyper-parameter setting in the previous work [72] and choose
the separate training strategy for all experiments.
6

## related_work
2.1
Datasets for Outside-in 3D Human Pose Estimation
Many datasets were proposed for 3D pose estimation with ground-truth anno-
tations. Some of them are captured with optical markers [61,41,65], while the
others use marker-less mocap systems [52,51,43,70]. However, these datasets are
mostly captured in the studio and usually lack the diversity of clothing, occlu-
sions, and environments.
In the meantime, synthetic datasets have become popular because no costly
mocap setups are required for annotations. Many such datasets are created by
compositing people on background images [67,57,39,58,51,53]. Because of such
composition, however, their images do not match real-world scenes in terms of the
local pixel intensity statistics and distributions. Butler et al. [30] provide images
rendered using underlying detailed 3D geometry and corresponding optical flows
that can be used for tracking purposes. However, this dataset does not provide
3D joint annotations unlike ours.
The recent works by Zhu et al. [73] and Patel et al. [56] use 3D modeling tools
and game engines [1,9,10] to render realistic images of rigged 3D human models
in 3D environments. Unfortunately, these datasets are designed for outside-in
pose estimation from an external camera viewpoint; they are not suitable and
cannot be easily repurposed for egocentric 3D pose estimation.
2.2
Datasets for Egocentric 3D Human Pose Estimation
There exist several datasets specifically recorded for egocentric 3D human poses.
Mo2Cap2 [69] is the first cap-based setup with a single wide-view fisheye cam-
era attached 8cm away from the user. With this setup, Xu et al. [69] create a
large-scale dataset by compositing SMPL models [48] on randomly-chosen back-
grounds (real images), resulting in 530k images with 15 annotated keypoints
per image. xR-EgoPose [63] approach uses a head-mounted display with a sin-
gle fisheye camera equipped 2cm away from a user’s nose. This work uses the
Mixamo motion dataset [5] to animate 3D human models and renders egocentric
views with HDR backgrounds with the help of the 3D rendering tool V-Ray [3].
Their dataset contains 380k photorealistic synthetic images with 25 body and 40
hand keypoints. However, both datasets contain only monocular images. They
feature only simple (every-day) human motions (due to the restrictions imposed
by their setups) and do not accurately model 3D environments and complex hu-
man trajectories in them. Hence, they do not cover most motions that can arise
in egocentric 3D human pose estimation using a compact eyeglass-based setup.
Ego4D [37] is a new large-scale dataset for egocentric vision. Unfortunately, it
does not contain 3D annotations of human poses.
On the other hand, existing stereo egocentric datasets have several limi-
tations. Rhodin et al. [59] proposed EgoCap, i.e., a headgear with a pair of
fisheye cameras equipped 25cm away from users to capture stereo views. Their
dataset contains only 30k stereo image pairs with a limited variety of motions

UnrealEgo: A New Dataset for Robust Egocentric 3D Human MoCap
5
Fig. 3: Comparison of datasets for egocentric 3D human pose estimation.
in a lab environment. More recently, EgoGlass [72] simplified the stereo setup
with eyeglasses and two cameras equipped on the glasses frames. Although Ego-
Glass captured a relatively large-scale of images, i.e., total 170k stereo pairs, the
dataset is captured only in a studio environment and is not publicly available.
In contrast to existing datasets, UnrealEgo addresses the above shortcomings.
Fig. 3 illustrates the differences among existing datasets and UnrealEgo. Firstly,
UnrealEgo provides stereo images in indoor and outdoor scenes. Secondly, it
offers the largest number of images, e.g., 15 times larger than EgoCap [59] and 2.5
times larger than EgoGlass [72]. Next, it contains naturalistic image sequences
with accurately modeled geometry changes in the surrounding 3D environments.
Also, it offers the largest number of body and hand keypoints. Furthermore, it
is the most challenging egocentric dataset in terms of motion variety.
2.3
Methods for Egocentric 3D Human Pose Estimation
Existing methods for egocentric 3D human pose estimation can be divided into
two groups in terms of egocentric settings. The first group aims at estimating 3D
keypoints from monocular views. Mo2Cap2 [69] is the first CNN-based system to
predict 3D poses. Tome et al. [63,62] follow a two-step approach using a multi-
branch autoencoder to capture uncertainty in their predicted 2D heatmaps and
to leverage rotation constraints [62]. Jiang et al. [42] predict 3D poses by utilizing
the information of surrounding environments and extremities of the user’s body.
Zhang et al. [71] estimate 3D poses with fisheye distortions using an automatic
calibration module. More recently, Wang et al. [68] proposed an optimization-
based approach with a motion prior learned from an additional dataset for global

6
H. Akada et al.
Table 1: Comparison of human motion capture datasets.
Dataset
Subjects Motions Minutes
Dataset
Subjects Motions Minutes
ACCAD [26]
20
252
26.74
KIT [50]
55
4232
661.84
BMLhandball [46]
10
649
101.98
MPI HDM05 [54]
4
215
144.54
BMLmovi [36]
89
1864
174.39
MPI Limits [27]
3
35
20.82
BMLrub [64]
111
3061
522.69
MPI MoSh [47]
19
77
16.53
CMU [31]
96
1983
543.49
MPI-INF-3DHP [51]
8
-
-
D-FAUST [29]
10
129
5.73
SFU [66]
7
44
15.23
DanceDB [28]
20
151
203.38
SSM [49]
3
30
1.87
EKUT [50]
4
349
30.74
TCD Hands [40]
1
62
8.05
Eyes Japan [35]
12
750
363.64
TotalCapture [65]
5
37
41.1
Human3D [41]
11
-
-
Transitions [49]
1
110
15.1
Human4D [33]
8
148
72.60
AMASS [49]
344
11265 2420.86
HumanEva [61]
3
28
8.48
Ours
17
45520 3174.63
3D human motion capture. Even with their competitive results, these monocular
methods often fail on complex motions (e.g., due to the depth ambiguity).
The second group follows multi-view settings, including our work. EgoCap [59]
is an optimization-based approach using a body-part detector and personalized
3D skeleton models. Cha et al. [32] developed a headset equipped with eight
cameras; they introduced a CNN-based method to reconstruct a human body
and an environment in 3D. EgoGlass [72] builds upon xR-EgoPose [63] and is one
of the most accurate methods; its architecture contains two separate UNets for
the stereo inputs in the 2D joint estimation module. In contrast to the reviewed
works, this paper proposes a simple yet effective idea of devising a new 2D joint
estimation module that accepts stereo inputs to significantly improve 3D pose
estimation compared with the existing best-performing methods.
3
UnrealEgo Dataset
This section provides details of the UnrealEgo dataset, focusing on our setup,
motions, and rendered egocentric data. Please also see our supplementary video
for dynamic visualizations and our supplementary asset list.
3.1
Setup
We use Unreal Engine [10] to synthetically design the eyeglasses with two fisheye
cameras equipped on the glasses frame as shown in Fig. 1-(a). The distance
between the cameras is 12cm. The cameras’ field of view amounts to 170◦. We
attach the glasses to 3D human models (RenderPeople) that perform different
motions in various 3D environments. Fig. 1-(b) and (c) show an example of the
human models in a Kyoto-inspired environment in Japan, and fisheye views.
Characters. We use 17 realistic RenderPeople 3D human models (commercially
available) [7], nine female and eight male. These models are rigged and skinned
based on the default 3D human skeleton of Unreal Engine [10]. Their skin color

UnrealEgo: A New Dataset for Robust Egocentric 3D Human MoCap
7
Table 2: Motion categories in our dataset.
Motion types
Motions Minutes
Motion types
Motions Minutes
1: jumping
1343
36.35
16: standing - whole body
3791
307.95
2: falling down
714
35.27
17: standing - upper body
5820
708.74
3: exercising
1225
82.07
18: standing - turning
1785
82.73
4: pulling
272
28.31
19: standing - to crouching
680
38.21
5: singing
1054
149.21
20: standing - forward
3417
93.68
6: rolling
136
4.69
21: standing - backward
1207
21.69
7: crawling
612
22.47
22: standing - sideways
1496
30.42
8: laying
612
30.92
23: dancing
5728
800.13
9: sitting on the ground
68
10.88
24: boxing
4012
160.53
10: crouching - normal
1802
127.90
25: wrestling
2958
119.63
11: crouching - turning
612
12.74
26: soccer
1892
69.63
12: crouching - to standing
850
29.46
27: baseball
476
27.31
13: crouching - forward
1020
29.50
28: basketball
272
7.54
14: crouching - backward
493
8.82
29: american football
85
6.07
15: crouching - sideways
646
11.69
30: golf
442
80.07
tones include pale white, white, light brown, moderate brown, dark brown, and
black. Their clothing types include athletic pants, jeans, shorts, tights, dress
pants, skirts, jackets, t-shirts, and long sleeves with diffident colors. Please see
Fig. 2 for an overview of the 3D human models we use. Also, please see our
supplement for detailed characteristics of each human model.
Motions. It is our top prior

## conclusion
We presented UnrealEgo, i.e., a new large-scale naturalistic dataset for egocen-
tric 3D human pose estimation. It allows a comprehensive evaluation of existing
and upcoming methods for egocentric 3D vision, including the temporal com-
ponent and global 3D poses. Our simple yet effective architecture for egocentric
3D human pose estimation brings significant improvement compared to previ-
ous best-performing methods qualitatively and quantitatively. In addition, our
extensive ablation studies validate our architectural design choices for the stereo
inputs and the training strategy. Although our method achieved state-of-the-art
results, there are still failure cases due to occlusions and complex motions. In
future work, we are interested in incorporating explicit 3D geometry obtained
from our stereo fisheye setup for further performance improvements.
Acknowledgements. We thank Silicon Studio Corp. for providing the fisheye plu-
gin. Hiroyasu Akada and Masaki Takahashi were supported by the Core Research for
Evolutional Science and Technology of the Japan Science and Technology Agency (JP-
MJCR19A1). Jian Wang, Soshi Shimada, Vladislav Golyanik and Christian Theobalt
were supported by the ERC Consolidator Grant 4DReply (770784).

UnrealEgo: A New Dataset for Robust Egocentric 3D Human MoCap
15