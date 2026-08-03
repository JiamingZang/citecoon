# DiffPose: Toward More Reliable 3D Pose Estimation

> 2023 · id: W4386075813 · arXiv: 2211.16940 · pdf: https://researchmgt.monash.edu/ws/files/484120613/484115041_oa.pdf · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Monocular 3D human pose estimation is quite challeng-
ing due to the inherent ambiguity and occlusion, which often
lead to high uncertainty and indeterminacy. On the other
hand, diffusion models have recently emerged as an effec-
tive tool for generating high-quality images from noise. In-
spired by their capability, we explore a novel pose estima-
tion framework (DiffPose) that formulates 3D pose estima-
tion as a reverse diffusion process. We incorporate novel de-
signs into our DiffPose to facilitate the diffusion process for
3D pose estimation: a pose-speciﬁc initialization of pose
uncertainty distributions, a Gaussian Mixture Model-based
forward diffusion process, and a context-conditioned re-
verse diffusion process. Our proposed DiffPose signiﬁcantly
outperforms existing methods on the widely used pose
estimation benchmarks Human3.6M and MPI-INF-3DHP.
Project page: https://gongjia0208.github.io/Diffpose/.

## introduction
3D human pose estimation, which aims to predict the 3D
coordinates of human joints from images or videos, is an
important task with a wide range of applications, including
augmented reality [5], sign language translation [21] and
human-robot interaction [40], attracting a lot of attention
in recent years [23, 46, 50, 52]. Generally, the mainstream
approach is to conduct 3D pose estimation in two stages: the
2D pose is ﬁrst obtained with a 2D pose detector, and then
2D-to-3D lifting is performed (where the lifting process is
the primary aspect that most recent works [2,10,16,17,19,
32, 54] focus on). Yet, despite the considerable progress,
monocular 3D pose estimation still remains challenging. In
particular, it can be difﬁcult to accurately predict 3D pose
from monocular data due to many challenges, including the
inherent depth ambiguity and the potential occlusion, which
often lead to high indeterminacy and uncertainty.
† Equal contribution; § Currently at Meta; ‡ Corresponding author
2D pose sequence
𝝐
𝝐
𝝐
𝝐
𝑯𝑲
𝑯𝒌
𝑯𝟎
Forward diffusion
Reverse diffusion
Context condition
Initial Pose 
Distribution
Desired 
3D Pose
𝒈
𝒈
𝒈
𝒈
Figure 1. Overview of our DiffPose framework. In the forward
process (denoted with blue dotted arrows), we gradually diffuse
a “ground truth” 3D pose distribution H0 with low indetermi-
nacy towards a 3D pose distribution with high uncertainty HK
by adding noise ϵ at every step, which generates intermediate dis-
tributions to guide model training. Before the reverse process, we
ﬁrst initialize the indeterminate 3D pose distribution HK from the
input. Then, during the reverse process (denoted with red solid
arrows), we use the diffusion model g, conditioned on the context
information from 2D pose sequence, to progressively transform
HK into a 3D pose distribution H0 with low indeterminacy.
On the other hand, diffusion models [12, 38] have re-
cently become popular as an effective way to generate high-
quality images [33]. Generally, diffusion models are capa-
ble of generating samples that match a speciﬁed data distri-
bution (e.g., natural images) from random (indeterminate)
noise through multiple steps where the noise is progres-
sively removed [12, 38]. Intuitively, such a paradigm of
progressive denoising helps to break down the large gap be-
tween distributions (from a highly uncertain one to a deter-
minate one) into smaller intermediate steps [39] and thus
successfully helps the model to converge towards smoothly
generating samples from the target data distribution.
Inspired by the strong capability of diffusion models to
generate realistic samples even from a starting point with
high uncertainty (e.g., random noise), here we aim to tackle
3D pose estimation, which also involves handling uncer-
tainty and indeterminacy (of 3D poses), with diffusion mod-
els. In this paper, we propose DiffPose, a novel framework
that represents a new brand of diffusion-based 3D pose es-
timation approach, which also follows the mainstream two-
stage pipeline. In short, DiffPose models the 3D pose esti-
arXiv:2211.16940v3  [cs.CV]  9 Apr 2023

mation procedure as a reverse diffusion process, where we
progressively transform a 3D pose distribution with high
uncertainty and indeterminacy towards a 3D pose with low
uncertainty.
Intuitively, we can consider the determinate ground truth
3D pose as particles in the context of thermodynamics,
where particles can be neatly gathered and form a clear
pose with low indeterminacy at the start; then eventually
these particles stochastically spread over the space, leading
to high indeterminacy. This process of particles evolving
from low indeterminacy to high indeterminacy is the for-
ward diffusion process. The pose estimation task aims to
perform precisely the opposite of this process, i.e., the re-
verse diffusion process. We receive an initial 2D pose that
is indeterminate and uncertain in 3D space, and we want
to shed the indeterminacy to obtain a determinate 3D pose
distribution containing high-quality solutions.
Overall, our DiffPose framework consists of two oppo-
site processes: the forward process and the reverse process,
as shown in Fig. 1. In short, the forward process generates
supervisory signals of intermediate distributions for training
purposes, while the reverse process is a key part of our 3D
pose estimation pipeline that is used for both training and
testing. Speciﬁcally, in the forward process, we gradually
diffuse a “ground truth” 3D pose distribution H0 with low
indeterminacy towards a 3D pose distribution with high in-
determinacy that resembles the 3D pose’s underlying uncer-
tainty distribution HK. We obtain samples from the inter-
mediate distributions along the way, which are used during
training as step-by-step supervisory signals for our diffu-
sion model g. To start the reverse process, we ﬁrst initialize
the indeterminate 3D pose distribution (HK) according to
the underlying uncertainty of the 3D pose. Then, our diffu-
sion model g is used in the reverse process to progressively
transform HK into a 3D pose distribution with low indeter-
minacy (H0). The diffusion model g is optimized using the
samples from intermediate distributions (generated in the
forward process), which guide it to smoothly transform the
indeterminate distribution HK into accurate predictions.
However, there are several challenges in the above for-
ward and reverse process. Firstly, in 3D pose estimation,
we start the reverse diffusion process from an estimated 2D
pose which has high uncertainty in 3D space, instead of
starting from random noise like in existing image genera-
tion diffusion models [12, 38]. This is a signiﬁcant differ-
ence, as it means that the underlying uncertainty distribution
of each 3D pose can differ. Thus, we cannot design the out-
put of the forward diffusion steps to converge to the same
Gaussian noise like in previous image generation diffusion
works [12, 38]. Moreover, the uncertainty distribution of
3D poses can be irregular and complicated, making it hard
to characterize via a single Gaussian distribution. Lastly,
it can be difﬁcult to perform accurate 3D pose estimation
with just HK as input. This is because our aim is not just to
generate any realistic 3D pose, but rather to predict accurate
3D poses corresponding to our estimated 2D poses, which
often requires more context information to achieve.
To address these challenges, we introduce several novel
designs in our DiffPose. Firstly, we initialize the indetermi-
nate 3D pose distribution HK based on extracted heatmaps,
which captures the underlying uncertainty of the desired 3D
pose. Secondly, during forward diffusion, to generate the
indeterminate 3D pose distributions that eventually (after
K steps) resemble HK, we add noise to the ground truth
3D pose distribution H0, where the noise is modeled by
a Gaussian Mixture Model (GMM) that characterizes the
uncertainty distribution HK. Thirdly, the reverse diffusion
process is conditioned on context information from the in-
put video or frame in order to better leverage the spatial-
temporal relationship between frames and joints. Then, to
effectively use the context information and perform the pro-
gressive denoising to obtain accurate 3D poses, we design a
GCN-based diffusion model g.
The contributions of this paper are threefold: (i) We pro-
pose DiffPose, a novel framework which represents a new
brand of method with the diffusion architecture for 3D pose
estimation, which can naturally handle the indeterminacy
and uncertainty of 3D poses. (ii) We propose various de-
signs to facilitate 3D pose estimation, including the initial-
ization of 3D pose distribution, a GMM-based forward dif-
fusion process and a conditional reverse diffusion process.
(iii) DiffPose achieves state-of-the-art performance on two
widely used human pose estimation benchmarks.

## method
MPJPE
P-MPJPE
[34]
42.1
34.4
Ours + [34]
39.3
31.8
[49]
40.9
32.6
Ours + [49]
36.9
28.7
Impact of context fST .
Another crucial component
to explore is the role of
spatial-temporal
context
fST in our method. Here,
we
evaluate
the
perfor-
mance when using various context encoders [34, 49] to
obtain fST . As shown in Tab. 6, our DiffPose achieves
good performance using both models. We also ﬁnd that
DiffPose signiﬁcantly outperforms both context encoders,
which veriﬁes the efﬁcacy of our approach.
Figure 4. Evaluation of
parameters K and N.
Impact
of
reverse
diffusion
steps K and sample number N. To
further investigate the characteristics
of our pose diffusion process, we
conduct several experiments with
different diffusion step numbers (K)
and sample numbers (N) and plot
the results in Fig. 4. We observe that MPJPE ﬁrst drops
signiﬁcantly till K = 50, and shows minor improvements
when K > 50. Thus, we use 50 diffusion steps (K = 50)
in our method, which can effectively and efﬁciently shed
indeterminacy.
On the other hand, we ﬁnd that model
performance improves with the number of samples N until
N = 5, where our performance stays roughly consistent.
Table 7. Analysis of speed.
Our

## experiments
We evaluate our method on two widely used datasets for
3D human pose estimation: Human3.6M [15] and MPI-
INF-3DHP [27]. Speciﬁcally, we conduct experiments to

evaluate the performance of our method in two scenarios:
video-based and frame-based 3D pose estimation.
Human3.6M [15] is the largest benchmark for 3D hu-
man pose estimation, consisting of 3.6 million images cap-
tured from four cameras, where 15 daily activities are per-
formed by 11 subjects. For video-based 3D pose estima-
tion, we follow previous works [3, 24, 32] to train on ﬁve
subjects (S1, S5, S6, S7, S8) and test on two subjects (S9
and S11). For frame-based 3D pose estimation, we follow
[46,51,52] to train on (S1, S5, S6, S7, S8) subjects and test
on (S9, S11) subjects. We report the mean per joint posi-
tion error (MPJPE) and Procrustes MPJPE (P-MPJPE). The
former computes the Euclidean distance between the pre-
dicted joint positions and the ground truth positions. The
latter is the MPJPE after the predicted results are aligned
to the ground truth via a rigid transformation. Due to page
limitations, we move P-MPJPE results to Supplementary.
MPI-INF-3DHP [27] is a large 3D pose dataset captured
in both indoor and outdoor environments, with 1.3 million
frames. Following [3,22,27,54], we train DiffPose using all
activities from 8 camera views in the training set and eval-
uate on valid frames in the test set. Here, we report metrics
of MPJPE, Percentage of Correct Keypoints (PCK) with the
threshold of 150 mm, and Area Under Curve (AUC) for a
range of PCK thresholds to compare our performance with
other methods on the video-based setting.
Implementation Details. We set the number of pose
samples N to 5 and number of reverse diffusion steps K to
50. We ﬁt ˆHK via a GMM model with 5 kernels (M = 5)
for forward diffusion, and accelerate our diffusion inference
procedure for all experiments via an acceleration technique
DDIM [38], where only ﬁve steps are required to complete
the reverse diffusion process. For video pose estimation, we
set the Context Encoder φST to follow [49], and for frame-
based pose estimation, we set φST to follow [52].
The
Context Encoder φST is pre-trained on the training set to
predict (x, y, z), then frozen during diffusion model train-
ing; we use it to produce features fST and also to initialize
the z distribution. For video-based pose estimation, we fol-
low [2, 32] to use detected 2D pose (using CPN [4]) and
ground truth 2D pose on Human3.6M, and use ground truth
2D pose on MPI-INF-3DHP. For frame-based pose estima-
tion, we follow [51,52] to use the 2D pose detected by [4]
and ground truth 2D pose to conduct experiments on Hu-
man3.6M. More details are in Supplementary.
5.1. Comparison with State-of-the-art Methods
Video-based Results on Human3.6M. We follow [32,
48,49] to use 243 frames for 3D pose estimation and com-
pare our method against existing works on Human3.6M in
Tab. 1. As shown in the top of Tab. 1, our method achieves
the best MPJPE results using the detected 2D pose, and sig-
niﬁcantly outperforms the SOTA method [49] by around
Table 1. Video-based results on Human3.6M in millimeters under
MPJPE. Top table shows the results on detected 2D poses. Bottom
table shows the results on ground truth 2D poses.
MPJPE(CPN) Dir Disc Eat Greet Phone Photo Pose Pur Sit SitD Smoke Wait WalkD Walk WalkT Avg
Pavllo [32]
45.2 46.7 43.3 45.6 48.1
55.1 44.6 44.3 57.3 65.8 47.1 44.0 49.0
32.8
33.9 46.8
Liu [24]
41.8 44.8 41.1 44.9 47.4
54.1 43.4 42.2 56.2 63.6 45.3 43.5 45.3
31.3
32.2 45.1
Zeng [48]
46.6 47.1 43.9 41.6 45.8
49.6 46.5 40.0 53.4 61.1 46.1 42.6 43.1
31.5
32.6 44.8
Zheng [54]
41.5 44.8 39.8 42.5 46.5
51.6 42.1 42.0 53.3 60.7 45.5 43.3 46.1
31.8
32.2 44.3
Li [19]
39.2 43.1 40.1 40.9 44.9
51.2 40.6 41.3 53.5 60.3 43.7 41.1 43.8
29.8
30.6 43.0
Shan [34]
38.4 42.1 39.8 40.2 45.2
48.9 40.4 38.3 53.8 57.3 43.9 41.6 42.2
29.3
29.3 42.1
Zhang [49]
37.6 40.9 37.3 39.7 42.3
49.9 40.1 39.8 51.7 55.0 42.1 39.8 41.0
27.9
27.9 40.9
Ours
33.2 36.6 33.0 35.6 37.6
45.1 35.7 35.5 46.4 49.9 37.3 35.6 36.5
24.4
24.1 36.9
MPJPE(GT)
Dir Disc Eat Greet Phone Photo Pose Pur Sit SitD Smoke Wait WalkD Walk WalkT Avg
Pavllo [32]
35.2 40.2 32.7 35.7 38.2
45.5 40.6 36.1 48.8 47.3 37.8 39.7 38.7
27.8
29.5 37.8
Liu [24]
34.5 37.1 33.6 34.2 32.9
37.1 39.6 35.8 40.7 41.4 33.0 33.8 33.0
26.6
26.9 34.7
Zeng [48]
34.8 32.1 28.5 30.7 31.4
36.9 35.6 30.5 38.9 40.5 32.5 31.0 29.9
22.5
24.5 32.0
Zheng [54]
30.0 33.6 29.9 31.0 30.2
33.3 34.8 31.4 37.8 38.6 31.7 31.5 29.0
23.3
23.1 31.3
Li [19]
27.7 32.1 29.1 28.9 30.0
33.9 33.0 31.2 37.0 39.3 30.0 31.0 29.4
22.2
23.0 30.5
Shan [34]
28.5 30.1 28.6 27.9 29.8
33.2 31.3 27.8 36.0 37.4 29.7 29.5 28.1
21.0
21.0 29.3
Zhang [49]
21.6 22.0 20.4 21.0 20.8
24.3 24.7 21.9 26.9 24.9 21.2 21.5 20.8
14.7
15.7 21.6
Ours
18.6 19.3 18.0 18.4 18.3
21.5 21.5 19.1 23.6 22.3 18.6 18.8 18.3
12.8
13.9 18.9
Table 3. Frame-based results on Human3.6M in millimeters under
MPJPE. Top table shows the results on detected 2D poses. Bottom
table shows the results on ground truth 2D poses.
MPJPE(CPN) Dir Disc Eat Greet Phone Photo Pose Pur Sit SitD Smoke Wait WalkD Walk WalkT Avg
Pavlakos [31] 67.4 71.9 66.7 69.1 72.0
77.0 65.0 68.3 83.7 96.5 71.7 65.8 74.9
59.1
63.2 71.9
Martinez [26] 51.8 56.2 58.1 59.0 69.5
78.4 55.2 58.1 74.0 94.6 62.3 59.1 65.1
49.5
52.4 62.9
Sun [41]
52.8 54.8 54.2 54.3 61.8
53.1 53.6 71.7 86.7 61.5 67.2 53.4 47.1
61.6
53.4 59.1
Yang [47]
51.5 58.9 50.4 57.0 62.1
65.4 49.8 52.7 69.2 85.2 57.4 58.4 43.6
60.1
47.7 58.6
Hossain [13] 48.4 50.7 57.2 55.2 63.1
72.6 53.0 51.7 66.1 80.9 59.0 57.3 62.4
46.6
49.6 58.3
Zhao [51]
48.2 60.8 51.8 64.0 64.6
53.6 51.1 67.4 88.7 57.7 73.2 65.6 48.9
64.8
51.9 60.8
Liu [23]
46.3 52.2 47.3 50.7 55.5
67.1 49.2 46.0 60.4 71.1 51.5 50.1 54.5
40.3
43.7 52.4
Xu [46]
45.2 49.9 47.5 50.9 54.9
66.1 48.5 46.3 59.7 71.5 51.4 48.6 53.9
39.9
44.1 51.9
Zhao [52]
45.2 50.8 48.0 50.0 54.9
65.0 48.2 47.1 60.2 70.0 51.6 48.7 54.1
39.7
43.1 51.8
Ours
42.8 49.1 45.2 48.7 52.1
63.5 46.3 45.2 58.6 66.3 50.4 47.6 52.0
37.6
40.2 49.7
MPJPE(GT)
Dir Disc Eat Greet Phone Photo Pose Pur Sit SitD Smoke Wait WalkD Walk WalkT Avg
Martinez [26] 37.7 44.4 40.3 42.1 48.2
54.9 44.4 42.1 54.6 58.0 45.1 46.4 47.6
36.4
40.4 45.5
Hossain [13] 35.2 40.8 37.2 37.4 43.2
44.0 38.9 35.6 42.3 44.6 39.7 39.7 40.2
32.8
35.5 39.2
Zhao [51]
37.8 49.4 37.6 40.9 45.1
41.4 40.1 48.3 50.1 42.2 53.5 44.3 40.5
47.3
39.0 43.8
Liu [23]
36.8 40.3 33.0 36.3 37.5
45.0 39.7 34.9 40.3 47.7 37.4 38.5 38.6
29.6
32.0 37.8
Xu [46]
35.8 38.1 31.0 35.3 35.8
43.2 37.3 31.7 38.4 45.5 35.4 36.7 36.8
27.9
30.7 35.8
Zhao [52]
32.0 38.0 30.4 34.4 34.7
43.3 35.2 31.4 38.0 46.2 34.2 35.7 36.1
27.4
30.6 35.2
Ours
28.8 32.7 27.8 30.9 32.8
38.9 32.2 28.3 33.3 41.0 31.0 32.1 31.5
25.9
27.5 31.6
4 mm. This shows that DiffPose can effectively improve
monocular 3D pose estimation. Moreover, we also conduct
experiments using the ground truth 2D pose as input, and re-
port our results at the bottom of Tab. 1. Our DiffPose again
outperforms all previous methods by a large margin.
Table 2. Video-based results
on MPI-INF-3DHP.

## related_work
3D Human Pose Estimation. Existing monocular 3D
pose estimation methods can roughly be categorized into
two groups: frame-based methods and video-based ones.
Frame-based methods predict the 3D pose from a single
RGB image. Some works [7–9,30,31,42] use Convolutional
Neural Networks (CNNs) to output a human pose from the
RGB image, while many works [26, 46, 51, 52] ﬁrst detect
the 2D pose and then use it to regress the 3D pose. On the
other hand, video-based methods tend to exploit temporal
dependencies between frames in the video clip. Most video-
based methods [2, 3, 6, 10, 14, 32, 34, 35, 44, 45, 54] extract
2D pose sequences from the input video clip via a 2D pose
detector, and focus on distilling the crucial spatial-temporal
information from these 2D pose sequences for 3D pose es-
timation. To encode spatial-temporal information, existing
works explore CNN-based frameworks with temporal con-
volutions [3, 32], GCNs [2, 6], or Transformers [34, 54].
Notably, several works [17, 19, 36] aim to alleviate the un-
certainty and indeterminacy in 3D pose estimation by de-
signing models that can generate multiple hypothesis solu-
tions from a single input. Different from all the aforemen-

tioned works, DiffPose is formulated as a distribution-to-
distribution transformation process, where we train a dif-
fusion model to smoothly denoise from the indeterminate
pose distribution to a pose distribution with low indetermi-
nacy. By framing the 3D pose estimation procedure as a
reverse diffusion process, DiffPose can naturally handle the
indeterminacy and uncertainty for 3D pose estimation.
Denoising Diffusion Probabilistic Models (DDPMs).
DDPMs (called diffusion models for short) have emerged
as an effective approach to learn a data distribution that
is straightforward to sample from.
Introduced by Sohl-
Dickstein et al. [37] for image generation, DDPMs have
been further simpliﬁed and accelerated [12, 38], and en-
hanced [1, 28, 29, 53] in recent years. Previous works have
explored applying diffusion models to various generation
tasks, including image inpainting [25] and text generation
[20]. Here, we explore using diffusion models to tackle 3D
pose estimation with our DiffPose framework. Unlike these
generation tasks [20,25] that often start the generation pro-
cess from random noise, our pose estimation process starts
from an estimated 2D pose with uncertainty and indeter-
minacy in 3D space, where the uncertainty distribution dif-
fers for each pose and can also be irregular and difﬁcult to
characterize. We also design a GCN-based architecture as
our diffusion model g, and condition it on spatial-temporal
context information to aid the reverse diffusion process and
obtain accurate 3D poses.
3. Background on Diffusion Models
Diffusion models [12,38] are a class of probabilistic gen-
erative models that learn to transform noise hK ∼N(0, I)
to a sample h0 by recurrently denoising hK, i.e., (hK →
hK−1 →... →h0).
This denoising process is called
reverse diffusion. Conversely, the process (h0 →h1 →
... →hK) is called forward diffusion.
To allow the diffusion model to learn the reverse diffu-
sion process, a set of intermediate noisy samples {hk}K−1
k=1
are needed to bridge the source sample h0 and the Gaus-
sian noise hK. Speciﬁcally, forward diffusion is conducted
to generate these samples, where the posterior distribution
q(h1:K|h0) from h0 to hK is formulated as:
q(h1:K|h0) :=
K
Y
k=1
q(hk|hk−1)
(1)
q(hk|hk−1) := Npdf
 hk

r αk
αk−1 hk−1, (1 −
αk
αk−1 )I

,
(2)
where Npdf(hk|·) refers to the likelihood of sampling hk
conditioned on the given parameters, and α1:K ∈(0, 1]K is
a ﬁxed decreasing sequence that controls the noise scaling
at each diffusion step. Using the known statistical results
for the combination of Gaussian distributions, the posterior
for the diffusion process to step k can be formulated as:
q(hk|h0) :=
Z
q(h1:k|h0)dh1:k−1
=Npdf(hk|√αkh0, (1 −αk)I).
(3)
Thus, hk can be expressed as a linear combination of the
source sample h0 and a noise variable ϵ, where each element
of ϵ is sampled from N(0, 1), as follows:
hk = √αkh0 +
p
(1 −αk)ϵ.
(4)
Hence, when a long decreasing sequence α1:K is set such
that αK ≈0, the distribution of hK will converge to a stan-
dard Gaussian, i.e., hK ∼N(0, I). This indicates that the
source signal h0 will eventually be corrupted into Gaussian
noise, which conforms to the non-equilibrium thermody-
namics phenomenon of the diffusion process [37].
Using the sample h0 and noisy samples {hk}K
k=1 gener-
ated by forward diffusion, the diffusion model g (which is
often a deep network parameterized by θ) is optimized to
approximate the reverse diffusion process. Speciﬁcally, al-
though the exact formulations may differ [12, 37, 38], each
reverse diffusion step can be expressed as a function f that
takes in hk and diffusion model g as input to generate an
output hk−1 as follows:
hk−1 = f(hk, g).
(5)
Finally, during testing, a Gaussian noise hK can be eas-
ily sampled, and the reverse diffusion step introduced in
Eq. 5 can be recurrently performed to generate a high-
quality sample h0 using the trained diffusion model g.
4. Proposed Method: DiffPose
Given an RGB image frame It or a video clip Vt =
{Iτ}(t+T )
τ=(t−T ), the goal of 3D human pose estimation is to
predict the 3D coordinates of all the J keypoints of the hu-
man body in It. In this paper, inspired by diffusion-based
generative models that can recurrently shed the indetermi-
nacy in an initial distribution (e.g., Gaussian distribution) to
reconstruct a high-quality determinate sample, we frame the
3D pose estimation task as constructing a determinate 3D
pose distribution (H0) from the highly indeterminate pose
distribution (HK) via diffusion models, which can handle
the uncertainty and indeterminacy of 3D poses.
As shown in Fig. 2, we conduct pose estimation in two
stages: (i) Initializing the indeterminate 3D pose distribu-
tion HK based on extracted heatmaps, which capture the
underlying uncertainty of the input 2D pose in 3D space;
(ii) Performing the reverse diffusion process, where we use
a diffusion model g to progressively denoise the initial dis-
tribution HK to a desired high-quality determinate distribu-
tion H0, and then we can sample h0 ∈R3×J from the pose
distribution H0 to synthesize the ﬁnal 3D pose hs.

X
Z
Y
X
Z
Y
GCN Layer
GCN Layer
𝒇𝑺𝑻
Context 
Encoder 𝝓𝑺𝑻
𝒇𝑫
𝒌
𝒉𝒌
𝟏
𝒉𝒌
𝟐
𝒉𝒌
𝑵
𝒉𝒌(𝟏
𝟏
𝒉𝒌(𝟏
𝟐
𝒉𝒌(𝟏
𝑵
Final XYZ 
pose 𝒉𝒔
Initial XYZ Pose 
distribution 𝑯𝑲
GCN Layer
GCN Layer
Attention Layer
GCN Layer
GCN Layer
Attention Layer
…
…
…
the 𝒌𝒕𝒉conditional reverse diffusion step
Determinate XYZ 
Pose distribution 𝑯𝟎
𝒇𝑺𝑻: Spatial-temporal context feature
𝒇𝑫  
𝒌:  The 𝒌𝒕𝒉diffusion step embedding
Diffusion model 𝒈
X
Z
Y
𝒌= 𝒌−𝟏
𝒌= 𝟎
𝒌= 𝑲
Heatmaps
Depth 
distribution
Figure 2. Illustration of our DiffPose framework during inference. First, we use the Context Encoder φST to extract the spatial-temporal
context feature fST from the given 2D pose sequence. We also generate diffusion step embedding f k
D for each kth diffusion step. Then,
we initialize the indeterminate pose distribution HK using heatmaps derived from an off-the-shelf 2D pose detector and depth distributions
that can either be computed from the training set or predicted by the Context Encoder φST . Next, we sample N noisy poses {hi
K}N
i=1 from
HK, which are required for performing distribution-to-distribution mapping. We feed these N poses into the diffusion model K times,
where diffusion model g is also conditioned on fST and f k
D at each step, to obtain {hi
0}N
i=1 which represents the high-quality determinate
distribution H0. Lastly, we use the mean of {hi
0}N
i=1 as our ﬁnal 3D pose hs.
In Sec. 4.1, we describe how to initialize the 3D distribu-
tion HK from an input 2D pose that effectively captures the
uncertainty in the 3D space. Then, we explain our forward
diffusion process in Sec. 4.2 and the reverse diffusion pro-
cess in Sec. 4.3. After that, we present the detailed training
and testing process in Sec. 4.4. Finally, the architecture of
our diffusion network is detailed in Sec. 4.5.
4.1. Initializing 3D Pose Distribution HK
In previous diffusion models [11,12,38], the reverse dif-
fusion process often starts from random noise, which is pro-
gressively denoised to generate a high-quality output. How-
ever, in 3D pose estimation, our input here is instead an
estimated 2D pose that has its own uncertainty character-
istics in 3D space. To aid our diffusion model in handling
the uncertainty and indeterminacy of each input 2D pose
in 3D space, we would like to initialize a corresponding 3D
pose distribution HK that captures the uncertainty of the 3D
pose. Thus, the reverse diffusion process can start from the
distribution HK with sample-speciﬁc knowledge (in con-
trast to Gaussian noise with no prior information), which
leads to better performance. Below, we describe how we
construct the x, y, and z uncertainty distribution for each
joint of an input pose.
Initializing 

## conclusion
This paper presents DiffPose, a novel diffusion-based
framework that handles the uncertainty and indeterminacy
in monocular 3D pose estimation.
DiffPose ﬁrst initial-
izes the indeterminate 3D pose distribution and then recur-
rently sheds the indeterminacy in this distribution to obtain
the ﬁnal high-quality 3D human pose distribution for reli-
able pose estimation. Extensive experiments show that the
proposed DiffPose achieves state-of-the-art performance on
two widely used benchmark datasets.

Acknowledgments. This work is supported by MOE AcRF Tier 2 (Pro-
posal ID: T2EP20222-0035), National Research Foundation Singapore un-
der its AI Singapore Programme (AISG-100E-2020-065), and SUTD SKI
Project (SKI 2021 02 06). This work is also supported by TAILOR, a
project funded by EU Horizon 2020 research and innovation programme
under GA No 952215.