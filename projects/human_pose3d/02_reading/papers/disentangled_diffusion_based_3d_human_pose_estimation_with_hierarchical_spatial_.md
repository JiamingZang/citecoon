# Disentangled Diffusion-Based 3D Human Pose Estimation with Hierarchical Spatial and Temporal Denoiser

> 2024 · id: W4393158891 · arXiv: 2403.04444 · pdf: https://ojs.aaai.org/index.php/AAAI/article/download/27847/27720 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Recently, diffusion-based methods for monocular 3D hu-
man pose estimation have achieved state-of-the-art (SOTA)
performance by directly regressing the 3D joint coordinates
from the 2D pose sequence. Although some methods decom-
pose the task into bone length and bone direction predic-
tion based on the human anatomical skeleton to explicitly
incorporate more human body prior constraints, the perfor-
mance of these methods is significantly lower than that of
the SOTA diffusion-based methods. This can be attributed
to the tree structure of the human skeleton. Direct applica-
tion of the disentangled method could amplify the accumu-
lation of hierarchical errors, propagating through each hier-
archy. Meanwhile, the hierarchical information has not been
fully explored by the previous methods. To address these
problems, a Disentangled Diffusion-based 3D Human Pose
Estimation method with Hierarchical Spatial and Temporal
Denoiser is proposed, termed DDHPose. In our approach:
(1) We disentangle the 3D pose and diffuse the bone length
and bone direction during the forward process of the diffu-
sion model to effectively model the human pose prior. A dis-
entanglement loss is proposed to supervise diffusion model
learning. (2) For the reverse process, we propose Hierarchical
Spatial and Temporal Denoiser (HSTDenoiser) to improve
the hierarchical modeling of each joint. Our HSTDenoiser
comprises two components: the Hierarchical-Related Spatial
Transformer (HRST) and the Hierarchical-Related Temporal
Transformer (HRTT). HRST exploits joint spatial informa-
tion and the influence of the parent joint on each joint for
spatial modeling, while HRTT utilizes information from both
the joint and its hierarchical adjacent joints to explore the hi-
erarchical temporal correlations among joints. Extensive ex-
periments on the Human3.6M and MPI-INF-3DHP datasets
show that our method outperforms the SOTA disentangled-
based, non-disentangled based, and probabilistic approaches
by 10.0%, 2.0%, and 1.3%, respectively. Code and models are
available at https://github.com/Andyen512/DDHPose

## introduction
3D Human Pose Estimation (HPE) has crucial applications
in virtual reality (Hagbi et al. 2010), human motion recog-
nition (Zhang et al. 2022b), and human-computer interac-
*These authors contributed equally.
†Corresponding author
Copyright © 2024, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.
Figure 1: Left: The hierarchy defined in our method and
the forward kinematic structure (drawn with brown dashed
lines) based on the Human3.6M dataset. Right: The MPJPE
of the hierarchy 1-5 joints comparison among Anatomy3D
(Chen et al. 2021), MixSTE (Zhang et al. 2022a), D3DP
(Shan et al. 2023) and our method.
tion (Kisacanin, Pavlovic, and Huang 2005). The goal is to
regress the 3D joints locations of a human in the 3D space
using the input of 2D pose sequence. Most of the methods
first derive predictions of 2D joints using estimators such as
HRNet (Wang et al. 2020), CPN (Chen et al. 2018), Open-
Pose (Cao et al. 2017) and AlphaPose (Fang et al. 2017), and
then perform 2D-to-3D lifting to obtain the final estimation
results.
Recently, monocular 3D human pose estimation has expe-
rienced significant advancements. Many methods have been
proposed to alleviate the depth ambiguity.
(Pavllo et al.
2019) considers this issue by exploring temporal informa-
tion with the convolutional network while the transformer-
based methods (Zheng et al. 2021; Zhang et al. 2022a) make
use of spatial-temporal information to compensate for the in-
formation loss in the 2D to 3D mapping process. Learning or
introducing human pose prior is another method to mitigate
the depth ambiguity. (Shan et al. 2023; Ci et al. 2023; Gong
et al. 2023) introduce the original pose distribution prior in
the training phase, and model 2D-to-3D lifting as a process
to denoise from the pose distribution with uncertain noise.
Moreover, some disentangle-based methods like (Xu et al.
2020; Chen et al. 2021; Wang et al. 2022) explicitly predict
the bone length and bone direction, subsequently composing
3D joints locations based on the forward kinematics of the
human skeleton. Such methods employ explicit pose con-
straints, integrating symmetry loss, joint angle limits (Xu
et al. 2020), and the consistent bone length in the videos
(Chen et al. 2021).

However, there are two problems existing in these meth-
ods: (1) Despite the advantages of disentangle-based tech-
niques in incorporating human pose priors, they come with
the drawback of amplified error accumulation, resulting
in decreased performance. Meanwhile, diffusion-based 3D
HPE methods
(Shan et al. 2023; Ci et al. 2023; Gong
et al. 2023) directly add noise to the original 3D pose which
is not conducive to learn the explicit human pose prior.
What if we disentangle the diffusion model by adding noise
to bone length and direction separately? This disentangle-
based model can separately focus on the temporal consis-
tency of bone length and joint angle variations, better en-
abling the diffusion model to learn human pose prior. (2)
Although the transformer-based methods have the ability to
explore the spatial-temporal context information, these mod-
els generally lack attention to the fine-grained hierarchical
information among joints. As shown in the left side of Fig-
ure 1, we group joints into six hierarchies based on the kine-
matic tree depth of the human body. The experiment results
in the right side of Figure 1 show a rising hierarchical accu-
mulation error when the hierarchy increases from 1 to 5.
To solve the problems mentioned above, (1) we introduce
the disentangled method in the forward process of diffu-
sion model instead of decomposing the 3D HPE task into
bone length and bone direction prediction task, which sim-
plifies learning the human pose prior. (2) For better model-
ing the hierarchical relation among joints, we propose HST-
Denoiser, which contains two modules: the Hierarchical-
Related Spatial Transformer (HRST) and the Hierarchical-
Related Temporal Transformer (HRTT). In HRST, due to the
spatial information of a joint is influenced by its parent joint,
we supply the joint’s attention with information from its par-
ent joint. Besides, in HRTT, we try to make cross-attention
of the joint and the adjacent joints to learn the temporal in-
terrelationships. HRST and HRTT make the joints pay more
attention to their hierarchical-related joints, which conse-
quently improves performance on higher-hierarchy joints
and contributes to overall better performance.
In conclusion, our contributions can be summarized as
follows:
• We propose the first Disentangled Diffusion-based 3D
human Pose Estimation method with Hierarchical Spa-
tial and Temporal Denoiser (DDHPose), which intro-
duces Hierarchical Information in two ways.
• We present the Disentangle Strategy for the forward dif-
fusion process based on hierarchical information to better
model explicit pose prior. Additionally, we incorporate a
disentanglement loss to guide the model’s training.
• The
HSTDenoiser
is
introduced,
comprising
the
Hierarchical-Related
Spatial
Transformer
(HRST)
and the Hierarchical-Related Temporal Transformer
(HRTT). This denoiser strengthens the relation among
the hierarchical joints by enhancing the attention weight
of adjacent joints in the reverse diffusion process.
• Our method outperforms the performance of disentangle-
based, non-disentangle based, and probabilistic methods
on 3D HPE benchmarks. The qualitative results show
that our method has better performance on the higher hi-
erarchy joints.

## method
The overview of our proposed DDHPose is in Figure 2(a). In
our framework, we decompose the 3D joint location into the
bone length and bone direction, adding noise in the forward
process. After the forward process, the noisy bone length,
noisy bone direction, and 2D pose are fed to HSTDenoiser,
which contains HRST and HRTT to reverse the 3D pose
from the noisy input. Further details will be introduced in
the following section.
3D POSE Disentanglement Strategy
We first introduce the motivation of why we use the
disentanglement strategy in our paper. The original non-
disentangle diffusion-based methods directly take the 3D
joint sequence as input without any skeleton structural prior.
Modeling the dependencies among each joint pair tends
to be challenging due to their complex and dense relation
which makes the optimization task more difficult. But in
our approach, Our disentangle-based method first decom-
poses ground truth 3D pose y0 ∈RN×J×3 to bone length
l0 ∈RN×(J−1)×1 and bone direction d0 ∈RN×(J−1)×3,
where N is the frame length of the input sequences, J
is the number of joints. This operation divides the dense
and high-dimensional problem into multiple sparse and low-
dimensional sub-problems, making the gradient-based op-
timization easier. Besides, The disentangling representa-
tion with bone length and direction makes it easier to add
structural constraints, such as temporal consistency in bone
length. The addition of bone length loss as a constraint en-
hances output certainty and shows effectiveness in the ex-
periment.
For the i-th bone, ground truth length li
0 and direction di
0
can be defined as:
li
0 = ∥yci
0 −ypi
0 ∥2 ,
di
0 =
yci
0 −ypi
0
∥yci
0 −ypi
0 ∥2
(1)
where ci and pi are the child joint and parent joint, which are
in the upstream and downstream of the i-th bone according
to the forward kinematic structure defined in the left portion
of Figure 1.
The disentangled bone length and bone direction are both
processed through the forward and reverse processes.
The Forward Process
The Forward Process is an approximate posterior that fol-
lows the Markov chain that gradually adds Gaussian noise
N(0, I) to the original data x0. Followed by DDPM (Ho,
Jain, and Abbeel 2020), the forward process can be defined
as:
q(xt|x0) := N(xt; √¯αtx0, (1 −¯αt)I)
(2)
where ¯αt := Qt
s=0αs and αs := 1−βs, βs is a noise sched-
ule and we adopt the cosine-schedule proposed by (Song

and Ermon 2020) which always increases as the sampling
step t increases.
During the training stage in Figure 2(a), when we get the
disentangled bone length l0 and bone direction d0, we can do
the forward process separately in Eq (2) to get the noisy bone
length lt and bone direction dt by adding t-step Gaussian
noise as:
lt = √¯αtl0 +
√
1 −¯αtϵ,
dt = √¯αtd0 +
√
1 −¯αtϵ (3)
where ϵ is the random Gaussian sampled at the t-step.
The Reverse Process
In the training stage, under the condition of a 2D pose se-
quence x ∈RN×J×2, the contaminated bone length lt and
direction dt from the forward process are concatenated. This
combined information is then processed through the HST-
Denoiser and a regression head, resulting in the denoised 3D
joints locations ˜y0. Then using our disentanglement strategy
to decompose bone length ˜l0 and bone direction ˜d0 for dis-
entanglement supervision during training.
At the inference stage, inspired by the method in
D3DP (Shan et al. 2023), we simultaneously sample H hy-
potheses from the Gaussian distribution as the initial noisy
bone length and direction. They are then denoised through
the trained denoiser, resulting in the denoised bone length
˜l0:H,0 and bone direction ˜d0:H,0. Then ˜l0:H,0 and ˜l0:H,0 are
employed to generate noisy samples ˜l0:H,t′ and ˜d0:H,t′ in the
next iteration at step t′ via DDIM (Song, Meng, and Ermon
2020):
˜l0:H,t′ = √¯αt′˜l0:H,0 +
q
1 −¯αt′ −σ2
t · ϵtl + σtϵ
˜d0:H,t′ = √¯αt′ ˜d0:H,0 +
q
1 −¯αt′ −σ2
t · ϵtd + σtϵ
(4)
where ϵtl
=
˜l0:H,t−√¯αt˜l0:H,0
√1−¯αt
, ϵtd
=
˜d0:H,t−√¯αt ˜d0:H,0
√1−¯αt
are the noise at step t and σt =
p
(1 −¯αt′)/(1 −¯αt) ·
p
1 −¯αt/¯αt′ controls the stochastic of the diffusion pro-
cess. We can control the hypothesis number H and iteration
times W in the whole process. Appropriately increasing H
and W can optimize the final prediction of the bone length
and bone direction and improve the performance of MPJPE
and P-MPJPE in our experiments.
Hierarchical Spatial and Temporal Denoiser
Both in
the training or inference phase, noisy bone length and
bone direction are fed into our HSTDenoiser to reconstruct
the original data. HSTDenoiser, which consists of HRST
and HRTT, is used to explore the hierarchical informa-
tion, specifically the relation among the joint, the parent
joint, and the child joint. The main architecture is shown
in Figure 2(b). We utilize a linear layer to enhance the in-
put feature and use the spatial-temporal transformer block
in MixSTE (Zhang et al. 2022a) to extract joint features.
We also introduce Hierarchical spatial position embedding
HSP for better spatial position modeling and temporal em-
bedding TP for better temporal position modeling. HSP
embedding not only contains the spatial position informa-
tion of each joint but also contains the joint hierarchy in-
formation. Inspired by (Li et al. 2023), we split the joints
Figure 3: The main components of our HSTDenoiser.
(a): Hierarchical-Related Spatial Transformer(HRST). (b):
Hierarchical-Related Temporal Transformer(HRTT).
into six hierarchies according to the joint’s depth of the hu-
man body tree-like structure to build hierarchical embed-
ding, which is shown in the left portion of Figure 1. It means
the joints in the same hierarchy share the same embedding.
Based on hierarchical embedding, the hierarchical-related
information can be well learned by our model. After one
layer of spatio-temporal transformer modeling, we utilize
the HRST and HRTT, which we introduce in the subsequent
section, to model the spatio-temporal correlations of joints
through d loops alternately.
Transformer
The Transformer model we used in our ap-
proach is followed by (Vaswani et al. 2017), the basic idea
of query, key, and value is that the query is used to match
with the key, and then according to the degree of matching,
to selectively focus on the value. This design allows the
output to selectively pay attention to the value based on
the query. The mechanism of attention can be formulated by:
Attention = Softmax(A)V,
A = QKT
√dm
(5)
where Q, K, V ∈RZ×dm are generated by the input fea-
ture, Z is the number of tokens and dm is the dimension of
feature. A ∈RZ×Z denotes the attention weight matrix.
The input of our transformer module is the noisy bone
length and direction generated by the forward diffusion pro-
cess. For better denoising, the 2D pose sequence is added as
the condition and concatenated with the 3D noisy data as the
whole input.
HRST
In HRST, we enhance the modeling of joint spa-
tial information with its parent joint feature. Based on the
forward kinematics structure, we define all the hierarchical-
related joints triplets in the human body as {Jp, J, Jc},
where Jp, J, Jc are the set of jp, j, jc. In each of the
hierarchical-related joints triplets {jp, j, jc}, jp is the parent

Algorithm 1: Hierarchical-Related Spatial Transformer
Input: Q, K, V generated by Joint feature f ∈RN×J×C
Parameter: Hierarchical Related joints triplets {Jp, J, Jc}
Output:Hierarchical-Related spatial attention map
1: A = QKT
√dm
2: for jp, j, jc in Jp, J, Jc do
3:
A[j][jc] += A[jp][j]
4:
A[jc][j] += A[jp][j]
5:
A[j][jc]/2.0 , A[jc][j]/2.0
6: end for
7: return A
joint of joint j and jc is the child joint of joint j accord-
ing to the forward kinematic structure. Because our method
decomposes the location of joints into bone length and di-
rection, we believe that the position of a joint is influenced
by the position of its parent joint, combined with the bone
length and direction. The attention of the parent joint signif-
icantly affects the attention of the joint. Therefore, in HRST,
we augment the parent joint’s influence on each joint at the
spatial level, specifically as illustrated in Algorithm 1. After
Algorithm 1, we derive the weight matrix A and then com-
pute the attention utilizing Eq (5).
HRTT
We propose HRTT to further introduce the inter-
relationship between each joint and its hierarchical adjacent
joints in the temporal dimension. In the process of exploring
joint temporal information, we believe that due to the tree-
like structure of the human skeleton, there exists a strong
temporal correlation among a joint, its parent joint, and its
child joints. Because we have enhanced the relation between
a joint and its parent joint, the temporal relation between a
joint and its child joints is more considered in HRTT.
Specifically, we primarily adopt a cross-attention mech-
anism to capture the relationship between the current joint
and its child joints. According to the kinematic chain struc-
ture of human joints, we comp

## experiments
Dataset
Human3.6M
(Ionescu et al. 2014) is widely used in 3D
HPE task. It contains 3.6 million 3D human poses and cor-
responding images with 11 professional actors and collected
in 17 scenarios. Following the previous work (Pavllo et al.
2019; Zheng et al. 2021; Zhang et al. 2022a), we use S1-S9
for training and use S9 and S11 for testing.
MPI-INF-3DHP
(Mehta et al. 2017) record 8 actors,
composed of 4 males and 4 females, each undertaking 8 dif-
ferent sets of activities. We use eight activities performed
by eight actors to train our model, while the test dataset has
seven different activities.
Metrics
We use the mean per joint position error (MPJPE) and pro-
crustes mean per joint position error(P-MPJPE) for evalua-
tion. MPJPE measures the Euclidean distance between the
ground truth and the predicted 3D positions of each joint
while P-MPJPE makes procrustes analysis involves scaling,
translating, and rotating the predicted pose to best align it
with the ground truth, providing a more fair comparison.
Following D3DP (Shan et al. 2023), we use J-AGG based
MPJPE and P-MPJPE to evaluate our results.

Deterministic methods: Disentangle-based model
Protocol #1: MPJPE
Dir. Disc. Eat Greet Phone Photo Pose Pur. Sit SitD. Smoke Wait WalkD. Walk WalkT. Avg.
DKA (Xu et al. 2020)(N=9)
37.4 43.5 42.7 42.7
46.6
59.7 41.3 45.1 52.7 60.2
45.8
43.1
47.7
33.7
37.1
45.6
Anatomy3D (Chen et al. 2021)(N=243)
41.4 43.5 40.1 42.9
46.6
51.9 41.7 42.3 53.9 60.2
45.4
41.7
46.0
31.5
32.7
44.1
Virtual Bones (Wang et al. 2022)(N=243) 42.4 43.5 41.0 43.5
46.7
54.6 42.5 42.1 54.9 60.5
45.7
42.1
46.5
31.7
33.7
44.8
Ours (N=243, H=1, W=1)
37.3 40.0 35.2 37.7
41.1
46.7 38.4 38.4 52.2 53.3
41.4
38.9
38.8
27.6
27.7
39.7
Deterministic methods: Non-Disentangle based model
Protocol #1: MPJPE
Dir. Disc. Eat Greet Phone Photo Pose Pur. Sit SitD. Smoke Wait WalkD. Walk WalkT. Avg.
VideoPose3D (Pavllo et al. 2019)(N=243) 45.2 46.7 43.3 45.6
48.1
55.1 44.6 44.3 57.3 65.8
47.1
44.0
49.0
32.8
33.9
46.8
PoseFormer (Zheng et al. 2021)(N=81)
41.5 44.8 39.8 42.5
46.5
51.6 42.1 42.0 53.3 60.7
45.5
43.3
46.1
31.8
32.2
44.3
P-STMO (Shan et al. 2022)(N=243)
38.9 42.7 40.4 41.1
45.6
49.7 40.9 39.9 55.5 59.4
44.9
42.2
42.7
29.4
29.4
42.8
MixSTE (Zhang et al. 2022a)(N=243)
37.6 40.9 37.3 39.7
42.3
49.9 40.1 39.8 51.7 55.0
42.1
39.8
41.0
27.9
27.9
40.9
PoseFormerV2 (Zhao et al. 2023)(N=243) 41.3 45.5 41.5 44.0
46.7
53.8 42.6 42.6 55.2 64.6
45.7
42.9
45.8
32.3
32.9
45.2
STCFormer (Tang et al. 2023)(N=243)
38.4 41.2 36.8 38.0
42.7
50.5 38.7 38.2 52.5 56.8
41.8
38.4
40.2
26.2
27.7
40.5
Ours (N=243, H=1, W=1)
37.3 40.0 35.2 37.7
41.1
46.7 38.4 38.4 52.2 53.3
41.4
38.9
38.8
27.6
27.7
39.7
Probabilistic methods
Protocol #1: MPJPE
Dir. Disc. Eat Greet Phone Photo Pose Pur. Sit SitD. Smoke Wait WalkD. Walk WalkT. Avg.
MHFormer (Li et al. 2022)(N =351, H=3) 39.2 43.1 40.1 40.9
44.9
51.2 40.6 41.3 53.5 60.3
43.7
41.1
43.8
29.8
30.6
43.0
GFPose (Ci et al. 2023)(H = 10)
39.9 44.6 40.2 41.3
46.7
53.6 41.9 40.4 52.1 67.1
45.7
42.9
46.1
36.5
38.0
45.1
D3DP (Shan et al. 2023)(N=243,∗)
37.3 39.5 35.6 37.8
41.3
48.2 39.1 37.6 49.9 52.8
41.2
39.2
39.4
27.2
27.1
39.5
Ours (N=243, H=5, W=1)
37.2 39.9 35.1 37.6
41.0
46.5 38.3 38.3 52.1 53.1
41.3
38.8
38.7
27.5
27.6
39.5
Ours (N=243, H=20, W=10)
36.4 39.5 34.9 37.6
40.1
45.9 37.8 37.8 51.5 52.2
40.8
38.3
38.3
27.0
27.0
39.0
Table 1: Results on Human3.6M in millimeters under MPJPE. N, H, W: the number of input frames, hypotheses, and iterations
used in the inference stage. In this table, we compare with the deterministic and probabilistic methods. The best results are
highlighted in bold. (∗)-For clarity, H=20, W=10 is omitted.

## related_work
3D Human Pose Estimation
3D HPE can be divided into two categories, one that di-
rectly regresses the 3D human pose from raw RGB images
and another that first detects the 2D human pose from raw
RGB images by using one of the 2D human pose estimation
methods like HRNet (Wang et al. 2020), CPN (Chen et al.
2018), OpenPose (Cao et al. 2017), AlphaPose (Fang et al.
2017) and then make a 2D-to-3D lifting to get the final es-
timation results. (Tekin et al. 2016; Pavlakos et al. 2017;
Sun et al. 2018) directly use convolutional neural network
to regress 3D pose from a feature volume. Based on the ac-
curacy improvement of 2D human pose estimation, (Pavllo
et al. 2019) uses a fully convolutional model based on dilated
temporal convolutions to estimate 3D poses and achieves
better results. (Zheng et al. 2021; Zhang et al. 2022a; Zhao
et al. 2023) demonstrate that 3D poses in the video can be ef-
fectively estimated with spatial-temporal transformer archi-
tecture. Due to the superior performance of two-stage meth-
ods, we also employ a two-stage approach for 3D human
pose estimation in this paper. While these models are capa-
ble of exploring spatial-temporal context information, they
always fail to incorporate fine-grained hierarchical informa-
tion. This leads to a higher hierarchical accumulation error
from hierarchy 1 to hierarchy 5 in the right portion of Fig-
ure 1. Therefore, we apply HRST and HRTT in our method,
providing more hierarchical features for better modeling.
Diffusion Model
The diffusion model belongs to a class of generative mod-
els, which has outstanding performance in image genera-
tion (Batzolis et al. 2021; Nichol et al. 2021; Ho et al. 2022),
image super-resolution (Saharia et al. 2022), semantic seg-
mentation (Baranchuk et al. 2021), multi-modal tasks (Fan
et al. 2023) and so on. The diffusion model is first introduced
by (Sohl-Dickstein et al. 2015), which defines two stages
which are the forward process and the reverse process. The
forward process refers to the gradual addition of Gaussian
noise to the data until it becomes random noise, while the re-
verse process is the denoising of noisy data to obtain the true
samples. The following works DDPM (Ho, Jain, and Abbeel
2020) and DDIM (Song, Meng, and Ermon 2020) simplify
and accelerate previous diffusion models which make a solid
foundation in this area.
Recent explorations
(Choi, Shim, and Kim 2022;
Holmquist and Wandt 2022; Ci et al. 2023; Shan et al.
2023) try to apply the diffusion model to 3D human pose
estimation. Note that (Gong et al. 2023) also uses a dif-
fusion model for 3D HPE, but they additionally introduce
the heatmap distribution of 2D pose, and the depth distribu-
tion to initialize 3D pose distribution, making a GMM-based
forward diffusion process, so that they have a better perfor-
mance than the other diffusion-based 3D HPE model. How-
ever, these approaches directly add t-step noise in the for-
ward process to the original 3D pose, which is not conducive

Figure 2: (a): The overview of DDHPose’s training pipeline. (b): The architecture of our HSTDenoiser, which contains HRST
and HRTT. HSP embedding and TP embedding are used in the spatial-temporal transformer to better modeling the hierarchi-
cal relation of spatial position information and temporal position information. f is the feature extracted by the spatial-temporal
transformer and fc is the child joint feature separated from f. The input consists of N frames for both 2D pose and 3D pose.
For better clarity, only three frames of input are illustrated here as an example.
to learning the explicit human pose prior. Additionally, (Xu
et al. 2020; Chen et al. 2021; Wang et al. 2022) have a higher
accumulation of errors that disentangle the 3D joint loca-
tion to the prediction of bone length and bone direction. We
introduce the disentanglement strategy in the forward pro-
cess of the diffusion model, integrating the explicit human
body prior to the diffusion model, and proposing the first
disentangle-based diffusion model for 3D HPE. As a result,
we achieve outstanding results on 3D HPE benchmarks.

## conclusion
In this paper, we propose DDHPose, a diffusion-based 3D
HPE method that introduces hierarchical information in two
ways: (1)We propose the Disentangle Strategy for the for-
ward diffusion process, which decomposes the 3D pose into
bone length and direction based on the Hierarchical Infor-
mation. This simplifies learning the human pose prior, re-
duces optimization dimension, and speeds up gradient de-
scent. (2)We propose HSTDenoiser to strengthen the rela-
tion among the hierarchical joints by enhancing the attention
weight of adjacent joints for each joint in the reverse dif-
fusion process. Extensive results on Human3.6M and MPI-
INF-3DHP reveal that our method surpasses the disentangle-
based method, non-disentangle based method, and the prob-
abilistic approaches on 3D HPE benchmarks.

Acknowledgments
This work is jointly supported by National Natural
Science Foundation of China (62276025, 62206022),
Beijing Municipal Science & Technology Commission
(Z231100007423015) and Shenzhen Technology Plan Pro-
gram (KQTD20170331093217368).
Appendix
Implementation Details
Our model is implemented in Pytorch (Paszke et al. 2019),
using AdamW (Loshchilov and Hutter 2017) as our opti-
mizer with the momentum parameters as β1, β2 = 0.9, 0.999.
We train our model for 400 epochs on NVIDIA Tesla A800
and the initial learning rate is 1e−4, with a shrink factor of
0.993. We set batch size to 4 and in each batch, we use the
length of 243 frames to train our model. We use a total of
eight layers of spatio-temporal transformers in our model.
The timestep t is sampled from U(0, T ) during training,
where T is the maximum number of timesteps and is set
to 1000. The first layer employed the spatio-temporal trans-
former in MixSTE (Zhang et al. 2022a), followed by seven
layers of HRST and HRTT used in a loop.
Inference Details
During the inference stage, we simultaneously sample H
bone length and bone direction hypotheses from a standard
normal Gaussian distribution N(0, I) at the initial timestep
T for the diffusion-based HSTDenoiser. The noisy 3D bone
length l0:H,T and bone direction d0:H,T , with the condition
of the 2D pose sequence, are denoised by our denoiser and
make predictions of 3D poses ˜y0:H,0. Then we can decom-
pose ˜y0:H,0 to get the feasible predictions of bone length
˜l0:H,0 and bone direction ˜d0:H,0.
The denoised ˜l0:H,0 and ˜d0:H,0 can be used to generate the
noisy input l0:H,t′ and d0:H,t′ at timestep t
′ via DDIM(Song,
Meng, and Ermon 2020), which is formulated in Eq (4) and
shown in Figure 5. The updated input l0:H,t′ and d0:H,t′ will
lead to another refined prediction and this procedure can be
repeated for W times to get the final predictions, which can
enhance the prediction performance to some extent.
Figure 5: The overview of the inference pipeline.
Additional Quantitative Results
The supplement comparison of P-MPJPE is shown in
Table 7. We compare our method with the SOTA de-
terministic methods and probabilistic methods. Based on
whether the regression of the 3D pose locations is decom-
posed into the regression of bone length and bone direc-
tion, we divide the methods into disentangle-based meth-
ods and non-disentangle based methods. For disentangle-
based methods, we can see from the table that our method
achieves the best P-MPJPE of 31.2mm, which outper-
forms Anatomy3D (Chen et al. 2021) by 3.7mm(10.6%).
While for non-disentangle based model, we improve STC-
Former (Tang et al. 2023) by 0.6mm(1.9%) under P-MPJPE.
And then we compare our method with probabilistic meth-
ods, our method reaches the SOTA MPJPE of 31.2mm, out-
erperforms D3DP (Shan et al. 2023) by 0.4mm(1.3%).
Qualitative results
We evaluate the qualitative results between our method,
MixSTE (Zhang et al. 2022a) and D3DP (Shan et al. 2023).
The visual results are in Figure 6. Our method has better re-
sults than MixSTE and D3DP for the high hierarchical joints
where the black dashed circle highlighting in the figure.
Figure 6: Qualitative comparison between our method,
MixSTE and D3DP on Human3.6M. The ground truth pose
is drawn in blue and the predicted pose is drawn in red.
The black dashed circle highlights the locations where our
method has better results.

Deterministic methods: Disentangle-based model
Protocol #2: P-MPJPE
Dir. Disc. Eat Greet Phone Photo Pose Pur. Sit SitD. Smoke Wait WalkD. Walk WalkT. Avg.
DKA (Xu et al. 2020)(N=9)
31.0 34.8 34.7 34.4
36.2
43.9 31.6 33.5 42.3 49.0
37.1
33.0
39.1
26.9
31.9
36.2
Anatomy3D (Chen et al. 2021)(N=243)
32.6 35.1 32.8 35.4
36.3
40.4 32.4 32.3 42.7 49.0
36.8
32.4
36.0
24.9
26.5
35.0
Virtual Bones (Wang et al. 2022)(N=243)
32.2 34.9 33.0 35.2
35.7
40.7 32.6 32.1 42.8 48.9
36.5
32.5
35.9
25.0
26.7
34.9
Ours(N=243, H=1, W=1)
29.5 31.9 28.7 30.1
31.3
35.5 29.7 29.5 41.8 42.9
33.4
29.7
30.4
21.7
22.6
31.2
Deterministic methods: Non-Disentangle-based model
Protocol #2: P-MPJPE
Dir. Disc. Eat Greet Phone Photo Pose Pur. Sit SitD. Smoke Wait WalkD. Walk WalkT. Avg.
VideoPose3D (Pavllo et al. 2019)(N=243) 34.1 36.1 34.4 37.2
36.4
42.2 34.4 33.6 45.0 52.5
37.4
33.8
37.8
25.6
27.3
36.5
PoseFormer (Zheng et al. 2021)(N=81)
34.1 36.1 34.4 37.2
34.4
39.2 32.0 31.8 42.9 46.9
35.5
32.0
34.4
23.6
25.2
33.9
P-STMO (Shan et al. 2022)(N =243)
31.3 35.2 32.9 33.9
35.4
39.3 32.5 31.5 44.6 48.2
36.3
32.9
34.4
23.8
23.9
34.4
MixSTE (Zhang et al. 2022a)(N =243)
30.8 33.1 30.3 31.8
33.1
39.1 31.1 30.5 42.5 44.5
34.0
30.8
32.7
22.1
22.9
32.6
PoseFormerV2 (Zhao et al. 2023)(N =243) 32.3 35.9 33.8 35.8
36.0
41.1 33.2 32.7 44.3 51.9
37.4
32.8
35.6
25.2
26.6
35.6
STCFormer (Tang et al. 2023)(N=243)
29.3 33.0 30.7 30.6
32.7
38.2 29.7 28.8 42.2 45.0
33.3
29.4
31.5
20.9
22.3
31.8
Ours(N=243, H=1, W=1)
29.5 31.9 28.7 30.1
31.3
35.5 29.7 29.5 41.8 42.9
33.4
29.7
30.4
21.7
22.6
31.2
Probabilistic methods
Protocol #2: P-MPJPE
Dir. Disc. Eat Greet Phone Photo Pose Pur. Sit SitD. Smoke Wait WalkD. Walk WalkT. Avg.
MHFormer (Li et al. 2022)(N=351,H=3)
31.5 34.9 32.8 33.6
35.3
39.6 32.0 32.2 43.5 48.7
36.4
32.6
34.3
23.9
25.1
34.4
GFPose (Ci et al. 2023)(H = 10)
32.0 39.5 34.4 34.7
38.6
44.3 32.7 31.9 49.0 60.1
38.9
36.6
42.2
28.3
32.3
38.4
D3DP (Shan et al. 2023)(N=243, ∗)
30.6 32.4 29.2 30.9
31.9
37.4 30.2 29.3 40.4 43.2
33.2
30.4
31.3
21.5
22.3
31.6
Ours(N=243,H=5, W=1)
29.5 31.8 28.7 30.1
31.3
35.6 29.7 29.5 41.7 42.9
33.4
29.6
30.4
21.7
22.6
31.2
Ours(N=243, H=20, W=10)
29.4 32.1 28.4 30.1
31.0
35.3 29.5 29.2 41.7 42.7
33.2
29.7
30.6
21.5
22.2
31.2
Table 7: Results on Human3.6M in millimeters under P-MPJPE. N, H, W: the number of input frames, hypotheses, and
iterations used in the inference stage. In this table, we are compared with the deterministic and probabilistic methods. The best
results are highlighted in bold. (∗)-For clarity, H=20, W=10 is omitted.