# Diff9D: Diffusion-Based Domain-Generalized Category-Level 9-DoF Object Pose Estimation

> 2025 · id: arxiv:2502.02525 · arXiv: 2502.02525 · pdf: https://arxiv.org/pdf/2502.02525 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
N
INE-degrees-of-freedom (9-DoF) object pose and size
estimation predicts the three-dimensional (3D) trans-
lation and 3D rotation of an object relative to the camera co-
ordinate system as well as its 3D size. This is a core problem
in augmented reality and robotic 3D scene understanding
[1], [2], [3], [4], [5].
Existing pose estimation approaches can be divided into
instance-level and category-level methods. Instance-level
methods [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16],
[17], [18] are restricted to specific objects the model has been
trained on, which greatly limits their practical applicability.
Category-level pose estimation methods exhibit a degree
of flexibility, and are able to estimate the pose of novel
objects within categories that are seen during training. Wang
et al. [19] proposed the first category-level method. Their
•
Jian Liu, Wei Sun, Hui Yang, and Chongpei Liu are with the National
Engineering Research Center for Robot Visual Perception and Control
Technology, College of Electrical and Information Engineering, School
of Robotics, and the State Key Laboratory of Advanced Design and
Manufacturing for Vehicle Body, Hunan University, Changsha 410082,
China. E-mail: (jianliu, wei sun, huiyang, chongpei56)@hnu.edu.cn
•
Pengchao Deng is with the Institute of Artificial Intelligence and
Robotics, Xi’an Jiaotong University, Xi’an 710049, China. E-mail:
dpc987003425@stu.xjtu.edu.cn
•
Nicu Sebe is with the Department of Information Engineering and
Computer Science, University of Trento, Trento 38123, Italy. E-mail:
sebe@disi.unitn.it
•
Hossein Rahmani is with the School of Computing and Communi-
cations, Lancaster University, LA1 4YW, United Kingdom. E-mail:
h.rahmani@lancaster.ac.uk
•
Ajmal Mian is with the Department of Computer Science, The
University of Western Australia, WA 6009, Australia. E-mail: aj-
mal.mian@uwa.edu.au.
This work was done while Jian Liu and Chongpei Liu were visiting Ph.D.
students with The University of Western Australia and the University of
Trento, respectively, supervised by Prof. Ajmal Mian and Prof. Nicu Sebe.
approach involves the design of a Normalized Object Co-
ordinate Space (NOCS) and the use of the Umeyama algo-
rithm for recovering object pose. However, NOCS exhibits
low accuracy due to its inability to effectively represent the
diverse shape variations among intra-class objects.
In response to the aforementioned challenge, some shape
prior-based methods have been proposed [20], [21], [22],
[23], [24], [25]. Although these methods significantly im-
prove accuracy, they are not trained in an end-to-end fash-
ion. Specifically, these approaches first need to extract the 3D
shape prior based on the CAD models of intra-class known
objects in offline mode. Then, they estimate the NOCS shape
of the intra-class unknown object using shape deformation.
Finally, the Umeyama algorithm is used to perform point
cloud registration to find the object pose. Constructing CAD
model libraries is time-consuming and requires significant
manual effort. To address these problems, some prior-free
methods have been introduced [28], [29], [30], [31], [32],
[33] to directly regress object pose, achieving better real-time
performance during inference. However, these methods still
require large amounts of real-world annotated data for
training, which is expensive to obtain.
In a recent development, to address the challenge of
inadequate real-world training data, some domain adapta-
tion methods [36], [38], [39], [40] and a test-time adaptation
method [41] have been proposed. The domain adaptation
methods require both labeled synthetic data and unlabeled
real-world data for training, whereas the test-time adapta-
tion method solely relies on labeled synthetic data during
the training process. Nevertheless, the performance of these
methods is limited by the huge domain gap between the
rendered synthetic domain and the real world.
Inspired by Noble Laureate Richard Feynman’s quote
arXiv:2502.02525v1  [cs.CV]  4 Feb 2025

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE
2
Fig. 1. Comparison of diffusion model-based image generation and
object pose estimation. (a): The process of diffusion model, where q
and pθ represent the forward (noising) and reverse (denoising) diffusion
processes, respectively. (b): Diffusion model-based image generation
task, which generates an image based on a prompt. (c): The Overall
pipeline of our Diff9D, which redefines the 9-DoF object pose and size
estimation task from a generative perspective, i.e., from Gaussian noise
pose to true object pose. tp, R, and s represent 3D translation, 3D
rotation, and 3D size, respectively.
“What I cannot create, I do not understand”, we propose a
novel paradigm to redefine object pose estimation from a
generative perspective, termed Diff9D. Figure 1 illustrates
an overview of the proposed Diff9D. ∗and ∧denote the
Gaussian noise pose and an intermediate pose of the de-
noising process, respectively. Our motivation is to leverage
the latent generalization ability of the diffusion model to
address the domain generalization challenge [42] in ob-
ject pose estimation. Specifically, we introduce a Denoising
Diffusion Probabilistic Model (DDPM)-based method for
domain-generalized category-level object pose estimation,
which is simple yet effective and does not rely on the use of
any 3D shape priors during training or inference, facilitating
generalization across various object categories. A major
challenge in taking a generative modeling approach to pose
estimation in robotics is that real-time performance is not
feasible since the reverse diffusion requires a large number
of denoising steps that must be performed sequentially. We
address this challenge by leveraging a Denoising Diffusion
Implicit Model (DDIM) [43] and achieving reverse diffusion
in as few as 3 steps, enabling near real-time performance.
Our main contributions and highlights are as follows:
•
We propose a DDPM-based method for domain-
generalized category-level 9-DoF object pose and size
estimation. Our method redefines the pose estima-
tion problem from a generative perspective to reduce
the impact of domain gap. Our model is trained
solely on rendered synthetic data and yet generalizes
to real-world data, eliminating the laborious human
effort required for data collection and annotation.
•
We design a simple yet effective object pose/size dif-
fusion model to directly diffuse the sparse pose data,
achieving near real-time performance by leveraging
the DDIM to perform reverse diffusion in as few as 3
steps. Our model is lightweight and does not require
any 3D shape priors during training or inference.
Specifically, we perform condition extraction based
on the lightweight ResNet18 and PointNet models,
then propose a transformer-based denoiser for de-
noising.
•
We build a robotic grasping system and deploy the
proposed method on it. Extensive experiments on
real-world robotic grasping scenes and two widely
used challenging datasets (REAL275 and Wild6D)
demonstrate that the proposed method achieves su-
perior domain generalization performance, which is
able to generalize to real-world grasping tasks at 17.2
frames per second (FPS).
The rest of this paper is organized as follows. The next
section reviews related works. Sec. 3 presents the proposed
method and Sec. 4 presents the designed robotic grasping
system, including hardware/software setup and workflow.
Next, extensive experimental results are reported in Sec. 5
to demonstrate the superior performance of the proposed
method. Finally, Sec. 6 concludes the paper.
2

## method
This section gives details of the proposed Diff9D. First, we
illustrate the pose diffusion process (Sec. 3.1), and then
describe the condition extraction for pose diffusion (Sec.
3.2). Next, we introduce the proposed transformer-based
denoiser for pose denoising (Sec. 3.3). Finally, we elaborate
the supervision method (Sec. 3.4).
3.1
Pose Diffusion
3.1.1
Forward Pose Diffusion Process
Given a 9-DoF object pose sampled from a real-world pose
distribution x0 ∼q (x), we define a forward diffusion pro-
cess where Gaussian noise is gradually added to the sample
in T time steps (T represents the maximum time step),
producing a sequence of noisy samples x1, · · ·, xT . The time
step is controlled by a variance schedule {βt ∈(0, 1)}T
t=1
and β1 < β2 < ··· < βT . The forward pose diffusion process
from xt−1 to xt is defined as [59]:
q (xt|xt−1) = N

xt;
p
1 −βtxt−1, βtI

,
(1)
where N
 µ, σ2
represents a Gaussian distribution. The
step-by-step diffusion process follows the Markov chain
assumption:
q (x1:T |x0) =
T
Y
t=1
q (xt|xt−1).
(2)
Specifically, xt can be represented as:
xt =
p
βtεt +
p
1 −βtxt−1,
(3)

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE
5
Shape
Encoder
Reverse Diffusion Process
1×512
Q
K V
Q
K V
MLP
ResNet18
Transformer -Based Denoiser
PointNet
n×576
1×128
MLP
condition
1×512
1×384
MLP
1×1536
1×256
Q
K V
Q
K V
Q
K V
Q
K V
…
Q
K V
Q
K V
MLP
Q
K V
MLP
Q
K V
Q
K V
MLP
Condition Extraction
Pose Diffusion
(Gaussian)
ForwardDiffusion Process
Q
K V Transformer Block
MLP Multi-Layer Perceptron
Feature Concatenate
… Multiple Iterations
𝑃𝑜𝑠𝑒𝑇
𝑃𝑜𝑠𝑒𝑇
𝑃𝑜𝑠𝑒0
𝑃𝑜𝑠𝑒𝑇−1
𝑃𝑜𝑠𝑒0
Shape
Estimator
𝑐𝑡𝑖𝑚𝑒𝑠𝑡𝑒𝑝
𝑐𝑟𝑔𝑏
𝑐𝑝𝑜𝑖𝑛𝑡
𝑐𝑠ℎ𝑎𝑝𝑒
𝐹𝑝𝑜𝑠𝑒
𝑐
𝜀𝜃𝑥𝑇, 𝑇, 𝑐
𝐷𝑖
RGB
Points
Time
Step
0~𝑇
Points Global Feature
Points Local Feature
𝐷𝑖
Fig. 3. Workflow of the proposed Diff9D, which includes three main parts (pose diffusion, condition extraction for pose diffusion, and transformer-
based denoiser for pose denoising). The input of Diff9D is RGB image, point cloud, and time step T and its corresponding noise pose PoseT . Note
that the image is first instance segmented by Mask R-CNN [60] before condition extraction. The condition extraction extracts the input condition
c. The pose diffusion consists of forward (noising) and reverse (denoising) diffusion processes. Forward diffusion continuously adds noise to the
ground-truth object pose Pose0. Reverse diffusion first concatenates the noise pose features Fpose and c to form the input Di for the denoiser. The
transformer-based denoiser then takes Di as input and predicts the pose noise εθ (xT , T, c). Finally, εθ (xT , T, c) can be used to denoise PoseT
through the reverse diffusion process based on the Markov chain to obtain PoseT −1. We directly use the translation, size, and rotation matrices to
represent the object pose, as shown in Fig. 1. Detailed architecture of the shape estimator and shape encoder is shown in Fig. 4.
where εt is a randomly sampled standard Gaussian noise at
time step t. Let αt = 1 −βt and ¯αt = Qt
i=1 αi, we can get:
xt = √¯αtx0 +
√
1 −¯αtε.
(4)
Hence, the forward pose diffusion process from x0 to xt can
be represented as:
q (xt|x0) = N
 xt; √¯αtx0, (1 −¯αt) I
 .
(5)
3.1.2
Reverse Pose Diffusion Process
As shown in Fig. 3, the reverse diffusion process aims to
recover the object pose from a standard Gaussian noise
input xT ∼N (0, I). However, obtaining q (xt−1|xt) is not
easy, so we learn a model pθ to approximate this conditional
probability to run the reverse diffusion process as:
pθ (x0:T ) = p (xT )
TQ
t=1
pθ (xt−1|xt),
pθ (xt−1|xt) = N (xt−1; µθ (xt, t, c) , P
θ (xt, t)) ,
(6)
where c denotes the condition (see Sec. 3.2 for more details).
Also let αt = 1 −βt and ¯αt = Qt
i=1 αi and follow DDPM
[59] to use Bayes’ theorem transform Eq. (6), then the
variance and mean of pθ (xt−1|xt) can be parameterized as
follows:
X
θ (xt, t) = 1/
αt
βt
+
1
1 −¯αt−1

· I = 1 −¯αt−1
1 −¯αt
· βt · I,
(7)
µθ (xt, t, c) =
√αt (1 −¯αt−1)
1 −¯αt
xt +
√¯αt−1βt
1 −¯αt
x0.
(8)
From the previous forward diffusion process Eq. (4), we can
obtain:
µθ (xt, t, c) =
1
√αt

xt −1 −αt
√1 −¯αt
εθ (xt, t, c)

.
(9)
where εθ denotes the predicted pose noise during the re-
verse diffusion process.
n×1088
Conv 512
Conv 256
Conv 3
Conv 512
Conv 256
Conv 3
Shape
NOCS
Shape
Conv 3
Conv 64
Conv 192
Conv 3
Conv 64
Conv 192
Max-pooling
Max-pooling
1×384
n×3
𝑅𝑠
𝑁𝑠
n×3
Fig. 4. Detailed architecture of the shape estimator and shape encoder.
We use two parallel branches to estimate and encode shape and NOCS
shape.
To improve the speed of the reverse diffusion process, we
utilize the DDIM [43] scheduler. Some visualizations of the
reverse diffusion process are shown in Fig. 2. Specifically, we
take a sample every T/S time steps to reduce the number
of sampling time steps from T to S. The new sampling
schedule is {τ1, · · ·, τS}. Now, the reverse diffusion process
can be expressed as:
pθ
 xτi−1|xτi
 =
N

xτi−1; √¯ατi−1x0 +
q
1 −¯ατi−1 −σ2τi
xτi−√
¯ατix0
√
1−¯ατi
, σ2
τiI
 ,
(10)
where σ2
t can be obtained from Eq. (6) and Eq. (7) as:
σ2
τi = 1 −¯ατi−1
1 −¯ατi
· βτi.
(11)
Overall, the reverse diffusion process predicts the pose
noise εθ (xt, t, c) by learning a model, and then utilizes the
DDIM [43] scheduler for denoising.
3.2
Condition Extraction for Pose Diffusion
However, it is difficult to directly perform the reverse dif-
fusion process using only xT ∼N (0, I) as the input of the

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE
6
Fig. 5. Some visualizations of the estimated shape and NOCS shape. Top, middle, and bottom rows denote the observed RGB images and their
corresponding estimated shape and NOCS shape, respectively. It can be seen that these two processes can introduce potential 3D geometric
information, making the pose diffusion process geometrically guided.
diffusion model. Therefore, we propose to incorporate con-
ditional information from the input to guide the diffusion
model to achieve more accurate pose predictions. Given that
obtaining RGB images and point clouds is straightforward
with an RGB-D camera, we utilize them along with the
time step as inputs for condition extraction. The detailed
architecture is shown in Fig. 3. Specifically, the diffusion
process is associated with the time step. To extract the
time step condition ctimestep, we follow DiffPose [61] to
employ a Multi-Layer Perceptron (MLP). Additionally, for
the observed RGB image and point cloud, we utilize the
lightweight ResNet [49] and PointNet [50] to extract the RGB
global condition crgb and point cloud condition (including
global features cpoint and local features) of the observed
object, respectively.
Inspired by the positive impact of the 3D shape recon-
struction on object pose estimation [20], [22], [24], [40], we
point-wise concatenate the point cloud condition with the
RGB global condition. The concatenated features are subse-
quently fed into a shape estimator-encoder network, thereby
incorporating supervision for the 3D shape of intra-class
unknown objects. The detailed architecture of the shape
estimator-encoder network is shown in Fig. 4. Specifically,
we use two parallel branches to perform decoupled esti-
mation and encoding for the shape and NOCS shape of
intra-class unknown objects. Some visualizations are shown
in Fig. 5. Subsequently, we perform max pooling on the
encoded two-branch features and concatenate them as the
shape condition cshape. Finally, we concatenate the obtained
time step, RGB global, point cloud global, and shape con-
ditions to obtain the conditional input of pose diffusion as
follows:
c = cat (ctimestep, crgb, cpoint, cshape) .
(12)
To supervise the condition extraction process, we follow
[20], [22], [24], [40] and use the Chamfer distance between
the ground-truth 3D model Mgt and the estimated shape
Rs, and the Smooth-L1 distance between the ground-truth
NOCS shape MNs and the estimated NOCS shape Ns as the
loss functions. MNs can be easily obtained from Mgt [19].
Chamfer distance Lcd can be expressed as:
Lcd (Rs, Mgt) =
1
2n×
 
P
a∈Rs
min
b∈Mgt ∥a −b ∥2
2 +
P
b∈Mgt
min
a∈Rs ∥a −b ∥2
2
!
,
(13)
where n denotes the number of points. The Smooth-L1
distance LS−L1 can be expressed as:
LS−L1 (Ns, MNs) = 1
n
nP
i=1
3P
k=1

5x2,
if
x ≤0.1,
x −0.05,
otherwise,
and
x =
N ik
s −M ik
Ns
 ,
(14)
where k denotes the dimension of the coordinates.
3.3
Transformer-Based Denoiser for Pose Denoising
Since εt is available at training time, we need to train a
network to predict the pose noise conditioned on c, i.e.,
predict εθ (xt, t, c). Then, the denoising loss term can be pa-
rameterized to minimize the difference from εt to εθ (xt, t, c)
as follows:
Ldiff = Et∼[1,T ],x0,εt
h
∥εt −εθ (xt, t, c)∥2i
= Et∼[1,T ],x0,εt
hεt −εθ
 √¯αtx0 + √1 −¯αtε, t, c
2i
.
(15)
Due to the sparsity of pose data, utilizing the cross-
attention mechanism, commonly employed in other Dif-
fusion models, is not feasible. Consequently, we propose
a transformer-based denoiser as the denoising model con-
sistin

## experiments
We first describe the benchmark datasets and evaluation
metrics (Sec. 5.1) and the implementation details (Sec. 5.2).
We train the proposed Diff9D using only the synthetic
dataset and test it on two challenging real-world datasets
to demonstrate its domain generalization ability (Sec. 5.3).
Next, we further test Diff9D in real-world robotic grasping
scenarios and deploy it on a robot to perform grasping task
(Sec. 5.4). Finally, we conduct some ablation studies for the
condition extraction module, transformer-based denoiser,
and the number of reverse diffusion time steps to explore
their impact on the performance of Diff9D (Sec. 5.5).
5.1
Benchmark Datasets and Evaluation Metrics
5.1.1
Benchmark Datasets
For synthetic datasets, we choose the large CAMERA25
dataset [19], which is currently the most widely used
synthetic dataset for category-level object pose estimation.
For real-world datasets, the challenging and widely used
REAL275 [19] and Wild6D [39] datasets are chosen for
testing.
CAMERA25 Dataset [19] contains 275K synthetic RGB-D
images for training, which includes 1085 instances from 6
categories of objects: bowl, bottle, can, camera, mug, and
laptop. Note that all the 3D object models in the CAMERA25
are selected from the synthetic ShapeNet [62] dataset. All
RGB-D images contain multiple instances and have segmen-
tation masks and 9-DoF pose labels.
REAL275 Dataset [19] is currently the most widely used real-
world dataset for category-level object pose estimation. It
contains 8K real-world RGB-D images from 18 videos. We
exclusively utilize the test set of this dataset, consisting of
2754 images from 6 videos, to evaluate the performance of
our proposed method. The test set includes 18 instances
from 6 categories of objects, and the object categories are
the same as in CAMERA25.
Wild6D Dataset [39] is a large dataset collected in the real
world for evaluating self-supervised category-level object
pose estimation methods. It provides annotations for only
486 test videos with different backgrounds, containing 162
objects from five categories (i.e., except “can” in CAMERA25
and REAL275). This paper only uses the test videos of
Wild6D for experiments to enrich the real-world evaluation.
5.1.2
Evaluation Metrics
For a fair comparison with previous methods, we select
the widely used 3D Intersection-over-Union (IoU3D) and
n◦mcm metrics for evaluation. IoU3D denotes the percent-
age of intersection and union of the ground-truth and the
predicted 3D bounding box, which can be expressed as:
IoU3D = PB ∩GB
PB ∪GB
,
(21)
where GB and PB denote the ground-truth and the pre-
dicted 3D bounding boxes, respectively. ∩and ∪denote the
intersection and union, respectively. The predicted object
pose is considered correct when the value of IoU3D is
greater than a predefined threshold.
n◦mcm directly represents the predicted rotation and
translation errors. The predicted object pose is considered
correct when the rotation and translation errors are less than
both n◦and mcm, respectively. We follow previous methods
[20], [22], [24], [38], [40], [41] to choose 50% and 75% as the
thresholds of IoU3D (termed as 3D50 and 3D75 [22]) and
select 5◦2cm, 5◦5cm, 10◦2cm, and 10◦5cm for evaluation.
5.2
Implementation Details
Following [20], [21], [22], [24], we set the number of points
n in Eq. (13) and the dimension of the coordinates k in Eq.
(14) to 1024 and 3, respectively. The diffusion time step T
in Eq. (2) and Eq. (6) is set to 1000. We take samples every
333 time steps (see Tab. 8 for reasoning) by utilizing the
DDIM [43] scheduler in the reverse diffusion process. The
number of heads m in Eq. (17) is set to 16 experimentally,
so the feature dimension of each head d in Eq. (18) is set
to 112 (calculated by 1792/16) [64], [65]. The initial RGB-
D image resolution is 640 × 480. We first utilize Mask R-
CNN [60] to perform instance segmentation for the initial
image, then the image is scaled to 192 × 192 to reduce

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE
9
Fig. 9. Qualitative comparison results on the real-world REAL275 dataset. Both SGPA [22] and Diff9D are trained using only synthetic data. For a
fair comparison, all results are based on the same segmentation, i.e., by Mask R-CNN [60]. The top, middle, and bottom rows denote the results of
SGPA [22], Diff9D, and ground truth, respectively. Arrows point to areas of focus. We can see that Diff9D performs better than SGPA [22].
further computation. The model weights are initialized via
the default initialization method of PyTorch. The learning
rate is dynamically adjusted between 1×10−4 and 1×10−6
through the CyclicLR function [66], [67], and the step size of
a cycle is set to 20K. The batch size of each step is set to 48.
To avoid damaging the gripper, we rotate the grasping angle
up 30◦and move the object center point up 2 centimeters as
the grasping position. Experiments are conducted using an
Intel Xeon Gold 6138 CPU and an NVIDIA RTX 3090 GPU.
5.3
Evaluation on Real-World Datasets
5.3.1
REAL275 Dataset
We only use the synthetic CAMERA25 dataset to train the
proposed Diff9D and compare it with one baseline method
[20] and nine state-of-the-art (SOTA) methods [21], [22], [24],
[35], [36], [38], [39], [40], [41] on the test set of the real-
world REAL275 dataset. Quantitative comparison results
are shown in Tab. 1. When using Mask R-CNN pretrained
on the ImageNet dataset for segmentation, Diff9D achieves
43.9% and 54.8% mean average precision (mAP) on 5◦5cm
and 10◦5cm, outperforming the baseline method SPD [20]
by 31.9% and 21.3%, the SOTA methods SGPA [22] by
16.2% and 18.3%, STG6D [24] by 16.0% and 15.8%, respec-
tively. In addition, Diff9D achieves 35.3% and 70.0% mAP
on 5◦2cm and 10◦2cm, outperforming the SOTA method
SAR-Net [35] by 3.7% and 1.7%, respectively. Note that
these four comparison methods all rely on shape priors,
while Diff9D outperforms them without using shape priors.
Moreover, SPD [20], CR-Net [21], and SGPA [22] also use the
real-world REAL275 dataset for training. Diff9D is 14.2%,
8.5%, and 5.1% better than SPD [20], CR-Net [21], and
SGPA [22] respectively on the 75% IoU3D metric using
only synthetic dataset for training. Furthermore, we also
compare with some SOTA self-supervised methods that use
labeled synthetic data and unlabeled real-world data for
training. Diff9D achieves 76.5% and 35.3% mAP on 50%
IoU3D and 5◦2cm, outperforming RePoNet [39] by 0.5% and
6.2%, SSC6D+ICP [36] by 3.8% and 6.7%, respectively. Some
qualitative results are shown in Fig. 9. Additionally, we
compare Diff9D with DiffusionNOCS [47], another domain-
generalized category-level pose estimation method based
on diffusion model. Both methods use Mask R-CNN pre-
trained on the ImageNet dataset for segmentation for a
fair comparison. We evaluate using the same metrics as
those in DiffusionNOCS [47]. The quantitative results in
Tab. 2 demonstrate that Diff9D exhibits stronger domain
generalization ability compared to DiffusionNOCS [47].
To ensure that no real-world data is involved in any stage
of the training, we use the synthetic CAMERA25 to retrain
Mask R-CNN for segmentation, as shown in Tab. 1. When
using the retrained Mask R-CNN, Diff9D achieves 69.2%
and 45.2% mAP on 50% IoU3D and 5◦5cm, outperforming
the SOTA self-supervised methods DPDN [40] by 2.0%
and 7.9%, respectively. In addition, Diff9D achieves 57.7%
and 72.2% mAP on 10◦2cm and 10◦5cm, outperforming
the SOTA domain adaptation method UDA-COPE [38] by
0.7% and 6.1%, respectively. Note that DPDN [40] also uses
shape priors for learning, and UDA-COPE [38] requires
real-world mask labels for learning, yet Diff9D outperforms
these methods without using shape priors and any real-
world data. Finally, Diff9D achieves 44.1% and 36.5% mAP
on the most stringent 75% IoU3D and 5◦2cm metrics, out-
performing the SOTA test-time adaptation method TTA-
COPE [41] by 4.4% and 6.3%, respectively. Moreover, we
also retrain VI-Net [33] and SecondPose [63] using only
the synthetic CAMERA25 dataset, observing a significant
accuracy drop in all metrics. We analyze that this is because
Diff9D samples a substantial amount of object pose data
along the Markov chain, leading to a more uniform pose
data distribution, thus effectively reducing the domain gap
between synthetic and real-world scenes. Furthermore, to
assess the upper-bound performance of Diff9D, we train
it using a 3:1 mix of CAMERA25 and REAL275 datasets,
following the setup in VI-Net [33]. The experimental results

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE
10
TABLE 1
Comparison on the REAL275 dataset in metrics of IoU3D (%) and n◦mcm (%). “✓” and “-” indicate with and without. “Syn”, “Real w Label”, and
“Real w/o Label” indicate synthetic dataset, real-world dataset with label, and real-world dataset without label, respectively. Note that we use the
IoU3D metrics of SSC6D [36] and CATRE [26], which correct the small error of NOCS [19] for size evaluation. The

## related_work
This section first reviews the object pose and size estimation
methods, dividing them into instance-level and category-
level methods, and then reviews recent diffusion model-
based methods and explains how our proposed method
differs from them. Finally, we review the object pose
estimation-based robotic grasping methods.
2.1
Instance-Level Methods
Instance-level methods are trained on known objects
[44]
and
can
be
mainly
divided
into
three
cate-
gories: correspondence-based, template-based, and direct
regression-based. Correspondence-based methods can be
further divided into 2D-3D correspondence and 3D-3D cor-
respondence. 2D-3D correspondence methods [6], [7] first
define the keypoints between RGB image and object CAD
model. This is followed by training a model to predict
the 2D keypoints and using the Perspective-n-Points (PnP)
algorithm to solve the object pose. 3D-3D correspondence
methods [8], [9] define the keypoints on the object CAD
model directly and use the observed point cloud to predict
the predefined 3D keypoints. Next, they apply the least
squares algorithm to solve the object pose. However, most
correspondence-based methods rely heavily on rich texture
information and may not work well when applied to tex-
tureless objects.
There are some point cloud-based template methods,
which are based on point cloud registration [10], [11].
Specifically, the template is the object CAD model with
the canonical pose, and the purpose of these methods is to
find the optimal relative pose that aligns the observed point
cloud with the template. Besides these methods, RGB-based
template methods [12], [13] also exist, which require collect-
ing and annotating object images from various perspectives
during the training phase to create templates. After that,
these methods train a template matching model to find the
closest template to the observed image and use the template
pose as the actual pose of the object. Overall, template-based
methods can be effectively applied to textureless objects,
however, the template-matching process is generally time-
consuming.
With the rapid advancement of deep learning technol-
ogy, direct regression-based methods [14], [15], [16], [17],

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE
3
[18] have recently gained popularity. These methods use
the ground-truth object poses for supervision and train
models to regress the object pose end-to-end. Specifically,
DenseFusion [14] fuses the RGB and depth features and
proposes a pixel-level dense fusion network for pose re-
gression. FFB6D [15] further designs a bidirectional feature
fusion network to fully fuse the RGB and depth features.
GDR-Net [16] proposes a geometry-guided network for
end-to-end monocular object pose regression. HFF6D [17]
designs a hierarchical feature fusion framework for object
pose tracking in dynamic scenes. Although instance-level
methods have achieved high accuracy, they are restricted
to fixed instances, meaning that they only work for specific
objects on which they are trained.
2.2
Category-Level Methods
Research in the domain of category-level methods has re-
ceived substantial attention given their potential for gener-
alization to unknown objects within the given object cat-
egories. NOCS [19] introduces a normalized object coor-
dinate space, providing a standardized representation for
a category of objects, and recovers object pose using the
Umeyama algorithm. SPD [20] leverages shape prior de-
formation to solve the problem of diverse shape variations
between intra-class objects. Due to the superior performance
achieved by SPD, other prior-based methods are also sub-
sequently proposed. CR-Net [21] designs a recurrent frame-
work for iterative residual refinement to improve the shape
prior-based deformation and coarse to fine object pose esti-
mation. SGPA [22] utilizes the structure similarity between
the shape prior and the observed intra-class unknown object
to dynamically adapt the shape prior. 6D-ViT [23] intro-
duces Pixelformer and Pointformer networks, based on the
Transformer architecture, to extract more refined features of
the observed objects. STG6D [24] goes a step further and
fuses the difference features between the shape prior and
the observed objects, enabling more refined deformation.
RBP-Pose [25] designs a geometry-guided residual object
bounding box projection network to solve the insufficient
pose-sensitive feature extraction. CATRE [26] proposes a
pose refinement method based on the alignment of the
observed point cloud and the shape prior, which can be
used to further refine the object pose estimated by the above
methods. GeoReF [27] builds upon CATRE [26] to tackle
the geometric variation issue by incorporating hybrid scope
layers and learnable affine transformations. Although prior-
based methods significantly improve accuracy, constructing
CAD model libraries is cumbersome and time-consuming.
Besides these prior-based methods, DualPoseNet [28]
introduces a dual pose encoder with refined learning of
pose consistency and regresses object pose via two par-
allel pose decoders. FS-Net [29] proposes a shape-based
3D graph convolution network and performs decoupled
regression for translation, rotation, and size. GPV-Pose [30]
harnesses geometry-guided point-wise voting to enhance
the learning of category-level pose-sensitive features. HS-
Pose [31] further proposes a hybrid scope feature extrac-
tion network, addressing the limitations associated with
the size and translation invariant properties of 3D graph
convolution. IST-Net [32] explores the necessity of shape
priors for category-level pose estimation and proposes an
implicit space transformation-based prior-free method. VI-
Net [33] addresses the problem of poor rotation estimation
by decoupling rotation into viewpoint and in-plane rota-
tions. While these methods do not depend on shape priors,
they still require large amounts of real-world annotated data
for training, which hinders their practical applicability.
To address the problem of insufficient real-world train-
ing data, CPPF [34] performs pose estimation in the wild
by introducing a category-level point pair feature voting
method. SAR-Net [35] proposes to explore the shape align-
ment of each intra-class unknown object against its cor-
responding shape prior without using real-world training
data. SSC6D [36] proposes a self-supervised method us-
ing DeepSDF [37] for deep implicit shape representation.
UDA-COPE [38] utilizes a teacher-student self-supervised
learning framework to achieve domain adaptation. RePoNet
[39] proposes a self-supervised method based on pose and
shape differentiable rendering. DPDN [40] designs a par-
allel deep prior deformation-based domain generalization
learning scheme. More recently, TTA-COPE [41] introduces
a test-time adaptation method, which initially trains the
model on labeled synthetic data and subsequently utilizes
the pretrained model for test-time adaptation in real-world
data during inference. Nevertheless, the performance of
these methods is limited by the huge domain gap between
the rendered synthetic domain and the real world.
2.3
Diffusion Model-Based Methods
More recently, diffusion models gained popularity in object
pose estimation. In terms of instance-level methods, Diffu-
sionReg [45] proposes a point cloud registration framework
leveraging the SE(3) diffusion model. This model gradually
perturbs the optimal rigid transformation of a pair of point
clouds by continuously injecting perturbations through the
SE(3) forward diffusion process. The SE(3) reverse denoising
process is then used to progressively denoise, approaching
the optimal transformation for precise pose estimation. 6D-
Diff [46] develops a diffusion-based framework that formu-
lates 2D keypoint detection as a denoising process, enabling
more accurate 2D-3D correspondences. As for category-level
methods, GenPose [52] introduces a score-based diffusion
model to tackle the multi-hypothesis issue in symmetric
objects and partial point clouds. Their approach first uses
the score-based diffusion model to generate multiple pose
candidates and then employs an energy-based diffusion
model to eliminate abnormal poses. DiffusionNOCS [47]
first diffuses the NOCS map of the object using multi-modal
input as a condition, and then uses an offline registration
algorithm to align and solve the object pose.
In general, DiffusionReg [45] and 6D-Diff [46] are
instance-level methods. GenPose [52] and DiffusionNOCS
[47] mainly focus on 6-DoF pose (excluding 3D size). More-
over, GenPose does not focus on solving the problem of
domain generalization, and the diffusion target of Diffusion-
NOCS is the NOCS map. Different from the above methods,
we aim to develop a category-level 9-DoF object pose esti-
mation method suitable for real-world robotic applications
using only rendered synthetic data for training. This ap-
proach faces two main challenges: 1) The significant domain
gap betw