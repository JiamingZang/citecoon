# 3D Human Pose Estimation Using Convolutional Neural Networks with 2D Pose Information

> 2016 · id: W2502928967 · arXiv: 1608.03075 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
Both 2D and 3D human pose recovery from images are important tasks since
the retrieved pose information can be used to other applications such as action
recognition, crowd behavior analysis, markerless motion capture and so on. How-
ever, human pose estimation is a challenging task due to the dynamic variations
of a human body. Various skin colors and clothes also make the estimation dif-
ﬁcult. Especially, pose estimation from a single image requires a model that is
robust to occlusion and viewpoint variations.
Recently, 2D human pose estimation achieved a great success with convo-
lutional neural networks (CNNs) [1,2,3]. Strong representation power and the
ability to disentangle underlying factors of variation are characteristics of CNNs
that enable learning discriminative features automatically [4] and show supe-
rior performance to the methods based on hand-crafted features. On the other
hands, 3D human pose estimation using CNNs has not been studied thoroughly
compared to the 2D cases. Estimating a 3D human pose from a single image is
more challenging than 2D cases due to the lack of depth information. However,
CNN can be a powerful framework for learning discriminative image features and

2
S. Park et al.
estimating 3D poses from them. In the case where the target object is ﬁxed such
as human body, it is able to learn useful features directly from images without
keypoint matching step in the typical 3D reconstruction tasks.
Though recent algorithms that are based on CNNs for 3D human pose esti-
mation have been proposed [5,6,7], they do not make use of 2D pose information
which can provide additional information for 3D pose estimation. From 2D pose
information, undesirable 3D joint positions which generate unnatural human
pose may be discarded. Therefore, if the information that contains the 2D posi-
tion of each joint in the input image is used, the results of 3D pose estimation
can be improved.
In this paper, we propose a simple yet powerful 3D human pose estimation
framework based on the regression of joint positions using CNNs. We introduce
two strategies to improve the regression results from the baseline CNNs. Firstly,
not only the image features but also 2D joint classiﬁcation results are used as
input features for 3D pose estimation. This scheme successfully incorporates the
correlation between 2D and 3D poses. Secondly, rather than estimating relative
positions with respect to only one root joint, we estimated the relative 3D posi-
tions with respect to multiple joints. This scheme eﬀectively reduces the error of
the joints that are far from the root joint. Experimental results validate the pro-
posed framework signiﬁcantly improves the baseline method and achieves com-
parable performance to the state-of-the-art methods on Human 3.6m dataset [8]
without utilizing the temporal information.
The rest of the paper is organized as follows. Related works are reviewed in
Section 2. The structure of CNNs used in this paper and two key ideas of our
method, 1) the integration of 2D joint classiﬁcation results into 3D pose estima-
tion and 2) multiple 3D pose regression from various root nodes, are explained
in Section 3. Details of implementation and training procedures are explained in
Section 4. Experimental results are illustrated in Section 5, and ﬁnally conclu-
sions are made in Section 6.
2

## experiments
We used Human 3.6m dataset [8] to evaluate our method and compared the
proposed method with the other 3D human pose estimation algorithms. The
dataset provides 3D human pose information acquired by a motion capture sys-
tem with synchronized RGB images. It consists of 15 diﬀerent sequences which

3D Human Pose Estimation Using CNNs with 2D Pose Information
9
Table 1. Quantitative results on Human 3.6m dataset. The best and the second best
methods for each sequence are marked as (1) and (2) respectively.
Directions Discussion
Eating
Greeting
Phoning
Photo
LinKDE [8]
132.71
183.55
132.37
164.39
162.12
205.94
Li and Chan [5]
-
148.79
104.01
127.17
-
189.08
Li et al. [6]
-
136.88
96.94
124.74
-
168.68
Tekin et al. [7]
-
129.06
91.43
121.68
-
162.17
Tekin et al. [20]
102.41
147.72
88.83(2)
125.28
118.02
182.73
Zhou et al. [19]
87.36(1)
109.31(1)
87.05(1) 103.16(1) 116.18(2) 143.32(1)
Our method
100.34(2) 116.19(2)
89.96
116.49(2) 115.34(1) 149.55(2)
Posing
Purchases
Sitting
Sitting
Smoking
Waiting
Down
LinKDE [8]
150.61
171.31
151.57
243.03
162.14
170.69
Li and Chan [5]
-
-
-
-
-
-
Li et al. [6]
-
-
-
-
-
-
Tekin et al. [7]
-
-
-
-
-
-
Tekin et al. [20] 112.38(2)
129.17
138.89
224.90
118.42
138.75
Zhou et al. [19] 106.88(1)
99.78(1)
124.52(1) 199.23(2) 107.42(2) 118.09(1)
Our method
117.57
106.94(2) 137.21(2) 190.82(1) 105.78(1) 125.12(2)
Walk
Walking
Walk
Average
Dog
Together
LinKDE [8]
177.13
96.60
127.88
162.14
Li and Chan [5]
146.59
77.60
-
-
Li et al. [6]
132.17
69.97
-
-
Tekin et al. [7]
130.53
65.75
-
-
Tekin et al. [20] 126.29(2)
55.07(1)
65.76(1)
124.97
Zhou et al. [19] 114.23(1)
79.39
97.70
113.01(1)
Our method
131.90
62.64(2)
96.18(2) 117.34(2)
contain speciﬁc actions such as discussion, eating, walking, and so on. There
are 7 diﬀerent persons who perform all 15 actions. We trained and tested each
action individually. Following the previous works on the dataset [5,19], we used
5 subjects (S1, S5, S6, S7, S8) as a training set, and 2 subjects (S9, S11) as a test
set. The training and the testing procedures are conducted on a single PC with
a Titan X GPU. Training procedure takes 7-10 hours for one action sequence
depending on the number of training images. For the evaluation metric, we used
the mean per joint position error (MPJPE).
First, we compared the performance of our method with the conventional
methods on Human 3.6m dataset. Table 1 shows the MPJPE of our method
and the previous works. The smallest and the second smallest errors for each
sequence are marked. Our method achieves the best performance in 3 sequences
and shows the second best performance in 9 sequences. Note that the methods of
[20] and [19] make use of temporal information from multiple frames. Meanwhile,
our method produce a 3D pose from a single image. Our method is also beneﬁcial
against [20] and [19] in terms of running time and the simplicity of the algorithm

10
S. Park et al.
Table 2. Comparison of our method with the baseline.

## related_work
Human pose estimation has been a fundamental task since early computer vision
literature, and numerous researches have been conducted on both 2D and 3D
human pose estimation. In this section, we will cover both 2D and 3D human
pose estimation methods focusing on the CNN-based methods.
Early works for 2D human pose estimation which are based on deformable
parts model [9], pictorial structure [10,11,12], or poselets [13] train the relation-
ship between body appearance and body joints using hand-crafted features. Re-
cently proposed CNN based methods drastically improve the performance over
the previous hand-crafted feature based methods. DeepPose [1] used CNN-based
structure to regress joint locations with multiple iterations. Firstly, it predicts
an initial pose using holistic view and reﬁne the currently predicted pose using
relevant parts of the image. Xiaochuan et al. [14] integrated both the local part
appearance and the holistic view of an image using dual-source CNN. Convolu-

3D Human Pose Estimation Using CNNs with 2D Pose Information
3
tional pose machine [3] is a systematic approach to improve prediction of each
stage. Each stage operates a CNN which accepts both the original image and
conﬁdence maps from preceding stages as an input. The performance is improved
by combining the joint prediction results from the previous step with features
from CNN. Joao et al. [2] proposed a self-correcting method by a top-down feed-
back. It iteratively learns a human pose using a self-correcting CNN model which
gradually improves the initial result by feeding back error predictions. Xiao et
al. [15] proposed an end-to-end learning system which captures the relationships
among feature maps of joints. Geometrical transform kernels are introduced to
learn features and their relationship jointly.
Similar to the 2D case, early stage of 3D human pose estimation is also
based on the low-level features such as local shape context [16] or segmentation
results [17]. With the extracted features, 3D pose estimation is formulated as a
regression problem using relevance vector machines [16], structured SVMs [17],
or random forest classiﬁers [18]. Recently, CNNs have drew a lot of attentions
also for the 3D human pose estimation tasks. Since search space in 3D is much
larger than 2D image space, 3D human pose estimation is often formulated as
a regression problem rather than a classiﬁcation task. Li and Chan [5] ﬁrstly
used CNNs to learn 3D human pose directly from input images. Relative 3D
position to the parent joint is learned by CNNs via regression. They also used
2D part detectors of each joints in a sliding window fashion. They found that loss
function which combines 2D joint classiﬁcation and 3D joint regression helps to
improve the 3D pose estimation results. Li et al. [6] improved the performance of
3D pose estimation by integrating a structured learning framework into CNNs.
Recently, Tekin et al. [7] proposed a structured prediction framework which
learns 3D pose representations using an auto-encoder. Temporal information
from video sequences also helps to predict more accurate pose estimation result.
Zhou et al. [19] used the result of 2D pose estimation to reconstruct a 3D pose.
They represented a 3D pose as a weighted sum of shape bases similar to typical
non-rigid structure from motion, and they designed an EM-algorithm which
formulates the 3D pose as a latent variable when 2D pose estimation results are
available. The method achieved the state-of-the-art performance for 3D human
pose estimation when combined with 2D pose predictions learned from CNN.
Tekin et al. [20] used multiple consecutive frames to build a spatio-temporal
features, and the features are fed to a deep neural network regressor to estimate
the 3D pose.
The method proposed in this paper aims to provide an end-to-end learning
framework to estimate 3D structure of a human body from a single image. Similar
to [5], 3D and 2D pose information are jointly learned in a single CNN. Unlike
the previous works, we directly propagate the 2D classiﬁcation results to the
3D pose regressors inside the CNNs. Using additional information such as 2D
classiﬁcation results and the relative distance from multiple joints, we improve
the performance of 3D human pose estimation over the baseline method.

4
S. Park et al.
Input Image (225 × 225)
7 × 7 / 2
Conv 1
64 maps
3 × 3 / 2
Pool 1
5 × 5 / 2
Conv 2
128 maps
3 × 3 / 2
Pool 2
3 × 3 / 1
Conv 3
192 maps
3 × 3 / 1
Conv 4
192 maps
3 × 3 / 1
Conv 5
192 maps
3 × 3 / 2
Pool 5
2048
fc1 3D
2048
fc1 2D
2048
fc2 3D
2048
fc2 2D
3D Euclidean
Loss
3 × (Nj −1)
2D Cross
Entropy Loss
Ng × Ng × Nj
Fig. 1. The baseline structure of CNN used in this paper. Convolutional and pooling
layers are shared for both 2D and 3D losses, and the losses are attached to diﬀerent
fully connected layers.
3
3D-2D Joint Estimation of Human Body Using CNN
The task of 3D human pose estimation is deﬁned as predicting the 3D joint
positions of a human body. Speciﬁcally, we estimate the relative 3D position
of each joint with respect to the root joint. The number of joints Nj is set to
17 in this paper according to the dataset used in the experiment. The key idea
of our method is to train CNN which performs 3D pose estimation using both
image features from the input image and 2D pose information retrieved from
the same CNN. In other words, the proposed CNN is trained for both 2D joint
classiﬁcation and 3D joint regression tasks simultaneously. Details of each part
is explained in the following subsections.
3.1
Structure of the Baseline CNN
The CNN used in this experiment consists of ﬁve convolutional layers, three
pooling layers, two parallel sets of two fully connected layers, and loss layers for
2D and 3D pose estimation tasks. The CNN accepts a 225 × 225 sized image as
an input. The sizes and the numbers of ﬁlters as well as the strides are speciﬁed
in Figure 1. The ﬁlter sizes of convolutional and pooling layers are the same as
those of ZFnet [21], but we reduced the number of feature maps to make the
network smaller.
Joint optimization using both 3D and 2D information helps CNN to learn
more meaningful features than the optimization using 3D regression alone. Li et
al. [5] trained a CNN both for 2D joint detection task and for 3D pose regression
task. Since both tasks share the same convolutional layers, features that are
useful for estimating both 2D and 3D positions of joints in an image are learned
in convolutional layers. Following the idea, we also used both 2D and 3D loss
functions in the CNN. Convolutional layers are shared, and the feature maps
after the last pooling layer are connected to two diﬀerent fully connected layers,
each of which is connected to 2D loss function and 3D loss function respectively
(See Figure 1).
We formulated 2D pose estimation as a classiﬁcation problem. For the 2D
classiﬁcation task, we divided an input image into Ng × Ng grids and treat each
grid as a separate class, which results in N 2
g classes per joint. The ground truth

3D Human Pose Estimation Using CNNs with 2D Pose Information
5
label is assigned in accordance with the ground truth position of each joint.
When the ground truth joint position is near the boundary of a grid, zero-one
labeling that is typically used for multi-class classiﬁcation may give unprecise
information. Therefore, we used a soft label which assigns non-zero probability
to the four nearest neighbor grids from the ground truth joint position. The
target probability for the ith grid gi of the jth joint is inversely proportional to
the distance from the ground truth position, i.e.,
ˆpj(gi) =
d−1(ˆyj, ci)I(gi)
PN 2
g
k=1 d−1(ˆyj, ck)I(gk)
,
(1)
where d−1(x, y) is the inverse of the Euclidean distance between the point x and
y in the 2D pixel space, ˆyj is the ground truth position of the jth joint in the
image, and ci is the center of the grid gi. I(gi) is an indicator function that is
equal to 1 if the grid gi is one of the four nearest neighbors, i.e.,
I(gi) =
(
1
if d(ˆyj, ci) < wg
0
otherwise,
(2)
where wg is the width of a grid. Hence, higher probability is assigned to the
grid closer to the ground truth joint position, and ˆpj(gi) is normalized so that
the sum of the class probabilities is equal to 1. Finally, the objective of the 2D
classiﬁcation task for the jth joint is to minimize the following cross entropy loss
function.
L2D(j) = −
N 2
g
X
i=1
ˆpj(gi) log pj(gi),
(3)
where pj(gi) is the probability that comes from the softmax output of the CNN.
On the other hand, estimating 3D position of joints is formulated as a regres-
sion task. Since the search space is much larger than the 2D case, it is undesirable
to solve 3D pose estimation as a classiﬁcation task. The 3D loss function is de-
signed as a square of the Euclidean distance between the prediction and the
ground truth. We estimate 3D position of each joint relative to the root node.
Hence, the loss function for the jth joint when the root node is t

## conclusion
Eating
Greeting
Phoning
Photo
Walking
Baseline CNN
125.45
95.21
120.69
119.66
153.76
72.55
Multi-reg
122.71
94.67
119.70
119.25
153.54
71.19
2D-cls
118.19
91.39
118.19
115.84
149.97
64.27
Multi-reg+2D-cls
116.19
89.96
116.49
115.34
149.55
62.64
epoch
0
7
14
21
28
loss
0.01
0.04
0.07
0.1
With 2D class info
Without 2D class info
(a)
epoch
4
8
12
16
20
24
28
loss
0.01
0.04
0.07
0.1
With 2D class info
Without 2D class info
(b)
Fig. 4. The 3D losses of Walking sequence with and without 2D classiﬁcation result
integration. (a) Losses for training data. (b) Losses for test data.
since the estimation is done by a forward pass of the CNN and simple averaging.
Moreover, from Table 1, it is justiﬁed that our method outperforms the CNN
based methods that predict 3D pose from a single image [5,6,7].
Next, we measured the eﬀect of our contribution, 1) the integration of 2D
classiﬁcation results and 2) regression from multiple root nodes, by comparing
their performance with the baseline CNN. Note that the 2D classiﬁcation loss is
also used in the baseline CNN. The diﬀerence of the baseline CNN is that 2D
classiﬁcation results are not propagated to the 3D loss part, i.e., probs 2D, fc
probs 2D and fc 2D-3D layers in Figure 2 are deleted in the baseline CNN. The
results are shown in Table 2. Multiple regression from diﬀerent root nodes and
the integration of 2D classiﬁcation results are denoted as Multi-reg and 2D-cls
respectively. Both modiﬁcations improve the result over the baseline CNN in
all tested sequences. 2D classiﬁcation integration showed larger error reduction
rate than the multiple regression strategy, which proves that the 2D classiﬁcation
information is indeed a useful feature for 3D pose estimation. Multiple regression
can be considered as an ensemble of diﬀerent estimation results, which improves
the overall performance. It can be found that the error reduction rate for the case
that both 2D classiﬁcation result integration and multiple regression are applied
is slightly bigger than the sum of the reduction rates when they are individually
applied in most sequences. Since each 3D pose regressor takes advantage of 2D
classiﬁcation feature, there is a synergy eﬀect between the two schemes.
We also analyzed the eﬀect of integrating 2D classiﬁcation result in terms of
3D losses. Training losses are measured every 50 iterations and testing losses are

3D Human Pose Estimation Using CNNs with 2D Pose Information
11
measured every 4 epochs. The results on the Walking sequence are illustrated
in Figure 4. For the training data, loss is slightly smaller when 2D classiﬁcation
information is not used (Figure 4(a)). However, test loss is much lower when 2D
classiﬁcation information is used(Figure 4(b)). This indicates that 2D classiﬁ-
cation information impose generalization power and reduce overﬁtting for CNN
regressor. Since the 2D joint probabilities provide more abstracted and subject-
independent information compared to the features obtained from an image, the
CNN model is able to learn representations that are robust to variability of
subjects in the image.
Finally, we illustrated qualitative results of our method in Figure 5. Input
images, ground truth poses, and the estimation results with and without 2D
classiﬁcation information are visualized. Diﬀerent colors are used to distinguish
the left and right sides of human bodies. It can be found that 2D pose estimation
results help reducing the error of 3D pose estimation. While the CNN which does
not use 2D classiﬁcation information gives poor results, the estimated results
are much more improved when 2D classiﬁcation information is used for 3D pose
estimation.
6