# Self-supervised Multi-view Synchronization Learning for 3D Pose Estimation

> 2021 · id: W3092971698 · arXiv: 2010.06218 · pdf: https://arxiv.org/pdf/2010.06218 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
The ability to accurately reconstruct the 3D human pose from a single real im-
age opens a wide variety of applications including computer graphics animation,
postural analysis and rehabilitation, human-computer interaction, and image
understanding [1]. State-of-the-art methods for monocular 3D human pose esti-
mation employ neural networks and require large training data sets, where each
sample is a pair consisting of an image and its corresponding 3D pose annotation
[2]. The collection of such data sets is expensive, time-consuming, and, because
it requires controlled lab settings, the diversity of motions, viewpoints, subjects,
appearance and illumination, is limited (see Fig. 1). Ideally, to maximize di-
versity, data should be collected in the wild. However, in this case precise 3D
annotation is diﬃcult to obtain and might require costly human intervention.
In this paper, we overcome the above limitations via self-supervised learning
(SSL). SSL is a method to build powerful latent representations by training a
neural network to solve a so-called pretext task in a supervised manner, but with-
out manual annotation [3]. The pretext task is typically an artiﬁcial problem,
where a model is asked to output the transformation that was applied to the
data. One way to exploit models trained with SSL is to transfer them to some
arXiv:2010.06218v1  [cs.CV]  13 Oct 2020

2
S. Jenni and P. Favaro
(a) Desired subject invariance.
(b) Desired pose equivariance.
Fig. 1: Monocular 3D human pose estimation. An ideal regressor would be
able to generalize across subjects regardless of their appearance, as shown in (a),
and be sensitive to small pose variations, such as those shown in (b).
target task on a small labeled data set via ﬁne-tuning [3]. The performance of
the transferred representations depends on how related the pretext task is to the
target task. Thus, to build latent representations relevant to 3D human poses,
we propose a pretext task that implicitly learns 3D structures. To collect data
suitable to this goal, examples from nature point towards multi-view imaging
systems. In fact, the visual system in many animal species hinges on the pres-
ence of two or multiple eyes to achieve a 3D perception of the world [4,5]. 3D
perception is often exempliﬁed by considering two views of the same scene cap-
tured at the same time and by studying the correspondence problem. Thus, we
take inspiration from this setting and pose the task of determining if two images
have been captured at exactly the same time. In general, the main diﬀerence
between two views captured at the same time and when they are not, is that the
former are always related by a rigid transformation and the latter is potentially
not (e.g., in the presence of articulated or deformable motion, or multiple rigid
motions). Therefore, we propose as pretext task the detection of synchronized
views, which translates into a classiﬁcation of rigid versus non-rigid motion.
As shown in Fig. 1, we aim to learn a latent representation that can general-
ize across subjects and that is sensitive to small pose variations. Thus, we train
our model with pairs of images, where the subject identity is irrelevant to the
pretext task and where the diﬀerence between the task categories are small pose
variations. To do so, we use two views of the same subject as the synchronized
(i.e., rigid motion) pair and two views taken at diﬀerent (but not too distant)
time instants as the unsynchronized (i.e., non-rigid motion) pair.1 Since these
1 If the subject does not move between the two chosen time instants, the unsynchro-
nized pair would be also undergoing a rigid motion and thus create ambiguity in the
training. However, these cases can be easily spotted as they simply require detecting

Multi-View Synchronization Learning for 3D Pose Estimation
3
pairs share the same subject (as well as the appearance) in all images, the model
cannot use this as a cue to learn the correct classiﬁcation. Moreover, we believe
that the weight decay term that we add to the loss may also help to reduce the
eﬀect of parameters that are subject-dependent, since there are no other incen-
tives from the loss to preserve them. Because the pair of unsynchronized images
is close in time, the deformation is rather small and forces the model to learn
to discriminate small pose variations. Furthermore, to make the representation
sensitive to left-right symmetries of the human pose, we also introduce in the
pretext task as a second goal the classiﬁcation of two synchronized views into
horizontally ﬂipped or not. This formulation of the SSL task allows to poten-
tially train a neural network on data captured in the wild simply by using a
synchronized multi-camera system. As we show in our experiments, the learned
representation successfully embeds 3D human poses and it further improves if
the background can also be removed from the images. We train and evaluate
our SSL pre-training on the Human3.6M data set [2], and ﬁnd that it yields
state-of-the-art results when compared to other methods under the same train-
ing conditions. We show quantitatively and qualitatively that our trained model
can generalize across subjects and is sensitive to small pose variations. Finally,
we believe that this approach can also be easily incorporated in other methods to
exploit additional available labeling (e.g., 2D poses). Code will be made available
on our project page https://sjenni.github.io/multiview-sync-ssl.
Our contributions are: 1) A novel self-supervised learning task for multi-view
data to recognize when two views are synchronized and/or ﬂipped; 2) Extensive
ablation experiments to demonstrate the importance of avoiding shortcuts via
the static background removal and the eﬀect of diﬀerent feature fusion strategies;
3) State-of-the-art performance on 3D human pose estimation benchmarks.
2

## method
MPJPE
NMPJPE
PMPJPE
All
Chen et al.[52]*
80.2
-
58.2
Kocabas et al.[50]*
51.8
51.6
45.0
Rhodin et al.[51]*
66.8
63.3
51.6
Rhodin et al.[51]*
-
95.4
-
Rhodin (UNet) et al.[57]
-
127.0
-
Rhodin et al.[57]*
-
115.0
-
Mitra et al.[58]*†
94.3
92.6
72.5
Ours
79.5
73.4
59.7
Ours*
72.6
68.5
54.5
Ours* (with background)
64.9
62.3
53.5
S1
Chen et al.[52]*
91.9
-
68.0
Kocabas et al.[50]*
-
67.0
60.2
Rhodin et al.[51]*
-
78.2
-
Rhodin et al.[51]*
-
153.3
128.6
Rhodin (UNet) et al.[57]
149.5
135.9
106.4
Rhodin et al.[57]*
131.7
122.6
98.2
Mitra et al.[58]*†
121.0
111.9
90.8
Ours
104.9
91.7
78.4
Ours*
101.2
89.6
76.9
Ours* (with background)
101.4
93.7
82.4
uses two diﬀerent protocols: 1) All, where transfer learning is performed on all
the training subjects and 2) S1, where only subject S1 is used for transfer learn-
ing. We use two diﬀerent network architectures: 1) A ResNet-18 initialized with
random weights, i.e., no ImageNet pre-training, and 2) A ResNet-50 initialized
with ImageNet pre-trained weights. Since prior work transfers to images with
backgrounds we include results with a ResNet-50 also ﬁne-tuned on images with
backgrounds. To reduce the domain gap between pre-training and transfer and
to eliminate shortcuts, we pre-train on both images, where backgrounds are re-
moved, and images with substituted backgrounds. We keep the ratio without
and with substituted backgrounds to 50:50 during pre-training. During transfer
learning we freeze the ﬁrst 3 residual blocks and ﬁne-tune the remaining lay-
ers for 200K iterations using a mini-batch size of 16. Training of the networks
was again performed using the AdamW optimizer [60] with default parameters,
initial learning rate of 10−4 with cosine annealing, and a weight decay of 10−4.

12
S. Jenni and P. Favaro
Fig. 3: Examples of predictions. We show predictions on unseen test subjects
(S9 and S11) of a model that was pre-trained using our self-supervised learning
task and ﬁne-tuned for 3D pose estimation. We show the test images on the
ﬁrst row, the ground truth poses on the second row, the predictions of a model
ﬁne-tuned on all training subjects on the third row, and the predictions of a
model ﬁne-tuned only on subject S1 on the last row.
The results and the comparisons to prior work are provided in Table 2. We
observe that our ﬁne-tuned network generalizes well and outperforms the rele-
vant prior work [51,57,58] by a considerable margin. Note that our ResNet-18
is the only method besides [57] that does not rely on any prior supervision. In-
terestingly, our model with substituted backgrounds performs better than our
model with removed backgrounds in protocol All and worse in protocol S1. The
diﬀerence is most severe in the scale sensitive metric MPJPE. We hypothesize
that the backgrounds might be useful to learn the correct absolute scale, given
a number of diﬀerent training subjects. We also list methods that make use of
large amounts of additional labelled 2D pose data. Using all the training sub-
jects for transfer learning, we even outperform the weakly supervised method
[52]. We show some qualitative predictions for the two test subjects in Fig. 3.
4.3
Evaluation of the synchronization task
Generalization across subjects. We qualitatively evaluate how well our model
generalizes to new subjects on the synchronization prediction task. The idea is
to look for synchronized frames, where the two frames contain diﬀerent subjects.
A query image x(q) of a previously unseen test subject is chosen. We look for

Multi-View Synchronization Learning for 3D Pose Estimation
13
Fig. 4: Synchronization performance on unseen test subjects. We eval-
uate the pre-trained synchronization network on test subjects S9 and S11. Un-
synchronized pairs are created with a time diﬀerence sampled from the interval
[∆t −7, ∆t + 8]. We also show the accuracy on two training subjects S1 and S5.
Fig. 5: Synchronization retrievals on test subjects. Query images are shown
on the top row and the corresponding predicted synchronized frames are shown
in the row below.
the training image x(a) of each training subject that maximizes the predicted
probability of synchronization, i.e., a = arg maxi F1((x(q), x(i))). Note that our
network was not explicitly trained for this task and that it only received pairs
of frames as input containing the same subject during training.
We show example retrievals in Fig. 6. We can observe that the retrieved
images often show very similar poses to the query image. The retrieved images
also cover diﬀerent viewpoints. The method generalizes surprisingly well across
subjects and manages to associate similar poses across subjects. This indicates
a certain degree of invariance to the person appearance. As discussed in the
Introduction, such an invariance is welcome in 3D human pose estimation and
could explain the good performance in the transfer experiments.
Synchronization performance on test subjects. We evaluate the perfor-
mance of the synchronization network also on the test subjects. To this end, we

14
S. Jenni and P. Favaro
Fig. 6: Generalization across subjects. We investigate if a network trained
on our self-supervised synchronization task generalizes across diﬀerent subjects.
The left-most column shows the query image from test subject S9. The remaining
columns shows images of training subjects S1, S5, S6, S7, and S8, respectively,
for which our model predicts the highest probability of being synchronized.
sample synchronized and unsynchronized pairs in equal proportion with varying
time gaps. We plot the accuracy in Fig. 4. Note that we use all the frames in the
test examples in this case. As expected we see an increase in performance with
a larger temporal gap between the two frames. We also show some qualitative
samples of synchronization retrievals in Fig. 5.
5

## experiments
Dataset. We perform an extensive experimental evaluation on the Human3.6M
data set [2]. The data set consists of 3.6 million 3D human poses with the corre-
sponding images. The data is captured in an indoor setting with 4 synchronized
and ﬁxed cameras placed at each corner of the room. Seven subjects perform 17
diﬀerent actions. As in prior work, we use the ﬁve subjects with the naming con-
vention S1, S5, S6, S7, and S8 for training and test on the subjects S9 and S11.
We ﬁlter the training data set by skipping frames without signiﬁcant movement.
On average, we skip every fourth frame in the training data set. On the test set,
we follow prior work and only evaluate our network on one every 64 frames.
2 We drop the subscripts ν, t indicating viewpoint and time in this notation.

Multi-View Synchronization Learning for 3D Pose Estimation
9
Metrics. To evaluate our method on 3D human pose regression we adopt the
established metrics. We use the Mean Per Joint Prediction Error (MPJPE) and
its variants Normalized MPJPE (NMPJPE) and Procrustes MPJPE (PMPJPE).
In NMPJPE the predicted joints are aligned to the ground truth in a least squares
sense with respect to the scale. PMPJE uses Procrustes alignment to align the
poses both in terms of scale and rotation.
4.1
Ablations
We perform ablation experiments to validate several design choices for our self-
supervised learning task. We illustrate the eﬀect of background removal with
respect to the shortcuts in the case of non-varying backgrounds (as is the case
for Human3.6M). We also demonstrate the eﬀects of the two self-supervision
signals (ﬂipping and synchronization) by themselves and in combination and
explore diﬀerent feature fusion strategies.
The baseline model uses background removal, the combination of both self-
supervision signals and element-wise multiplication for feature fusion. The net-
works are pre-trained for 200K iterations on all training subjects (S1, S5, S6, S7,
and S8) using our SSL task and without using annotations. We use a ResNet-
18 architecture [59] and initialize with random weights (i.e., we do not rely on
ImageNet pre-training). Transfer learning is performed for an additional 200K
iterations using only subject S1 for supervision. We freeze the ﬁrst 3 residual
blocks and only ﬁnetune the remaining layers. Training of the networks was per-
formed using the AdamW optimizer [60] with default parameters and a weight
decay of 10−4. We decayed the learning rate from 10−4 to 10−7 over the course
of training using cosine annealing [61]. The batch size is set to 16.
We perform the following set of ablation experiments and report transfer
learning performance on the test set in Table 1:
(a)-(d) Inﬂuence of background removal: We explore the inﬂuence of back-
ground removal on the performance of randomly initialized networks and
networks initialized with weights from our self-supervised pre-training. We
observe that background removal provides only a relatively small gain in
the case of training from scratch. The improvement in the case of our self-
supervised learning pre-training is much more substantial. This suggests that
the static background introduces shortcuts for the pretext task;
(e)-(f) Combination of self-supervision signals: We compare networks ini-
tialized by pre-training (e) only to detect ﬂipping, (f) only to detect synchro-
nization, and (g) on the combination of both. We observe that the synchro-
nization on its own leads to better features compared to ﬂipping alone. This
correlates with the pretext task performance, where we observe that the ac-
curacy on the ﬂipping task is higher than for synchronization. Interestingly,
the combination of both signals gives a substantial boost. This suggests that
both signals learn complementary features;
(h)-(k) Fusion of features: Diﬀerent designs for the fusion of features in the
Siamese architecture are compared. We consider (h) concatenation, (i) addi-
tion, (j) subtraction, and (k) element-wise multiplication. We observe that

10
S. Jenni and P. Favaro
Table 1: Ablation experiments. We investigate the inﬂuence of scene back-
ground (a)-(d), the inﬂuence of the diﬀerent self-supervision signals (e)-(g), the
use of diﬀerent fusion strategies (h)-(k), and the importance of synchronized
multiview data (l) and (m). The experiments were performed on Human3.6M
using only subject S1 for transfer learning.
Ablation
MPJPE
NMPJPE
PMPJPE
(a) Random with background
167.8
147.1
124.5
(b) Random without background
165.5
145.4
114.5
(c) SSL with background
138.4
128.1
100.8
(d) SSL without background
104.9
91.7
78.4
(e) Only ﬂip
129.9
117.6
101.1
(f) Only sync
126.7
115.2
91.8
(g) Both sync & ﬂip
104.9
91.7
78.4
(h) Fusion concat
180.0
162.5
130.5
(i) Fusion add
169.4
158.4
122.8
(j) Fusion diﬀ
108.2
93.9
80.4
(k) Fusion mult
104.9
91.7
78.4
(l) Single-view SSL
158.3
142.0
106.9
(m) Multi-view SSL
104.9
91.7
78.4
multiplication performs the best. Multiplication allows a simple encoding of
similarity or dissimilarity in feature space solely via the sign of each entry.
The fused feature is zero (due to the ReLU) if the two input features do not
agree on the sign;
(l)-(m) Necessity of multi-view data: We want to test how important syn-
chronized multi-view data is for our approach. We compare to a variation
of our task, where we do not make use of multi-view data. Instead, we cre-
ate artiﬁcial synchronized views via data augmentation. The images in the
input pairs are therefore from the same camera view, but with diﬀerent aug-
mentations (i.e., random cropping, scaling and rotations). This corresponds
to using a 2D equivalent of our self-supervised learning task. We observe
drastically decreased feature performance in this case.
4.2
Comparison to prior work
We compare our method to prior work by Rhodin et al. [51,57] and Mitra et
al. [58] on Human3.6M. To the best of our knowledge, these are the only methods
that use the same training settings as in our approach. Other methods, which
we also report, train their networks with additional weak supervision in the form
of 2D pose annotation (see Table 2).
We pretrain the networks on our self-supervised learning task for 200K iter-
ations on all training subjects (namely, S1, S5, S6, S7, and S8). Our comparison

Multi-View Synchronization Learning for 3D Pose Estimation
11
Table 2: Comparison to prior work. We compare to other prior work. All
methods are pre-trained on all the training subjects of Human3.6M. Transfer
learning is performed either using all the training subjects or only S1 for su-
pervision. Methods using large amounts of data with 2D pose annotation are
italicized. * indicates the use of an ImageNet pretrained ResNet-50. † indicates
a viewpoint invariant representation of the 3D pose.
Supervision

## related_work
In this section, we brieﬂy review literature in self-supervised learning, human
pose estimation, representation learning and synchronization, that is relevant to
our approach.
Self-supervised learning. Self-supervised learning is a type of unsupervised
representation learning that has demonstrated impressive performance on image
and video benchmarks. These methods exploit pretext tasks that require no hu-
man annotation, i.e., the labels can be generated automatically. Some methods
are based on predicting part of the data [6,7,8], some are based on contrastive
learning or clustering [9,10,11], others are based on recognizing absolute or rel-
ative transformations [12,13,14]. Our task is most closely related to the last
category since we aim to recognize a transformation in time between two views.
Unsupervised learning of 3D. Recent progress in unsupervised learning has
shown promising results for learning implicit and explicit generative 3D models
no motion over time. Besides standing still, the probability that a moving subject
performs a rigid motion is extremely low. In practice, we found experimentally that
a simple temporal sub-sampling was suﬃcient to avoid these scenarios.

4
S. Jenni and P. Favaro
from natural images [15,16,17]. The focus in these methods is on modelling 3D
and not performance on downstream tasks. Our goal is to learn general purpose
3D features that perform well on downstream tasks such as 3D pose estimation.
Synchronization. Learning the alignment of multiple frames taken from dif-
ferent views is an important component of many vision systems. Classical ap-
proaches are based on ﬁtting local descriptors [18,19,20]. More recently, methods
based on metric learning [21] or that exploit audio [22] have been proposed. We
provide a simple learning based approach by posing the synchronization problem
as a binary classiﬁcation task. Our aim is not to achieve synchronization for its
own sake, but to learn a useful image representation as a byproduct.
Monocular 3D pose estimation. State-of-the-art 3D pose estimation meth-
ods make use of large annotated in-the-wild 2D pose data sets [23] and data sets
with ground truth 3D pose obtained in indoor lab environments. We identify
two main categories of methods: 1) Methods that learn the mapping to 3D pose
directly from images [24,25,26,27,28,29,30,31,32] often trained jointly with 2D
poses [33,34,35,36,37], and 2) Methods that learn the mapping of images to 3D
poses from predicted or ground truth 2D poses [38,39,40,41,42,43,44,45,46]. To
deal with the limited amount of 3D annotations, some methods explored the use
of synthetic training data [47,48,49]. In our transfer learning experiments, we
follow the ﬁrst category and predict 3D poses directly from images. However, we
do not use any 2D annotations.
Weakly supervised methods. Much prior work has focused on reducing the
need for 3D annotations. One approach is weak supervision, where only 2D anno-
tation is used. These methods are typically based on minimizing the re-projection
error of a predicted 3D pose [50,51,52,53,54,55,56]. To resolve ambiguities and
constrain the 3D, some methods require multi-view data [50,51,52], while others
rely on unpaired 3D data used via adversarial training [54,55]. [53] solely relies
on 2D annotation and uses an adversarial loss on random projections of the 3D
pose. Our aim is not to rely on a weaker form of supervision (i.e., 2D annota-
tions) and instead leverage multi-view data to learn a representation that can
be transferred on a few annotated examples to the 3D estimation tasks with a
good performance.
Self-supervised methods for 3D human pose estimation. Here we con-
sider methods that do not make use of any additional supervision, e.g., in the
form of 2D pose annotation. These are methods that learn representations on
unlabelled data and can be transferred via ﬁne-tuning using a limited amount
of 3D annotation. Rhodin et al. [57] learn a 3D representation via novel view
synthesis, i.e., by reconstructing one view from another. Their method relies on
synchronized multi-view data, knowledge of camera extrinsics and background
images. Mitra et al. [58] use metric learning on multi-view data. The distance in
feature space for images with the same pose, but diﬀerent viewpoint is minimized
while the distance to hard negatives sampled from the mini-batch is maximized.
By construction, the resulting representation is view invariant (the local camera
coordinate system is lost) and the transfer can only be performed in a canoni-
cal, rotation-invariant coordinate system. We also exploit multi-view data in our

Multi-View Synchronization Learning for 3D Pose Estimation
5
pre-training task. In contrast to [57], our task does not rely on knowledge of
camera parameters and unlike [58] we successfully transfer our features to pose
prediction in the local camera system.
3
Unsupervised learning of 3D pose-discriminative
features
Our goal is to build image features that allow to discriminate diﬀerent 3D human
poses. We achieve this objective by training a network to detect if two views
depict the exact same scene up to a rigid transformation. We consider three
diﬀerent cases: 1) The two views depict the same scene at exactly the same
time (rigid transformation); 2) The two views show the same scene, but at a
diﬀerent time (non-rigid transformation is highly likely); 3) The two views show
the same scene, but one view is horizontally mirrored (a special case of a non-
rigid transformation), which, for simplicity, we refer to as ﬂipped.
To train such a network, we assume we have access to synchronized multi-
view video data. The data set consists of N examples (in the Human3.6M data
set N = S × A, where S is the number of subjects and A the number of actions)
{x(i)
ν,t}i=1,...,N, where ν ∈V(i) = {1, . . . , ν(i)} indicates the diﬀerent views of the
i-th example and t ∈T (i) = {1, . . . , t(i)} indicates the time of the frame. Let F
denote the neural network trained to solve the self-supervised task. We will now
describe the diﬀerent self-supervision signals.
3.1
Classifying synchronized and ﬂipped views
To train our neural network we deﬁne three types image pairs, each correspond-
ing to a diﬀerent 3D deformation:
Synchronized pairs. In this case, the scenes in the two views are related by a
rigid transformation. This is the case when the two images are captured at
the same time instant, i.e., they are synchronized. The input for this cate-
gory is a sample pair xp = (x(i)
ν1,t, x(i)
ν2,t), where i ∈{1, . . . , N}, ν1̸ = ν2 ∈V(i),
and t ∈T (i), i.e., a pair with diﬀerent views taken at the same time.
Unsynchronized pairs. Pairs of images that are captured at diﬀerent times
(i.e., unsynchronized) are likely to undergo a non-rigid deformation (by as-
suming that objects are non static between these two time instants). To cre-
ate such image pairs we sample xn = (x(i)
ν1,t1, x(i)
ν2,t2), where i ∈{1, . . . , N},
ν1̸ = ν2 ∈V(i), and t1, t2 ∈T (i) such that dmin < |t2 −t1| < dmax, where
dmin and dmax deﬁne the range in time for sampling unsynchronized pairs.
In our experiments, we set dmin = 4 and dmax = 128 and sample uniformly
within this range.

6
S. Jenni and P. Favaro
!1
!1
!2
!2
F
synchronized
ﬂipped
not ﬂipped
not synchronized
F
F
F
¯x⌫2,t1
x⌫1,t1
x⌫2,t1
x⌫3,t2
Fig. 2: An overview of the proposed self-supervised task. We train a
network F to learn image representations that transfer well to 3D pose estimation
by learning to recognize if two views of the same scene are synchronized and/or
ﬂipped. Our model uses a Siamese architecture. The inputs are pairs of frames
of the same scene under diﬀerent views. Frames are denoted by xν,t, where ν
indicates the viewpoint and t the time (¯xν,t denotes a ﬂipped frame). Pairs are
classiﬁed into whether the frames are synchronized or not synchronized, and
whether one of the frames was ﬂipped or not.
Flipped pairs. The last class of image pairs consists of images from two views
which are synchronized, but where one of them has been horizontally mir-
rored. Let ¯x denote the image obtained by applying horizontal mirroring
to the sample x. Flipped pairs are deﬁned as xf = (x(i)
ν1,t, ¯x(i)
ν2,t) or xf =
(¯x(i)
ν1,t, x(i)
ν2,t), where i ∈{1, . . . , N}, ν1̸ = ν2 ∈V(i) and t ∈T (i). In the
case of approximately symmetric objects (such as with human bodies) and
in the absence of background cues (since we mirror the entire image) dis-
tinguishing this relative transformation between views requires accurate 3D
pose discriminative features.
Although both unsynchronized and ﬂipped pairs exhibit non-rigid defor-
mations, they have distinct characteristics. In the case of ﬂipped pairs, the 3D
pose in the second view is heavily constrained by the one in the ﬁrst view. In
contrast, given a non negligible temporal gap between the frames, the 3D pose
is much less constrained in the case of unsynchronized views.
We deﬁne our self-supervised learning task as a combination of

## conclusion
We propose a novel self-supervised learning method to tackle monocular 3D
human pose estimation. Our method delivers a high performance without re-
quiring large manually labeled data sets (e.g., with 2D or 3D poses). To avoid
such detailed annotation, we exploit a novel self-supervised task that leads to
a representation that supports 3D pose estimation. Our task is to detect when
two views have been captured at the same time (and thus the scene is related
by a rigid transformation) and when they are horizontally ﬂipped with respect

Multi-View Synchronization Learning for 3D Pose Estimation
15
to one another. We show on the well-known Human3.6M data set that these two
objectives build features that generalize across subjects and are highly sensitive
to small pose variations.
Acknowledgements. This work was supported by grant 169622 of the Swiss
National Science Foundation (SNSF).