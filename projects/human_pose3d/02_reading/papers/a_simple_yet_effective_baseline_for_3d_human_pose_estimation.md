# A Simple Yet Effective Baseline for 3d Human Pose Estimation

> 2017 · id: W2612706635 · arXiv: 1705.03098 · pdf: https://arxiv.org/pdf/1705.03098 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Following the success of deep convolutional networks,
state-of-the-art methods for 3d human pose estimation have
focused on deep end-to-end systems that predict 3d joint
locations given raw image pixels. Despite their excellent
performance, it is often not easy to understand whether
their remaining error stems from a limited 2d pose (visual)
understanding, or from a failure to map 2d poses into 3-
dimensional positions.
With the goal of understanding these sources of error,
we set out to build a system that given 2d joint locations
predicts 3d positions. Much to our surprise, we have found
that, with current technology, “lifting” ground truth 2d joint
locations to 3d space is a task that can be solved with a
remarkably low error rate: a relatively simple deep feed-
forward network outperforms the best reported result by
about 30% on Human3.6M, the largest publicly available
3d pose estimation benchmark. Furthermore, training our
system on the output of an off-the-shelf state-of-the-art 2d
detector (i.e., using images as input) yields state of the art
results – this includes an array of systems that have been
trained end-to-end speciﬁcally for this task. Our results in-
dicate that a large portion of the error of modern deep 3d
pose estimation systems stems from their visual analysis,
and suggests directions to further advance the state of the
art in 3d human pose estimation.

## introduction
The vast majority of existing depictions of humans are
two dimensional, e.g. video footage, images or paintings.
These representations have traditionally played an impor-
tant role in conveying facts, ideas and feelings to other peo-
ple, and this way of transmitting information has only been
possible thanks to the ability of humans to understand com-
plex spatial arrangements in the presence of depth ambi-
guities. For a large number of applications, including vir-
tual and augmented reality, apparel size estimation or even
autonomous driving, giving this spatial reasoning power to
machines is crucial. In this paper, we will focus on a partic-
ular instance of this spatial reasoning problem: 3d human
pose estimation from a single image.
More formally, given an image – a 2-dimensional rep-
resentation – of a human being, 3d pose estimation is the
task of producing a 3-dimensional ﬁgure that matches the
spatial position of the depicted person. In order to go from
an image to a 3d pose, an algorithm has to be invariant to
a number of factors, including background scenes, lighting,
clothing shape and texture, skin color and image imperfec-
tions, among others. Early methods achieved this invariance
through features such as silhouettes [1], shape context [28],
SIFT descriptors [6] or edge direction histograms [40].
While data-hungry deep learning systems currently outper-
form approaches based on human-engineered features on
tasks such as 2d pose estimation (which also require these
invariances), the lack of 3d ground truth posture data for im-
ages in the wild makes the task of inferring 3d poses directly
from colour images challenging.
Recently, some systems have explored the possibility of
directly inferring 3d poses from images with end-to-end
deep architectures [33,45], and other systems argue that 3d
reasoning from colour images can be achieved by training
on synthetic data [38, 48]. In this paper, we explore the
power of decoupling 3d pose estimation into the well stud-
ied problems of 2d pose estimation [30, 50], and 3d pose
estimation from 2d joint detections, focusing on the latter.
Separating pose estimation into these two problems gives
us the possibility of exploiting existing 2d pose estimation
systems, which already provide invariance to the previously
mentioned factors. Moreover, we can train data-hungry al-
gorithms for the 2d-to-3d problem with large amounts of
3d mocap data captured in controlled environments, while
working with low-dimensional representations that scale
well with large amounts of data.
Our main contribution to this problem is the design and
analysis of a neural network that performs slightly better
than state-of-the-art systems (increasing its margin when
1
arXiv:1705.03098v2  [cs.CV]  4 Aug 2017

the detections are ﬁne-tuned, or ground truth) and is fast (a
forward pass takes around 3ms on a batch of size 64, allow-
ing us to process as many as 300 fps in batch mode), while
being easy to understand and reproduce. The main reason
for this leap in accuracy and performance is a set of simple
ideas, such as estimating 3d joints in the camera coordinate
frame, adding residual connections and using batch normal-
ization. These ideas could be rapidly tested along with other
unsuccessful ones (e.g. estimating joint angles) due to the
simplicity of the network.
The experiments show that inferring 3d joints from
groundtruth 2d projections can be solved with a surprisingly
low error rate – 30% lower than state of the art – on the
largest existing 3d pose dataset. Furthermore, training our
system on noisy outputs from a recent 2d keypoint detec-
tor yields results that slightly outperform the state-of-the-art
on 3d human pose estimation, which comes from systems
trained end-to-end from raw pixels.
Our work considerably improves upon the previous best
2d-to-3d pose estimation result using noise-free 2d detec-
tions in Human3.6M, while also using a simpler archi-
tecture. This shows that lifting 2d poses is, although far
from solved, an easier task than previously thought. Since
our work also achieves state-of-the-art results starting from
the output of an off-the-shelf 2d detector, it also suggests
that current systems could be further improved by focus-
ing on the visual parsing of human bodies in 2d images.
Moreover, we provide and release a high-performance, yet
lightweight and easy-to-reproduce baseline that sets a new
bar for future work in this task. Our code is publicly avail-
able at https://github.com/una-dinosauria/
3d-pose-baseline.
2. Previous work
Depth from images
The perception of depth from purely
2d stimuli is a classic problem that has captivated the atten-
tion of scientists and artists at least since the Renaissance,
when Brunelleschi used the mathematical concept of per-
spective to convey a sense of space in his paintings of Flo-
rentine buildings.
Centuries later, similar perspective cues have been ex-
ploited in computer vision to infer lengths, areas and dis-
tance ratios in arbitrary scenes [57]. Apart from perspective
information, classic computer vision systems have tried to
use other cues like shading [53] or texture [25] to recover
depth from a single image. Modern systems [12,26,34,39]
typically approach this problem from a supervised learning
perspective, letting the system infer which image features
are most discriminative for depth estimation.
Top-down 3d reasoning
One of the ﬁrst algorithms for
depth estimation took a different approach: exploiting the
known 3d structure of the objects in the scene [37]. It has
been shown that this top-down information is also used by
humans when perceiving human motion abstracted into a
set of sparse point projections [8]. The idea of reasoning
about 3d human posture from a minimal representation such
as sparse 2d projections, abstracting away other potentially
richer image cues, has inspired the problem of 3d pose esti-
mation from 2d joints that we are addressing in this work.
2d to 3d joints
The problem of inferring 3d joints from
their 2d projections can be traced back to the classic work
of Lee and Chen [23]. They showed that, given the bone
lengths, the problem boils down to a binary decision tree
where each split correspond to two possible states of a
joint with respect to its parent.
This binary tree can be
pruned based on joint constraints, though it rarely resulted
in a single solution.
Jiang [20] used a large database
of poses to resolve ambiguities based on nearest neigh-
bor queries.
Interestingly, the idea of exploiting nearest
neighbors for reﬁning the result of pose inference has been
recently revisited by Gupta et al. [14], who incorporated
temporal constraints during search, and by Chen and Ra-
manan [9]. Another way of compiling knowledge about 3d
human pose from datasets is by creating overcomplete bases
suitable for representing human poses as sparse combina-
tions [2, 7, 36, 49, 55, 56], lifting the pose to a reproducible
kernel Hilbert space (RHKS) [18] or by creating novel pri-
ors from specialized datasets of extreme human poses [2].
Deep-net-based 2d to 3d joints
Our system is most re-
lated to recent work that learns the mapping between 2d
and 3d with deep neural networks.
Pavlakos et al. [33]
introduced a deep convolutional neural network based on
the stacked hourglass architecture [30] that, instead of re-
gressing 2d joint probability heatmaps, maps to probabil-
ity distributions in 3d space. Moreno-Noguer [27] learns
to predict a pairwise distance matrix (DM) from 2-to-3-
dimensional space.
Distance matrices are invariant up
to rotation, translation and reﬂection; therefore, multi-
dimensional scaling is complemented with a prior of human
poses [2] to rule out unlikely predictions.
A major motivation behind Moreno-Noguer’s DM re-
gression approach, as well as the volumetric approach of
Pavlakos et al., is the idea that predicting 3d keypoints
from 2d detections is inherently difﬁcult.
For example,
Pavlakos et al. [33] present a baseline where a direct 3d
joint representation (such as ours) is used instead (Table 1
in [33]), with much less accurate results than using volumet-
ric regression1 Our work contradicts the idea that regress-
ing 3d keypoints from 2d joint detections directly should
1This approach, however, is slightly different from ours, as the input is
still image pixels, and the intermediate 2d body representation is a series
of joint heatmaps – not joint 2d locations.

Linear 
1024
Batch norm
RELU
Dropout 0.5
+
x2
Linear 
1024
Batch norm
RELU
Dropout 0.5
Figure 1. A diagram of our approach. The building block of our network is a linear layer, followed by batch normalization, dropout and a
RELU activation. This is repeated twice, and the two blocks are wrapped in a residual connection. The outer block is repeated twice. The
input to our system is an array of 2d joint positions, and the output is a series of joint positions in 3d.
be avoided, and shows that a well-designed and simple net-
work can 

## conclusion
Looking at Table 2, we see a generalized increase in er-
ror when training with SH detections as opposed to training
with ground truth 2d across all actions – as one may well
expect. There is, however, a particularly large increase in
the classes taking photo, talking on the phone, sitting and
sitting down. We hypothesize that this is due to the se-
vere self-occlusions in these actions – for example, in some
phone sequences, we never get to see one of the hands of
the actor. Similarly, in sitting and sitting down, the legs are
often aligned with the camera viewpoint, which results in
large amounts of foreshortening.
Further improvements
The simplicity of our system
suggests multiple directions of improvement in future work.
For example, we note that stacked hourglass produces ﬁ-
nal joint detection heatmaps of size 64 × 64, and thus a
larger output resolution might result in more ﬁne-grained
detections, moving our system closer to its performance
when trained on ground truth. Another interesting direc-
tion is to use multiple samples from the 2d stacked hour-
glass heatmaps to estimate an expected gradient – `a la pol-
icy gradients, commonly used in reinforcement learning –
so as to train a network end-to-end. Yet another idea is to
emulate the output of 2d detectors using 3-dimensional mo-
cap databases and “fake” camera parameters for data aug-
mentation, perhaps following the adversarial approach of
Shrivastava et al. [41]. Learning to estimate coherently the
depth of each person in the scene is an interesting research
path, since it would allow our system to work on 3d pose
estimation of multiple people. Finally, our architecture is
simple, and it is likely that further research into network
design could lead to better results on 2d-to-3d systems.
5.1. Implications of our results
We have demonstrated that a relatively simple deep feed-
forward neural network can achieve a remarkably low error
rate on 3d human pose estimation. Coupled with a state-of-
the-art 2d detector, our system obtains the best results on 3d
pose estimation to date.

Figure 3. Qualitative results on the MPII test set. Observed image, 2d detection with Stacked Hourglass [30], (in green) our 3d prediction.
The bottom 3 examples are typical failure cases, where either the 2d detector has failed badly (left), or slightly (right). In the middle, the 2d
detector does a ﬁne job, but the person is upside-down and Human3.6M does not provide any similar examples – the network still seems
to predict an average pose.
Our results stand in contrast to recent work, which has
focused on deep, end-to-end systems trained from pixels to
3d positions, and contradicts the underlying hypothesis that
justify the complexity of recent state-of-the-art approached
to 3d human pose estimation. For example, the volumetric
regression approach of [33] et al. is based on the hypothe-
sis that directly regressing 3d points is inherently difﬁcult,
and regression in a volumetric space would provide easier
gradients for the network (see Table 1 in [33]). Although
we agree that image content should help to resolve chal-
lenging ambiguous cases (consider for example the classic
turning ballerina optical illusion), competitive 3d pose es-
timation from 2d points can be achieved with simple high
capacity systems. This might be related to the latent in-
formation about subtle body and motion traits existing in
2d joint stimuli, such as gender, which can be perceived
by people [47]. Similarly, the use of a distance matrix as
a body representation in [27] is justiﬁed by the claim that
invariant, human-designed features should boost the accu-
racy of the system. However, our results show that well
trained systems can outperform these particular features in
a simple manner. It would be interesting to see whether a
combination of joint distances and joint positions boost the
performance even further – we leave this for future work.
6. Conclusions and future work
We have shown that a simple, fast and lightweight deep
neural network can achieve surprisingly accurate results in
the task of 2d-to-3d human pose estimation; and coupled
with a state-of-the-art 2d detector, our work results in an
easy-to-reproduce, yet high-performant baseline that out-
performs the state of the art in 3d human pose estimation.
Our accuracy in 3d pose estimation from 2d ground
truth suggest that, although 2d pose estimation is consid-
ered a close to solved problem, it remains as one of the
main causes for error in the 3d human pose estimation task.
Moreover, our work represents poses in simple 2d and 3d
coordinates, which suggests that ﬁnding invariant (and more
complex) representations of the human body, as has been
the focus of recent work, might either not be crucial, or have
not been exploited to its full potential.
Finally, given its simplicity and the rapid development
in the ﬁeld, we like to think of our work as a future base-
line, rather than a full-ﬂedged system for 3d pose esti-
mation. This suggests multiple directions of future work.
For one, our network currently does not have access to vi-
sual evidence; we believe that adding this information to
our pipeline, either via ﬁne-tuning of the 2d detections or
through multi-sensor fusion will lead to further gains in per-
formance. On the other hand, our architecture is similar to a
multi-layer perceptron, which is perhaps the simplest archi-
tecture one may think of. We believe that a further explo-
ration of the network architectures will result in improved
performance. These are all interesting areas of future work.
Acknowledgments
The authors thank NVIDIA for the
donation of GPUs used in this research. Julieta was sup-
ported in part by the Perceiving Systems group at the Max
Planck Institute for Intelligent Systems. This research was
supported in part by the Natural Sciences and Engineering
Research Council of Canada (NSERC).