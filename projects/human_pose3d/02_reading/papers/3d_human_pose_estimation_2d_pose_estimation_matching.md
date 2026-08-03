# 3D Human Pose Estimation = 2D Pose Estimation + Matching

> 2017 · id: W2583372902 · arXiv: 1612.06524 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
We explore 3D human pose estimation from a single
RGB image. While many approaches try to directly pre-
dict 3D pose from image measurements, we explore a sim-
ple architecture that reasons through intermediate 2D pose
predictions. Our approach is based on two key observa-
tions (1) Deep neural nets have revolutionized 2D pose es-
timation, producing accurate 2D predictions even for poses
with self-occlusions (2) ”Big-data”sets of 3D mocap data
are now readily available, making it tempting to “lift” pre-
dicted 2D poses to 3D through simple memorization (e.g.,
nearest neighbors). The resulting architecture is straight-
forward to implement with off-the-shelf 2D pose estimation
systems and 3D mocap libraries. Importantly, we demon-
strate that such methods outperform almost all state-of-the-
art 3D pose estimation systems, most of which directly try
to regress 3D pose from 2D measurements.

## introduction
Inferring 3D human pose from image measurements is
classic task in computer vision, dating back to the iconic
work of Hogg [11] and O’Rourke and Badler [22]. Such a
technology has immediate applications in various tasks such
as action understanding, surveillance, human-robot interac-
tion, and motion capture, to name a few. As such, it has
a long and storied history. We refer the reader to various
surveys for a broad overview of the popular topic [8, 19].
Previous approaches often make use of a highly sensored
environment, including video streams [37, 30], multiview
cameras [3, 10], depth images [23, 36, 27]. In this work, we
focus on the ”pure” and challenging setting of recovering
3D body pose with a single 2D RGB image [17, 35, 25, 32].
Our key insight to the problem is leveraging recent ad-
vances in 2D image understanding, made possible through
the undeniable impact of deep learning.
While origi-
nally explored for coarse recognition tasks such as im-
age classiﬁcation, recent methods have extended such net-
work architectures to “ﬁne-grained” human pose estima-
tion, where the task is formulated as one of 2D heatmap
Estimated 2D Pose
Input Image
3D Pose Library
CNN
Depth added by
3D Exemplar 
Output 3D Pose
Figure 1. Overview of our approach for 3D pose estimation: given
an input image, ﬁrst estimate a 2D pose and then estimate its depth
by matching to a library of 3D poses. The ﬁnal prediction is given
by the colored skeleton, while the ground-truth is shown in gray.
Our approach works surprisingly well because 2D pose estimation
is accurate even during occlusions (as illustrated by both wrists
above), suggesting that 2D pose estimates need only be reﬁned by
adding depth values.
prediction [33, 21, 31, 12]. One of the long standing chal-
lenges in 2D human pose estimation has been estimating
poses under self-occlusions. Indeed, reasoning about occlu-
sions has been one of the underlying motivations for work-
ing in a 3D coordinate frame rather than 2D. But one of
our salient conclusions is that state-of-the-art methods do a
surprisingly good job of 2D pose estimation even under oc-
clusion. Given this observation, the remaining challenge is
predicting depth values for the estimated 2D joints.
Inferring 3D structure from 2D correspondences is also a
1
arXiv:1612.06524v2  [cs.CV]  11 Apr 2017

well-studied problem in computer vision, often addressed in
multiview setting as structure from motion. In the context of
monocular human pose estimation, the relevant cues seem
to be semantic rather than geometric. One can estimate 3D
postures from a 2D skeleton based on high-level knowl-
edge derived from anthropometric, kinematic, and dynamic
constraints. Inspired by the success of data-driven architec-
tures, we explore a simple non-parametric encoding of such
high-level constraints: given a 3D pose library, we gener-
ate a large number of 2D projections (from virtual camera
views). Given this training set of paired (2D,3D) data and
predictions from a 2D pose estimation algorithm, we report
back the depths from the 3D pose associated with the closest
matching 2D example from our library. Our entire pipeline
is summarized in Fig. 1.
Generalization: One desirable property of our two-
stage approach is generalization. Due to the difﬁculty of
annotation in 3D, training datasets with 3D labels are typi-
cally collected in a lab environment, while 2D datasets tend
to be more diverse. Our two-stage pipeline makes use of
different training sets for different stages, resulting a sys-
tem that can predict 3D poses from “in-the-wild” images.
Evaluation: Though we present qualitative results on
in-the-wild-imagery, we also perform an extensive quanti-
tative evaluation of our method on widely benchmarked 3D
human-pose datasets, such as Human3.6M [13]. We fol-
low standard train/test protocol splits, but our analysis re-
veals that there has been inconsistent reporting in the lit-
erature, both in terms of test sets and evaluation criteria.
To make our results as transparent as possible, we report
performance for all metrics and splits we could ﬁnd. One
of our surprising ﬁndings is the impressive performance of
our simple pipeline: we outperform essentially all prior
work on all metrics. Our entire pipeline, even given the
non-parametric matching step, returns a 3D pose given a
2D image in under 200ms (160ms for 2D estimation by a
CNN, 26ms for exemplar matching with a training library of
200,000 poses). Finally, to promote future progress, we per-
form an exhaustive analysis of additional baselines with up-
per bounds that reveal the continued beneﬁt of working with
intermediate 2D representations and data-driven encoding
of 3D constraints.

## method
In this section, we describe our method for estimating 3D
human pose given a single RGB image. We make use of a
probabilistic formulation over variables including the image
I, the 3D pose X ∈RN×3, and the 2D pose x ∈RN×2,
where N is the number of articulated joints. We write the
joint probability as:
p(X, x, I) = p(X|x, I) · p(x|I) · p(I)
(1)
where the above makes no limiting assumptions by itself.
Conditional independence: Let us now assume that the
3D pose X is conditionally independent of image I given
the 2D pose x. This is equivalent to the implication that
given a 2D skeleton, the prediction of its corresponding 3D
skeleton would not be affected by 2D image measurements.
While this is not quite true (we show a counter example
in Fig. 2), it seems to be a reasonable ﬁrst-order approxi-
mation. Moreover, this factorization still allows for p(x|I)
to be arbitrarily complex, which is likely needed to accu-
rately model complex interactions between 2D projections
2

Figure 2. A failure case where the 3D pose is not conditionally in-
dependent of the image given the 2D pose: p(X|x, I)̸ = p(X|x).
We show the output of our system given the ground-truth 2D pose,
with the (incorrect) best-matching 3D exemplar on the right (vi-
sualized from a novel viewpoint, where the estimated camera is
drawn as a view frustum). Our experiments suggest that such cases
are rare, and that much of the time 3D can be inferred from 2D
projections.
and image features during occlusions. Given this condi-
tional Independence, one can write:
p(X, x, I) = p(X|x)
| {z }
NN
· p(x|I)
| {z }
CNN
·p(I)
(2)
We tackle the second term with a image-based CNN that
predicts 2D keypoint heatmaps. We tackle the ﬁrst term
with a non-parametric nearest-neighbor (NN) model. We
describe each term in turn below.
3.1. Image-Based 2D Pose Estimation
Given the above Independence assumption, we would
ﬁrst like to predict 2D pose given image measurements. We
model the conditional of 2D pose given an image as
P(x|I) = CNN(I)
(3)
where we assume CNN is a nonlinear function that returns
N 2D heatmaps (or marginal distributions over the location
of individual joints). We make use of convolutional pose
machines (CPMs) [33], which return precisely N heatmaps
for individual body joints. We normalize the heatmaps so
that they can be interpreted as marginal distributions for
each joint. CPM is a near-state-of-the-art pose estimation
system (88.5% PCKh on MPII dataset [4], quite close to
the state-of-the-art value of 90.9%
[21]). Note the off-
the-shelf CPM model was trained on MPII dataset, which
is a somewhat limited dataset in that annotations are pro-
vided through manual inspection. We ﬁne-tune this model
on the large scale Human3.6M [13] training set, which con-
tains annotations acquired by a mocap system (allowing for
larger-scale labeling).
3.2. Nonparametric 3D shape model
We model P(X|x) with a non-parametric nearest neigh-
bor model. We will follow a notational convention where
X = [X, Y, Z] and x = [x, y]. Assume that we have library
Xi
Xi
*
Figure 3. On the left, we show the 3D exemplar Xi that best
matches the ground-truth 2D pose x. While the overall pose is
roughly correct, the arms and legs are bent incorrectly. By sim-
ply copying the depth values from the exemplar (and copying the
(x, y) values from the 2D pose under a weak perspective model,
as given in (6)), we can obtain a warped exemplar X∗
i that better
matches the 2D pose.
of 3D poses {Xi} paired with a particular camera projection
matrix {Mi}, such that the associated 2D poses are given by
{Mi(Xi)}. If we want to consider multiple cameras for a
single 3D pose, we add another copy of the 3D pose with a
different camera matrix to our library. We deﬁne a distribu-
tion over 3D poses based on reprojection error:
P(X = Xi|x) ∝e−1
σ2 ||Mi(Xi)−x||2
(4)
where the MAP estimate is given by the 1-nearest neigh-
bor (1NN). We explore two extensions to the above basic
framework.
Virtual cameras: We can further reduce the squared re-
projection error by searching over small perturbations of
each camera. This involves solving a camera resectioning
problem [9], where an iterative solver can be initialized with
Mi:
M ∗
i = argmin
M
||M(Xi) −x||2
(5)
In practice, we construct a shortlist of k candidates that
score well according to (4), and resort them according to
optimal camera matrix. We found that optimizing over cam-
eras produced a small but noticeable improvement in our ex-
periments. Unless otherwise speciﬁed, we choose k = 10
in our experiments.
Warped exemplars: Much previous work on exemplars
introduce methods for warping exemplars to better match
the 2D pose estimates, often formulated as an inverse kine-
matics optimization problem.
We describe an extremely
lightweight method for doing so here. We ﬁrst align the 3D
exemplar to the camera-coordinate system used to compute
the projection x. This is done with a 3D rigid transforma-
tion given by the camera extrinsics encoded in Mi (or M ∗
i ).
In practice we use a training set {Xi} where 3D exemplars
are already aligned to their projections {xi}, implying that
extrinsics in Mi reduce to an identity matrix (which is the
case for the Human3.6M dataset [13], since 3D poses are
3

speciﬁed in camera coordinates of their associated image
projections). Given this alignment, we simply replace the
(Xi, Yi) exemplar coordinates with their scaled 2D coun-
terparts (x, y) under a weak perspective camera model:
X∗
i =

sx
sy
Zi

,
where
s = average(Zi)
f
(6)
where f is the focal length of the camera (given by the in-
trinsics in Mi) and average(Zi) is the average depth of the
3D joints. Such weak-perspective approximations are com-
monly used to initialize algorithms for perspective (PnP)
camera calibration [18], and will be reasonable when the
depth variation of the human skeleton is small relative to
the overall distance to the camera. Our results suggest that
such closed-form solutions for 3D warping rival the accu-
racy of complex energy minimization methods (see Fig. 3).

## experiments
In our experiments, we test a variety of variations of our
proposed pipeline.
Qualitative results: We ﬁrst present some qualitative
results. Fig. 4 shows results on challenging examples from
subject S11 of Human3.6M. We choose examples with self-
occlusions and sitting poses. To demonstrate the accuracy
of the 3D predictions, we visualize novel viewpoints. We
then apply the proposed method on Leeds Sports Pose(LSP)
dataset [15] to test cross-dataset generalization. We posit
that our pipeline will generalize across image variation (due
to the underlying robustness of our 2D pose estimation sys-
tem) but maybe limited in the 3D estimates due to the li-
brary used (from Human3.6M). Importantly, our approach
produces plausible 3D poses even when the activity class is
not included in Human3.6M. This implies that our method
can reliably estimate 3D poses in the wild!
4.1. Evaluation protocols
We use Human3.6M for quantitative evaluation and anal-
ysis. It appears that multiple train/test splits have been used
in the literature, as well as different approaches to comput-
ing mean per joint position error (MPJPE), measured in mil-
limeters. We summarize them here.
Protocol 1: In [35, 16, 25], the entire dataset was par-
titioned into six training subjects (S1, S5, S6, S7, S8, S9),
and one testing subject (S11). Evaluation is performed on
every 64th frame of S11’s video clips. In this conﬁguration,
there are total 1.8 million 3D poses available in the training
set. MPJPE between the ground truth 3D pose and the es-
timated 3D pose is computed by ﬁrst aligning poses with a
rigid transformation [16].
Protocol 2: Others [37, 30, 17] use ﬁve subjects (S1,
S5, S6, S7, S8) for training, and two subjects (S9, S11)
for testing. We follow [37]’s setup that downsamples the
videos from 50 fps to 10 fps. Here, MPJPE is evaluated
without a rigid transformation, following the original h36m
protocol: both the ground-truth and predicted 3D pose is
centered with respect to a root joint (ie. pelvis). In contrast
to Protocol 1, this evaluation can be sensitive to a single
poorly-predicted joint, particularly if it is the root [13].
To compare to published performance numbers, we use
the appropriate protocol as needed. From our own experi-
ence, we ﬁnd Protocol 1 to be simpler and more intuitive,
and so focus on it for our diagnostic evaluations.
4.2. Comparison to state-of-the-art (Protocol 1)
Final system: Table 1 compare MPJPE for each activ-
ity class. Our approach clearly outperforms [35] and [25].
(”Ours” in the tables of comparison throughout the exper-
iment refers to the warped exemplar X∗described in Sec-
tion 3.2)
Performance given ground-truth 2D: A common di-
agnostic is evaluating performance given ground-truth 2D
poses, written as gt. Table 2 shows that our simple matching
+ warping outperforms [35], who use a complex iterative al-
gorithm for matching and warping exemplars to image ev-
idence. Our diagnostics will later show that even matching
exemplars without warping outperforms prior art, indicating
the remarkable power of a simple NN baseline.
Size of trainset: Table 3 shows the MPJPE versus the
training data size. Since approaches deal with 2D and 3D
sources differently, we list both sizes. Yasin et al. [35]
project multiple 2D poses from each 3D exemplar (with
virtual cameras) to create 2D poses for matching, and Ro-
gez et al. [25] directly synthesize 2D images for training.
Our approach makes use of the default training data in Hu-
man3.6M, where each 3D pose is paired with a single 2D
projection. We max out performance with a modest pose
library of 180k 3D-2D pairs, but produce competitive accu-
racy even for 18k. The slight increase in MPJPE for larger
training sets seems to be related to noise from 2D pose es-
timation, since we observe a monotonically decrease when
ground truth 2D poses are given (Fig. 7).
4.3. Comparison to state-of-the-art (Protocol 2)
Final system: Table 4 provides the comparison to [37]
and [30] using Protocol 2. Note that in both these works,
temporal smoothness was exploited by taking a short image
sequences as input. Even though we do not use temporal
information, our system is quite close to state-of-the-art. A
qualitative comparison to [37] is also provided in Fig. 5.
Performance given ground-truth 2D: Our strong per-
formance in Fig. 5 might be attributed to better 2D pose
estimation. Therefore, we investigate the case given ground
truth 2D pose, following Zhou’s diagnostic protocol [37]:
evaluate MPJPE up to a 3D rigid body transformation in-
cluding scale, only on the ﬁrst 30 seconds of the ﬁrst cam-
4

Images with
2D pose
Estimation
3D pose in a 
novel view
Figure 4. We show qualitative results on Human3.6M-test (top) and LSP-test (bottom). Our method produces plausible results for chal-
lenging images with self-occlusions and extreme poses, and can generalize to activities and poses not in the train set (Human3.6M-train).
era in Human3.6M. For a fair comparison, we make use the
same training set of 3D-2D training data for both methods.
The results are shown in Table 5. With a shortlist of k = 10
matches, camera resectioning (5) and exemplar warping (6)
produces a slightly lower error than [37]’s approach with-
out a 3D prior. Qualitative results are provided in Fig. 6.
Our approach produces lower 2D reprojection error, while
Zhou’s method appears to suffers from the restriction of 3D
poses to a low-dimensional subspace.
4.4. Diagnostics
We now perform an extensive set of diagnostics to reveal
the strength of our individual components, as well as upper-
bound analysis that is useful for guiding future work. For
simplicity, we restrict ourselves to Protocol 1.
5

Mean Per Joint Position Error (MPJPE), in mm

## related_work
Here we review related works on 3D human pose predic-
tion most relevant to our approach.
(Deep) Regression: Most existing work that makes use
of deep features tends to formulate the problem as a di-
rect 2D image to 3D pose regression task. Li et al. [17]
use deep learning to train a regression model to predict 3D
pose directly from images. Tekin et al. [30] integrate spatio-
temporal features via an image sequence to learn regression
model for 3D pose mapping. We provide both a theoretical
and empirical analysis that suggests that 2D pose may be a
useful intermediate representation.
Intermediate 2D pose: Other approaches have explored
pipelines that use 2D poses as an intermediate result. Most
focus on the second-stage that lifts 2D estimates to 3D. This
is classically treated as a constrained optimization problem
who’s objective minimizes the 2D reprojection error of an
unknown 3D pose and unknown camera [37, 32, 24, 2].
The optimization problem is often subject to kinematic con-
straints [34, 29], and sometimes 3D poses are assumed
to live a in low-dimensional subspace to better condition
the optimization [37]. Such optimization-based approaches
could be sensitive to initialization and local minima, and
often require expensive constrained solvers. We use data-
driven matching, that when combined with a simple closed-
form warping algorithm, yields a fast and accurate 3D solu-
tion.
Exemplar-based:
Previous work has also explored
example-based methods, dating back at least to [26]. A cen-
tral challenge is generalization to novel poses outside the
training set. [14] propose matching upper and lower bodies
individually, to allow for novel compositions at test-time.
[35] adapt exemplars to better match image measurements
with an energy minimization approach. [25] synthesize new
2D images with image-based rendering. Other methods also
warp 3D exemplars to 2D image descriptors, often based on
shape-context [1, 20] or silhouette features [5]. In our work,
we show that a modest number of exemplars (200,000),
combined with a simple closed-form algorithm for warp-
ing a 3D exemplar to exactly project to 2D pose estimates,
outperforms more complex methods.

## conclusion
We present an simple approach to 3D human pose esti-
mation by performing 2D pose estimation, followed by 3D
exemplar matching. The simplicity and efﬁciency of our
method, combined with its state-of-the-art performance on
both benchmark datasets and unconstrained “in-the-wild”
Figure 8. Median MPJPE by Protocol 1 versus database size. Me-
dian error is lower than mean error in Fig. 7, suggesting that a few
joints are responsible for a large mean error. Other trends follow
the mean error curves from Fig. 7.
Walk
Jog
Throw
Gestures
Box
Avg.
Catch
Warped
64.46
69.88
59.99
67.89
79.22
68.29
Unwarped
90.17
95.27
82.74
88.82
103.85
92.17
Table 9. We evaluate a Human3.6M-trained model on HumanEva
(under Protocol 1). To isolate the impact of 3D matching, we use
ground-truth 2D keypoints. As a point of comparison, average er-
ror on Human3.6M test is 70.93 (unwarped) and 57.5 (warped)
(Table 9). These results suggest that 3D exemplars do generalize
across datasets, and importantly, warping signiﬁcantly increases
the amount of generalization. Note that the two datasets use dif-
ferent deﬁnitions of skeletons, implying that learning a mapping
should reduce error even further.
imagery, suggests that such simple baselines should be used
for future benchmarking in 3D pose estimation. A notable
advantage of intermediate 2D representations is modular
training – 2D datasets (which are typically larger and more
diverse because of ease of annotation) can be used to train
the initial-image processing module, while 3D motion cap-
ture data can be used to train the subsequent 3D-reasoning
module. This allows our system to take immediate advan-
tage of advances in 2D pose estimation, such as multi-body
analysis [7]. Our results also suggest that 3D inference is,
in some sense, “all about 2D”, at least in the case of articu-
lated objects. Indeed, one of our surprising ﬁndings was the
high performance of 2D pose estimation systems even un-
der occlusions, suggesting that 2D estimates can in fact be
reliably estimated without directly reasoning about depth.
Given such reliable 2D estimates, we show that one can
efﬁciently impute depth through simple memorization and
warping of a 3D pose library.
Acknowledgements: This work was supported by NSF
Grant 1618903, NSF Grant 1208598, the Intel Science and
Technology Center for Visual Cloud Systems (ISTC-VCS),
Google, and Amazon.
8