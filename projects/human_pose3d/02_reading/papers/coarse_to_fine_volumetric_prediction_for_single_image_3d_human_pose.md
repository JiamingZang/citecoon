# Coarse-to-Fine Volumetric Prediction for Single-Image 3D Human Pose

> 2017 · id: W2554247908 · arXiv: 1611.07828 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
This paper addresses the challenge of 3D human pose
estimation from a single color image. Despite the general
success of the end-to-end learning paradigm, top perform-
ing approaches employ a two-step solution consisting of a
Convolutional Network (ConvNet) for 2D joint localization
and a subsequent optimization step to recover 3D pose. In
this paper, we identify the representation of 3D pose as a
critical issue with current ConvNet approaches and make
two important contributions towards validating the value of
end-to-end learning for this task. First, we propose a ﬁne
discretization of the 3D space around the subject and train a
ConvNet to predict per voxel likelihoods for each joint. This
creates a natural representation for 3D pose and greatly im-
proves performance over the direct regression of joint coor-
dinates. Second, to further improve upon initial estimates,
we employ a coarse-to-ﬁne prediction scheme. This step ad-
dresses the large dimensionality increase and enables iter-
ative reﬁnement and repeated processing of the image fea-
tures. The proposed approach outperforms all state-of-the-
art methods on standard benchmarks achieving a relative
error reduction greater than 30% on average. Additionally,
we investigate using our volumetric representation in a re-
lated architecture which is suboptimal compared to our end-
to-end approach, but is of practical interest, since it enables
training when no image with corresponding 3D groundtruth
is available, and allows us to present compelling results for
in-the-wild images.

## introduction
Estimating the full-body 3D pose of a human from a sin-
gle monocular image is an open challenge, which has gar-
nered signiﬁcant attention since the early days of computer
vision [18].
Given its ill-posed nature, researchers have
generally approached 3D human pose estimation in sim-
pliﬁed settings, such as assuming background subtraction
is feasible [1], relying on groundtruth 2D joint locations
to estimate 3D pose [26, 42], employing additional cam-
era views [7, 15], and capitalizing on temporal consistency
to improve upon single frame predictions [37, 3]. This di-
Image
ConvNet
Volumetric Output
Figure 1: Illustration of our volumetric representation for
3D human pose. We discretize the space around the subject
and use a ConvNet to predict per voxel likelihoods for each
joint from a single color image.
versity of assumptions and additional information sources
exempliﬁes the challenge presented by the task.
With the introduction of more powerful discriminative
approaches, such as Convolutional Networks (ConvNets),
many of these restrictive assumptions have been relaxed.
End-to-end learning approaches attempt to estimate 3D
pose directly from a single image by addressing it as co-
ordinate regression [19, 34], nearest neighbor between im-
ages and poses [20], or classiﬁcation over a set of pose
classes [27]. Yet to date, these approaches have been out-
performed by more traditional two-step pipelines, e.g., [44,
6]. In these cases, ConvNets are used only for 2D joint
localization and 3D poses are generated during a post-
processing optimization step. Combining accurate 2D joint
localization with strong and expressive 3D priors has been
proven to be very effective. In this work, we show that
ConvNets are able to provide much richer information than
simply 2D joint locations.
To fully exploit the potential of ConvNets in the context
of 3D human pose, we propose the following items, and jus-
tify them empirically. First, we cast 3D pose estimation as
a keypoint localization problem in a discretized 3D space.
Instead of directly regressing the coordinates of the joints
(e.g., [19, 34]), we train a ConvNet to predict per voxel
likelihoods for each joint in this volume. This volumetric
representation, illustrated in Figure 1, is much more sensi-
ble for the 3D nature of our problem and improves learning.
1
arXiv:1611.07828v2  [cs.CV]  26 Jul 2017

Effectively, for every joint, the volumetric supervision pro-
vides the network with groundtruth for each voxel in the 3D
space. This provides much richer information than a set of
world coordinates. The empirical results also validate the
superiority of our proposed form of supervision.
Second, to deal with the increased dimensionality of the
volumetric representation, we propose a coarse-to-ﬁne pre-
diction scheme. As demonstrated in the 2D pose case, inter-
mediate supervision and iterative estimation are particularly
effective strategies [39, 8, 21]. For our volumetric repre-
sentation though, naively stacking an increasing number of
components and reﬁning the estimates is not an effective so-
lution, as shown empirically. Instead, we gradually increase
the resolution of the supervision volume for the most chal-
lenging z-dimension (depth), during the processing. This
coarse-to-ﬁne supervision, illustrated schematically in Fig-
ure 2, allows for more accurate estimates after each step.
We empirically demonstrate the advantage of this practice
over naively stacking more components together.
Our proposed approach achieves state-of-the-art results
on standard benchmarks, outperforming both ConvNet-only
and hybrid approaches that post-process the 2D output of a
ConvNet. Additionally, we investigate using our volumet-
ric representation within a related architecture that decou-
ples 2D joint localization and 3D joint reconstruction. In
particular, we use two separate networks (the output of one
serves as the input to the other) and two non-corresponding
data sources, i.e., 2D labeled imagery to train the ﬁrst com-
ponent and an independent 3D data source (e.g., MoCap) to
train the second one separately. While this architecture has
practical beneﬁts (e.g., predicting 3D pose for in-the-wild
images), we show empirically that it underperforms com-
pared to our end-to-end approach when images with cor-
responding 3D groundtruth are available for training. This
ﬁnding further underlines the beneﬁt of predicting 3D pose
directly from an image, whenever this is possible, instead
of using 2D joint localization as an intermediate step.
In summary, we make the four following contributions:
• we are the ﬁrst to cast 3D human pose estimation as
a 3D keypoint localization problem in a voxel space
using the end-to-end learning paradigm;
• we propose a coarse-to-ﬁne prediction scheme to deal
with the large dimensionality of our representation and
enable iterative processing to realize further beneﬁts;
• our proposed approach achieves state-of-the-art results
on standard benchmarks, surpassing both ConvNet-
only and hybrid approaches that employ ConvNets for
2D pose estimation, with a relative error reduction that
exceeds 30% on average;
• we show the practical use of our volumetric representa-
tion in cases when end-to-end training is not an option
and present compelling results on in-the-wild images.

## related_work
The literature on 3D human pose estimation is vast with
approaches addressing the problem in a variety of settings.
Here, we survey works that are most relevant to ours with a
focus on ConvNet-based approaches; we refer the reader to
a recent survey [29] for a more complete literature review.
The majority of recent ConvNet-only approaches cast 3D
pose estimation as a coordinate regression task, with the tar-
get output being the spatial x, y, z coordinates of the human
joints with respect to a known root joint, such as the pelvis.
Li and Chan [19] pretrain their network with maps for 2D
joint classiﬁcation. Tekin et al. [34] include a pretrained
autoencoder within the network to enforce structural con-
straints on the output. Ghezelghieh et al. [13] employ view-
point prediction as a side task to provide the network with
global joint conﬁguration information. Zhou et al. [43] em-
bed a kinematic model to guarantee the validity of the re-
gressed pose. Park et al. [22] concatenate the 2D joint pre-
dictions with image features to improve 3D joint localiza-
tion. Tekin et al. [35] include temporal information in the
joint predictions by extracting spatiotemporal features from
a sequence of frames. In contrast to all these approaches, we
adopt a volumetric representation of the human pose, and
regress the per voxel likelihood for each joint separately.
This proves to have signiﬁcant advantages for the network
performance and provides a richer output compared to the
low-dimensional vector of joint coordinates.
An alternative approach to the classical regression
paradigm is proposed by Li et al. [20]. During training,
they learn a common embedding between color images and
3D poses. At test time, the test image is coupled with each
candidate pose and forwarded through the network; the in-
put image is assigned the candidate pose with the maximum
network score. This is a form of nearest neighbor classiﬁ-
cation which is highly inefﬁcient due to the requirement of
multiple forward network passes. On the other hand, Ro-
gez and Schmid [27] cast pose estimation as a classiﬁcation
problem. Given a predeﬁned set of pose classes, each im-
age is assigned to the class with the highest score. This
guarantees a valid global pose prediction, but the approach
is constrained by the poses in the original classes and thus
returns only a rough pose estimate. In contrast to the inef-
ﬁcient nearest neighbor approach and the coarse classiﬁca-
tion approach, our volume regression allows for much more
accurate 3D joint localization, while also being efﬁcient.
Despite the interest in end-to-end learning, ConvNet-
only
approaches
underperform
those
that
employ
a
ConvNet for the 2D localization of joints, and produce 3D
pose with a subsequent optimization step. Zhou et al. [44]
utilize a standard 2D pose ConvNet to localize the joints
and retrieve the 3D pose using an optimization scheme over
a sequence of monocular images. Similarly, Du et al. [10]
include height-maps of the human body to improve 2D joint
2

Figure 2: Illustration of our coarse-to-ﬁne volumetric approach for 3D human pose estimation from a single image. The input
is a single color image and the output is a dense 3D volume with separate per voxel likelihoods for each joint. The network
consists of multiple fully convolutional components [21], which are supervised in a coarse-to-ﬁne fashion, to deal with the
large dimensionality of our representation. 3D heatmaps are synthesized for supervision by increasing the resolution for the
most challenging z-dimension (depth) after each component. The dashed lines indicate that the intermediate heatmaps are
fused with image features to produce the input for the next fully convolutional component. For presentation simplicity, the
illustrated heatmaps correspond to the location of only one joint.
localization. Bogo et al. [6] use the joints predicted by a
2D ConvNet and ﬁt a statistical body shape model to re-
cover the full shape of the human body. In contrast, our
approach achieves state-of-the-art results with a single net-
work. Furthermore, it provides a rich 3D output, amenable
to post-processing, such as pictorial structures optimization
to constrain limb lengths, or temporal ﬁltering.
Another issue that has been addressed in the context of
using ConvNets for 3D human pose is the scarcity of train-
ing data. Chen et al. [9] use a graphics renderer to create
images with known groundtruth. Similarly, Ghezelghieh et
al. [13] augment the training set with synthesized examples.
A collage approach is proposed by Rogez and Schmid [27],
where parts from in-the-wild images are combined to create
additional images with known 3D poses. However, there
is no guarantee that the statistics of the synthetic exam-
ples match those of real images. To investigate the data
scarcity issue, we take inspiration from the 3D Interpreter
Network [40], which decouples the 3D pose estimation task
into 2D localization and 3D reconstruction within a single
ConvNet. In contrast, rather than using a predeﬁned linear
basis for 3D reconstruction, we predict 3D joint locations
directly with our volumetric representation. This demon-
strates the practical use of our volumetric representation
even when end-to-end training is not an option.
Finally, while we do not compare explicitly with multi-
view pose estimation work (e.g., [12, 31, 4, 11]), it is inter-
esting to note that the representation of 3D human pose in
a discretized 3D space has also been previously adopted in
multi-view settings [7, 15, 23], where it was used to accom-
modate predictions from different viewpoints. For single
view pose estimation, it has been considered in the context
of random forests [16]. This approach suffered from large
execution time (around three minutes), and required an ad-
ditional reﬁnement step using a pictorial structures model.
In stark contrast, our network can provide complete volume
predictions with a single forward pass in a few milliseconds,
needs no additional reﬁnement (although it is still a possi-
bility) to provide state-of-the-art results, and is integrated
within a coarse-to-ﬁne prediction scheme to deal with ex-
cessive dimensionality.
3. Technical approach
The following subsections summarize our technical ap-
proach. Section 3.1 describes the proposed volumetric rep-
resentation for 3D human pose and discusses its merits.
Next, Section 3.2 describes our coarse-to-ﬁne prediction ap-
proach that addresses the high dimensional nature of our
output representation. Finally, Section 3.3 describes the use
of our volumetric representation within a related decoupled
architecture and discusses its relative merits compared to
our coarse-to-ﬁne volumetric prediction approach.
3.1. Volumetric representation for 3D human pose
The problem of 3D human pose estimation using
ConvNets has been primarily approached as a coordinate
regression problem. In this case, the target of the network
is a 3N-dimensional vector comprised of the concatenation
of the x, y, z coordinates of the N joints of the human body.
For training, an L2 regression loss is employed:
L =
N
X
n=1
∥xn
gt −xn
pr∥2
2,
(1)
where xn
gt is the groundtruth and xn
pr is the predicted lo-
cation for joint n. The location of each joint is expressed
globally, with respect to a root joint, or locally, with re-
spect to its parent joint in the kinematic tree. The second
formulation has some beneﬁts, as discussed also by Li et
3

al. [19] (e.g., easier to learn to predict small, local devi-
ations), but still suffers from the fact that small errors can
easily propagate hierarchically to children joints of the kine-
matic tree. In general, despite its simplicity, the coordinate
regression approach makes the problem highly non-linear
and presents problems for the learning procedure. These
issues have previously been identiﬁed in the context of 2D
human pose [36, 24].
To improve learning, we propose a volumetric represen-
tation for 3D human pose. The volume around the subject
is discretized uniformly in each dimension. For each joint
we create a volume of size w×h×d. Let pn
(i,j,k) denote the
predicted likelihood of joint n being in voxel (i, j, k). To
train this network, the supervision is also provided in vol-
umetric form. The target for each joint is a volume with
a 3D Gaussian centered around the groundtruth position
xn
gt = (x, y, z) of the joint in the 3D grid:
Gi,j,k(xn
gt) =
1
2πσ2 e−(x−i)2+(y−j)2+(z−k)2
2σ2
,
(2)
where the value σ = 2 is used for our experiments. For
training, we use the mean squared error loss:
L =
N
X
n=1
X
i,j,k
∥G(i,j,k)(xn
gt) −pn
(i,j,k)∥2.
(3)
In theory, the output of the network is four dimensional,
i.e., (w × h × d × N), but in practice we organize it in
channels, thus our output is three dimensional, i.e., w ×h×
dN. The voxel with the maximum response in each 3D grid
is selected as the joint’s 3D location.
A major advantage of the volumetric representation is
that it casts the highly non-linear problem 