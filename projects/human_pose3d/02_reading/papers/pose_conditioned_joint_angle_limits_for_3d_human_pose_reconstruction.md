# Pose-conditioned joint angle limits for 3D human pose reconstruction

> 2015 · id: W1943191679 · 来源: web-agent
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

Pose-Conditioned Joint Angle Limits for 3D Human Pose Reconstruction
Ijaz Akhter and Michael J. Black
Max Planck Institute for Intelligent Systems, T¨ubingen, Germany
{ijaz.akhter, black}@tuebingen.mpg.de
Abstract
Estimating 3D human pose from 2D joint locations is
central to the analysis of people in images and video. To ad-
dress the fact that the problem is inherently ill posed, many
methods impose a prior over human poses. Unfortunately
these priors admit invalid poses because they do not model
how joint-limits vary with pose. Here we make two key con-
tributions. First, we collect a motion capture dataset that
explores a wide range of human poses. From this we learn
a pose-dependent model of joint limits that forms our prior.
Both dataset and prior are available for research purposes.
Second, we deﬁne a general parametrization of body pose
and a new, multi-stage, method to estimate 3D pose from 2D
joint locations using an over-complete dictionary of poses.
Our method shows good generalization while avoiding im-
possible poses. We quantitatively compare our method with
recent work and show state-of-the-art results on 2D to 3D
pose estimation using the CMU mocap dataset. We also
show superior results using manual annotations on real im-
ages and automatic detections on the Leeds sports pose
dataset.
1. Introduction
Accurate modeling of priors over 3D human pose is fun-
damental to many problems in computer vision. Most pre-
vious priors are either not general enough for the diverse
nature of human poses or not restrictive enough to avoid
invalid 3D poses. We propose a physically-motivated prior
that only allows anthropometrically valid poses and restricts
the ones that are invalid.
One can use joint-angle limits to evaluate whether two
connected bones are valid or not. However, it is established
in biomechanics that there are dependencies in joint-angle
limits between certain pair of bones [12, 17]. For example
how much one can ﬂex one’s arm depends on whether it
is in front of, or behind, the back. Medical textbooks only
provide joint-angle limits in a few positions [2, 26] and the
complete conﬁguration of pose-dependent joint-angle limits
for the full body is unknown.
Figure 1. Joint-limit dataset. We captured a new dataset for learn-
ing pose-dependent joint angle limits. This includes an extensive
variety of stretching poses. A few sample images are shown here.
We found that existing mocap datasets (like the CMU
dataset) are insufﬁcient to learn true joint angle limits, in
particular limits that are pose dependent. Therefore we cap-
tured a new dataset of human motions that includes an ex-
tensive variety of stretching poses performed by trained ath-
letes and gymnasts (see Fig. 1). We learn pose-dependent
joint angle limits from this data and propose a novel prior
based on these limits.
The proposed prior can be used for problems where
estimating 3D human pose is ambiguous.
Our pose
parametrization is particularly simple and general in that
the 3D pose of the kinematic skeleton is deﬁned by the
two endpoints of each bone in Cartesian coordinates. Con-
straining a 3D pose to remain valid during an optimization
simply requires the addition of our penalty term in the ob-
jective function. We also show that our prior can be com-
bined with a sparse representation of poses, selected from
an overcomplete dictionary, to deﬁne a general yet accurate
parametrization of human pose.
We use our prior to estimate 3D human pose from 2D
joint locations. Figure 2 demonstrates the main difﬁculty
in this problem. Given a single view in Fig. 2(a), the 3D
pose is ambiguous [27] and there exist several plausible 3D
poses as shown in Fig. 2(b), all resulting in the same 2D
observations. Thus no generic prior information about static
body pose is sufﬁcient to guarantee a single correct 3D pose.
Here we seek the most probable, valid, human pose.
We show that a critical step for 3D pose estimation given
1

(a) 2D Frame
(b) 3D Pose Interpretations
Figure 2. Given only 2D joint locations in (a), there are several
valid 3D pose interpretations resulting in the same image observa-
tion. Some of them are shown as colored points in (b), while the
gray points represent the ground truth. Here we display the pose
from a different 3D view so that the difference is clear, but all these
poses project to exactly the same 2D observations.
2D point locations is the estimation of camera parame-
ters. Given the diversity of human poses, incorrect cam-
era parameters can lead to an incorrect pose estimate. To
solve this problem we propose a grouping of body parts,
called the “extended-torso,” consisting of the torso, head,
and upper-legs. Exploiting the fact that the pose variations
for the extended-torso are fewer than for the full-body, we
estimate its 3D pose and the corresponding camera parame-
ters more easily. The estimated camera parameters are then
used for full-body pose estimation. The proposed multi-step
solution gives substantially improved results over previous
methods.
We evaluate 3D pose estimation from 2D for a wide
range of poses and camera views using activities from the
CMU motion capture dataset1. These are more complex and
varied than the data used by previous methods and we show
that previous methods have trouble in this case. We also re-
port superior results on manual annotations and automatic
part-based detections [16] on the Leeds sports pose dataset.
The data used for evaluation and all software is available for
other researchers to compare with our results [1].
2. Related Work
The literature on modeling human pose priors and the
estimation of 3D pose from points, images, video, depth
data, etc. is extensive. Most previous methods for model-
ing human pose assume ﬁxed joint angle limits [7, 24, 28].
Herda et al. [14] model dependencies of joint angle limits
on pose for the elbow and shoulder joint. Their model can-
not be used for our 2D to 3D estimation problem because
it requires the unobserved rotation around the bone axis to
be known. Hauberg et al. [13] suggest modeling such priors
in terms of a distribution over the endpoints of the bones in
the space of joint angles. We go a step further to deﬁne our
model entirely on the 3D bone locations.
There are a number of papers on 3D human pose estima-
1The CMU data was obtained from http://mocap.cs.cmu.edu.
The
database was created with funding from NSF EIA-0196217.
tion from 2D points observed in a static camera. All such
methods must resolve the inherent ambiguities by using ad-
ditional information. Methods vary in how this is done.
Lee and Chen [18] recover pose by pruning a binary in-
terpretation tree representing all possible body conﬁgura-
tions. Taylor [29] resolves the depth ambiguity using man-
ual intervention. Barr`on and Kakadiaris [4] use joint angle
limit constraints to resolve this ambiguity. Parameswaran
and Chellappa [21] use 3D model-based invariants to re-
cover the joint angle conﬁguration.
BenAbdelkader and
Yacoob [5] estimate limb lengths by exploiting statistical
limits on their ratios. Guan et al. [11] use a database of
body measurements and the known gender and height of a
person to predict bone lengths. Bourdev and Malik [6] esti-
mate pose from key points followed by manual adjustment.
Jiang [15] uses Taylor’s method and proposes an exemplar-
based approach to prune the hypotheses. Ramakrishna et
al. [23] propose an over-complete dictionary of actions to
estimate 3D pose. These methods do not impose joint angle
limits and can potentially estimate an invalid 3D pose.
Some of the ambiguities in monocular pose estimation
are resolved by having a sequence (but not always). Wei
and Chai [31] and Valmadre and Lucey [30] estimate 3D
pose from multiple images and exploit joint angle limits.
To apply joint angle limits, one must ﬁrst have a kinematic
tree structure in which the coordinate axes are clearly de-
ﬁned. Given only two points per bone, this is itself a se-
riously ill-posed problem requiring prior knowledge. Val-
madre and Lucey require manual resolution to ﬁx this is-
sue. Our body representation simpliﬁes this problem since
it does not represent unobserved rotations about the limbs.
We believe ours is the ﬁrst work to propose joint-angle lim-
its for a kinematic skeleton in Cartesian coordinates, where
only two points per bone are known.
In Computer Graphics, there also exist methods for hu-
man pose animation from manual 2D annotations. Grochow
et al. [10] proposed a scaled Gaussian latent variable model
as a 3D pose prior. Space complexity of their method is a
quadratic in the size of the training data. Wei and Chai [32]
and Lin et al. [19] require additional constraints, like the
distance between joints or the ground plane to be known, to
resolve ambiguity in pose estimation. Yoo et al. [33] and
Choi et al. [8] propose a sketching interface for 3D pose es-
timation. Their methods only works for the poses present in
the training data.
Discriminative approaches also exist in the literature that
do not require 2D point correspondence and directly esti-
mate human pose from 2D image measurements [3, 20, 22,
25, 34]. Discriminative approaches are generally restricted
to the viewpoints learned from training data. Though our
dataset can be used for the training of discriminative meth-
ods, it will likely require retraining for each new applica-
tion. In contrast, our prior can be easily incorporated into

generative approaches of pose estimation and tracking.
3. Pose-Conditioned Pose Prior
We observe that existing mocap datasets are not designed
to explore pose-dependent joint angle limits. Consequently,
we captured a new set of human motions performed by ﬂex-
ible people such as gymnasts and martial artists. Our cap-
ture protocol was designed to elicit a wide range of pair-
wise conﬁgurations of connected limbs in a kinematic tree
(Fig. 4 (a)). We captured two types of movements. In the
range of motion captures, participants were asked to keep
their upper-arm ﬁxed, fully ﬂex and extend their lower-arms
and then turn them inwards to outwards. This movement
was repeated for a number of horizontal and vertical pos-
tures of the upper-arm. The same procedure was adopted
for the legs. They were also asked to perform a number of
stretching exercises (Fig. 1). From this data, we estimate a
17-point kinematic skeleton and learn joint angle limits.
We represent the human pose as a concatenation of 3D
coordinates of P points X =

XT
1
· · ·
XT
P
T
∈
R3P ×1. Let δ(.) be an operator that returns the relative co-
ordinates of a joint with respect to its parent in the kinematic
skeleton. We extend δ for vectors and matrices of points.
The goal is to ﬁnd a function
isvalid(δX) : R3×N →{0, 1}N,
where N denotes the number of bones, and value 1 is re-
turned if the corresponding bone is in a valid pose and 0
otherwise. Given a kinematic skeleton we ﬁrst ﬁnd a local
coordinate system for each bone as we discuss next.
3.1. Global to Local Coordinate Conversion
In order to estimate joint-angle limits, we need to ﬁrst
ﬁnd the local coordinate systems for all the joints. We can
uniquely ﬁnd a coordinate axis in 3D with respect to two
non-parallel vectors u and v. The three coordinate axes
can be found using Gram-Schmidt on u, v, and u × v. We
propose a conversion from δX to local coordinates ˜X in
Algorithm 1. For upper-arms, upper-legs and the head, u
and v are deﬁned with the help of the torso “bones” (spine,
left/right hip, left/right shoulder) (lines 3-8). The selection
of the coordinate system for every other bone, b, is arbi-
trary and is deﬁned with the help of an arbitrary vector, a,
and the parent bone, pa(b), of b (lines 10-11). Ru is the
estimated rotation of this parent bone. Varying the values
of the input vector, a, can generate different coordinate sys-
tems and by keeping its value ﬁxed we ensure consistency
of the local coordinate system. Finally the local coordinate
axes are found using Gram-Schmidt (line 12) and the local
coordinates ˜b are computed (line 13).
Algorithm 1 Global to Local Coordinate Conversion
1: Input δX and a constant arbitrary 3D vector a.
2: for b ∈δX
3:
if ( b is an upper-arm or head)
4:
u = Left-shldr −Right-shldr;
5:
v = back-bone;
6:
else if (b is an upper-leg)
7:
u = Left-hip −Right-hip;
8:
v = back-bone;
9:
else
10:
u = pa(b);
11:
v = Rua × u;
12:
Rb = GramSchmidt(u, v, u × v);
13:
˜b = RT
bb;
14: Return ˜X = {˜b};
3.2. Learning Joint-Angle Limits
We convert the local coordinates of the upper-arms,
upper-legs and the head into spherical coordinates. Using
our dataset, we then deﬁne a binary occupancy matrix for
these bones in discretized azimuthal and polar angles, θ and
φ respectively. A bone is considered to be in a valid posi-
tion if its azimuthal and radial angles give a value 1 in the
corresponding occupancy matrix (Fig. 3(a)).
The validity of every other bone b is decided conditioned
on the position of its parent with a given θ and φ. Under this
conditioning the bone can only lie on a hemisphere or even
a smaller part of it. To exploit this we propose two types of
constraints to check the validity of b. First we ﬁnd a half-
space, bT n + d < 0, deﬁned by a separating plane with the
normal vector n and the distance to origin d. Second we
project all the instances of b in the dataset to the plane and
ﬁnd a bounding box enclosing these projections. A bone
is considered to be valid if it lies in the half-space and its
projection is inside the bounding-box (Fig. 3(b)). The sep-
arating plane is estimated by the following optimization,
min
n,d d2
subject to
AT n < −d1,
(1)
where A is a column-vise concatenation of all the instances
of b in the dataset.
Figure 3 shows a visualization of our learned joint-angle
limits. It shows that the joint angle limits for the wrist are
different for two different positions of the elbow.
3.3. Augmenting 3D Pose Sparse Representation
To represent 3D pose, a sparse representation is proposed
in [23], which uses a linear combination of basis poses

θ
φ
 A
 B
 C
 L
−120
−60
0
60
120
180
−60
−30
0
30
60
90
(a) elbow distribution
(b) cond-wrist distribution for A & B
C
D-G
H-K
(c) valid samples (points C to K)
L
M
N
(d) invalid samples (L to N)
Figure 3. Pose-dependent joint-angle limit. (a) Occupancy matrix for right elbow in azimuthal and polar angles: green/sky-blue areas
represent valid/invalid poses as observed in our capture data. (b) Given the elbow locations at A and B, the wrist can only lie on the green
regions of the spheres. These valid wrist positions project to a box on the plane separating valid and invalid poses. The plots show that the
valid poses of the wrist depend on the position of the elbow. (c) and (d) illustrate the valid (in green) and invalid (in sky-blue) elbow and
wrist positions for the corresponding selected points in plots (a) and (b).
(a) δX
(b) ex-torso
(c) Effect of prior
Figure 4. Representation and ambiguity. (a) The δ operator com-
putes relative coordinates by considering the parent as the origin.
(b) The Bayesian network for the extended-torso exploits the rel-
atively rigid locations of the joints within the torso and the cor-
relation of left and right knee. (c) The over-complete dictionary
representation allows invalid poses. Left to right: i) A 3D pose,
where the right lower-arm violates the joint-angle limits is shown.
ii) The over-complete dictionary represents this invalid 3D pose
with a small number of basis poses (20 in comparison with the
full dimensionality of 51). iii) Applying our joint-angle-limit prior
makes the invalid pose valid.
B1, B2, · · · , BK, plus the mean pose µ,
ˆX = µ +
K
X
i=1
ωiBi = µ + B∗ω,
{Bi}i∈IB∗∈B∗⊂B,
(2)
where ω is a vector of pose coefﬁcients, ωi, the matrix B∗
is a column-wise concatenation of basis poses Bi selected
with column indices IB∗from an over-complete dictionary
B. B is computed by concatenating the bases of many ac-
tions and each basis is learned using Principal Component
Analysis (PCA) on an action class. ˆX denotes the approx-
imate 3D pose aligned with the basis poses and is related
to the estimated pose, X, by the camera rotation R as,
X ≈(IP ×P ⊗R) ˆX. This sparse representation provides
better generalization than PCA [23].
We observe that despite good generalization, the sparse
representation also allows invalid poses. It is very easy to
stay in the space spanned by the basis vectors, yet move
outside the space of valid poses. Figure 4(c) shows that
a small number of basis poses can reconstruct an invalid
3D pose, whereas our joint-angle-limit prior prevents the
invalid conﬁguration. We estimate this pose by solving the
following optimization problem
min
ω ∥X −(I ⊗R) (B∗ω + µ) ∥2
2 + Cp,
(3)
where ∥.∥2 denotes the L2 norm and where Cp = 0, if
all the bones in δ ˆX are valid according to the function
isvalid(.) and inf otherwise. Deﬁning Cp this way is equiv-
alent to adding nonlinear inequality constraints using the
isvalid(.) function.
4. 3D Pose Estimation
4.1. Preliminaries
Recall that human pose is represented as a con-
catenation
of
3D
coordinates
of
P
points
X
=

XT
1
· · ·
XT
P
T ∈R3P ×1. Under a scaled ortho-
graphic camera model, the 2D coordinates of the points in
the image are given by
x = s (IP ×P ⊗R1:2) X + t ⊗1P ×1,
(4)
where x ∈R2P ×1 and s, R, and t denote the camera
scale, rotation and translation parameters, ⊗denotes the
Kronecker product and the subscript 1 : 2 gives the ﬁrst two
rows of the matrix. We can make t = 0, under the assump-
tion that the 3D centroid gets mapped to the 2D centroid and
these are the origins of the world and the camera coordinate
systems. Once the 3D pose is known, the actual value of t
can be estimated using Equation (4).
Ramakrishna et al. [23] exploit the sparse representation
in Equation (2) to ﬁnd the unknown 3D pose X. They min-
imize the following reprojection error to ﬁnd ω, IB∗, s, and
R using a greedy Orthogonal Matching Pursuit (OMP) al-
gorithm subject to an anthropometric regularization,
Cr(ω, IB∗, s, R) = ∥x−s (I ⊗R1:2) (B∗ω + µ) ∥2
2. (5)
Once ω, IB∗, and R are known, the pose is estimated as
X = (I ⊗R) (B∗ω + µ) .
(6)

4.2. The Objective Function
Our method for 3D pose estimation given 2D joint lo-
cations exploits the proposed pose prior and the fact that
bone-lengths follow known proportions. To learn the over-
completely dictionary we choose the same CMU mocap se-
quences as were selected by Ramakrishna et al. [23] and add
two further action classes “kicks” and “pantomine.” To fo-
cus on pose and not body proportions we take the approach
of Fan et al. [9] and normalize all training bodies to have
the same mean bone length and all bodies to have the same
proportions, giving every training subject the same bone
lengths. We align the poses using Procrustes alignment of
the extended-torso, deﬁned below. We learn the PCA ba-
sis on each action class and concatenate the bases to get the
over-complete dictionary. We also learn the PCA basis and
the covariance matrix for the extended-torso, which we use
for its pose estimation in the next section.
We estimate the 3D pose by minimizing,
min
ω,s,R Cr + Cp + βCl,
(7)
where β is a normalization constant and the cost Cl penal-
izes the difference between the squares of the estimated ith
bone length ∥δ( ˆXi)∥2 and the normalized mean bone length
li, Cl = PN
i=1
∥δ( ˆXi)∥2
2 −l2
i
, where |.| denotes the ab-
solute value and ˆX is estimated using Equation (2). We use
an axis-angle representation to parameterize R. We do not
optimize for the basis vectors but estimate them separately
as discussed Section 4.4.
An important consideration in minimizing the cost, given
in Equation (7) as well as the objective function in previ-
ous methods [9, 22], is the sensitivity to initialization. In
particular a good guess of the camera rotation matrix R
is required to estimate the correct 3D pose. To solve this
problem we notice that an extended-torso, consisting of the
torso, head and upper-legs exhibits less diversity of poses
than the full body and its pose estimation can give a more
accurate estimate of the camera matrix.
4.3. Pose Estimation for Extended-Torso
To estimate the 3D pose for the extended-torso, we min-
imize a cost similar to Equation (7), but instead of the full-
body, X, we only consider points in the extended torso X′.
We learn a PCA basis B′ for the extended torso with mean
µ′. Hence a basis-aligned pose is given by, ˆX′ = B′ω′+µ′.
Even the PCA-based modelling of the extended torso
is not enough to constrain its 3D pose estimation from
2D. We model a prior on ˆX′ by exploiting the inter-
dependencies between points in the form of a Bayesian net-
work (Fig. 4(b)). This network exploits the fact that the hu-
man torso is almost rigid and often left and right knees move
in correlation. Hence, the probability of a pose is given by
p

δ ˆX′
=
Y
i
p

δ ˆ
X′i|δ ˆ
X′I

,
(8)
where δ ˆ
X′I denotes a vector obtained by concatenating the
3D coordinates of the points in the conditioning set deﬁned
by the Bayesian network. Under the assumption that the
pair

δ ˆ
X′i, δ ˆ
X′I

is Gaussian distributed, we show in the
Appendix that the prior on pose can be written as a linear
constraint, Apω′ = 0, where Ap is computed using the
basis B′ and the covariance matrix of δ ˆX′. Hence, the prior
term for the extended torso becomes, C′
p = ∥Apω′∥2
2. We
estimate the pose for the extended torso by minimizing the
following objective analogous to Equation (7),
min
ω′,s,R C′
r + αC′
p + βC′
l.
(9)
We initialize the optimization by ﬁnding R and s using Pro-
crustes alignment between the 2D joint locations x′ and µ′.
We ﬁnd the solution using Quasi-Newton optimization. The
estimated ω′, s, and R are used for the basis estimation for
the full body in the next stage.
4.4. The Basis Estimation
Algorithm 2 Orthogonal Matching Pursuit (OMP)
rp0 = x −s (I ⊗R1:2) µ;
2: rd0 = δZ(Id) −s (I ⊗R3) δµ(Id);
while t < K do
4:
imax = arg max
i
 ⟨rpt, s (I ⊗R1:2) Bi⟩+
⟨rdt, s (I ⊗R3) δBi(Id)⟩

;
B∗= [B∗Bimax];
6:
ω∗= arg min
ω
 ∥x −s (I ⊗R1:2) (B∗ω + µ) ∥2
2 +
∥δZ(Id) −s (I ⊗R3) (δB∗(Id)ω + δµ(Id)) ∥2
2

;
R = arg min
R
∥x −s (I ⊗R1:2) (B∗ω∗+ µ) ∥2
2;
8:
if !isvalid (δ (B∗ω + µ))
remove Bimax and go to step 4
10:
rpt = x −s (I ⊗R1:2) (B∗ω∗+ µ);
rdt = δZ(Id)−
s (I ⊗R3) (δB∗(Id)ω∗+ δµ(Id));
12: Return {R, B∗};
In this step we estimate the basis B∗using an OMP al-
gorithm similar to Ramakrishna et al. [23]. The difference
is that here we already know the depth of a few of the bones
by exploiting the joint-angle limit constraints. Additionally,
we do not impose a hard constraint that the bone lengths
have to sum to a predeﬁned number.
Let Z denote the vector of unknown depths of all the
points in 3D pose. Given the mean bone lengths li and the

0
20
40
60
80
Belly
Neck
Face
R−hip
R−knee
R−ankle
R−foot
L−hip
L−knee
L−ankle
L−foot
R−shldr
R−elbow
R−wrist
L−shldr
L−elbow
R−wrist
alignAvg
unalignAvg
Reconstruction Error (%)
 
 
baseline
ex−torso
Final
Figure 5. Impact of the extended-torso initialization and the pro-
posed pose prior: Reconstruction error is average Euclidean dis-
tance per joint between the estimated and the ground-truth 3D pose
and is measured as a fraction of the back-bone length. Error de-
creases monotonically with the addition of each module.
estimated orthographic scale s, we estimate the absolute rel-
ative depths |δZ| using Taylor’s method [29]. Since natu-
ral human poses are not completely arbitrary, the unknown
signs of the relative depths can be estimated for some of
the bones by exploiting joint-angle limits. We generate all
signs of the bones in an arm or leg and test whether they
correspond to a valid pose using the function isvalid(δX).
The sign of a bone is taken to be positive if, according to
our prior, a negative sign is not possible in any of the com-
binations for the corresponding arm or leg. If not positive,
we do the same test in the other direction to see if the sign
can be negative. If neither is possible, we must rely on the
overcomplete basis. The indices of the depths estimated this
way are denoted as Id.
Given the 2D joint locations, x, the relative depths es-
timated above, δZ(Id), the current estimate of s and R,
OMP, given in Algorithm 2, proceeds in a greedy fashion.
The algorithm starts with a current estimate of 3D pose as
µ and computes the initial residual for the 2D projection
and known relative depths (line 1,2). At each iteration a ba-
sis vector from B is chosen and added to B∗that is most
aligned with the residual under the current estimate of rota-
tion (line 4,5). Then given B∗, the pose coefﬁcients ω∗and
camera rotations R are re-estimated (line 6,7). We remove
the basis vector if it makes the resulting pose invalid and
consider the basis vector with the next highest dot product
(line 8,9). The residual is updated using B∗, ω∗, and the
new estimate of R (line 10,11). The algorithm terminates
when B∗has reached a predeﬁned size.
Finally, the estimated B∗, ω∗, and R are used to initial-
ize the optimization in Equation (7).
5. Experiments
We compare the pose-prior learned from our dataset with
the same prior learned from the CMU dataset. We classify
0
20
40
60
80
100
Belly
Neck
Face
R−shldr
R−elbow
R−wrist
L−shldr
L−elbow
R−wrist
R−hip
R−knee
R−ankle
R−foot
L−hip
L−knee
L−ankle
L−foot
Reconstruction Error (%)
 
 
Ramakrisnat et al.
Fan et al.
Our Method
Figure 6. The proposed method gives consistently smaller recon-
struction error in comparison with the other two methods.
0
5
10
15
20
20
40
60
80
100
120
Reconstruction Error (%)
Noise σ (%)
 
 
Ramakrisnat et al.
Fan et al.
Our Method
−5
0
5
Figure 7. The proposed method is robust to a fairly large range
of noise in comparison with the previous methods. Noise σ is
proportional to the back-bone length. A sample input frame at σ =
20% and our estimated 3D pose with error=50% is also shown in
two views (gray: ground-truth, colored: estimated).
all the poses in our dataset as valid or invalid using the prior
learned from CMU. We ﬁnd that out of a 110 minutes of
data about 12% is not explained by the CMU-based prior.
This suggests that the CMU dataset does not cover the full
range of human motions. A similar experiment shows that
out of 9.5 hours of CMU data about 8% is not explained by
our prior. A closer investigation reveals that CMU contains
many mislabeled markers. This inﬂates the space of valid
CMU poses to include invalid ones. Removing the invalid
poses would likely increase the percentage of our poses that
are not explained and would decrease the amount of CMU
data unexplained by our prior.
We quantitatively evaluate our method using all CMU
mocap sequences of four actors (103, 111, 124, and 125)
for a total of 69 sequences.
We create two sets of syn-
thetic images, called testset1 and testset2, by randomly se-
lecting 3000 and 10000 frames from these sequences and
projecting them using random camera viewpoints. We re-
port reconstruction error per joint as the average Euclidean
distance between the estimated and the ground-truth pose.
Like previous methods [9, 23] we Procrustes align the esti-
mated 3D pose with the ground-truth to compute the error.
To ﬁx arbitrary scale, we divide the ground by back-bone
length and Procrustes align this with the estimated pose. We
also evaluate the camera matrix estimation.
We ﬁrst evaluate the impact of the extended-torso ini-
tialization and joint-angle prior in the proposed method on

Our Method
Ramakrishna et al. [23]
Fan et al. [9]
Figure 8. Real results with manual annotation. We demonstrate substantial improvement over the previous methods. The proposed method
gives an anthropometrically valid interpretation of 2D joint locations whereas the previous methods often give invalid 3D poses.
Figure 9. Real results with automatic part-based detections [16] on the Leeds sports pose dataset on a few frames. Despite the outliers in
detections, our method gives valid 3D pose interpretations. Please note that feet were not detected in the images but with the help of our
pose prior their 3D location is estimated.
testset1. We start with a baseline consisting of just the pro-
jected matching pursuit algorithm and test its accuracy. We
initialize this by ﬁnding R and s by Procrustes alignment
between x and µ. Then we include the initialization using
pose estimation for the extended-torso and the ﬁnal joint
optimization to enforce length constraints. Finally, we in-
clude the depth estimation using joint-angle limits and the
proposed pose prior in the joint optimization. In Fig. 5 we
report the mean reconstruction errors per joint for this ex-
periment. The results show a monotonic decrease in error
with the addition of each of these modules. We also report
the overall mean reconstruction error with and without Pro-
crustes alignment. For the later case we multiply the camera
rotation with the 3D pose and adopt a canonical camera con-
vention. Observing that both the errors are roughly equal we
conclude that the estimated camera matrices are correct.
Next we compare the accuracy of our method against the
previous methods on testset2. The source code for the pre-
vious methods were kindly provided by the authors. Note
that the method by Fan et al. is customized for a few classes
of actions, including walking, running, jumping, boxing,
and climbing and its accuracy is expected to degrade on
other types of actions. Figure 6 shows that the proposed
method outperforms the other two methods. In Fig. 7 we
test the sensitivity of our algorithm against Gaussian noise
and compare it against the methods by Ramakrishna et al.
[23] and Fan et al. [9]. We add noise proportional to the
backbone length in 3D, project the noisy points using ran-
dom camera matrices and report our pose estimation accu-
racy. The results demonstrate that the proposed method is
signiﬁcantly more robust than the previous methods. Our
experiments show that the proposed method gives a small
reprojection error and an anthropometrically valid 3D pose
interpretation, whereas the previous methods often estimate
an invalid 3D pose. A further investigation reveals that the
reconstruction error in canonical camera convention for the
previous methods is signiﬁcantly worse than the one with
Procrustes alignment (55% and 44% vs. 143% and 145%
respectively), whereas for our method the errors are not sig-
niﬁcantly different (34% vs. 45%). This implies that an im-
portant reason for the failure of previous methods is the in-
correct estimation of the camera matrix. This highlights the

contribution of extended-torso initialization.
It is important to mention the inherent ambiguities in 3D
pose estimation (see Fig. 2), which imply that given 2D
point locations, a correct pose estimation can never be in-
sured and only a probable 3D pose can be estimated. Re-
sults show that the proposed method satisﬁes this criterion.
Figure 8 shows results on real images with manual an-
notations of joints and compares them with the previous
methods by showing the 3D pose in two arbitrary views.
Again the results show that our method gives a valid 3D
pose whereas the previous methods often do not. Figure 9
shows results with automatic part-based detections [16] on
the Leeds sports pose dataset on a few frames.
Results
show that despite signiﬁcant noise in detection, the pro-
posed method is able to recover a valid 3D pose. For more
results please see supplementary material [1].
6. Conclusion
We propose pose-conditioned joint angle limits and for-
mulate a prior for human pose. We believe that this is the
ﬁrst general prior to consider pose dependency of joint lim-
its. We demonstrate that this prior restricts invalid poses in
2D-to-3D human pose reconstruction. Additionally we pro-
vide a new algorithm for estimating 3D pose that exploits
our prior. Our method signiﬁcantly outperforms the current
state of the art methods both quantitatively and qualitatively.
Our prior and the optimization framework can be applied
to many problems in human pose estimation beyond the
application described here. In future we will consider de-
pendencies of siblings in the kinematic tree on joint-angle
limits. We are also working on temporal models of 2D-to-
3D pose estimation that further reduce ambiguities. Future
work should also consider the temporal dependency of joint
limits since, during motion, the body can reach states that
may not be possible statically.
7. Appendix
We model the pose prior on the extended-torso as the
following Bayesian network,
p

ˆX

=
Y
i
p

δ ˆ
X′i|δ ˆ
X′I

,
(10)
where I denotes the indices of the joints in the conditioning
set deﬁned by the Bayesian network shown in Fig. 4(b) and
δ ˆ
X′I is a vector obtained by concatenating their 3D coor-
dinates. We consider the combined Gaussian distribution of
a joint i and its conditioning set as,
 δ ˆ
X′i
δ ˆ
X′I

∼N
 δµ′
i
δµ′
I

,
 Σ′
ii
Σ′
iI
Σ′
Ii
Σ′
II

, (11)
where the relative pose δ ˆX′ satisﬁes,
δ ˆX′ = δB′ω′ + δµ′.
(12)
Given Equation (11) the conditional distribution can be
written as,

δ ˆ
X′i|δ ˆ
X′I = a

∼N

δµ′
i, Σ
′
ii

, where
δµ′
i = δµ′
i + Σ′
iIΣ′−1
II(a −δµ′
I),
Σ
′
ii = Σ′
ii −Σ′
iIΣ′−1
IIΣ′
Ii.
(13)
The above pose prior can be combined with Equation (12)
by noticing that (a −δµ′
I) = δB′
Iω′, where δB′
I consists
of the rows from δB′ corresponding to the points I. Using
this relation the complete vector δµ′ can be estimated as,
δµ′ −δµ′ = Gω′,
(14)
where G is formed by stacking the matrices Σ′iIΣ′−1
IIδB′
I
for all i. Equation (14) provides the mean 3D pose under
the Gaussian network prior. The covariance of pose Σ
′ is
formed by stacking all conditional covariances Σ
′
ii from
Equation (13). This prior on 3D pose is used to formulate a
prior on ω′ ∼N
 µω′, Σω′
using Equation (12) as,
µω′ = δB′†Gω′,
and
Σω′ = δB′†Σ
′δB′†T ,
(15)
where the superscript † denotes MoorePenrose pseudoin-
verse. This prior can be used to formulate a MAP estimate
of ω′. The likelihood equation for ω′ is the following,
x′ = s (I ⊗R1:2) (B′ω′ + µ′) .
(16)
The above becomes linear if the camera matrix and ortho-
graphic scale-factor are known and can be written as a ma-
trix multiplication, Aω′ = b. Therefore the likelihood dis-
tribution can be written as b|ω′ ∼N (Aω′, α′I), where α′
is the variance of the noise. Using this the MAP estimate
of ω can be found by minimizing the sum of Mahalanobis
distances of both the prior and likelihood distributions,
c(ω′) = 1
α∥Aω′ −b∥2 + (ω′ −µω′)T Σ
−1
ω′ (ω′ −µω′) .
By taking the partial derivatives of c with respect to ω′, a
linear system of equations can be made and the MAP esti-
mate of ω′ can be found as the following,

AT A + αDT Σ
−1
ω′ D

ω′ = AT b,
(17)
where D = I−δB′†G. Solving this linear system is equiv-
alent to solving two sets of equations, Aω′ = b, and
√αApω′ = 0,
(18)
where AP is a Cholesky decomposition of DT Σ
−1
ω′ D. We
use Equation (18) to add a prior for the extended-torso.

8. Acknowledgements
We thank Andrea Keller, Sophie Lupas, Stephan Streu-
ber, and Naureen Mahmood for joint-limit dataset prepara-
tion. We beneﬁted from discussions with Jonathan Taylor,
Peter Gehler, Gerard Pons-Moll, Varun Jampani, and Kashif
Murtza. We also thank Varun Ramakrishna and Xiaochuan
Fan for providing us their source code.
References
[1] http://poseprior.is.tue.mpg.de/.
[2] U. S. N. Aeronautics and S. Administration.
NASA-STD-
3000: Man-systems integration standards. Number v. 3 in
NASA-STD. National Aeronautics and Space Administra-
tion, 1995.
[3] M. Andriluka, S. Roth, and B. Schiele. Monocular 3D pose
estimation and tracking by detection. In Computer Vision
and Pettern Recognition, pages 623–630, 2010.
[4] C. Barr`on and I. Kakadiaris. Estimating anthropometry and
pose from a single uncalibrated image. Computer Vision and
Image Understanding, 81(3):269–284, March 2001.
[5] C. BenAbdelkader and Y. Yacoob.
Statistical estimation
of human anthropometry from a single uncalibrated im-
age. In Methods, Applications, and Challenges in Computer-
assisted Criminal Investigations, Studies in Computational
Intelligence. Springer-Verlag, 2008.
[6] L. Bourdev and J. Malik.
Poselets: Body part detectors
trained using 3D human pose annotations. In International
Conference on Computer Vision, pages 1365–1372, Sept.
2009.
[7] J. Chen, S. Nie, and Q. Ji. Data-free prior model for upper
body pose estimation and tracking. IEEE Trans. Image Proc.,
22(12):4627–4639, Dec. 2013.
[8] M. G. Choi, K. Yang, T. Igarashi, J. Mitani, and J. Lee. Re-
trieval and visualization of human motion data via stick ﬁg-
ures. Computer Graphics Forum, 31(7):2057–2065, 2012.
[9] X. Fan, K. Zheng, Y. Zhou, and S. Wang.
Pose locality
constrained representation for 3d human pose reconstruction.
In Computer Vision–ECCV 2014, pages 174–188. Springer,
2014.
[10] K. Grochow, S. L. Martin, A. Hertzmann, and Z. Popovi´c.
Style-based inverse kinematics.
ACM Transactions on
Graphics (TOG), 23(3):522–531, 2004.
[11] P. Guan, A. Weiss, A. Balan, and M. J. Black. Estimating
human shape and pose from a single image. In Int. Conf. on
Computer Vision, ICCV, pages 1381–1388, Sept. 2009.
[12] H. Hatze. A three-dimensional multivariate model of pas-
sive human joint torques and articular boundaries. Clinical
Biomechanics, 12(2):128–135, 1997.
[13] S. Hauberg, S. Sommer, and K. Pedersen. Gaussian-like spa-
tial priors for articulated tracking. In K. Daniilidis, P. Mara-
gos, and N. Paragios, editors, Computer Vision ECCV 2010,
volume 6311 of Lecture Notes in Computer Science, pages
425–437. Springer Berlin Heidelberg, 2010.
[14] L. Herda, R. Urtasun, and P. Fua. Hierarchical implicit sur-
face joint limits for human body tracking. Computer Vision
and Image Understanding, 99(2):189–209, 2005.
[15] H. Jiang. 3D human pose reconstruction using millions of
exemplars. In Pattern Recognition (ICPR), 2010 20th Inter-
national Conference on, pages 1674–1677. IEEE, 2010.
[16] M. Kiefel and P. Gehler. Human pose estimation with ﬁelds
of parts. In D. Fleet, T. Pajdla, B. Schiele, and T. Tuytelaars,
editors, Computer Vision – ECCV 2014, volume 8693 of Lec-
ture Notes in Computer Science, pages 331–346. Springer
International Publishing, Sept. 2014.
[17] T. Kodek and M. Munich.
Identifying shoulder and el-
bow passive moments and muscle contributions. In IEEE
Int. Conf. on Intelligent Robots and Systems, volume 2, pages
1391–1396, 2002.
[18] H. J. Lee and Z. Chen. Determination of 3D human body
postures from a single view. Computer Vision Graphics and
Image Processing, 30(2):148–168, 1985.
[19] J. Lin, T. Igarashi, J. Mitani, M. Liao, and Y. He. A sketch-
ing interface for sitting pose design in the virtual environ-
ment. Visualization and Computer Graphics, IEEE Transac-
tions on, 18(11):1979–1991, 2012.
[20] G. Mori and J. Malik. Recovering 3D human body conﬁgu-
rations using shape contexts. Pattern Analysis and Machine
Intelligence, 28(7):1052–1062, 2006.
[21] V. Parameswaran and R. Chellappa. View independent hu-
man body pose estimation from a single perspective image.
In Computer Vision and Pettern Recognition, pages 16–22,
2004.
[22] I. Radwan, A. Dhall, and R. Goecke.
Monocular image
3D human pose estimation under self-occlusion. In Inter-
national Conference on Computer Vision, pages 1888–1895,
2013.
[23] V. Ramakrishna, T. Kanade, and Y. Sheikh. Reconstructing
3D human pose from 2D image landmarks. European Con-
ference on Computer Vision, pages 573–586, 2012.
[24] J. M. Rehg, D. D. Morris, and T. Kanade. Ambiguities in
visual tracking of articulated objects using two-and three-
dimensional models. The International Journal of Robotics
Research, 22(6):393–418, 2003.
[25] G. Rogez, J. Rihan, C. Orrite-Uru˜nuela, and P. H. Torr. Fast
human pose detection using randomized hierarchical cas-
cades of rejectors. International Journal of Computer Vision,
99(1):25–52, 2012.
[26] M. Sch¨unke, E. Schulte, and U. Schumacher. Prometheus:
Allgemeine Anatomie und Bewegungssystem : LernAtlas der
Anatomie.
Prometheus Lern