# On the Continuity of Rotation Representations in Neural Networks

> 2019 · id: W2949924544 · arXiv: 1812.07035 · pdf: https://arxiv.org/pdf/1812.07035 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
In neural networks, it is often desirable to work with var-
ious representations of the same space. For example, 3D
rotations can be represented with quaternions or Euler an-
gles. In this paper, we advance a deﬁnition of a continuous
representation, which can be helpful for training deep neu-
ral networks. We relate this to topological concepts such as
homeomorphism and embedding. We then investigate what
are continuous and discontinuous representations for 2D,
3D, and n-dimensional rotations. We demonstrate that for
3D rotations, all representations are discontinuous in the
real Euclidean spaces of four or fewer dimensions. Thus,
widely used representations such as quaternions and Eu-
ler angles are discontinuous and difﬁcult for neural net-
works to learn. We show that the 3D rotations have con-
tinuous representations in 5D and 6D, which are more suit-
able for learning.
We also present continuous represen-
tations for the general case of the n dimensional rotation
group SO(n). While our main focus is on rotations, we also
show that our constructions apply to other groups such as
the orthogonal group and similarity transforms. We ﬁnally
present empirical results, which show that our continuous
rotation representations outperform discontinuous ones for
several practical problems in graphics and vision, includ-
ing a simple autoencoder sanity test, a rotation estimator
for 3D point clouds, and an inverse kinematics solver for
3D human poses.

## introduction
Recently, there has been an increasing number of appli-
cations in graphics and vision, where deep neural networks
are used to perform regressions on rotations. This has been
done for tasks such as pose estimation from images [13, 32]
and from point clouds [15], structure from motion [30], and
skeleton motion synthesis, which generates the rotations of
∗Authors have equal contribution.
joints in skeletons [31]. Many of these works represent 3D
rotations using 3D or 4D representations such as quater-
nions, axis-angles, or Euler angles.
However, for 3D rotations, we found that 3D and 4D rep-
resentations are not ideal for network regression, when the
full rotation space is required. Empirically, the converged
networks still produce large errors at certain rotation an-
gles. We believe that this actually points to deeper topolog-
ical problems related to the continuity in the rotation rep-
resentations. Informally, all else being equal, discontinuous
representations should in many cases be “harder” to approx-
imate by neural networks than continuous ones. Theoreti-
cal results suggest that functions that are smoother [34] or
have stronger continuity properties such as in the modulus
of continuity [33, 10] have lower approximation error for a
given number of neurons.
Based on this insight, we ﬁrst present in Section 3 our
deﬁnition of the continuity of representation in neural net-
works. We illustrate this deﬁnition based on a simple exam-
ple of 2D rotations. We then connect it to key topological
concepts such as homeomorphism and embedding.
Next, we present in Section 4 a theoretical analysis of the
continuity of rotation representations. We ﬁrst investigate in
Section 4.1 some discontinuous representations, such as Eu-
ler angle and quaternion representations. We show that for
3D rotations, all representations are discontinuous in four or
fewer dimensional real Euclidean space with the Euclidean
topology. We then investigate in Section 4.2 some continu-
ous rotation representations. For the n dimensional rotation
group SO(n), we present a continuous n2 −n dimensional
representation. We additionally present an option to reduce
the dimensionality of this representation by an additional 1
to n −2 dimensions in a continuous way. We show that
these allow us to represent 3D rotations continuously in 6D
and 5D. While we focus on rotations, we show how our con-
tinuous representations can also apply to other groups such
as orthogonal groups O(n) and similarity transforms.
Finally, in Section 5 we test our ideas empirically. We
conduct experiments on 3D rotations and show that our
6D and 5D continuous representations always outperform
1
arXiv:1812.07035v4  [cs.LG]  8 Jun 2020

the discontinuous ones for several tasks, including a rota-
tion autoencoder “sanity test,” rotation estimation for 3D
point clouds, and 3D human pose inverse kinematics learn-
ing. We note that in our rotation autoencoder experiments,
discontinuous representations can have up to 6 to 14 times
higher mean errors than continuous representations. Fur-
thermore they tend to converge much slower while still pro-
ducing large errors over 170◦at certain rotation angles even
after convergence, which we believe are due to the discon-
tinuities being harder to ﬁt.
This phenomenon can also
be observed in the experiments on different rotation repre-
sentations for homeomorphic variational auto-encoding in
Falorsi et al. [14], and in practical applications, such as 6D
object pose estimation in Xiang et al. [32].
We also show that one can perform direct regression on
3x3 rotation matrices. Empirically this approach introduces
larger errors than our 6D representation as shown in Sec-
tion 5.2. Additionally, for some applications such as inverse
and forward kinematics, it may be important for the network
itself to produce orthogonal matrices. We therefore require
an orthogonalization procedure in the network. In particu-
lar, if we use a Gram-Schmidt orthogonalization, we then
effectively end up with our 6D representation.
Our contributions are: 1) a deﬁnition of continuity for
rotation representations, which is suitable for neural net-
works; 2) an analysis of discontinuous and continuous rep-
resentations for 2D, 3D, and n-D rotations; 3) new formulas
for continuous representations of SO(3) and SO(n); 4) em-
pirical results supporting our theoretical views and that our
continuous representations are more suitable for learning.

## related_work
In this section, we will ﬁrst establish some context for
our work in terms of neural network approximation theory.
Next, we discuss related works that investigate the continu-
ity properties of different rotation representations. Finally,
we will report the types of rotation representations used in
previous learning tasks and their performance.
Neural network approximation theory. We review a
brief sampling of results from neural network approxima-
tion theory. Hornik [18] showed that neural networks can
approximate functions in the Lp space to arbitrary accuracy
if the Lp norm is used. Barron et al. [6] showed that if a
function has certain properties in its Fourier transform, then
at most O(ϵ−2) neurons are needed to obtain an order of ap-
proximation ϵ. Chapter 6.4.1 of LeCun et al. [24] provides
a more thorough overview of such results. We note that
results for continuous functions indicate that functions that
have better smoothness properties can have lower approxi-
mation error for a particular number of neurons [33, 10, 34].
For discontinuous functions, Llanas et al. [25] showed that
a real and piecewise continuous function can be approx-
imated in an almost uniform way.
However, Llanas et
al. [25] also noted that piecewise continuous functions when
trained with gradient descent methods require many neu-
rons and training iterations, and yet do not give very good
results. These results suggest that continuous rotation rep-
resentations might perform better in practice.
Continuity for rotations. Grassia et al. [16] pointed out
that Euler angles and quaternions are not suitable for ori-
entation differentiation and integration operations and pro-
posed exponential map as a more robust rotation represen-
tation. Saxena et al. [29] observed that the Euler angles and
quaternions cause learning problems due to discontinuities.
However, they did not propose general rotation representa-
tions other than direct regression of 3x3 matrices, since they
focus on learning representations for objects with speciﬁc
symmetries.
Neural networks for 3D shape pose estimation. Deep
networks have been applied to estimate the 6D poses of ob-
ject instances from RGB images, depth maps or scanned
point clouds.
Instead of directly predicting 3x3 matri-
ces that may not correspond to valid rotations, they typ-
ically use more compact rotation representations such as
quaternion [32, 22, 21] or axis-angle [30, 15, 13].
In
PoseCNN [32], the authors reported a high percentage of
errors between 90◦and 180◦, and suggested that this is
mainly caused by the rotation ambiguity for some symmet-
ric shapes in the test set. However, as illustrated in their
paper, the proportion of errors between 90◦to 180◦is still
high even for non-symmetric shapes. In this paper, we ar-
gue that discontinuity in these representations could be one
cause of such errors.
Neural networks for inverse kinematics. Recently, re-
searchers have been interested in training neural networks
to solve inverse kinematics equations. This is because such
networks are faster than traditional methods and differen-
tiable so that they can be used in more complex learning
tasks such as motion re-targeting [31] and video-based hu-
man pose estimation [20]. Most of these works represented
rotations using quaternions or axis-angle [19, 20]. Some
works also used other 3D representations such as Euler an-
gles and Lie algebra [20, 35], and penalized the joint posi-
tion errors. Csiszar et al. [11] designed networks to output
the sine and cosine of the Euler angles for solving the in-
verse kinematics problems in robotic control. Euler angle
representations are discontinuous for SO(3) and can result
in large regression errors as shown in the empirical test in
Section 5. However, those authors limited the rotation an-
gles to be within a certain range, which avoided the discon-
tinuity points and thus achieved very low joint alignment
errors in their test. However, many real-world tasks require
the networks to be able to output the full range of rotations.
In such cases, continuous rotation representations will be a
better choice.
3. Deﬁnition of Continuous Representation
In this section, we begin by deﬁning the terminology we
will use in the paper. Next, we analyze a simple motivat-
ing example of 2D rotations. This allows us to develop our
general deﬁnition of continuity of representation in neural
networks. We then explain how this deﬁnition of continuity
is related to concepts in topology.
Terminology. To denote a matrix, we typically use M,
and Mij refers to its (i, j) entry. We use the term SO(n) to
denote the special orthogonal group, the space of n dimen-
2

Mapping g
Connected Set
of Rotations in S1
Original Space
Disconnected Set of Angular 
Representations in [0, 2π]
0
2π
Representation Space
Figure 1. A simple 2D example, which motivates our deﬁnition of
continuity of representation. See Section 3 for details.
Representation 
Space 
R
Input 
signal
Original 
Space
X
Mapping f
Neural 
Network
Mapping g
Figure 2. Our deﬁnition of continuous representation, as well as
how it can apply in a neural network. See the body for details.
sional rotations. This group is deﬁned on the set of n × n
real matrices with MM T = M T M = I and det(M) = 1.
The group operation is multiplication, which results in the
concatenation of rotations. We denote the n dimensional
unit sphere as Sn = {x ∈Rn+1 : ||x|| = 1}.
Motivating example: 2D rotations. We now consider
the representation of 2D rotations. For any 2D rotation M ∈
SO(2), we can also express the matrix as:
M =

cos(θ)
−sin(θ)
sin(θ)
cos(θ)

(1)
We can represent any rotation matrix M ∈SO(2) by
choosing θ ∈R, where R is a suitable set of angles, for ex-
ample, R = [0, 2π]. However, this particular representation
intuitively has a problem with continuity. The problem is
that if we deﬁne a mapping g from the original space SO(2)
to the angular representation space R, then this mapping is
discontinuous. In particular, the limit of g at the identity ma-
trix, which represents zero rotation, is undeﬁned: one direc-
tional limit gives an angle of 0 and the other gives 2π. We
depict this problem visually in Figure 1. On the right, we
visualize a connected set of rotations C ⊂SO(2) by visual-
izing their ﬁrst column vector [cos(θ), sin(θ)]T on the unit
sphere S1. On the left, after mapping them through g, we
see that the angles are disconnected. In particular, we say
that this representation is discontinuous because the map-
ping g from the original space to the representation space
is discontinuous. We argue that these kind of discontinu-
ous representations can be harder for neural networks to ﬁt.
Contrarily, if we represent the 2D rotation M ∈SO(2) by
its ﬁrst column vector [cos(θ), sin(θ)]T , then the represen-
tation would be continuous.
Continuous representation: We can now deﬁne what
we consider a continuous representation. We illustrate our
deﬁnitions graphically in Figure 2. Let R be a subset of
a real vector space equipped with the Euclidean topology.
We call R the representation space: in our context, a neu-
ral network produces an intermediate representation in R.
This neural network is depicted on the left side of Figure 2.
We will come back to this neural network shortly. Let X
be a compact topological space. We call X the original
space. In our context, any intermediate representation in
R produced by the network can be mapped into the orig-
inal space X. Deﬁne the mapping to the original space
f : R →X, and the mapping to the representation space
g : X →R. We say (f, g) is a representation if for every
x ∈X, f(g(x)) = x, that is, f is a left inverse of g. We say
the representation is continuous if g is continuous.
Connection with neural networks: We now return to
the neural network on the left side of Figure 2. We imag-
ine that inference runs from left to right. Thus, the neural
network accepts some input signals on its left hand side,
outputs a representation in R, and then passes this repre-
sentation through the mapping f to get an element of the
original space X. Note that in our context, the mapping f
is implemented as a mathematical function that is used as
part of the forward pass of the network at both training and
inference time. Typically, at training time, we might im-
pose losses on the original space X. We now describe the
intuition behind why we ask that g be continuous. Suppose
that we have some connected set C in the original space,
such as the one shown on the right side of Figure 1. Then if
we map C into representation space R, and g is continuous,
then the set g(C) will remain connected. Thus, if we have
continuous training data, then this will effectively create a
continuous training signal for the neural network. Contrar-
ily, if g is not continuous, as shown in Figure 1, then a con-
nected set in the original space may become disconnected
in the representation space. This could create a discontinu-
ous training signal for the 

## conclusion
We investigated the use of neural networks to approx-
imate the mappings between various rotation representa-
tions. We found empirically that neural networks can better
ﬁt continuous representations. For 3D rotations, the com-
monly used quaternion and Euler angle representations have
discontinuities and can cause problems during learning. We
present continuous 5D and 6D rotation representations and
demonstrate their advantages using an auto-encoder sanity
test, as well as real world applications, such as 3D pose es-
timation and human inverse kinematics.
7. Acknowledgements
We thank Noam Aigerman, Kee Yuen Lam, and Sitao
Xiang for fruitful discussions; Fangjian Guo, Xinchen Yan,
and Haoqi Li for helping with the presentation. This re-
search was conducted at USC and Adobe and was funded
by in part by the ONR YIP grant N00014-17-S-FO14, the
CONIX Research Center, one of six centers in JUMP, a
Semiconductor Research Corporation (SRC) program spon-
sored by DARPA, the Andrew and Erna Viterbi Early Ca-
reer Chair, the U.S. Army Research Laboratory (ARL)
8

under contract number W911NF-14-D-0005, Adobe, and
Sony. This project was not funded by Pinscreen, nor has
it been conducted at Pinscreen or by anyone else afﬁliated
with Pinscreen. The content of the information does not
necessarily reﬂect the position or the policy of the Govern-
ment, and no ofﬁcial endorsement should be inferred.