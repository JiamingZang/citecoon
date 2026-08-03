# Confronting Ambiguity in 6D Object Pose Estimation via Score-Based Diffusion on SE(3)

> 2024 · id: W4402816866 · arXiv: 2305.15873 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Addressing pose ambiguity in 6D object pose estimation
from single RGB images presents a significant challenge,
particularly due to object symmetries or occlusions. In re-
sponse, we introduce a novel score-based diffusion method
applied to the SE(3) group, marking the first application of
diffusion models to SE(3) within the image domain, specif-
ically tailored for pose estimation tasks. Extensive evalu-
ations demonstrate the method’s efficacy in handling pose
ambiguity, mitigating perspective-induced ambiguity, and
showcasing the robustness of our surrogate Stein score for-
mulation on SE(3). This formulation not only improves the
convergence of denoising process but also enhances com-
putational efficiency. Thus, we pioneer a promising strategy
for 6D object pose estimation.

## introduction
Estimating the six degrees of freedom (DoF) pose of ob-
jects from a single RGB image remains a formidable task,
primarily due to the presence of ambiguity induced by
symmetric objects and occlusions. Symmetric objects ex-
hibit identical visual appearance from multiple viewpoints,
whereas occlusions arise when key aspects of an object are
concealed either by another object or its own structure. This
can complicate the determination of its shape and orienta-
tion. Pose ambiguity presents a unique challenge as it trans-
forms the direct one-to-one correspondence between an im-
age and its associated object pose into a complex one-to-
many scenario, which can potentially leads to significant
performance degradation for methods reliant on one-to-one
correspondence. Despite extensive exploration in the prior
object pose estimation literature [10, 19, 21, 39, 41], pose
ambiguity still remains a persisting and unresolved issue.
Recent advancements in pose regression have introduced
the use of symmetry-aware annotations to improve pose es-
timation accuracy [39, 44, 60, 64]. These methods typically
employ symmetry-aware losses that can tackle the pose am-
Ground Truth
Figure 1. Visualization of the denoising process of our score-based
diffusion method on SE(3) for 6DoF pose estimation.
biguity problem. The efficacy of these losses, nevertheless,
depend on the provision of symmetry annotations, which
can be particularly challenging to obtain for objects with in-
tricate shapes or under occlusion. An example is a texture-
less cup, where the true orientation becomes ambiguous if
the handle is not visible.The manual labor and time required
to annotate the equivalent views of each object under such
circumstances is impractical.
Several contemporary studies have sought to eliminate
the reliance on symmetry annotations by treating ‘equiv-
alent poses’ as a multi-modal distribution, reframing the
original pose estimation problem as a density estima-
tion problem.
Methods such as Implicit-PDF [41] and
HyperPose-PDF [23] leverage neural networks to implic-
itly characterize the non-parametric density on the rota-
tion manifold SO(3). While these advances are notewor-
thy, they also introduce new complexities. For instance,
the computation during training requires exhaustive sam-
pling across the whole SO(3) space. Moreover, the accu-
racy of inference is dependent on the resolution of the grid
search, which necessitates a significant amount of grid sam-
pling. These computational limitations are magnified when
extending to larger spaces such as SE(3) due to the sub-
stantial memory requirements.
Recognizing these challenges, the research community
is pivoting towards diffusion models (DMs) [16, 56–58],
which are effective in handling multi-modal distributions.
Their effectiveness lies in the iterative sampling process,
which incorporates noises and enables a more focus ex-
ploration of the pose space while reducing computational
arXiv:2305.15873v2  [cs.CV]  8 Apr 2024

demands. As diffusion models refrain from explicit den-
sity estimation, this property enables them to handle large
spaces and high-dimensional distributions. In prior endeav-
ors, the authors in [28, 33] applied the denoising diffusion
probabilistic model (DDPM) [16] and score-based genera-
tive model (SGM) [58] to the SO(3) rotation manifold, ef-
fectively recovering unknown densities on the SO(3) space.
On the other hand, other research efforts [61, 71] have ex-
tended the application of diffusion models to the more com-
plex SE(3) space, which enlightens the potential applica-
bility of diffusion models in object pose estimation tasks.
In light of the above motivations, we introduce a novel
approach that applies diffusion models to the SE(3) group
for object pose estimation tasks, specifically aimed at ad-
dressing the pose ambiguity problem. This method draws
its inspiration from the correlation observed between rota-
tion and translation distributions, a phenomenon often re-
sultant from the perspective effect inherent in image projec-
tion. We propose that by jointly estimating the distribution
of rotation and translation on SE(3), we may secure more
accurate and reliable results as shown in Fig. 1. To the best
of our knowledge, this is the first work to apply diffusion
models to SE(3) within the context of image space. To sub-
stantiate our approach, we have developed a new synthetic
dataset, called SYMSOL-T, based on the original SYMSOL
dataset [41]. It enhances the original dataset with randomly
sampled translations, offering a more rigorous testbed to
evaluate our method’s effectiveness in capturing the joint
density of object rotations and translations.
Following the motivations discussed above, we have ex-
tensively evaluated our SE(3) diffusion model using the
synthetic SYMSOL-T dataset and a real-world T-LESS [20]
dataset. The experimental results affirm the model’s com-
petence in handling SE(3), which successfully addresses
the pose ambiguity problem in 6D object pose estimation.
Moreover, the SE(3) diffusion model has proven effective
in enhancing rotation estimation accuracy and robustness.
Importantly, the surrogate Stein score formulation we pro-
pose on SE(3) exhibits improved convergence in the de-
noising process compared to the score calculated via auto-
matic differentiation. This not only highlights the robust-
ness of our method, but also demonstrates its potential to
handle complex dynamics in object pose estimation tasks.

## method
Given an RGB image I that displays the object of interest,
our goal is to estimate the 6D object poses X = (R, T) ∈
SE(3), which represent the transformation from the cam-
era frame to the object. This estimation involves sampling
poses from a conditional distribution X ∼p(X|I), which
captures the inherent pose uncertainty of the object depict
in I. To facilitate this process, our method employs a score-
based generative model on SE(3) to recover this underlying
distribution. Poses are then sampled via a reverse process
that gradually refines noisy pose hypotheses ˜X ∼p( ˜X)
drawn from a known prior distribution p( ˜X), specifically
a Gaussian distribution on SE(3). Both the forward and
reverse processes are performed on Lie groups and lever-
age the associated group operations. It is important to note
that our approach does not utilize 3D models of the objects
or symmetry annotations during either the training or in-
ference phases, instead relying exclusively on RGB images
and the associated ground truth (GT) poses for training.
4.1. Score-Based Pose Diffusion on a Lie Group
To apply score-based generative modeling to a Lie group G,
we first establish a perturbation kernel on G that conforms
to the Gaussian distribution [8, 54]. The kernel is given by:
pΣ(Y |X) := NG(Y ; X, Σ)
≜
1
ζ(Σ) exp

−1
2 Log(X−1Y )⊤Σ−1Log(X−1Y )

,
(3)
where Σ is the covariance matrix with diagonal entries pop-
ulated by σ for representing the scale of the perturbation,
ζ(Σ) is the normalizing constant, and X, Y ∈G denote the
group elements. The score on G then corresponds to the gra-
dient of the log-density of the data distribution with respect
to the group element Y . It can be formulated as follows:
∇Y log pΣ(Y |X) = −J−⊤
r
(Log(X−1Y ))Σ−1Log(X−1Y ).
(4)
This term can be expressed in closed form if the inverse
of the right-Jacobian J−1
r
on G exists in a closed form.
Nevertheless, an alternative approach suggested by the au-
thors in [61] would be to compute this term using automatic
differentiation [45]. By substituting Y with ˜X, assuming
˜X = XExp(z), z ∼N(0, σ2
i I), and integrating the above
definition, the score on G can be reformulated as follows:
∇˜
X log pσ( ˜X|X) = −1
σ2 J−⊤
r
(z)z.
(5)
A score model sθ( ˜X, σ) can then be trained using the DSM
objective shown in Eq. (1), which takes the following form:
θ∗= arg min
θ
L(θ; σ)
≜1
2 Epdata(X)E ˜
X∼NG (X,Σ)
sθ( ˜
X, σ) −∇˜
X log pσ( ˜
X|X)

2
2

. (6)
For the denoising process, we employ a variant of the
Geodesic Random Walk [5], tailored to the Lie group con-
text, as a means to generate a sample from a noise distribu-
tion. The procedure is expressed as follows:
˜
Xi+1 = ˜
XiExp(ϵisθ( ˜
Xi, σi) +
√
2ϵizi),
zi ∼N(0, I). (7)
4.2. Efficient Computation of the Stein Score
Even with the above derivation, obtaining the closed-form
score remains challenging due to its dependency on the se-
lected distribution. For instance, deriving the closed-form
score for the IGSO(3) distribution [42] poses difficulties.
Furthermore, computing the score depends on the existence
of a closed-form expression for the Jacobian matrix on G.
Even if such an expression exists, it may not guarantee com-
putational efficiency compared to automatic differentiation.
Therefore, we next discuss a simplification method of the
Stein score under certain conditions for reducing computa-
tional costs on G. This can be expressed in a closed-form

if the Jacobian matrix on G is invertible and if the left and
right Jacobian matrices conform to the following relation:
Jl(z) = J⊤
r (z),
J−1
l
(z) = J−⊤
r
(z),
(8)
where z ∈g. As pointed out by [55], SO(3) exhibits this
property. Its closed-form score can then be simplified by
utilizing the following property, which holds on any G as
Jl(z)z = z. The derivation is in the supplementary mate-
rial. The score on SO(3) can then be expressed as follows:
∇˜
X log pσ( ˜X|X) = −1
σ2 J−1
l
(z)z = −1
σ2 z.
(9)
This shows that the score on SO(3) can be simplified to the
sampled Gaussian noise z scaled by −1/σ2, thus eliminat-
ing the need for both automatic differentiation and Jacobian
calculations. Similarly, the score on R3SO(3) also has a
closed-form as its Jacobians satisfy the relations in Eq. (8):
Jl(z) = (I, Jl(ϕ)) = (I, J⊤
r (ϕ)) = J⊤
r (z),
(10)
where in the case of R3SO(3), z = (T, ϕ) ∈⟨R3, so(3)⟩.
This implies that the score on R3SO(3) can also be simpli-
fied according to the formulation represented by Eq. (9).
4.3. Surrogate Stein Score Calculation on SE(3)
While the score on SO(3) and R3SO(3) can be simpli-
fied as described in the preceding sections, it can be shown
that SE(3) does not possess the property in Eq. (8). Con-
sider the inverse of the left-Jacobian on SE(3) at z =
(ρ, ϕ) ∈se(3), expressed as J−1
l
(z) =
h J−1
l
(ϕ) Z(ρ,ϕ)
0
J−1
l
(ϕ)
i
,
where Z(ρ, ϕ) = −J−1
l
(ϕ)Q(ρ, ϕ)J−1
l
(ϕ). The complete
form of Q(ρ, ϕ) can be found in [4, 55] and our supplemen-
tary material. The property Q⊤(−ρ, −ϕ) = Q(ρ, ϕ), as
derived in the references, leads to the following inequality:
J−⊤
r
(z) = (J−1
l
(−z))⊤=

J−1
l
(ϕ)
0
Z(ρ,ϕ) J−1
l
(ϕ)
̸
= J−1
l
(z).
(11)
This inequality indicates the potential discrepancy between
the score vector and the denoising direction due to the cur-
vature of the manifold, which may impede the convergence
of the reverse process and necessitate additional denoising
steps. To address this problem, we turn to higher-order ap-
proximation methods by breaking one step of reverse pro-
cess into multiple smaller sub-steps. Fig. 2 (right) illustrates
this one-step denoising process on SE(2) from a noisy sam-
ple ˜X = XExp(z) to its cleaned counterpart X, with con-
tour lines representing the distance to X in 2D Euclidean
space. We observe that increasing the number of sub-steps
eventually leads the integral of those small transformations
approaches the inverse of z. As a result, we propose substi-
tuting the true score in Eq. (5) with a surrogate score in our
training objective of Eq. (6) on SE(3), defined as follows:
˜sX( ˜
X, σ) ≜−1
σ2 z.
(12)
Note that the detailed training and sampling procedures are
described and elaborated in our supplementary material.
4.4. The Proposed Framework
Fig. 2 (left) presents an overview of our framework, which
consists of a conditioning part and a denoising part. The
conditioning part is responsible for generating the condition
variable c, which is crucial for guiding the denoising pro-
cess. This variable c can be derived either from an image en-
coder which extracts features from an image, or from a po-
sitional embedding module [62] that encodes a time index
i. In our experiments, we employ ResNet [14] as the image
encoder. The separation of the two parts in our framework
eliminates the need of image feature extraction in every de-
noising step, which offers efficiency in the inference phase.
For the denoising part, our score model is composed of
multiple multi-layer perceptron (MLP) blocks. This struc-
ture is inspired by the recent conditional generative mod-
els [16, 57], while we have modified their approaches by
substituting linear layers for the convolutional ones. The
score model processes a noisy pose ˜xi ∈g embedded us-
ing a positional encoding. It then computes an estimated
score sθ(˜xi, σi). This estimated score is subsequently uti-
lized in the denoising process (i.e., Eq. (7)). Please note that
the input and output of the denoising part are represented in
vector forms within the corresponding Lie algebra space.
Regarding the design of the conditioning mechanism in
MLPs, a few prior studies [16, 57] employ scale-bias condi-
tion, which is formulated as f(x, c) = A(c)x+B(c). Nev-
ertheless, our empirical observations suggest that this con-
ditioning mechanism does not perform satisfactorily when
learning distributions on SO(3). This may be attributable to
the limited expressivity of the underlying neural networks.
Inspired by [34, 73], we introduce a modified Fourier-based
conditioning mechanism, which is formulated as follows:
fi(x, c) =
d−1
X
j=0
Wij (Aj(c) cos(πxj) + Bj(c) sin(πxj)) ,
(13)
where d represents the dimension of our linear layer.
This form bears similarity to the Fourier series f(t) =
P∞
k=0 Ak cos
  2πkt
P

+ Bk sin
  2πkt
P

.
Our motivation
stems from the fact that the pose distribution on SO(3) is
circular, and can therefore be represented as periodic func-
tions. By the definition of periodic functions, their deriva-
tives are also periodic. It is worth noting that this condition-
ing mechanism does not introduce additional parameters in
our neural network design, as Wij is provided by the sub-
sequent linear layer. Our experimental findings suggest that
this conditioning scheme enhances the ability of neural net-
work to capture periodic features of score fields on SO(3).

Conditioning
Image
Encoder
i
Time Index
Positional  
Embedding
x
Noisy 
Pose
MLP
Block
Positional  
Embedding
MLP
Block
Denoising
MLP Block
˜xi
c
c
Linear
Linear
Linear
Condition
Operation
Linear
⨁
sθ(˜xi, σi)
Score


## experiments
In this section, we demonstrate that our score-based dif-
fusion model can produce precise pose estimation on both
SO(3) and SE(3) compared with previous probabilistic ap-
proaches. In addition, we present our method’s superior
performance on the real-world T-LESS [20] dataset without
relying on reconstructed 3D models or symmetric annota-
tions. Note that, to the best of our knowledge, our approach
is the first probabilistic model that conduct the experiments
on the complete T-LESS dataset and reports the accuracy,
in contrast to previous methods confined to a limited subset
of objects. The extensive evaluation substantiate the robust-
ness and scalability of our score-based diffusion model.
5.1. Experimental Setups
SYMSOL.
SYMSOL is a dataset specifically designed
for evaluating density estimators in the SO(3) space. This
dataset, first introduced by [41], comprises 250k images
of five texture-less and symmetric objects, with each sub-
ject to random rotations. The objects include tetrahedron
(tet.), cube, icosahedron (icosa.), cone, and cylinder (cyl.),
with each exhibiting unique symmetries that introduce var-
ious degrees of pose ambiguity. For this dataset, our score
model is compared in the SO(3) space with several recent
works [10, 23, 37, 41]. The baseline models compared with
utilize a pre-trained ResNet50 [15] as their backbones. Note
that we report the average angular distances in degrees.
SYMSOL-T.
To extend our evaluation into the SE(3)
space, we developed the SYMSOL-T dataset by incorporat-
ing random translations based on SYMSOL, which intro-
duces an additional layer of complexity due to perspective-
induced ambiguity.
Similar to SYMSOL, it features the
same five symmetric shapes and the same number of ran-
dom samples.
For SYMSOL-T, we benchmark our pro-
posed methods against two pose regression methods. These
two methods are trained using a symmetry-aware loss, but
with different strategies: one directly estimates the pose
from an image, while the other employs iterative refine-
ment. We report the average angular distances in degrees
for rotation and the average distances for translation.
T-LESS.
T-LESS [20] has been recognized as a challeng-
ing benchmark in the BOP challenge [22], which consists
of thirty texture-less industrial objects. The objects in this
dataset are characterized by a range of discrete and contin-
uous symmetries. In this dataset, the pose ambiguities arise
not only from the intrinsic object symmetries but also the
environmental factors such as occlusion and self-occlusion
due to its cluttered settings. The T-LESS dataset features a
training set with 50k physically based rendering (PBR) [22]
images from synthetic images, and an additional 37k im-
ages from real-world scanning. The testing set encompasses
10k real-world scanned images. The evaluation methods
employed in our study include three standard metrics from
the BOP challenge: Maximum Symmetry-Aware Projec-
tion Distance (MSPD), Maximum Symmetry-Aware Sur-
face Distance (MSSD), and Visible Surface Discrepancy
(VSD). To reflect the emphasis of our work on symme-
try, we further introduced symmetry-aware metrics: R@2,
R@5, and R@10, which represent predictions with rota-
tional errors within 2, 5, and 10 degrees, respectively. Sim-
ilarly, T@2, T@5, and T@10 are estimations with transla-
tional errors within 2, 5, and 10 centimeters, respectively.
Visualization
To visualize the density predictions, we
adopt the strategy employed in [41] to represent the rota-
tion densities generated by our model in the SO(3) space.
Specifically, we use the Mollweide projection for visual-
izing the SO(3) space, with longitude and latitude values
representing the yaw and pitch of the object’s rotation, re-
spectively. The color in the SO(3) space indicates the roll
of the object’s rotation. The circles denote sets of equivalent
poses, with each dot representing a single sample. For each
plot, we generate a total of 1, 000 random samples from our
model. For the translation part, we illustrate the rendered
results of the estimated poses below their original images.

Table 2. Evaluation results on SYMSOL.

## related_work
3.1. Methodologies for Dealing with Pose Ambiguity
Non-probabilistic modeling.
In the realm of object pose
estimation, pose ambiguity remains a significant challenge,
often stemming from an object that exhibits identical vi-
sual appearances from different perspectives [39]. A va-
riety of strategies have been explored in the literature to
directly address this issue, including the application of
symmetry supervisions and point matching algorithms [1,
66]. Regression-based approaches, such as those presented
in [11, 32, 60, 64], aim to minimize pose discrepancy by
selecting the closest candidate within a set of ambiguous
poses. Some researchers [46, 48], on the other hand, intro-
duce constraints to the regression targets (especially regard-
ing rotation angles) to mitigate ambiguity. Moreover, cer-
tain approaches [25, 44, 65] suggest regressing to a prede-
termined set of geometric features derived from symmetry
annotations. These prior arts often necessitate manual an-
notations of equivalent poses and are limited in dealing with
other sources of pose ambiguities, such as those caused by
occlusion and self-occlusion [39].
Probabilistic modeling.
On the other hand, several stud-
ies have investigated methods to model the inherent uncer-
tainty in pose ambiguity. This involves the quantification
and representation of uncertainty associated with the esti-
mated poses. Some works have employed parametric dis-
tributions such as Bingham distributions [10, 12, 43] and
von-Mises distributions [47, 72] to model orientation un-
certainty. Other approaches, such as in [38], utilize nor-
malizing flows [50] to model distributions within rotational
space.
A number of studies [23, 31, 41] employ non-
parametric distributions to implicitly represent rotation un-
certainty on SO(3).
These methods primarily focus on
modeling distributions on SO(3), leaving the joint distri-
bution modeling of rotation and translation unexplored.
3.2. Diffusion Probabilistic Models and Their Ap-
plication Domains
Diffusion models on Euclidean space.
Diffusion prob-
abilistic models [16, 56–58, 68] represent a class of gen-
erative models designed to learn the underlying probability
distribution of data. They have been applied to various gen-
erative tasks, and have shown impressive results in several
application domains, including image [2, 3, 7, 49, 51–53],
video [17, 18, 69], audio [26, 67], and natural language pro-
cessing [13, 35]. In the realm of human pose estimation,
diffusion models have also been found useful in addressing
joint location ambiguity, which arises from the projection
of 2D keypoints into 3D space [9, 24].
Diffusion models on non-Euclidean space.
To accom-
modate data residing on a manifold, the authors in [5]
extended diffusion models to Riemannian manifolds, and
leveraged Geodesic Random Walk [29] for sampling. Other
studies [28, 33] applied the Denoising Diffusion Probabilis-
tic Models (DDPM) [16] and score-based generative mod-
els [57, 58] to the SO(3) manifold to recover the density of
data on SO(3). Further extensions of diffusion models have
been attempted for tasks such as unfolding protein struc-
tures [71] and arm manipulations [61]. These approaches
typically used R3SO(3) parametrization, which treated ro-
tation and translation as separate entities for diffusion.
3.3. Diffusion Models on Lie Groups
Diffusion models on Lie groups have been explored in a
range of applications [28, 33, 61, 71]. Nevertheless, these
implementations vary in their choices of distributions and
computational methods, which lead to diverse outcomes
and different levels of computational efficiency. Table 1
presents a comparison of several previous diffusion model

Table 1. Comparison of different methods. △means closed form but with approximation. NSE(3) please refer to Eq. (3).
Baselines
Group
Distribution
Closed Form
Diffusion Method
Diffusion Space
App. Domain
Leach et al. [33]
SO(3)
IGSO(3)
✗
DDPM
SO(3)
Vector
Jagvaral et al. [28]
SO(3)
IGSO(3)
✗
Score / Autograd
SO(3)
Vector
Urain et al. [61]
R3SO(3)
NR3 × NSO(3)
✓
Score / Autograd
R3SO(3)
Vector
Yim et al. [71]
R3SO(3)
NR3 × IGSO(3)
✗
Score / Autograd
⟨R3, so(3)⟩
Vector
Ours
SE(3)
NSE(3)
△
Score / Closed Form
SE(3)
Image
approaches along with our own. It highlights the distinct
groups, distributions, methods, as well as diffusion spaces
each method utilizes. Several earlier studies [28, 33] have
introduced techniques that operate within the SO(3) space,
and adopted normal distributions defined on SO(3) [42]
(denoted as IGSO(3)). Unfortunately, a primary drawback
of IGSO(3) is its absence of a closed form, which poses
challenges in its computational efficiency. In a similar vein,
the authors in [71] developed a method that operates in the
tangent space of R3SO(3). This method’s distribution also
does not possess a closed form, which complicates the com-
putational procedure. On the other hand, the authors in [61]
employed a joint Gaussian distribution within the R3 and
SO(3) spaces. This distribution benefits from the presence
of a closed form and thus offers the potential for increased
computational efficiency. However, this approach is con-
fined to the R3 ×SO(3) space and treats rotation and trans-
lation as separate entities for diffusion. As a result, it may
not be able to offer the advantages that SE(3) can provide.

## conclusion
In this paper, we presented a novel approach that applies
diffusion models to the SE(3) group for object pose es-
timation, effectively addressing the pose ambiguity issue.
Inspired by the correlation between rotation and transla-
tion distributions caused by image projection effects, we
jointly estimated their distributions on SE(3) for improved
accuracy. This is the first work to apply diffusion models
to SE(3) in the image domain. To validate it, we devel-
oped the SYMSOL-T dataset, which enriches the original
SYMSOL dataset with randomly sampled translations. Our
experiments confirmed the applicability of our SE(3) dif-
fusion model in the image domain and the advantage of
SE(3) parametrization over R3SO(3). Moreover, our ex-
periments on T-LESS exhibits the efficacy of our SE(3)

diffusion model in real-world applications.
7. Acknowledgement
The authors gratefully acknowledge the support from the
National Science and Technology Council (NSTC) in Tai-
wan under grant numbers MOST 111-2223-E-007-004-
MY3, Taiwan. The authors would also like to express their
appreciation for the donation of the GPUs from NVIDIA
Corporation and NVIDIA AI Technology Center (NVAITC)
used in this work. Furthermore, the authors extend their
gratitude to the National Center for High-Performance
Computing (NCHC) for providing the necessary computa-
tional and storage resources.