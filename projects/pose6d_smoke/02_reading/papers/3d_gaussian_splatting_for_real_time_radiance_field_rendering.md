# 3D Gaussian Splatting for Real-Time Radiance Field Rendering

> 2023 · id: W4385318467 · arXiv: 2308.04079 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
The input to our method is a set of images of a static scene, together
with the corresponding cameras calibrated by SfM [Schönberger
and Frahm 2016] which produces a sparse point cloud as a side-
effect. From these points we create a set of 3D Gaussians (Sec. 4),
defined by a position (mean), covariance matrix and opacity 𝛼, that
allows a very flexible optimization regime. This results in a reason-
ably compact representation of the 3D scene, in part because highly
anisotropic volumetric splats can be used to represent fine structures
compactly. The directional appearance component (color) of the
radiance field is represented via spherical harmonics (SH), following
standard practice [Fridovich-Keil and Yu et al. 2022; Müller et al.
2022]. Our algorithm proceeds to create the radiance field represen-
tation (Sec. 5) via a sequence of optimization steps of 3D Gaussian
parameters, i.e., position, covariance, 𝛼and SH coefficients inter-
leaved with operations for adaptive control of the Gaussian density.
The key to the efficiency of our method is our tile-based rasterizer
(Sec. 6) that allows 𝛼-blending of anisotropic splats, respecting visi-
bility order thanks to fast sorting. Out fast rasterizer also includes
a fast backward pass by tracking accumulated 𝛼values, without a
limit on the number of Gaussians that can receive gradients. The
overview of our method is illustrated in Fig. 2.
4
DIFFERENTIABLE 3D GAUSSIAN SPLATTING
Our goal is to optimize a scene representation that allows high-
quality novel view synthesis, starting from a sparse set of (SfM)
points without normals. To do this, we need a primitive that inherits
the properties of differentiable volumetric representations, while
at the same time being unstructured and explicit to allow very fast
rendering. We choose 3D Gaussians, which are differentiable and
can be easily projected to 2D splats allowing fast 𝛼-blending for
rendering.
Our representation has similarities to previous methods that use
2D points [Kopanas et al. 2021; Yifan et al. 2019] and assume each
point is a small planar circle with a normal. Given the extreme
sparsity of SfM points it is very hard to estimate normals. Similarly,
optimizing very noisy normals from such an estimation would be
very challenging. Instead, we model the geometry as a set of 3D
Gaussians that do not require normals. Our Gaussians are defined
by a full 3D covariance matrix Σ defined in world space [Zwicker
et al. 2001a] centered at point (mean) 𝜇:
𝐺(𝑥) = 𝑒−1
2 (𝑥)𝑇Σ−1(𝑥)
(4)
. This Gaussian is multiplied by 𝛼in our blending process.
However, we need to project our 3D Gaussians to 2D for rendering.
Zwicker et al. [2001a] demonstrate how to do this projection to
image space. Given a viewing transformation 𝑊the covariance
matrix Σ′ in camera coordinates is given as follows:
Σ′ = 𝐽𝑊Σ 𝑊𝑇𝐽𝑇
(5)
where 𝐽is the Jacobian of the affine approximation of the projective
transformation. Zwicker et al. [2001a] also show that if we skip the
third row and column of Σ′, we obtain a 2×2 variance matrix with
the same structure and properties as if we would start from planar
points with normals, as in previous work [Kopanas et al. 2021].
An obvious approach would be to directly optimize the covariance
matrix Σ to obtain 3D Gaussians that represent the radiance field.
However, covariance matrices have physical meaning only when
they are positive semi-definite. For our optimization of all our pa-
rameters, we use gradient descent that cannot be easily constrained
to produce such valid matrices, and update steps and gradients can
very easily create invalid covariance matrices.
As a result, we opted for a more intuitive, yet equivalently ex-
pressive representation for optimization. The covariance matrix Σ
of a 3D Gaussian is analogous to describing the configuration of an
ellipsoid. Given a scaling matrix 𝑆and rotation matrix 𝑅, we can
find the corresponding Σ:
Σ = 𝑅𝑆𝑆𝑇𝑅𝑇
(6)
To allow independent optimization of both factors, we store them
separately: a 3D vector 𝑠for scaling and a quaternion 𝑞to represent
rotation. These can be trivially converted to their respective matrices
and combined, making sure to normalize 𝑞to obtain a valid unit
quaternion.
To avoid significant overhead due to automatic differentiation
during training, we derive the gradients for all parameters explicitly.
Details of the exact derivative computations are in appendix A.
This representation of anisotropic covariance – suitable for op-
timization – allows us to optimize 3D Gaussians to adapt to the
geometry of different shapes in captured scenes, resulting in a fairly
compact representation. Fig. 3 illustrates such cases.
5
OPTIMIZATION WITH ADAPTIVE DENSITY
CONTROL OF 3D GAUSSIANS
The core of our approach is the optimization step, which creates
a dense set of 3D Gaussians accurately representing the scene for
free-view synthesis. In addition to positions 𝑝, 𝛼, and covariance
Σ, we also optimize SH coefficients representing color 𝑐of each
Gaussian to correctly capture the view-dependent appearance of
the scene. The optimization of these parameters is interleaved with
steps that control the density of the Gaussians to better represent
the scene.
ACM Trans. Graph., Vol. 42, No. 4, Article 1. Publication date: August 2023.

3D Gaussian Splatting for Real-Time Radiance Field Rendering
•
1:5
Diﬀerentiable
Tile Rasterizer
Adaptive
Density Control
Projection
Initialization
SfM Points
3D Gaussians
Image
Camera
Gradient Flow
Operation Flow
Fig. 2. Optimization starts with the sparse SfM point cloud and creates a set of 3D Gaussians. We then optimize and adaptively control the density of this set
of Gaussians. During optimization we use our fast tile-based renderer, allowing competitive training times compared to SOTA fast radiance field methods.
Once trained, our renderer allows real-time navigation for a wide variety of scenes.
Original
Shrunken
Gaussians
Fig. 3. We visualize the 3D Gaussians after optimization by shrinking them
60% (far right). This clearly shows the anisotropic shapes of the 3D Gaussians
that compactly represent complex geometry after optimization. Left the
actual rendered image.
5.1
Optimization
The optimization is based on successive iterations of rendering and
comparing the resulting image to the training views in the captured
dataset. Inevitably, geometry may be incorrectly placed due to the
ambiguities of 3D to 2D projection. Our optimization thus needs to
be able to create geometry and also destroy or move geometry if it
has been incorrectly positioned. The quality of the parameters of the
covariances of the 3D Gaussians is critical for the compactness of
the representation since large homogeneous areas can be captured
with a small number of large anisotropic Gaussians.
We use Stochastic Gradient Descent techniques for optimization,
taking full advantage of standard GPU-accelerated frameworks,
and the ability to add custom CUDA kernels for some operations,
following recent best practice [Fridovich-Keil and Yu et al. 2022;
Sun et al. 2022]. In particular, our fast rasterization (see Sec. 6) is
critical in the efficiency of our optimization, since it is the main
computational bottleneck of the optimization.
We use a sigmoid activation function for 𝛼to constrain it in
the [0 −1) range and obtain smooth gradients, and an exponential
activation function for the scale of the covariance for similar reasons.
We estimate the initial covariance matrix as an isotropic Gaussian
with axes equal to the mean of the distance to the closest three points.
We use a standard exponential decay scheduling technique similar
to Plenoxels [Fridovich-Keil and Yu et al. 2022], but for positions
only. The loss function is L1 combined with a D-SSIM term:
L = (1 −𝜆)L1 + 𝜆LD-SSIM
(7)
We use 𝜆= 0.2 in all our tests. We provide details of the learning
schedule and other elements in Sec. 7.1.
5.2
Adaptive Control of Gaussians
We start with the initial set of sparse points from SfM and then apply
our method to adaptively control the number of Gaussians and their
density over unit volume1, allowing us to go from an initial sparse
set of Gaussians to a denser set that better represents the scene, and
with correct parameters. After optimization warm-up (see Sec. 7.1),
we densify every 100 iterations and remove any Gaussians that are
essentially transparent, i.e., with 𝛼less than a threshold 𝜖𝛼.
Our adaptive control of the Gaussians needs to populate empty
areas. It focuses on regions with missing geometric features (“under-
reconstruction”), but also in regions where Gaussians cover large
areas in the scene (which often correspond to “over-reconstruction”).
We observe that both have large view-space positional gradients.
Intuitively, this is likely because they correspond to regions that are
not yet well reconstructed, and the optimization tries to move the
Gaussians to correct this.
Since both cases are good candidates for densification, we den-
sify Gaussians with an average magnitude of view-space pos

## related_work
We first briefly overview traditional reconstruction, then discuss
point-based rendering and radiance field work, discussing their
similarity; radiance fields are a vast area, so we focus only on directly
related work. For complete coverage of the field, please see the
excellent recent surveys [Tewari et al. 2022; Xie et al. 2022].
2.1
Traditional Scene Reconstruction and Rendering
The first novel-view synthesis approaches were based on light fields,
first densely sampled [Gortler et al. 1996; Levoy and Hanrahan 1996]
then allowing unstructured capture [Buehler et al. 2001]. The advent
of Structure-from-Motion (SfM) [Snavely et al. 2006] enabled an
entire new domain where a collection of photos could be used to
synthesize novel views. SfM estimates a sparse point cloud during
camera calibration, that was initially used for simple visualization
of 3D space. Subsequent multi-view stereo (MVS) produced im-
pressive full 3D reconstruction algorithms over the years [Goesele
et al. 2007], enabling the development of several view synthesis
algorithms [Chaurasia et al. 2013; Eisemann et al. 2008; Hedman
et al. 2018; Kopanas et al. 2021]. All these methods re-project and
blend the input images into the novel view camera, and use the
geometry to guide this re-projection. These methods produced ex-
cellent results in many cases, but typically cannot completely re-
cover from unreconstructed regions, or from “over-reconstruction”,
when MVS generates inexistent geometry. Recent neural render-
ing algorithms [Tewari et al. 2022] vastly reduce such artifacts and
avoid the overwhelming cost of storing all input images on the GPU,
outperforming these methods on most fronts.
2.2
Neural Rendering and Radiance Fields
Deep learning techniques were adopted early for novel-view synthe-
sis [Flynn et al. 2016; Zhou et al. 2016]; CNNs were used to estimate
blending weights [Hedman et al. 2018], or for texture-space solutions
[Riegler and Koltun 2020; Thies et al. 2019]. The use of MVS-based
geometry is a major drawback of most of these methods; in addition,
the use of CNNs for final rendering frequently results in temporal
flickering.
Volumetric representations for novel-view synthesis were ini-
tiated by Soft3D [Penner and Zhang 2017]; deep-learning tech-
niques coupled with volumetric ray-marching were subsequently
proposed [Henzler et al. 2019; Sitzmann et al. 2019] building on a con-
tinuous differentiable density field to represent geometry. Rendering
using volumetric ray-marching has a significant cost due to the large
number of samples required to query the volume. Neural Radiance
Fields (NeRFs) [Mildenhall et al. 2020] introduced importance sam-
pling and positional encoding to improve quality, but used a large
Multi-Layer Perceptron negatively affecting speed. The success of
NeRF has resulted in an explosion of follow-up methods that address
quality and speed, often by introducing regularization strategies; the
current state-of-the-art in image quality for novel-view synthesis is
Mip-NeRF360 [Barron et al. 2022]. While the rendering quality is
outstanding, training and rendering times remain extremely high;
we are able to equal or in some cases surpass this quality while
providing fast training and real-time rendering.
The most recent methods have focused on faster training and/or
rendering mostly by exploiting three design choices: the use of spa-
tial data structures to store (neural) features that are subsequently
interpolated during volumetric ray-marching, different encodings,
ACM Trans. Graph., Vol. 42, No. 4, Article 1. Publication date: August 2023.

3D Gaussian Splatting for Real-Time Radiance Field Rendering
•
1:3
and MLP capacity. Such methods include different variants of space
discretization [Chen et al. 2022b,a; Fridovich-Keil and Yu et al. 2022;
Garbin et al. 2021; Hedman et al. 2021; Reiser et al. 2021; Takikawa
et al. 2021; Wu et al. 2022; Yu et al. 2021], codebooks [Takikawa
et al. 2022], and encodings such as hash tables [Müller et al. 2022],
allowing the use of a smaller MLP or foregoing neural networks
completely [Fridovich-Keil and Yu et al. 2022; Sun et al. 2022].
Most notable of these methods are InstantNGP [Müller et al. 2022]
which uses a hash grid and an occupancy grid to accelerate compu-
tation and a smaller MLP to represent density and appearance; and
Plenoxels [Fridovich-Keil and Yu et al. 2022] that use a sparse voxel
grid to interpolate a continuous density field, and are able to forgo
neural networks altogether. Both rely on Spherical Harmonics: the
former to represent directional effects directly, the latter to encode
its inputs to the color network. While both provide outstanding
results, these methods can still struggle to represent empty space
effectively, depending in part on the scene/capture type. In addition,
image quality is limited in large part by the choice of the structured
grids used for acceleration, and rendering speed is hindered by the
need to query many samples for a given ray-marching step. The un-
structured, explicit GPU-friendly 3D Gaussians we use achieve faster
rendering speed and better quality without neural components.
2.3
Point-Based Rendering and Radiance Fields
Point-based methods efficiently render disconnected and unstruc-
tured geometry samples (i.e., point clouds) [Gross and Pfister 2011].
In its simplest form, point sample rendering [Grossman and Dally
1998] rasterizes an unstructured set of points with a fixed size, for
which it may exploit natively supported point types of graphics APIs
[Sainz and Pajarola 2004] or parallel software rasterization on the
GPU [Laine and Karras 2011; Schütz et al. 2022]. While true to the
underlying data, point sample rendering suffers from holes, causes
aliasing, and is strictly discontinuous. Seminal work on high-quality
point-based rendering addresses these issues by “splatting” point
primitives with an extent larger than a pixel, e.g., circular or elliptic
discs, ellipsoids, or surfels [Botsch et al. 2005; Pfister et al. 2000; Ren
et al. 2002; Zwicker et al. 2001b].
There has been recent interest in differentiable point-based render-
ing techniques [Wiles et al. 2020; Yifan et al. 2019]. Points have been
augmented with neural features and rendered using a CNN [Aliev
et al. 2020; Rückert et al. 2022] resulting in fast or even real-time
view synthesis; however they still depend on MVS for the initial
geometry, and as such inherit its artifacts, most notably over- or
under-reconstruction in hard cases such as featureless/shiny areas
or thin structures.
Point-based 𝛼-blending and NeRF-style volumetric rendering
share essentially the same image formation model. Specifically, the
color 𝐶is given by volumetric rendering along a ray:
𝐶=
𝑁
∑︁
𝑖=1
𝑇𝑖(1 −exp(−𝜎𝑖𝛿𝑖))c𝑖
with 𝑇𝑖= exp ©­
«
−
𝑖−1
∑︁
𝑗=1
𝜎𝑗𝛿𝑗ª®
¬
,
(1)
where samples of density 𝜎, transmittance 𝑇, and color c are taken
along the ray with intervals 𝛿𝑖. This can be re-written as
𝐶=
𝑁
∑︁
𝑖=1
𝑇𝑖𝛼𝑖c𝑖,
(2)
with
𝛼𝑖= (1 −exp(−𝜎𝑖𝛿𝑖)) and 𝑇𝑖=
𝑖−1
Ö
𝑗=1
(1 −𝛼𝑖).
A typical neural point-based approach (e.g., [Kopanas et al. 2022,
2021]) computes the color 𝐶of a pixel by blending N ordered points
overlapping the pixel:
𝐶=
∑︁
𝑖∈N
𝑐𝑖𝛼𝑖
𝑖−1
Ö
𝑗=1
(1 −𝛼𝑗),
(3)
where c𝑖is the color of each point and 𝛼𝑖is given by evaluating a
2D Gaussian with covariance Σ [Yifan et al. 2019] multiplied with a
learned per-point opacity.
From Eq. 2 and Eq. 3, we can clearly see that the image formation
model is the same. However, the rendering algorithm is very differ-
ent. NeRFs are a continuous representation implicitly representing
empty/occupied space; expensive random sampling is required to
find the samples in Eq. 2 with consequent noise and computational
expense. In contrast, points are an unstructured, discrete represen-
tation that is flexible enough to allow creation, destruction, and
displacement of geometry similar to NeRF. This is achieved by opti-
mizing opacity and positions, as shown by previous work [Kopanas
et al. 2021], while avoiding the shortcomings of a full volumetric
representation.
Pulsar [Lassner and Zollhofer 2021] achieves fast sphere rasteri-
zation which inspired our tile-based and sorting renderer. However,
given the analysis above, we want to maintain (approximate) con-
ventional 𝛼-blending on sorted splats to have the advantages of vol-
umetric representations: Our rasterization respects visibility order
in contrast to their order-independent method. In addition, we back-
propagate gradients on all splats in a pixel and rasterize anisotropic
splats. These elements all contribute to the high visual quality of
our results (see Sec. 7.3). In addition, previous methods mentioned
above also use CNNs for rendering, which results in temporal in-
stability. Nonetheless, the rendering speed of Pulsar [Lassner and
Zollhofer 2021] and ADOP [Rückert et al. 2022] served as motivation
to develop our fast rendering solution.
While focusing on specular effects, the diffuse point-based ren-
dering track of Neura

## conclusion
Our method is not without limitations. In regions where the scene
is not well observed we have artifacts; in such regions, other meth-
ods also struggle (e.g., Mip-NeRF360 in Fig. 11). Even though the
anisotropic Gaussians have many advantages as described above,
our method can create elongated artifacts or “splotchy” Gaussians
(see Fig. 12); again previous methods also struggle in these cases.
We also occasionally have popping artifacts when our optimiza-
tion creates large Gaussians; this tends to happen in regions with
view-dependent appearance. One reason for these popping artifacts
is the trivial rejection of Gaussians via a guard band in the rasterizer.
A more principled culling approach would alleviate these artifacts.
Another factor is our simple visibility algorithm, which can lead to
Gaussians suddenly switching depth/blending order. This could be
addressed by antialiasing, which we leave as future work. Also, we
currently do not apply any regularization to our optimization; doing
so would help with both the unseen region and popping artifacts.
While we used the same hyperparameters for our full evaluation,
early experiments show that reducing the position learning rate can
be necessary to converge in very large scenes (e.g., urban datasets).
ACM Trans. Graph., Vol. 42, No. 4, Article 1. Publication date: August 2023.

3D Gaussian Splatting for Real-Time Radiance Field Rendering
•
1:11
Ground  
Truth
Full
Isotropic
Ground  
Truth
Full
Isotropic
Ground  
Truth
Full
Isotropic
Fig. 10. We train scenes with Gaussian anisotropy disabled and enabled. The use of anisotropic volumetric splats enables modelling of fine structures and has
a significant impact on visual quality. Note that for illustrative purposes, we restricted Ficus to use no more than 5k Gaussians in both configurations.
Even though we are very compact compared to previous point-
based approaches, our memory consumption is significantly higher
than NeRF-based solutions. During training of large scenes, peak
GPU memory consumption can exceed 20 GB in our unoptimized
prototype. However, this figure could be significantly reduced by a
careful low-level implementation of the optimization logic (similar
to InstantNGP). Rendering the trained scene requires sufficient GPU
memory to store the full model (several hundred megabytes for
large-scale scenes) and an additional 30–500 MB for the rasterizer,
depending on scene size and image resolution. We note that there
are many opportunities to further reduce memory consumption
of our method. Compression techniques for point clouds is a well-
studied field [De Queiroz and Chou 2016]; it would be interesting to
see how such approaches could be adapted to our representation.
Fig. 11. Comparison of failure artifacts: Mip-NeRF360 has “floaters” and
grainy appearance (left, foreground), while our method produces coarse,
anisoptropic Gaussians resulting in low-detail visuals (right, background).
Train scene.
Fig. 12. In views that have little overlap with those seen during training,
our method may produce artifacts (right). Again, Mip-NeRF360 also has
artifacts in these cases (left). DrJohnson scene.
8
DISCUSSION AND CONCLUSIONS
We have presented the first approach that truly allows real-time,
high-quality radiance field rendering, in a wide variety of scenes
and capture styles, while requiring training times competitive with
the fastest previous methods.
Our choice of a 3D Gaussian primitive preserves properties of
volumetric rendering for optimization while directly allowing fast
splat-based rasterization. Our work demonstrates that – contrary to
widely accepted opinion – a continuous representation is not strictly
necessary to allow fast and high-quality radiance field training.
The majority (∼80%) of our training time is spent in Python code,
since we built our solution in PyTorch to allow our method to be
easily used by others. Only the rasterization routine is implemented
as optimized CUDA kernels. We expect that porting the remaining
optimization entirely to CUDA, as e.g., done in InstantNGP [Müller
et al. 2022], could enable significant further speedup for applications
where performance is essential.
We also demonstrated the importance of building on real-time
rendering principles, exploiting the power of the GPU and speed of
software rasterization pipeline architecture. These design choices
are the key to performance both for training and real-time render-
ing, providing a competitive edge in performance over previous
volumetric ray-marching.
It would be interesting to see if our Gaussians can be used to per-
form mesh reconstructions of the captured scene. Aside from prac-
tical implications given the widespread use of meshes, this would
allow us to better understand where our method stands exactly in
the continuum between volumetric and surface representations.
In conclusion, we have presented the first real-time rendering
solution for radiance fields, with rendering quality that matches the
best expensive previous methods, with training times competitive
with the fastest existing solutions.
ACKNOWLEDGMENTS
This research was funded by the ERC Advanced grant FUNGRAPH
No 788065 http://fungraph.inria.fr. The authors are grateful to Adobe
for generous donations, the OPAL infrastructure from Université
Côte d’Azur and for the HPC resources from GENCI–IDRIS (Grant
2022-AD011013409). The authors thank the anonymous reviewers
for their valuable feedback, P. Hedman and A. Tewari for proof-
reading earlier drafts also T. Müller, A. Yu and S. Fridovich-Keil for
helping with the comparisons.
ACM Trans. Graph., Vol. 42, No. 4, Article 1. Publication date: August 2023.

1:12
•
Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis