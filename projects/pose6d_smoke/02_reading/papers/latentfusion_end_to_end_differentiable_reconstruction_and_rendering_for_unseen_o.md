# LatentFusion: End-to-End Differentiable Reconstruction and Rendering for Unseen Object Pose Estimation

> 2020 · id: arxiv:1912.00416 · arXiv: 1912.00416 · pdf: https://arxiv.org/pdf/1912.00416 · 来源: backfill-arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

LatentFusion: End-to-End Differentiable Reconstruction and Rendering
for Unseen Object Pose Estimation
Keunhong Park1,2∗
Arsalan Mousavian2
Yu Xiang2
Dieter Fox1,2
1University of Washington
2NVIDIA
1. Reconstruction and Rendering
Reference
Modeler
Latent Object
…
…
2. Pose Estimation
…
…
Pose
Image
Based
Renderer
…
RGB
Query
(RGB + Mask + Depth)
𝜽
Pose
Renderer
Predicted Depth
Query Depth
Gradients
Novel Pose
Renderer
Depth + Mask
Figure 1: We present an end-to-end differentiable reconstruction and rendering pipeline. We use this pipeline to perform pose
estimation on unseen objects using simple gradient updates in a render-and-compare fashion.
Abstract
Current 6D object pose estimation methods usually re-
quire a 3D model for each object. These methods also re-
quire additional training in order to incorporate new ob-
jects. As a result, they are difﬁcult to scale to a large number
of objects and cannot be directly applied to unseen objects.
We propose a novel framework for 6D pose estimation
of unseen objects. We present a network that reconstructs a
latent 3D representation of an object using a small num-
ber of reference views at inference time.
Our network
is able to render the latent 3D representation from arbi-
trary views. Using this neural renderer, we directly opti-
mize for pose given an input image. By training our net-
work with a large number of 3D shapes for reconstruc-
tion and rendering, our network generalizes well to un-
seen objects. We present a new dataset for unseen object
pose estimation–MOPED. We evaluate the performance of
our method for unseen object pose estimation on MOPED
as well as the ModelNet and LINEMOD datasets.
Our
method performs competitively to supervised methods that
are trained on those objects. Code and data will be avail-
able at https://keunhong.com/publications/latentfusion/.
1. Introduction
The pose of an object deﬁnes where it is in space and
how it is oriented. An object pose is typically deﬁned by
a 3D orientation (rotation) and translation comprising six
degrees of freedom (6D). Knowing the pose of an object is
∗Work done while author was an intern at NVIDIA.
crucial for any application that involves interacting with real
world objects. For example, in order for a robot to manip-
ulate objects it must be able to reason about the pose of the
object. In augmented reality, 6D pose estimation enables
virtual interaction and re-rendering of real world objects.
In order to estimate the 6D pose of objects, current state-
of-the-art methods [49, 8, 45] require a 3D model for each
object. Methods based on renderings [42] usually need high
quality 3D models typically obtained using 3D scanning de-
vices. Although modern 3D reconstruction and scanning
techniques such as [26] can generate 3D models of objects,
they typically require signiﬁcant effort. It is easy to see how
building a 3D model for every object is an infeasible task.
Furthermore, existing pose estimation methods require
extensive training under different lighting conditions and
occlusions. For methods that train a single network for mul-
tiple objects [49], the pose estimation accuracy drops sig-
niﬁcantly with the increase in the number of objects. This
is due to large variation of object appearances depending
on the pose. To remedy this mode of degradation, some ap-
proaches train a separate network for each object [42, 41, 8].
This approach is not scalable to a large number of ob-
jects. Regardless of using a single or multiple networks, all
model-based methods require extensive training for unseen
test objects that are not in the training set.
In this paper, we investigate the problem of constructing
a 3D object representations for 6D object pose estimation
without 3D models and without extra training for unseen
objects during test time. The core of our method is a novel
neural network that takes a set of reference RGB images of
a target object with known poses, and internally builds a 3D
1
arXiv:1912.00416v3  [cs.CV]  12 Jun 2020

representation of the object. Using the 3D representation,
the network is able to render arbitrary views of the object.
To estimate object pose, the network compares the input im-
age with its rendered images in a gradient descent fashion to
search for the best pose where the rendered image matches
the input image. Applying the network to an unseen ob-
ject only requires collecting views with registered poses us-
ing traditional techniques [26] and feeding a small subset
of those views with the associated poses to the network, in-
stead of training for the new object which takes time and
computational resources.
Our network design is inspired by space carving [18].
We build a 3D voxel representation of an object by comput-
ing 2D latent features and projecting them to a canonical 3D
voxel space using a deprojection unit inspired by [27]. This
operation can be interpreted as space carving in latent space.
Rendering a novel view is conducted by rotating the latent
voxel representation to the new view and projecting it into
the 2D image space. Using the projected latent features, a
decoder generates a new view image by ﬁrst predicting the
depth map of the object at the query view and then assign-
ing color for each pixel by combining corresponding pixel
values at different reference views.
To reconstruct and render unseen objects, we train the
network on the ShapeNet dataset [4] randomly textured
with images from the MS-COCO dataset [21] under random
lighting conditions. Our experiments show that the network
generalizes to novel object categories and instances. For
pose estimation, we assume that the object of interest is seg-
mented with a generic object instance segmentation method
such as [50]. The pose of the object is estimated by ﬁnding
a 6D pose that minimizes the difference between a predicted
rendering and the input image. Since our network is a dif-
ferentiable renderer, we optimize by directly computing the
gradients of the loss with respect to the object pose. Fig. 1
illustrates our reconstruction and pose estimation pipeline.
Some key beneﬁts of our method are:
1. Ease of Capture – we perform pose estimation given
just a few reference images rather than 3D scans;
2. Robustness to Appearance – we create a latent rep-
resentation from images rather than relying on a 3D
model with baked appearance; and
3. Practicality – our zero-shot formulation requires only
one neural network model for all objects and requires
no training for novel objects.
In addition, we introduce the Model-free Object Pose Es-
timation Dataset (MOPED) for evaluating pose estimation
in a zero-shot setting. Existing pose estimation benchmarks
provide 3D models and rendered training images sequences,
but typically do not provide casual real-world reference im-
ages. MOPED provides registered reference and test images
for evaluating pose estimation in a zero-shot setting.
2. Related Work
Pose Estimation. Pose estimation methods fall into three
major categories. The ﬁrst category tackles the problem of
pose estimation by designing network architectures that fa-
cilitate pose estimation [25, 17, 15]. The second category
formulates the pose estimation by predicting a set of 2D
image features, such as the projection of 3D box corners
[42, 45, 14, 33] and direction of the center of the object [49],
then recovering the pose of the object using the predictions.
The third category estimates the pose of objects by aligning
the rendering of the 3D model to the image. DeepIM [20]
trains a neural network to align the 3D model of the ob-
ject to the image. Another approach is to learn a model
that can reconstruct the object with different poses [41, 8].
These methods then use the latent representation of the ob-
ject to estimate the pose. A limitation of this line of work is
that they need to train separate auto-encoders for each ob-
ject category and there is a lack of knowledge transfer be-
tween object categories. In addition, these methods require
high-ﬁdelity textured 3D models for each object which are
not trivial to build in practice since it involves specialized
hardware [37]. Our method addresses these limitations: our
method works with a set of reference views with registered
poses instead of a 3D model. Without additional training,
our system builds a latent representation from the reference
views which can be rendered to color and depth for arbitrary
viewpoints. Similar to [41, 8], we seek to ﬁnd a pose that
minimizes the difference in latent space between the query
object and the test image. Differentiable mesh renderers
have been explored for pose estimation [29, 5] but still re-
quire 3D models leaving the acquisition problem unsolved.
3D shape learning and novel view synthesis. Inferring
shapes of objects at the category level has recently gained
a lot of attention. Shape geometry has been represented as
voxels [6], Signed Distance Functions (SDFs) [30], point
clouds [51], and as implicit functions encoded by a neural
network [39]. These methods are trained at the category
level and can only represent different instances within the
categories they were trained on. In addition, these models
only capture the shape of the object and do not model the
appearance of the object. To overcome this limitation, re-
cent works [28, 27, 39] decode appearance from neural 3D
latent representations that respect projective geometry, gen-
eralizing well to novel viewpoints. Novel views are gen-
erated by transforming the latent representation in 3D and
projecting it to 2D. A decoder then generates a novel view
from the projected features. Some methods ﬁnd a nearest
neighbor shape proxy and infer high quality appearances
but cannot handle novel categories [46, 32]. Differentiable

2D
U-Net
2D 
à
3D
3D
U-Net
1. Reconstruction
View Features
(Object Frame)
Input
(RGB + Mask)
3D
Transform
View
Fusion
Latent Object
(Object Frame)
…
3D
Transform
3D
U-Net
3D 
à
2D
2D
U-Net
…
View Features
(Camera Frame)
3D Object
(Camera Frame)
Camera Parameters
2. Rendering
Output
(Depth + Mask)
Camera
Parameters
…
…
…
…
Figure 2: A high level overview of our architecture. 1) Our modeling network takes an image and mask and predicts a feature
volume for each input view. The predicted feature volumes are then fused into a single canonical latent object by the fusion
module. 2) Given the latent object, our rendering network produces a depth map and a mask for any camera pose.
rendering [19, 27, 22] systems seek to implement the ren-
dering process (rasterization and shading) in a differentiable
manner so that gradients can be propagated to and from neu-
ral networks. Such methods can be used to directly opti-
mize parameters such as pose or appearance. Current dif-
ferentiable rendering methods are limited by the difﬁculty
of implemented complex appearance models and require a
3D mesh. We seek to combine the best of these methods
by creating a differentiable rendering pipeline that does not
require a 3D mesh by instead building voxelized latent rep-
resentations from a small number of reference images.
Multi-View Reconstruction. Our method takes inspiration
from multi-view reconstruction methods. It is most simi-
lar to space carving [18] and can be seen as a latent-space
extension of it.
Dense fusion methods such as [26, 47]
generate dense point clouds of the objects from RGB-D
sequences. Recent works [44, 43] have explored ways to
learn object representations from unaligned views. These
methods recover coarse geometry and pose given an image,
but require large amounts of training data for a single ob-
ject category. Our method builds on both approaches: we
train a network to reconstruct an object; however, instead
of training per-object or per-category, we provide multiple
reference images at inference time to create a 3D latent rep-
resentation which can be rendered from novel viewpoints.
3. Overview
We present an end-to-end system for novel view recon-
struction and pose estimation. We present our system in two
parts. Sec. 4 describes our reconstruction pipeline which
takes a small collection of reference images as input and
produces a ﬂexible representation which can be rendered
from novel viewpoints. We leverage multi-view consistency
to construct a latent representation and do not rely on cat-
egory speciﬁc shape priors. This key architecture decision
enables generalization beyond the distribution of training
objects. We show that our reconstruction pipeline can ac-
curately reconstruct unseen object categories from real im-
ages. In Sec. 5, we formulate the 6D pose estimation prob-
lem using our neural renderer. Since our rendering process
is fully differentiable, we directly optimize for the camera
parameters without the need for additional training or code-
book generation for new objects.
Camera Model. Throughout this paper we use a perspec-
tive pinhole camera model with an intrinsic matrix
K =


fu
0
u0
0
fv
v0
0
0
1

,
(1)
and a homogeneous extrinsic matrix E = [R|t], where fu
and fv are the focal lengths, u0 and v0 are the coordinates
of the camera principal point, and R and t are rotation and
translation of the camera, respectively. We also deﬁne a
viewport cropping parameter c = (u−, v−, u+, v+) which
represents a bounding box around the object in pixel coordi-
nates. For brevity, we refer to the collection of these camera
parameters as θ = {R, t, c}.
4. Neural Reconstruction and Rendering
Given a set of N reference images with associated object
poses and object segmentation masks, we seek to construct
a representation of the object which can be rendered with
arbitrary camera parameters. Building on the success of re-
cent methods [28, 39], we represent the object as a latent 3D
voxel grid. This representation can be directly manipulated
using standard 3D transformations–naturally accommodat-
ing our requirement of novel view rendering. The overview
of our method is shown in Fig. 2. There are two main com-
ponents in our reconstruction pipeline: 1) Modeling the ob-
ject by predicting per-view feature volumes and fusing them
into a single canonical latent representation; 2) Rendering
the latent representation to depth and color images.

Object 
Centroid
(𝑐#)
Near Plane
(𝑧& −𝑟)
Far Plane
(𝑧& + 𝑟)
Object Radius  𝑟
𝑀
𝑀
𝑀
𝑧&
𝒄= (𝑢., 𝑣., 𝑢1, 𝑣1)
Figure 3: The M × M × M per-view feature volumes
computed in the modeling network corresponds a depth
bounded camera frustum.
The blue box on the image
plane is determined by the camera crop parameter c =
(u−, v−, u+, v+) and together with the depth determines
the bounds of the frustum.
4.1. Modeling
Our modeling step is inspired by space carving [18] in
that our network takes observations from multiple views
and leverages multi-view consistency to build a canonical
representation. However, instead of using photometric con-
sistency, we use latent features to represent each view which
allows our network to learn features useful for this task.
Per-View Features.
We begin by generating a feature
volume for each input view Ii ∈{I1, . . . , IN}. Each
feature volume corresponds to the camera frustum of the
input camera, bounded by the viewport parameter c =
(u−, v−, u+, v+) and depth-wise by z ∈[zc −r, zc + r]
where zc is the distance to the object center and r is the
radius of the object.
Fig. 3 illustrates the generation of
the per-view features. Similar to [38], we use U-Nets [34]
for their property of preserving spatial structure. We ﬁrst
compute 2D features gpix(xi) ∈RC×H×W by passing the
input xi (an RGB image Ii, a binary mask Mi, and op-
tionally depth Di) through a 2D U-Net. The deprojection
unit (p↑) then lifts the 2D image features in RC×H×W to
3D volumetric features in R(C/D)×D×H×W by factoring
the 2D channel dimension into the 3D channel dimension
C′ = C/D and depth dimension D. This deprojection op-
eration is the exact opposite of the projection unit presented
in [27]. The lifted features are then passed through a 3D U-
Net gcam to produce the volumetric features for the camera:
Φi = gcam ◦p↑◦gpix(xi) ∈RC′×M×M×M.
Camera to Object Coordinates. Each voxel in our fea-
ture volume represents a point in 3D space. Following re-
cent works [27, 28, 38], we transform our feature volumes
directly using rigid transformations. Consider a continu-
ous function φ(x) ∈RC′ deﬁning our camera-space latent
representation, where x ∈R3 is a point in camera coordi-
nates. The feature volume Φ is a discrete sample of this
function. This representation in object space is given by
ψ(x′) = φ(W −1x′) where x′ is a point in object coordi-
Per-View
…
GRU
GRU
GRU
…
Canonical
2. Recurrent Fusion
1. Pooling Fusion
Per-View
…
Channel
Mean
Canonical
Figure 4: We illustrate two methods of fusion per-view fea-
ture volumes. (1) Simple channel-wise average pooling and
(2) a recurrent fusion module similar to that of [38]
nates and W = [R|t] is an object-to-camera extrinsic ma-
trix. We compute the object-space volume ˆΨ by sampling
φ(W −1x′
ijk) for each object-space voxel coordinate x′
ijk.
In practice, this is done by trilinear sampling the voxel grid
and edge-padding values that fall outside. Given this trans-
formation operation Tc→o, the object-space feature volume
is given by ˆΨi = Tc→o(Φi).
View Fusion. We now have a collection of feature volumes
ˆΨi ∈{ ˆΨi, . . . , ˆΨN}, each associated with an input view.
Our fusion module f fuses all views into a single canonical
feature volume: Ψ = f( ˆΨ1, . . . , ˆΨN).
Simple channel-wise average pooling yields good results
but we found that sequentially integrating each volume us-
ing a Recurrent Neural Network (RNN) similarly to [38]
slightly improved reconstruction accuracy (see Sec. 6.5).
Using a recurrent unit allows the network to keep and ignore
features from views in contrast to average pooling. This fa-
cilitates comparisons between different views allowing the
network to perform operations similar to the photometric
consistency criterion used in space carving [18]. We use a
Convolutional Gated Recurrent Unit (ConvGRU) [1] so that
the network can leverage spatial information.
4.2. Rendering
Our rendering module takes the fused object volume Ψ
and renders it given arbitrary camera parameters θ. Ideally,
the rendering module would directly regress a color image.
However, it is challenging to preserve high frequency de-
tails through a neural network. U-Nets [34] introduce skip
connections between equivalent-scale layers allowing high
frequency spatial structure to propagate to the end of the
network, but it is unclear how to add skip connections in
the presence of 3D transformations. Existing works such
as [38, 23] train a single network for each scene allowing
the decoder to memorize high frequency information while
the latent representation encodes state information. Trying

to predict color without skip connections results in blurry
outputs. We side-step this difﬁculty by ﬁrst rendering depth
and then using an image-based rendering approach to pro-
duce a color image.
Decoding Depth. Depth is a 3D representation, making it
easier for the network to exploit the geometric structure we
provide. In addition, depth tends to be locally smoother
compared to color allowing more information to be com-
pactly represented in a single voxel.
Our rendering network is a simple inversion of the recon-
struction network and bears many similarities to Render-
Net [27]. First, we pass the canonical object-space volume
Ψ through a small 3D U-Net (hobj) before transforming it to
camera coordinates using the method described in Sec. 4.1.
We perform the transformation with an object-to-camera
extrinsic matrix E instead of the inverse E−1. A second
3D U-Net (hcam) then decodes the resulting volume to pro-
duce a feature volume: Ψ′ = hcam ◦To→c ◦hobj(Ψ) which
is then ﬂattened to a 2D feature grid Φ′ = p↓(Ψ′) using the
projection unit (p↓) from [27] by ﬁrst collapsing the depth
dimension into the channel dimension and applying a 1x1
convolution. The resulting features are decoded by a 2D U-
Net (hpix) with two output branches for depth (hdepth) and
for a segmentation mask (hmask). The outputs of the render-
ing network are given by ydepth(Φ′) = hdepth ◦hpix(Φ′) and
ymask(Φ′) = hmask ◦hpix(Φ′).
Image Based Rendering (IBR). We use image-based
rendering [36] to leverage the reference images to pre-
dict output color.
Given the camera intrinsics K and
depth map for an output view, we can recover the 3D
object-space position of each output pixel (u, v) as X =
E−1 
u−u0
fu z, v−v0
fv z, z, 1
T
, which can be transformed to
the input image frame as x′
i = KiWiX for each input
camera θi = {Ki, Wi}. The output pixel can then copy
the color of the corresponding input pixel to produce a re-
projected color image.
The resulting reprojected image will contain invalid pix-
els due occlusions. There are multiple strategies to weight-
ing each pixel including 1) weighting by reprojected depth
error, 2) weighting by similarity between input and query
cameras, 3) using a neural network. The ﬁrst choice suf-
fers from artifacts in the presence of depth errors or thin
surfaces.
The second approach yields reasonable results
but produces blurry images for intermediate views.
We
opt for the third option. Following deep blending [10], we
train a network that predicts blend weights Wi for each
reprojected input I′
i: Io = P
i Wi ⊙I′
i, where ⊙is an
element-wise product. The blend weights are predicted by
a 2D U-Net. The inputs to this network are 1) the depth pre-
dicted by our reconstruction pipeline, 2) each reprojected
input image I′
i, and 3) a view similarity score s based on
the angle between the input and query poses.
4.3. Implementation Details
Training Data. We train our reconstruction network on
shapes from ShapeNet [4] which contains around 51,300
shapes. We exclude large models for efﬁcient data loading
resulting in around 30,000 models. We generate UV maps
using Blender’s smart UV projection [3] to facilitate textur-
ing. We normalize all models to unit diameter. When ren-
dering, we sample a random image from MS-COCO [21]
for each component of the model. We render with the Beck-
mann model [2] with randomized parameters and also ren-
der uniformly colored objects with a probability of 0.5.
Network Input. We generate our training data at a resolu-
tion of 640 × 480. However, the input to our network is a
ﬁxed size 128 × 128. To keep our inputs consistent and our
network scale-invariant, we ‘zoom’ into the object such that
all images appear to be from the same distance. This is done
by computing a bounding box size (wb, hb) = ( d′w′
fudw, d′h′
fvdh)
where (w, h) is the current image width and height, d is
the distance to the centroid co (See Fig. 3), (w′, h′) is the
desired output size, and d′ is the desired ‘zoom’ distance
and cropping around object centroid projected to image
coordinates (cu, cv). This deﬁnes the viewport parameter
c = (cu −wb/2, cv −hb/2, cu + wb/2, cv + hb/2). The
cropped image is scaled to 128 × 128.
Training. In each iteration of training, we sample a 3D
model and then sample 16 random reference poses and 16
random target poses. Each pose is sampled by uniformly
sampling a unit quaternion and translation such that the ob-
ject stays within frame. We train our network using the
Adam optimizer [16] with a ﬁxed learning rate of 0.001 for
1.5M iterations. Each batch consists of 20 objects with 16
input views and 16 target views. We use an L1 reconstruc-
tion loss for depth and binary cross-entropy for the mask.
We apply the losses to both the input and output views. We
randomly orient our canonical coordinate frame in each iter-
ation by uniformly sampling a random unit quaternion. This
prevents our network from overﬁtting to the implementation
of our latent voxel transformations. We also add motion
blur, color jitter, and pixel noise to the color inputs and add
noise to the input masks using the same procedure as [24].
5. Object Pose Estimation
Given an image I, and a depth map D, a pose estima-
tion system provides a rotation R and a translation t which
together deﬁne an object-to-camera coordinate transforma-
tion E = [R|t] referred to as the object pose. In this sec-
tion, we describe how we use our reconstruction pipeline
described in Sec. 4 to directly optimize for the pose. We
ﬁrst ﬁnd a coarse pose using only forward inference and

then reﬁne it using gradient optimization.
Formulation. Pose is deﬁned by a rotation R and a trans-
lation t. Our formulation also includes the viewport pa-
rameter c deﬁned in Sec. 3. Deﬁning a viewport allows us
to efﬁciently pass the input to the reconstruction network
while also providing scale invariance. We encode the rota-
tion as a quaternion q and translation as t. We assume we
are given an RGB image I, an object segmentation mask
M, and depth D comprising the input x = {I, M, D}.
5.1. Losses
In order to estimate pose, we must provide a crite-
rion which quantiﬁes the quality of the pose.
We use
four loss functions.
One is a standard L1 depth recon-
struction loss Ldepth(D∗, D) = ∥D∗−D∥1 which disam-
biguates the object scale and measures how well the pre-
dicted depth D matches the input depth D∗. We also use
a pixel-wise binary cross entropy loss Lmask on the pre-
dicted mask as well as intersection over union (IoU) loss
Liou(M∗, M) = log U −log I where U is the sum of
the pixels in the union and I is the sum of the pixels in
the intersection of the masks M∗and M. Finally, we in-
troduce a novel latent loss which leverages our reconstruc-
tion network F. Given the input x = {I, M, D}, a la-
tent object Ψ, and a pose θ, the latent loss is deﬁned as
Llatent(x, θ; Ψ) = ∥Hθ(Gθ(x)) −Hθ(Ψ)∥1, where Hθ is
the rendering network up to the projection layer and Gθ is
the modeling network as described in Sec. 4. This loss dif-
fers from auto-encoder based approaches such as [41, 8] in
that 1) our network is not trained on the object, and 2) the
loss is computed directly given the image and camera pose.
Our pose estimation problem is given by:
argmin
θ
Ldepth + λLlatent + γLmask + ηLiou,
(2)
where λ, γ, η are the weights of the lossees. The parameters
of the losses are omitted for clarity.
Parameterization. We parameterize the rotation in the log
quaternion form ω = (0, ω1, ω2, ω3) which ensures that all
updates to the parameters result in a valid unit quaternion:
q = exp (ω) =

cos∥ω∥
ω
∥ω∥sin ∥ω∥

.
(3)
Coarse Initialization. Although we have a differentiable
renderer, the space of poses is non-convex which can lead to
bad local minima when using gradient based optimization.
We therefore bootstrap the pose by computing a coarse es-
timate. This also has the beneﬁt of speeding up inference
since it only requires forward evaluation.
We begin by estimating the translation of the object t =
(x, y, z) as the centroid of the bounding cube deﬁned by
1. Target Image
2. GT Depth
3. Our Depth
4. Depth Error
5.37° Error
1.74° Error
a.
b.
Figure 5: Two examples from the ModelNet experiment.
(1) target image, (2) ground truth depth, (3) optimized pre-
dicted depth, and (4) L1 error between the ground truth and
our prediction. (a) illustrates how a pose with low depth
error can still result in a relatively high angular error.
Target
Image
Predicted
Color
Target
Depth
Predicted
Depth
Depth
Error
a.
b.
c.
Figure 6: Qualitative results on the MOPED dataset.
the mask bounding box c and corresponding depth values.
We initialize k poses using the estimated translation. To get
good coverage of possible orientations, we evenly sample
azimuth and elevation angles using a Fibonacci lattice [9]
then uniformly sample a random yaw angle. We use the
cross entropy method [7] to optimize the translation and log
quaternion parameters, and use a Gaussian Mixture Model
as the probability distribution.
Pose Optimization. Our entire pipeline is differentiable
end-to-end. We can therefore optimize Eq. (2) using gradi-
ent optimization. Given a latent object Ψ and a coarse pose
estimate θ, we compute the loss and propagate the gradi-
ents back to the camera pose. This step only requires the
rendering network and does not use the modeling network.
The image-based rendering network is also not used in this
step. We jointly optimize the rotation q, translation t, and
viewport c using Adam [16].
6. Experiments
We evaluate our method on LINEMOD [11], Model-
Net [48] and our new dataset MOPED. We aim to evaluate
pose estimation accuracy on unseen objects.

Table 1: Evaluation on LINEMOD. We report the ADD recall metric [12], comparing against DeepIM and pix2pose. Sym-
metric objects are indicated by ∗and the pose is considered correct if it is ﬂipped along the z axis.
Method
Input
ape
benchvice
camera
can
cat
driller
duck
eggbox∗
glue∗
holepunch
iron
lamp
phone
mean
Ours
RGB-D
88.0
92.4
74.4
88.8
94.5
91.7
68.1
96.3
94.9
82.1
74.6
94.7
91.5
87.1
DeepIM [20]
RGB
77.0
97.5
93.5
96.5
82.1
95.0
77.7
97.1
99.4
52.8
98.3
97.5
87.7
88.6
pix2pose [31]
RGB
58.1
91.0
60.9
84.4
65.0
76.3
43.8
96.8
79.4
74.8
83.4
82.0
45.0
72.4
6.1. Evaluation Metrics
We use four main evaluation metrics. 1) (k◦, kcm): a
pose is considered correct if it is within k◦and kcm of the
ground truth target pose where the angular metric is the an-
gle between the orientations. 2) ADD [12]: the average dis-
tance between points after being transformed by the ground
truth and predicted poses. 3) ADD-S: A modiﬁcation to
ADD that computes the average distance to the closest point
rather than the ground truth point to account for symmetric
objects. 4) Proj.2D: the pixel distance between the projected
points of the ground truth and predicted pose.
6.2. Experiments on the LINEMOD Dataset
We evaluate our method on the LINEMOD dataset. We
compare our results with DeepIM [20] and pix2pose [31].
Both of these methods are trained on the LINEMOD
dataset. DeepIM uses the 3D models in an online fashion to
reﬁne the pose. We do not train on the dataset. Instead, our
network is given 16 reference views of each object during
inference time. We use the provided segmentation masks
during inference. We follow the train/test split of [13] and
sample our reference views from the training set. We follow
the same evaluation methodology of [12], reporting the per-
centage of poses with ADD metric less than 10% of the ob-
ject diameter. Table. 1 shows the results. Our experiments
show that our method performs on-par with state of the art
supervised methods despite having never seen the objects.
6.3. Experiments on the ModelNet Dataset
We conduct experiments on ModelNet [48] to evaluate
the generalization of our method toward unseen object cat-
egories. To this end, we train our network on all the meshes
in ShapeNetCore [4] excluding the categories we are going
to evaluate on. We closely follow the evaluation protocol of
[20] here. The model is evaluated on 7 unseen categories:
bathtub, bookshelf, guitar, range hood, sofa, wardrobe, and
TV stand. For each category, 50 pairs of initial and target
object pose are sampled. We compared with [20] and [40]
where all the methods initialized with the initial pose and
evaluated on how successful they are on estimating the tar-
get pose. We report three metrics: (5◦, 5cm), ADD within
10% of the object diameter, and Proj.2D within 5 pixels.
Table 2 shows the quantitative results on the Model-
Net dataset.
On average, our method achieves state-of-
the-art results on all the metrics thanks to our ability to
Table 2: ModelNet pose reﬁnement experiments compared
to DeepIM (DI) [20] and Multi-Path Learning (MP) [40].
(5◦, 5cm)
ADD (0.1d)
Proj2D (5px)
DI
MP
Ours
DI
MP
Ours
DI
MP
Ours
bathtub
71.6
85.5
85.0
88.6
91.5
92.7
73.4
80.6
94.9
bookshelf
39.2
81.9
80.2
76.4
85.1
91.5
51.3
76.3
91.8
guitar
50.4
69.2
73.5
69.6
80.5
83.9
77.1
80.1
96.9
range hood
69.8
91.0
82.9
89.6
95.0
97.9
70.6
83.9
91.7
sofa
82.7
91.3
89.9
89.5
95.8
99.7
94.2
86.5
97.6
tv stand
73.6
85.9
88.6
92.1
90.9
97.4
76.6
82.5
96.0
wardrobe
62.7
88.7
91.7
79.4
92.1
97.0
70.0
81.1
94.2
Mean
64.3
84.8
85.5
83.6
90.1
94.3
73.3
81.6
94.7
Table 3: AUC metrics on MOPED by reference view count.
# Views
1
2
4
8
16
32
ADD
23.9
34.2
49.8
63.1
62.7
61.1
ADD-S
78.7
82.8
89.2
91.1
90.8
90.8
Proj.2D
9.5
15.9
30.4
44.7
46.8
42.6
Table 4: AUC metrics for different view fusion strategies
ADD
ADD-S
Proj.2D
Avg Pool
57.5
90.1
40.1
ConvGRU
63.1
91.1
44.7
perform continuous optimization on pose.
However, for
the (5◦, 5cm) metric, there are object categories that our
method performs worse despite performing well on all other
metrics. One reason is the image and spatial resolution.
The input and output images to our network have resolu-
tion 128 × 128. The resolution of our voxel representation
is 16×16×16. The limited resolution can hinder the perfor-
mance for small objects or objects that are distant from the
camera. Small changes in the depth of each pixel may dis-
proportionately affect the rotation of the object compared
to our losses. Fig. 5 shows examples from the ModelNet
experiment illustrating this limitation.
6.4. Experiments on the MOPED Dataset
We introduce the Model-free Object Pose Estimation
Dataset (MOPED). MOPED consists of 11 objects, shown
in Fig. 7. For each object, we take multiple RGB-D videos
cover all views the object. We ﬁrst use KinectFusion [26]
to register frames from a single capture and then use a
combination of manual annotation and automatic registra-
tion [52, 35, 53] to align separate captures. We generate
object segmentation maps using [50]. For each object, we

Table 5: Quantitative Results on MOPED Dataset. We report the Area Under Curve (AUC) for each metric. † shows the
average with the symmetric objects rinse aid and cheezit excluded. Ours (D) is our method with the depth loss only and Ours
(D+L) is our method with both the depth and latent losses. In some cases, the latent loss helps resolve pose ambiguities not
possible with only depth.
PoseRBPF
Ours (D)
Ours (D+L)
Input
Textured 3D Mesh
Image + Camera Pose
Training
Yes
No
# Networks
Per-Object
Single Universal
Object
ADD
ADD-S
Proj.2D
ADD
ADD-S
Proj.2D
ADD
ADD-S
Proj.2D
black drill
75.3
91.7
54.2
90.3
95.4
78.1
89.4
95.1
76.2
duplo dude
84.1
93.5
61.5
89.4
94.9
77.4
88.9
94.7
76.2
duster
72.2
90.6
51.5
40.0
88.3
17.8
47.2
87.9
15.5
graphics card
71.8
81.3
51.2
73.1
79.3
62.2
73.0
79.2
62.5
orange drill
70.3
87.5
43.9
78.4
93.7
62.3
78.3
93.5
61.7
pouch
26.5
80.7
14.1
61.9
91.8
44.2
54.8
91.7
31.4
remote
49.6
82.1
25.1
58.2
92.0
19.0
60.0
91.9
26.3
toy plane
48.6
88.6
29.5
55.5
89.7
46.5
82.9
92.2
70.5
vim mug
59.0
92.9
39.4
51.5
91.8
22.7
40.0
91.7
8.0
rinse aid
87.2
94.7
68.7
71.3
93.2
41.7
67.4
93.3
34.7
cheezit
64.3
93.8
55.6
24.8
92.4
20.5
24.6
92.2
20.2
Mean
64.4
88.9
45.0
63.1
91.1
44.7
64.2
91.2
43.9
Mean w/o sym
61.9
87.7
41.2
66.5
90.8
47.8
68.3
90.9
47.6
a.
b.
c.
e.
d.
f.
g.
h.
i.
j.
k.
Figure 7: Objects in MOPED–a new dataset for model-free
pose estimation. The objects shown are: (a) toy plane, (b)
duplo dude, (c) cheezit, (d) duster, (e) black drill, (f) or-
ange drill, (g) graphics card, (h) remote, (i) rinse aid, (j)
vim mug, and (k) pouch.
select reference frames with farthest point sampling to en-
sure good coverage of the object. For test sequences, we
capture each object in 5 different environments. We sam-
ple every other frame for evaluation videos. This results
in approximately 300 test images per object. We evaluate
our method and baselines using three metrics for which we
provide the Area Under Curve (AUC): 1) ADD with thresh-
old between 0 −10cm, 2) ADD-S with threshold between
0−10cm, and 3) Proj.2D with threshold between 0−40px.
We compute all metrics for all sampled frames.
We compare our method with PoseRBPF [8], a state-
of-the-art model-based pose estimation method.
Since
PoseRBPF requires textured 3D models, we reconstruct
a mesh for each object by aggregating point clouds from
reference captures and building a TSDF volume.
The
point clouds are integrated into the volume using Kinect-
Fusion [26]. The meshes have artifacts such as washed out
high frequency details and shadow vertices due to slight
misalignment (see supplementary materials). Table 5 shows
quantitative comparisons on the MOPED dataset.
Note
that our method is not trained on the test objects while
PoseRBPF has a separate encoder for each object.
Our
method achieves superior performance on both ADD and
ADD-S. We evaluate different version of our method with
different combinations of loss functions. Compared to our
combined loss, optimizing only Ldepth performs better for
geometrically asymmetric objects but worse on textured
objects such as the cheezit box. Optimizing both losses
achieves better results on textured objects. Fig. 6 shows
estimated poses for different test images. Please see sup-
plementary materials for qualitative examples.
6.5. Ablation Studies
In this section, we analyze the effect of different design
choices and how they affect the robustness of our method.
Number of reference views. We ﬁrst evaluate the sensitiv-
ity of our method to the number of input reference views.
Novel view synthesis is easier with more reference views
because there is a higher chance that a query view will be
close to a reference view. Table 3 shows that the accuracy

increases with the number of reference views. In addition,
having more than 8 reference views only yields marginal
performance gains shows that our method does not require
many views to achieve good pose estimation results.
View Fusion. We compare multiple strategies for aggregat-
ing the latent representations from each reference view. The
naive way is to use a simple pooling function such as av-
erage/max pooling. Alternatively, we can integrate the vol-
umes using an RNN such as a ConvGRU so that the network
can reason across views. Table 4 shows the quantitative
evaluation of these two variations. Although the average
performance of the objects are very similar, the ConvGRU
variation performs better than the average pooling variation.
This indicates the importance of spatial relationship in the
voxel representation for pose estimation.
7. Conclusion
We have presented a novel framework for building 3D
object representations at inference time using a small num-
ber of reference images, as well as an accompanying neu-
ral renderer to render the 3D representation from arbitrary
6D viewpoints. Our networks are trained on thousands of
shapes with random textures rendered under various light-
ing conditions allowing it to robustly generalize to unseen
objects without additional training.
We leverage our reconstruction and rendering pipeline
for zero-shot pose estimation. We perform pose estimation
given just a small number of reference views and without
needing to train any network. This greatly simpliﬁes the
process for performing pose estimation on novel objects as
a detailed 3D model is not required. In addition, we have
a single universal network which works for all objects in-
cluding unseen ones. For future work, we plan to investi-
gate unseen object pose estimation in cluttered scenes with
occlusions. We also plan to speed up the pose estimation
process by applying network optimization techniques.
Acknowledgements
We thank Xinke Deng for helpful discussions.
References
[1] Nicolas Ballas, Li Yao, Chris Pal, and Aaron Courville.
Delving deeper into convolutional networks for learning
video representations.
arXiv preprint arXiv:1511.06432,
2015. 4
[2] Petr Beckmann and Andre Spizzichino. The scattering of
electromagnetic waves from rough surfaces. Norwood, MA,
Artech House, Inc., 1987. 5
[3] Blender Online Community. Blender - a 3D modelling and
rendering package. Blender Foundation, 2019. 5
[4] Angel X. Chang, Thomas Funkhouser, Leonidas Guibas, Pat
Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Mano-
lis Savva, Shuran Song, Hao Su, Jianxiong Xiao, Li Yi,
and Fisher Yu. ShapeNet: An Information-Rich 3D Model
Repository.
Technical Report arXiv:1512.03012 [cs.GR],
2015. 2, 5, 7
[5] Wenzheng Chen, Huan Ling, Jun Gao, Edward Sm