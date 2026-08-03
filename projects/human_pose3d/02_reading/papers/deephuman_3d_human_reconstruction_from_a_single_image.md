# DeepHuman: 3D Human Reconstruction From a Single Image

> 2019 · id: W2991621301 · arXiv: 1903.06473 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
We propose DeepHuman, an image-guided volume-to-
volume translation CNN for 3D human reconstruction from
a single RGB image. To reduce the ambiguities associated
with the surface geometry reconstruction, even for the re-
construction of invisible areas, we propose and leverage a
dense semantic representation generated from SMPL model
as an additional input.
One key feature of our network
is that it fuses different scales of image features into the
3D space through volumetric feature transformation, which
helps to recover accurate surface geometry. The visible sur-
face details are further reﬁned through a normal reﬁnement
network, which can be concatenated with the volume gen-
eration network using our proposed volumetric normal pro-
jection layer. We also contribute THuman, a 3D real-world
human model dataset containing about 7000 models. The
network is trained using training data generated from the
dataset. Overall, due to the speciﬁc design of our network
and the diversity in our dataset, our method enables 3D hu-
man model estimation given only a single image and out-
performs state-of-the-art approaches.

## introduction
Image-based reconstruction of a human body is an im-
portant research topic for VR/AR content creation [6], im-
age and video editing and re-enactment [21, 52], holopor-
tation [48] and virtual dressing [51]. To perform full-body
3D reconstruction, currently available methods require the
fusion of multiview images [7, 29, 22] or multiple tempo-
ral images [2, 1] of the target. Recovering a human model
from a single RGB image remains a challenging task that
has so far attracted little attention. Using only a single im-
age, available human parsing studies have covered popular
topics starting from 2D pose detection [50, 5, 47], advanc-
ing to 3D pose detection [41, 53, 78], and ﬁnally expanding
to body shape capture [31] using a human statistic template
such as SMPL [39]. However, the statistic template can cap-
ture only the shape and pose of a minimally clothed body
and lack the ability to represent a 3D human model under
Figure 1. Given only a single RGB image, our method automati-
cally reconstructs the surface geometry of clothed human body.
a normal clothing layer. Although the most recent work,
BodyNet[64], has pioneered research towards this goal, it
only generates nearly undressed body reconstruction results
with occasionally broken body parts. We believe that 3D
human reconstruction under normal clothing from a single
image, which needs to be further studied, will soon be the
next hot research topic.
Technically, human reconstruction from a single RGB
image is extremely challenging, not only because of the re-
quirement to predict the shape of invisible parts but also due
to the need for the geometry recovery for visible surface.
Therefore, a method capable of accomplishing such a task
should meet two requirements: ﬁrst, the degrees of freedom
of the output space should be constrained to avoid unrea-
sonable artifacts (e.g., broken body parts) in invisible areas;
second, the method should be able to efﬁciently extract ge-
ometric information from the input image, such as clothing
styles and wrinkles, and fuse them into the 3D space.
In this paper, we propose DeepHuman, a deep learning-
based framework aiming to address these challenges.
Speciﬁcally, to provide a reasonable initialization for the
network and constrain the degrees of freedom of the output
space, we propose to leverage parametric body models by
generating a 3D semantic volume and a corresponding 2D
semantic map as a dense representation after estimating the
shape and pose parameters of a parametric body template
(e.g., SMPL[39]) for the input image. Note that the require-
ment of inferring a corresponding SMPL model for an im-
1
arXiv:1903.06473v2  [cs.CV]  28 Mar 2019

age is not strict; rather, several accurate methods are avail-
able for SMPL prediction from a single image[4, 31]. The
input image and the semantic volume&map are fed into an
image-guided volume-to-volume translation CNN for sur-
face reconstruction. To accurately recover surface geometry
like the hairstyle or cloth contours to the maximum possible
extent, we propose a multi-scale volumetric feature trans-
formation so that those different scales of image guidance
information can be fused into the 3D volumes. Finally, we
introduce a volumetric normal projection layer to further re-
ﬁne and enrich visible surface details according to the input
image. This layer is designed to concatenate the volume
generation network and the normal reﬁnement network and
enables end-to-end training. In summary, we perform 3D
human reconstruction in a coarse-to-ﬁne manner by decom-
posing this task into three subtasks: a) parametric body esti-
mation from the input image, b) surface reconstruction from
the image and the estimated body, and c) visible surface de-
tail reﬁnement according to the image.
The available 3D human dataset [65] used for network
training in BodyNet [64] is essentially a set of synthe-
sized images textured over SMPL models [39]. No large-
scale human 3D dataset with surface geometry under nor-
mal clothing is publicly available. To ﬁll in this gap, we
present the THuman dataset. We leverage the state-of-the-
art DoubleFusion [76] technique for real-time human mesh
reconstruction and propose a capture pipeline for fast and
efﬁcient capture of outer geometry of human bodies wear-
ing casual clothes with medium-level surface detail and tex-
ture. Based on this pipeline, we perform capture and recon-
struction of the THuman dataset, which contains about 7000
human meshes with approximately 230 kinds of clothes un-
der randomly sampled poses.
Our network learns from the training corpus synthesized
from our THuman dataset. Beneﬁting from the data diver-
sity of the dataset, the network generalizes well to natu-
ral images and provides satisfactory reconstruction given
only a single image. We demonstrate improved efﬁciency
and quality compared to current state-of-the-art approaches.
We also show the capability and robustness of our method
through an extended application on monocular videos.

## method
4.1. Network Architecture
Our network consists of 3 components, namely an im-
age feature encoder G, a volume-to-volume (vol2vol) trans-
lation network H and a normal reﬁnement network R, as
shown in Fig.3. The image feature encoder G aims to ex-
tract multi-scale 2D feature maps M(k)
f (k = 1, . . . , K)
from the combination of I and Ms.
The vol2vol net-
work is a volumetric U-Net [73], which takes Vs and
M(k)
f (k = 1, . . . , K) as input, and outputs an occupancy
volume Vo representing the surface. Our vol2vol network
H fuses multi-scale semantic features M(k)
f (k = 1, . . . , K)
into its encoder through a multi-scale volumetric feature
transformer. After generating Vo, a normal reﬁnement U-
Net [54] R further reﬁnes the normal map N after calculat-
ing it directly from Vo through a volume-to-normal projec-
tion layer. All operations in the network are differentiable,
and therefore, it can be trained or ﬁne-tuned in an end-to-
end manner.
4.1.1
Multi-scale Volumetric Feature Transformer
In this work, we extend the Spatial Feature Transformer
(SFT) layer [68] to handle 2D-3D data pairs in the multi-
scale feature pyramid, and propose multi-scale Volumet-
ric Feature Transformer (VFT). SFT was ﬁrst used in [68]
to perform image super-resolution conditioned on semantic
categorical priors to avoid the regression-to-the-mean prob-
lem. A SFT layer learns to output a modulation parameter
pair (α, β) based on the input priors. Then transformation
on the feature map F is carried out as:
SFT (F) = α ⊙F + β
(1)
where ⊙is Hadamard product.
In our network, at each level k, a feature volume V(k)
f
(blue cubes in Fig.3) and a feature map M(k)
f
(orange
squares in Fig.3) are provided by previous encoding layers.
Similar to [68], we ﬁrst map the feature map M(k)
f
to mod-
ulation parameters (αk, βk) through convolution+activation
layers (see the second row of Fig.4). Note that the operation
in Eqn.(1) cannot be applied directly on V(k)
f
and M(k)
f
be-
cause of dimension inconsistency (V(k)
f
has a z-axis while
(αk, βk) doesn’t.) Therefore, we slice the feature volume
along the z-axis into a series of feature slices, each of which
has a thickness of 1 along the z-axis. Then we apply the
same element-wise afﬁne transformation to each feature z-
slice independently:
VFT

V(k)
f (zi)

= αk ⊙V(k)
f (zi) + βk
(2)
where V(k)
f (zi) is the feature slice on plane z = zi, zi =
1, 2, . . . , Z and Z is the maximal z-axis coordinate. The
output of a VFT layer is the re-combination of transformed
feature slices. The operations applied by VFT layers are
illustrated in Fig.4.
The superiority of VFT is three-fold. Firstly, compared
to converting feature volumes/maps into a latent codes and
concatenating them at the network bottleneck, it preserves
the shape primitiveness of feature maps and thus encodes
more local information. Second, it is efﬁcient. Using VFT,
feature fusion can be achieved in one single pass of afﬁne
transformation, without requiring extra convolutions, full
connection or other operations. Third, it is ﬂexible. VFT
can be performed on either the original image/volume or
downsampled feature maps/volumes, which makes it pos-
sible to fuse different scales of features, enabling much
deeper feature transfer.
In order to integrate image features to the maximum
possible extent, we perform volumetric feature transforma-
tion on the multi-scale feature pyramid; see the blue ar-
rows/lines in Fig.3 for illustration. We only perform VFT
in the encoder part of our vol2vol network; however, the
4

Figure 3. Network architecture. Our network is mainly composed of an image feature encoder (orange), a volume-to-volume translation
network (blue & green) and a normal reﬁnement network (yellow).
Figure 4. Illustration of volumetric feature transformation at level
k.
Figure 5. Illustration of differentiable depth projection.
transformation information can be propagated to the de-
coder through skip-connections. As discussed in Sec.6.3,
the multi-scale feature transformation helps preserve geom-
etry details compared to directly concatenating latent vari-
ables at the network bottleneck.
4.1.2
Volume-to-normal Projection Layer
Our goal is to recover geometrical details (e.g. wrinkles
and cloth boundary) on the visible surface of the human
model. However, a volume-based representation is unable
to recover such ﬁne-grain details due to resolution limita-
tions. Thus, we encode the frontal geometrical details on
2D normal maps, which can be directly calculated from
the occupancy volume using our differentiable volume-to-
normal projection layer. The layer ﬁrst projects a depth map
directly from the occupancy volume, transforms the depth
map into a vertex map, and then calculates the normal maps
through a series of mathematical operations.
Fig.5 is a 2D illustration explaining how the layer
projects depth maps. In Fig.5(a), the blue circle is the model
we aim to reconstruct, and the voxels occupied by the circle
are marked in grey. Consider the pixel p = (xp, yp) on the
image plane as an example. To calculate depth value D(p)
of p according to Vo, a straightforward method is to con-
sider a ray along the z-axis and record the occupancy status
of all voxels along that ray (Fig.5(b)). Afterwards, we can
determine D(p) by ﬁnding the nearest occupied voxel. For-
mally, D(p) is obtained according to
D(p) = inf
n
z|V(xpypz)
o
= 1
o
(3)
where V(xpypz)
o
denotes the value of the voxel at coordinate
(xp, yp, z). Although this method is straightforward, it is
difﬁcult to incorporate the operation, inf{·} into neural net-
works due to the complexity of differentiating through it.
Therefore, we transform the occupancy volume to a depth
volume Vd by applying a transformation f:
V(xyz)
d
= f(V(xyz)
o
) = M(1 −V(xyz)
o
) + zV(xyz)
o
(4)
where M is a sufﬁciently large constant. Then D(p) can be
computed as:
D(p) = min
z
f(V(xpypz)
d
),
(5)
5

as illustrated in Fig.5(c).
After depth projection, we transform the depth map to a
vertex map Mv by assigning x and y coordinates to depth
pixels according to their positions on the images. Then So-
bel operators are used to calculate the directional deriva-
tive of the vertex map along both the x and y directions:
Gx = Sx ∗Mv, Gy = Sy ∗Mv, where Sx and Sy are
Sobel operators. The normal at pixel p = (xp, yp) can be
calculated as:
N(xpyp) = Gx(p) × Gy(p),
(6)
where × denotes cross product. Finally, N is up-sampled
by a factor of 2 and further reﬁned by a U-Net.
4.2. Loss Functions
Our loss functions used to train the network parame-
ters consist of reconstruction errors for the 3D occupancy
ﬁeld and 2D silhouette, as well as the reconstruction loss
for normal map reﬁnement. We use extended Binary Cross-
Entropy (BCE) loss for the reconstruction of occupancy vol-
ume [25]:
LV = −
1
 ˆVo

X
x,y,z
γ ˆV(xyz)
o
log V(xyz)
o
+
(1 −γ)

1 −ˆV(xyz)
o

log

1 −V(xyz)
o

(7)
where ˆVo is the ground-truth occupancy volume corre-
sponding to Vo, V(xyz)
o
and ˆV(xyz)
o
are voxels in the re-
spective volumes at coordinate (x, y, z), and γ is a weight
used to balance the loss contributions of occupied and un-
occupied voxels. Similar to [64], we use a multi-view re-
projection loss on the silhouette as additional regularization:
LF S = −
1
ˆSfv

X
x,y
S(xy)
fv
log S(xy)
fv +

1 −ˆS(xy)
fv

log

1 −S(xy)
fv

(8)
where LF S denotes the front-view silhouette re-projection
loss, Sfv is the silhouette re-projection of Vo, ˆSfv is the
corresponding ground-truth silhouette, and S(xy)
fv
and ˆS(xy)
fv
denote their respective pixel values at coordinate (x, y).
Assuming a weak-perspective camera, we can easily ob-
tain S(xy)
fv
through orthogonal projection [64]: S(xy)
fv
=
maxz V(xyz)
o
. The side-view re-projection loss LSS is de-
ﬁned similarly.
For normal map reﬁnement, we use the cosine distance
to measure the difference between predicted normal maps
and the corresponding ground truth:
LN =
1
 ˆN

X
x,y
1 −< N(xy), ˆN(xy) >
|N(xy)| · | ˆN(xy)|
(9)
Table 1. Network Architecture Details.
Net
Layer
Kernel
Stride
Output
G
conv+lrelu
4
2
96 × 64 × 8
conv+lrelu
4
2
48 × 32 × 16
conv+lrelu
4
2
24 × 16 × 32
conv+lrelu
4
2
12 × 8 × 64
conv+lrelu
4
2
6 × 4 × 128
H
conv+lrelu
4
2
64 × 96 × 64 × 8
conv+lrelu
4
2
32 × 48 × 32 × 16
conv+lrelu
4
2
16 × 24 × 16 × 32
conv+lrelu
4
2
8 × 12 × 8 × 64
conv+lrelu
4
2
4 × 6 × 4 × 128
transconv+lrelu
4
2
8 × 12 × 8 × 64
transconv+lrelu
4
2
16 × 24 × 16 × 32
transconv+lrelu
4
2
32 × 48 × 32 × 16
transconv+lrelu
4
2
64 × 96 × 64 × 8
transconv+lrelu
4
2
128 × 192 × 128 × 4
conv+sigmoid
3
1
128 × 192 × 128 × 1
R
conv+lrelu
4
2
192 × 128 × 16
conv+lrelu
4
2
96 × 64 × 32
conv+lrelu
4
2
48 × 32 × 32
conv+lrelu
4
2
24 × 16 × 32
conv+lrelu
4
2
12 × 8 × 32
transconv+lrelu
4
2
24 × 16 × 32
transconv+lrelu
4
2
48 × 32 × 32
transconv+lrelu
4
2
96 × 64 × 32
transconv+lrelu
4
2
192 × 128 × 16
transconv+lrelu
4
2
384 × 256 × 8
conv+tanh
3
1
384 × 256 × 3
* The term “conv” is convolution for short, “transconv” is transp

## experiments
We demonstrate our approach with a range of photos of
people in Fig.6. In this ﬁgure, image (A) and (B) are test-
ing images synthesized from our THUman dataset, while
image (C)-(H) are natural images sampled from the LIP
dataset[15]. Note that the subjects in image (A) and (B) do
not appear in our training data. As shown in Fig.6 our ap-
proach is able to reconstruct both the 3D human models and
surface details like cloth wrinkles (see Fig.6(a,b,c)). From
Fig.6(d,e,f) we can also see that our method can not only
reconstruct the 3D meshes according to input images and
detected SMPL models, but also recover discontinuities on
the surface like a belt and the hem of a dress. In Fig.7 we
show an extended application on 3D human performance
capture from a single-view RGB video. It should be noted
that the reconstruction results are generated by applying our
method on each the video frame independently, without any
temporal smoothness involved. The results demonstrate the
ability of our method to tackle various human poses and its
robust performance. Please see the supplemental video for
more details.
6.2. Comparison
6.2.1
Competing Approaches
We compare our method against two two state-of-the-art
deep learning based approaches for single view 3D human
reconstruction. To eliminate the effect of dataset bias, we
ﬁne-tuned the pre-trained model of both network with the
same training data as we use to train our network.
(1) HMR. In [31], Kanazawa et al.
proposed a neu-
ral network to directly regress the shape and pose param-
eter of SMPL from an RGB image. The output of HMR
7

Figure 6. Reconstruction results on synthesis images and natural images. The input images are presented in the ﬁrst row, while last four
rows show the results of HMR[31] (in orange), BodyNet[64] (in green) and our method (in blue). For BodyNet and our method we render
the results from two views, i.e., a front view and a side view.
is a 75-D vector, which can be used to generate a trian-
gular mesh of SMPL through linear shape blending and
pose skinning[39]. It is the state-of-the-art among avail-
able methods for single-view pose and shape estimation[31,
4, 16, 60, 63]. We ﬁne-tuned the pretrained HMR model
using the color images and the corresponding ground-truth
shape/pose parameters in our synthesis training data.
(2) BodyNet. In [64], Varol et al. proposed a neural net-
work for direct inference of volumetric body shape from a
single image. The output of BodyNet is a 128 × 128 × 128
occupancy volume with similar deﬁnition in Sec.3.
Bo-
dyNet is the most related work to this paper. We ﬁne-tuned
the whole network of BodyNet using the color images and
the ground-truth occupancy volumes in our training set.
6.2.2
Comparison Results
We compare against HMR and BodyNet both qualitatively
and quantitatively.
(1) For qualitative comparison, we feed all the net-
works with the same images and convert the network output
into a triangular mesh. The results are rendered in Fig.6.
As shown in the ﬁgure, our method is able to achieve much
more detailed reconstruction than HMR and BodyNet (See
Fig.6(a∼f)).
In addition, our method has higher robust-
ness than BodyNet when some body parts are occluded (See
Fig.6(b,g,h)).
(2) The quantitative comparison is conducted on the
testing set of our synthetic data.
We convert the out-
put of HMR to occupancy volumes with a resolution of
128×192×128. We also upsampled the output of BodyNet
8

Figure 7. 3D reconstruction from a monocular video using our method.
by a factor of 1.5 using trilinear interpolation, and then crop
the volume to make it have the same resolution. After that
we use the mean Intersection-over-Union (IoU score) be-
tween predicted 3D volumes and their ground-truth as the
comparison metric. It should be noted that the predicted
volume and the ground-truth may be unaligned along the
z-axis because of depth ambiguities. Therefore, we shift
the predicted volume along z-axis to search for the best
alignment (i.e., to maximize IoU score between the ground-
truth volumes and the predict ones), and regard the maxi-
mum IoU score as the ﬁnal score. The results are presented
in Tab.2. As shown by the numerical results, our method
achieves the most accurate reconstruction among all the ap-
proaches. BodyNet occasionally produces broken bodies
and consequently gets the lowest score.
Table 2. Quantitative comparison using 3D IOU score.

## related_work
Human Models from Multiview Images.
Previous
studies focused on using multiview images for human
model reconstruction [30, 58, 36]. Shape cues like silhou-
ette [42, 38], stereo and shading cues have been integrated
in both passive [58, 36, 71] and active illumination [69, 67]
modes to improve the reconstruction performance. State-of-
the-art real-time [10, 9] and extremely high-quality [7] re-
construction results have also been demonstrated with tens
or even hundreds of cameras using binocular [11] or multi-
view stereo matching [12] algorithms. To capture detailed
motions of multiple interacting characters, more than six
hundred cameras have been used to overcome the occlu-
sion challenges [28, 29]. However, all these multi-camera
systems require complicated environment setups including
camera calibration, synchronization and lighting control.
To reduce the difﬁculty of system setup, human model
reconstruction from extremely sparse camera views has re-
cently been investigated by using CNNs for learning silhou-
ette cues [14] and stereo cues [22]. These systems require
about 4 camera views for a coarse-level surface detail cap-
ture. Note also that although temporal deformation systems
using lightweight camera setups (usually with about eight
cameras) [66, 8, 13] have been developed for dynamic hu-
man model reconstruction using skeleton tracking [66, 37]
or human mesh-based template deformation [8], these sys-
tems assume a pre-scanned subject-speciﬁc human template
as a key model for deformation.
Human Models from Temporal Images. To explore
low-cost and convenient human model capture, many stud-
ies try to capture a human using only a single RGB or
RGBD camera. Since only a single-view camera is needed,
methods in this category require the aggregation of multiple
temporal frames for full-body model generation.
For RGBD images, DynamicFusion [46] breaks the
static scene assumption and deforms the non-rigid target
for TSDF fusion on a canonical static model. Many of the
following approaches have tried to improve the robustness
by adding color features [23], shading constraints [19] and
articulated prior [75] and dealing with topology changes
[56, 57]. The recently appeared DoubleFusion [76] method
introduced a human shape prior into the fusion pipeline and
achieved state-of-the-art real-time efﬁciency, robustness,
and loop closure performance for efﬁcient human model re-
construction even in cases of fast motions. There are also
ofﬂine methods for global registration of multiple RGBD
images to obtain a full-body model [35]. To reconstruct
a human body using a single-view RGB camera, methods
have been proposed for rotating the camera while the tar-
get remains as static as possible [79], or keeping the camera
static while the target rotates [2, 1]. Recently, human perfor-
mance capture that can reconstruct dynamic human models
using only a single RGB camera has been proposed [72] and
sped up to be run in real-time [20], however, similar to the
multicamera scenario [66, 8, 13], such approaches require a
pre-scanned human model obtained using a laser scanner or
temporal images-based like [79, 2, 1].
Human Parsing from a Single Image.
Parsing
human from a single image has recently been a popu-
lar topic in computer vision.
The research can be cate-
gorized into sparse 2D parsing (2D skeleton estimation)
[70, 50, 5, 47], sparse 3D parsing (3D skeleton estimation)
[41, 49, 53, 78, 59, 62, 44, 74], dense 2D parsing [18, 17]
and dense 3D parsing (shape and pose estimation). Dense
2

3D parsing from a single image has attracted substantial in-
terest recently because of the emergence of human statisti-
cal models like SCAPE [3] and SMPL [39]. For example,
by ﬁtting the SCAPE or SMPL model to the detected 2D
skeleton and other shape cues of an image [4, 34], or by
regressing [31, 60, 63] the SMPL model using CNNs, the
shape and pose parameters can be automatically obtained
from a single image.
Regarding single-view human model reconstruction,
there are only two recent works by Varol et al. [64] and
Jackson et al. [26]. In the former study, the 3D human
datasets used for the training process are essentially syn-
thesized human imagery textured over SMPL models (lack-
ing geometry details), leading to SMPL-like voxel geome-
tries in their outputs. The latter study shows the ability to
output high-quality details, but their training set is highly
constrained, leading to difﬁculty in generalization, e.g., to
different human poses.
3D Human Body Datasets. Most of the available 3D
human datasets are used for 3D pose and skeleton detection.
Both the HumanEva [55] and Human3.6M [24] datasets
contain multiview human video sequences with ground-
truth 3D skeleton motion obtained from a marker-based mo-
tion capture system. Because of the need to wear markers or
special suits, both of these datasets have limited apparel di-
vergence. MPI-INF-3DHP [43] dataset enriches the cloth
appearance by using a multiview markerless mocap sys-
tem. However, all the above datasets lack a 3D model of
each temporal frame. Recently, with the emergence of the
requirement of pose and shape reconstruction from a sin-
gle image, the synthesized SURREAL [65] datasets have
been created for this task by rendering SMPL models with
different shape and pose parameters under different cloth-
ing textures. The “Unite the People” dataset [34] provides
real-world human images annotated semi-automatic with
3D SMPL models. These two datasets, in contrasts to our
dataset, do not contain surface geometry details.

## conclusion
Limitations.
Our method relies on HMR to generate a
dense semantic representation from SMPL model. Conse-
quently, we cannot give an accurate reconstruction if the
estimation of SMPL model is erroneous; see Fig.12 for an
example. Additionally, the reconstruction of invisible areas
is over-smoothed; using a generative adversarial network
may force the network to learn to add realistic details to
these areas. Our method also fails to recover ﬁne-scale de-
tails such as facial expression and hands’ motion. This issue
can be addressed using methods that focus on face/hand re-
construction.
Conclusion.
In this paper, we have presented a deep-
learning based framework to reconstruct a 3D human model
from a single image. Based on the three-stage task decom-
position, the dense semantic representation, the proposed
network design and the 3D real-world human dataset, our
method is able to estimate a plausible geometry of the target
in the input image. We believe both our dataset and network
will enable convenient VR/AR content creation and inspire
further research on 3D vision for humans.