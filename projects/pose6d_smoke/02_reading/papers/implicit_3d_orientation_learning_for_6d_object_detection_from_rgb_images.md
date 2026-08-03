# Implicit 3D Orientation Learning for 6D Object Detection from RGB Images

> 2018 · id: W2895439318 · arXiv: 1902.01275 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
One of the most important components of modern com-
puter vision systems for applications such as mobile
1German Aerospace Center (DLR), 82234 Wessling, Germany
2Technical University of Munich, 80333 Munich, Germany
E-mail: {martin.sundermeyer, zoltan.marton,
maximilian.durner, rudolph.triebel}@dlr.de
1 https://github.com/DLR-RM/AugmentedAutoencoder
robotic manipulation and augmented reality is a reli-
able and fast 6D object detection module. Although,
there are very encouraging recent results from Xiang
et al. (2017); Kehl et al. (2017); Hodaˇn et al. (2017);
Wohlhart and Lepetit (2015); Vidal et al. (2018); Hin-
terstoisser et al. (2016); Tremblay et al. (2018), a gen-
eral, easily applicable, robust and fast solution is not
available, yet. The reasons for this are manifold. First
and foremost, current solutions are often not robust
enough against typical challenges such as object oc-
clusions, diﬀerent kinds of background clutter, and dy-
namic changes of the environment. Second, existing meth-
ods often require certain object properties such as enough
textural surface structure or an asymmetric shape to
avoid confusions. And ﬁnally, current systems are not
eﬃcient in terms of run-time and in the amount and
type of annotated training data they require.
Therefore, we propose a novel approach that di-
rectly addresses these issues. Concretely, our method
operates on single RGB images, which signiﬁcantly in-
creases the usability as no depth information is required.
We note though that depth maps may be incorporated
optionally to reﬁne the estimation. As a ﬁrst step, we
build upon state-of-the-art 2D Object Detectors of (Liu
et al. (2016); Lin et al. (2018)) which provide object
bounding boxes and identiﬁers. On the resulting scene
crops, we employ our novel 3D orientation estimation
algorithm, which is based on a previously trained deep
network architecture. While deep networks are also used
in existing approaches, our approach diﬀers in that we
do not explicitly learn from 3D pose annotations dur-
ing training. Instead, we implicitly learn representa-
tions from rendered 3D model views. This is accom-
plished by training a generalized version of the Denois-
ing Autoencoder from Vincent et al. (2010), that we
call ’Augmented Autoencoder (AAE)’, using a novel Do-
arXiv:1902.01275v2  [cs.CV]  17 Jul 2019

2
Martin Sundermeyer1 et al.
Hcam2obj
^
2D Object Detector
Distance Estimator
Perspective
Correction
R'cam2obj
^
tcam2obj
^
objid
Augmented
Autoencoder
objbox
Hcam2obj
^
(refined)
Fig. 1: Our full 6D Object Detection pipeline: after detecting an object (2D Object Detector), the object is
quadratically cropped and forwarded into the proposed Augmented Autoencoder. In the next step, the bounding
box scale ratio at the estimated 3D orientation ˆR′obj2cam is used to compute the 3D translation ˆtobj2cam. The
resulting euclidean transformation ˆ
H′obj2cam ∈R4x4 already shows promising results as presented in Sundermeyer
et al. (2018), however it still lacks of accuracy given a translation in the image plane towards the borders. Therefore,
the pipeline is extended by the Perspective Correction block which addresses this problem and results in more
accurate 6D pose estimates ˆHobj2cam for objects which are not located in the image center. Additionally, given
depth data, the result can be further reﬁned ( ˆH(refined)
obj2cam ) by applying an Iterative Closest Point post-processing
(bottom).
main Randomization strategy. Our approach has sev-
eral advantages: First, since the training is indepen-
dent from concrete representations of object orienta-
tions within SO(3) (e.g. quaternions), we can handle
ambiguous poses caused by symmetric views because
we avoid one-to-many mappings from images to orien-
tations. Second, we learn representations that specif-
ically encode 3D orientations while achieving robust-
ness against occlusion, cluttered backgrounds and gen-
eralizing to diﬀerent environments and test sensors. Fi-
nally, the AAE does not require any real pose-annotated
training data. Instead, it is trained to encode 3D model
views in a self-supervised way, overcoming the need of
a large pose-annotated dataset. A schematic overview
of the approach based on Sundermeyer et al. (2018) is
shown in Fig 1.

## method
In the following, we mainly focus on the novel 3D ori-
entation estimation technique based on the AAE.
3.1 Autoencoders
The original AE, introduced by Rumelhart et al. (1985),
is a dimensionality reduction technique for high dimen-
sional data such as images, audio or depth. It con-
sists of an Encoder Φ and a Decoder Ψ, both arbi-
trary learnable function approximators which are usu-
ally neural networks. The training objective is to re-
construct the input x ∈RD after passing through a
low-dimensional bottleneck, referred to as the latent
representation z ∈Rn with n << D :
ˆx = (Ψ ◦Φ)(x) = Ψ(z)
(1)
The per-sample loss is simply a sum over the pixel-wise
L2 distance
ℓ2 =
X
i∈D
∥xi −ˆxi ∥2
(2)
The resulting latent space can, for example, be used for
unsupervised clustering.
Denoising Autoencoders introduced by Vincent
et al. (2010) have a modiﬁed training procedure. Here,
artiﬁcial random noise is applied to the input images
x ∈RD while the reconstruction target stays clean. The
trained model can be used to reconstruct denoised test
images. But how is the latent representation aﬀected?
Hypothesis 1: The Denoising AE produces latent
representations which are invariant to noise because it
facilitates the reconstruction of de-noised images.
We will demonstrate that this training strategy ac-
tually enforces invariance not only against noise but

6
Martin Sundermeyer1 et al.
a)
b)
c)
Fig. 4: Training process for the AAE; a) reconstruction target batch x of uniformly sampled SO(3)
object views; b) geometric and color augmented input; c) reconstruction ˆxˆxˆx after 40000 iterations
against a variety of diﬀerent input augmentations. Fi-
nally, it allows us to bridge the domain gap between
simulated and real data.
3.2 Augmented Autoencoder
The motivation behind the AAE is to control what the
latent representation encodes and which properties are
ignored. We apply random augmentations faugm(.) to
the input images x ∈RD against which the encoding
should become invariant. The reconstruction target re-
mains eq. (2) but eq. (1) becomes
ˆx = (Ψ ◦Φ ◦faugm)(x) = (Ψ ◦Φ)(x′) = Ψ(z′)
(3)
To make evident that Hypothesis 1 holds for geomet-
ric transformations, we learn latent representations of
binary images depicting a 2D square at diﬀerent scales,
in-plane translations and rotations. Our goal is to en-
code only the in-plane rotations r ∈[0, 2π] in a two
dimensional latent space z ∈R2 independent of scale
or translation. Fig. 3 depicts the results after training
a CNN-based AE architecture similar to the model in
Fig. 5. It can be observed that the AEs trained on re-
constructing squares at ﬁxed scale and translation (1)
or random scale and translation (2) do not clearly en-
code rotation alone, but are also sensitive to other la-
tent factors. Instead, the encoding of the AAE (3) be-
comes invariant to translation and scale such that all
squares with coinciding orientation are mapped to the
same code. Furthermore, the latent representation is
much smoother and the latent dimensions imitate a
shifted sine and cosine function with frequency f =
4
2π
respectively. The reason is that the square has two per-
pendicular axes of symmetry, i.e. after rotating π
2 the
square appears the same. This property of representing
the orientation based on the appearance of an object
rather than on a ﬁxed parametrization is valuable to
avoid ambiguities due to symmetries when teaching 3D
object orientations.
3.3 Learning 3D Orientation from Synthetic Object
Views
Our toy problem showed that we can explicitly learn
representations of object in-plane rotations using a ge-
ometric augmentation technique. Applying the same ge-
ometric input augmentations we can encode the whole
SO(3) space of views from a 3D object model (CAD
or 3D reconstruction) while being robust against inac-
curate object detections. However, the encoder would
still be unable to relate image crops from real RGB
sensors because (1) the 3D model and the real object
diﬀer, (2) simulated and real lighting conditions dif-
fer, (3) the network can’t distinguish the object from
background clutter and foreground occlusions. Instead
of trying to imitate every detail of speciﬁc real sensor
recordings in simulation we propose a Domain Random-
ization (DR) technique within the AAE framework to
make the encodings invariant to insigniﬁcant environ-
ment and sensor variations. The goal is that the trained
encoder treats the diﬀerences to real camera images as
just another irrelevant variation. Therefore, while keep-
ing reconstruction targets clean, we randomly apply ad-
ditional augmentations to the input training views: (1)

Augmented Autoencoders: Implicit 3D Orientation Learning for 6D Object Detection
7
Fig. 5: Autoencoder CNN architecture with occluded test input, ”resize2x” depicts nearest-neighbor upsampling
Table 1: Augmentation Parameters of AAE; Scale
and translation is in relation to image shape and
occlusion is in proportion of the object mask
50% chance
light (random position)
(30% per channel)
& geometric
add
U(−0.1, 0.1)
ambient
0.4
contrast
U(0.4, 2.3)
diﬀuse
U(0.7, 0.9)
multiply
U(0.6, 1.4)
specular
U(0.2, 0.4)
invert
scale
U(0.8, 1.2)
gaussian blur
σ ∼U(0.0, 1.2)
translation
U(−0.15, 0.15)
occlusion
∈[0, 0.25]
Fig. 6: AAE decoder reconstruction of LineMOD
(left) and T-LESS (right) scene crops
rendering with random light positions and randomized
diﬀuse and specular reﬂection (simple Phong model (Phong,
1975) in OpenGL), (2) inserting random background
images from the Pascal VOC dataset (Everingham et al.,
2012), (3) varying image contrast, brightness, Gaussian
blur and color distortions, (4) applying occlusions using
random object masks or black squares. Fig. 4 depicts an
exemplary training process for synthetic views of object
5 from T-LESS (Hodaˇn et al., 2017).
3.4 Network Architecture and Training Details
The convolutional Autoencoder architecture that is used
in our experiments is depicted in Fig. 5. We use a
bootstrapped pixel-wise L2 loss, ﬁrst introduced by Wu
et al. (2016). Only the pixels with the largest recon-
struction errors contribute to the loss. Thereby, ﬁner
details are reconstructed and the training does not con-
verge to local minima like reconstructing black images
for all views. In our experiments, we choose a boot-
strap factor of k = 4 per image, meaning that 1
4 of all
pixels contribute to the loss. Using OpenGL, we ren-
der 20000 views of each object uniformly at random
3D orientations and constant distance along the cam-
era axis (700mm). The resulting images are quadrat-
ically cropped using the longer side of the bounding
box and resized (nearest neighbor) to 128 × 128 × 3
as shown in Fig. 4. All geometric and color input aug-
mentations besides the rendering with random lighting
are applied online during training at uniform random
strength, parameters are found in Tab. 1. We use the
Adam (Kingma and Ba, 2014) optimizer with a learn-
ing rate of 2 × 10−4, Xavier initialization (Glorot and
Bengio, 2010), a batch size = 64 and 40000 iterations
which takes ∼4 hours on a single Nvidia Geforce GTX
1080.
3.5 Codebook Creation and Test Procedure
After training, the AAE is able to extract a 3D object
from real scene crops of many diﬀerent camera sensors
(Fig. 6). The clarity and orientation of the decoder re-
construction is an indicator of the encoding quality. To
determine 3D object orientations from test scene crops
we create a codebook (Fig. 7 (top)):

8
Martin Sundermeyer1 et al.
5
Fig. 7: Top: creating a codebook from the encodings of discrete synthetic object views; bottom: object
detection and 3D orientation estimation using the nearest neighbor(s) with highest cosine similarity from
the codebook
1) Render clean, synthetic object views at nearly equidis-
tant viewpoints from a full view-sphere (based on a
reﬁned icosahedron (Hinterstoisser et al., 2008))
2) Rotate each view in-plane at ﬁxed intervals to cover
the whole SO(3)
3) Create a codebook by generating latent codes z ∈
R128 for all resulting images and assigning their cor-
responding rotation Rcam2obj ∈R3x3
At test time, the considered object(s) are ﬁrst de-
tected in an RGB scene. The image is quadratically
cropped using the longer side of the bounding box mul-
tiplied with a padding factor of 1.2 and resized to match
the encoder input size. The padding accounts for impre-
cise bounding boxes. After encoding we compute the co-
sine similarity between the test code ztest ∈R128 and
all codes zi ∈R128 from the codebook:
cosi =
zi ztest
∥zi∥∥ztest∥
(4)
The highest similarities are determined in a k-Nearest-
Neighbor (kNN) search and the corresponding rotation
matrices {RkNN} from the codebook are returned as
estimates of the 3D object orientation. For the quanti-
tative evaluation we use k = 1, however the next neigh-
bors can yield valuable information on ambiguous views
and could for example be used in particle ﬁlter based
tracking. We use cosine similarity because (1) it can be
very eﬃciently computed on a 

## experiments
We evaluate the AAE and the whole 6D detection pipeline
on the T-LESS (Hodaˇn et al., 2017) and LineMOD
(Hinterstoisser et al., 2011) datasets.
4.1 Test Conditions
Few RGB-based pose estimation approaches (e.g. Kehl
et al. (2017); Ulrich et al. (2009)) only rely on 3D model
information. Most methods like Wohlhart and Lepetit
(2015); Balntas et al. (2017); Brachmann et al. (2016)
make use of real pose annotated data and often even
train and test on the same scenes (e.g. at slightly diﬀer-
ent viewpoints, as in the oﬃcial LineMOD benchmark).
It is common practice to ignore in-plane rotations or to
only consider object poses that appear in the dataset
(Rad and Lepetit, 2017; Wohlhart and Lepetit, 2015)
which also limits applicability. Symmetric object views
are often individually treated (Rad and Lepetit, 2017;
Balntas et al., 2017) or ignored (Wohlhart and Lepetit,
2015). The SIXD challenge (Hodan, 2017) is an attempt
to make fair comparisons between 6D localization algo-
rithms by prohibiting the use of test scene pixels. We
follow these strict evaluation guidelines, but treat the
harder problem of 6D detection where it is unknown
which of the considered objects are present in the scene.
This is especially diﬃcult in the T-LESS dataset since
objects are very similar. We train the AAEs on the re-
constructed 3D models, except for object 19-23 where
we train on the CAD models because the pins are miss-
ing in the reconstructed plugs.
We noticed, that the geometry of some 3D recon-
struction in T-LESS is slightly inaccurate which badly
inﬂuences the RGB-based distance estimation (Sec. 3.6.2)
since the synthetic bounding box diagonals are wrong.
Therefore, in a second training run we only train on the
30 CAD models.
4.2 Metrics
Hodaˇn et al. (2016) introduced the Visible Surface Dis-
crepancy (errvsd), an ambiguity-invariant pose error
function that is determined by the distance between
the estimated and ground truth visible object depth
surfaces. As in the SIXD challenge, we report the recall
of correct 6D object poses at errvsd < 0.3 with toler-
ance τ = 20mm and > 10% object visibility. Although
the Average Distance of Model Points (ADD) metric in-
troduced by Hinterstoisser et al. (2012b) cannot handle
pose ambiguities, we also present it for the LineMOD
dataset following the oﬃcial protocol in Hinterstoisser
et al. (2012b). For objects with symmetric views (egg-
box, glue), Hinterstoisser et al. (2012b) adapts the met-
ric by calculating the average distance to the closest
model point. Manhardt et al. (2018) has noticed inac-
curate intrinsics and sensor registration errors between
RGB and D in the LineMOD dataset. Thus, purely syn-
thetic RGB-based approaches, although visually cor-
rect, suﬀer from false pose rejections. The focus of our
experiments lies on the T-LESS dataset.
In our ablation studies we also report the AUCvsd,
which represents the area under the ’errvsd vs. recall’
curve:
AUCvsd =
Z 1
0
recall(errvsd) derrvsd
(10)

Augmented Autoencoders: Implicit 3D Orientation Learning for 6D Object Detection
11
Table 5: Ablation study on color augmentations for diﬀerent test sensors. Object 5 tested
on all scenes, T-LESS Hodaˇn et al. (2017). Standard deviation of three runs in brackets.
Train RGB
Test RGB
dyn. light
add
contrast
multiply
invert
AUCvsd
3D Reconstruction
Primesense

0.472 (± 0.013)
(synthetic)
(real)


0.611 (± 0.030)



0.825 (± 0.015)




0.876 (± 0.019)





0.877 (± 0.005)



0.861 (± 0.014)
Primesense (real)
Primesense (real)



0.890 (± 0.003)
3D Reconstruction
Kinect

0.461 (± 0.022)
(synthetic)
(real)


0.580 (± 0.014)



0.701 (± 0.046)




0.855 (± 0.016)





0.897 (± 0.008)



0.903 (± 0.016)
Kinect (real)
Kinect (real)



0.917 (± 0.007)
(a) Eﬀect of latent space size, stan-
dard deviation in red
(b) Training on CAD model (bottom) vs. textured 3D
reconstruction (top)
Fig. 9: Testing object 5 on all 504 Kinect RGB views of scene 2 in T-LESS
4.3 Ablation Studies
To assess the AAE alone, in this subsection we only
predict the 3D orientation of Object 5 from the T-
LESS dataset on Primesense and Kinect RGB scene
crops. Table 5 shows the inﬂuence of diﬀerent input
augmentations. It can be seen that the eﬀect of diﬀer-
ent color augmentations is cumulative. For textureless
objects, even the inversion of color channels seems to
be beneﬁcial since it prevents overﬁtting to synthetic
color information. Furthermore, training with real ob-
ject recordings provided in T-LESS with random Pas-
cal VOC background and augmentations yields only
slightly better performance than the training with syn-
thetic data. Fig. 9a depicts the eﬀect of diﬀerent latent
space sizes on the 3D pose estimation accuracy. Perfor-
mance starts to saturate at dim = 64.
4.4 Discussion of 6D Object Detection results
Our RGB-only 6D Object Detection pipeline consists
of 2D detection, 3D orientation estimation, projective
distance estimation and perspective error correction.
Although the results are visually appealing, to reach
the performance of state-of-the-art depth-based meth-
ods we also need to reﬁne our estimates using a depth-
based ICP. Table 6 presents our 6D detection evalua-
tion on all scenes of the T-LESS dataset, which contains
a high amount of pose ambiguities. Our pipeline out-
performs all 15 reported T-LESS results on the 2018
BOP benchmark from Hodan et al. (2018) in a fraction
of the runtime. Table 6 shows an extract of competing
methods. Our RGB-only results can compete with the
RGB-D learning-based approaches of Brachmann et al.
(2016) and Kehl et al. (2016). Previous state-of-the-art
approaches from Vidal et al. (2018); Drost et al. (2010)
perform a time consuming reﬁnement search through
multiple pose hypotheses while we only perform the
ICP on a single pose hypothesis. That being said, the
codebook is well suited to generate multiple hypotheses
using k > 1 nearest neighbors. The right part of Ta-
ble 6 shows results with ground truth bounding boxes
yielding an upper bound on the pose estimation perfor-
mance.

12
Martin Sundermeyer1 et al.
Table 6: T-LESS: Object recall for errvsd < 0.3 on all Primesense test scenes (SIXD/BOP benchmark from
Hodan et al. (2018)). RGB† depicts training with 3D reconstructions, except objects 19-23 −→CAD models;
RGB‡ depicts training on untextured CAD models only
AAE
AAE
AAE
SSD
RetinaNet
RetinaNet
Brachmann et al.
Kehl et al.
Vidal et al.
Drost et al.
w/ GT 2D BBs
Data RGB† RGB† RGB‡ RGB†+Depth(ICP)
RGB-D
RGB-D +ICP Depth +ICP Depth +edge RGB† +Depth(ICP)
1
5.65
9.48
12.67
67.95
8
7
43
53
12.67
85.98
2
5.46
13.24
16.01
70.62
10
10
47
44
11.47
86.27
3
7.05
12.78
22.84
78.39
21
18
69
61
13.32
90.80
4
4.61
6.66
6.70
57.00
4
24
63
67
12.88
84.20
5
36.45
36.19
38.93
77.18
46
23
69
71
67.37
90.14
6
23.15
20.64
28.26
72.75
19
10
67
73
54.21
90.58
7
15.97
17.41
26.56
83.39
52
0
77
75
38.10
86.94
8
10.86
21.72
18.01
78.08
22
2
79
89
24.83
91.79
9
19.59
39.98
33.36
88.64
12
11
90
92
49.06
91.09
10
10.47
13.37
33.15
84.47
7
17
68
72
15.67
84.67
11
4.35
7.78
17.94
56.01
3
5
69
64
16.64
77.01
12
7.80
9.54
18.38
63.23
3
1
82
81
33.57
79.32
13
3.30
4.56
16.20
43.55
0
0
56
53
15.29
64.38
14
2.85
5.36
10.58
25.58
0
9
47
46
50.14
71.37
15
7.90
27.11
40.50
69.81
0
12
52
55
52.01
73.90
16
13.06
22.04
35.67
84.55
5
56
81
85
36.71
87.58
17
41.70
66.33
50.47
74.29
3
52
83
88
81.44
78.88
18
47.17
14.91
33.63
83.12
54
22
80
78
55.48
85.64
19
15.95
23.03
23.03
58.13
38
35
55
55
53.07
82.71
20
2.17
5.35
5.35
26.73
1
5
47
47
38.97
70.87
21
19.77
19.82
19.82
53.48
39
26
63
55
53.45
86.83
22
11.01
20.25
20.25
60.49
19
27
70
56
49.95
84.20
23
7.98
19.15
19.15
62.69
61
71
85
84
36.74
76.40
24
4.74
4.54
27.94
62.99
1
36
70
59
11.75
84.38
25
21.91
19.07
51.01
73.33
16
28
48
47
37.73
87.53
26
10.04
12.92
33.00
67.00
27
51
55
69
29.82
90.26
27
7.42
22.37
33.61
82.16
17
34
60
61
23.30
84.43
28
21.78
24.00
30.88
83.51
13
54
69
80
43.97
89.84
29
15.33
27.66
35.57
74.45
6
86
65
84
57.82
88.58
30
34.63
30.53
44.33
93.65
5
69
84
89
72.81
95.01
Mean
14.67
19.26
26.79
68.57
17.84
24.60
66.51
67.50
38.34
84.05
Time(s) 0.024
0.077
0.077
0.4
13.5
1.8
4.7
21.5
0.006
0.33
Table 7: Eﬀect of Perspective Corrections on T-LESS

## related_work
Depth-based methods (e.g. using Point Pair Features
(PPF) from Vidal et al. (2018); Hinterstoisser et al.
(2016)) have shown robust pose estimation performance
on multiple datasets, winning the SIXD challenge (Ho-
dan, 2017; Hodan et al., 2018). However, they usu-
ally rely on the computationally expensive evaluation of
many pose hypotheses and do not take into account any
high level features. Furthermore, existing depth sensors
are often more sensitive to sunlight or specular object
surfaces than RGB cameras.
Convolutional Neural Networks (CNNs) have revo-
lutionized 2D object detection from RGB images (Ren
et al., 2015; Liu et al., 2016; Lin et al., 2018). But,
in comparison to 2D bounding box annotation, the ef-
fort of labeling real images with full 6D object poses

Augmented Autoencoders: Implicit 3D Orientation Learning for 6D Object Detection
3
is magnitudes higher, requires expert knowledge and a
complex setup (Hodaˇn et al., 2017).
Nevertheless, the majority of learning-based pose es-
timation methods, namely Tekin et al. (2017); Wohlhart
and Lepetit (2015); Brachmann et al. (2016); Rad and
Lepetit (2017); Xiang et al. (2017), use real labeled im-
ages that you only obtain within pose-annotated datasets.
In consequence, Kehl et al. (2017); Wohlhart and
Lepetit (2015); Tremblay et al. (2018); Zakharov et al.
(2019) have proposed to train on synthetic images ren-
dered from a 3D model, yielding a great data source
with pose labels free of charge. However, naive training
on synthetic data does not typically generalize to real
test images. Therefore, a main challenge is to bridge
the domain gap that separates simulated views from
real camera recordings.
2.1 Simulation to Reality Transfer
There exist three major strategies to generalize from
synthetic to real data:
2.1.1 Photo-Realistic Rendering
The works of Movshovitz-Attias et al. (2016); Su et al.
(2015); Mitash et al. (2017); Richter et al. (2016) have
shown that photo-realistic renderings of object views
and backgrounds can in some cases beneﬁt the general-
ization performance for tasks like object detection and
viewpoint estimation. It is especially suitable in sim-
ple environments and performs well if jointly trained
with a relatively small amount of real annotated im-
ages. However, photo-realistic modeling is often imper-
fect and requires much eﬀort. Recently, Hodan et al.
(2019) have shown promising results for 2D Object De-
tection trained on physically-based renderings.
2.1.2 Domain Adaptation
Domain Adaptation (DA) (Csurka, 2017) refers to lever-
aging training data from a source domain to a tar-
get domain of which a small portion of labeled data
(supervised DA) or unlabeled data (unsupervised DA)
is available. Generative Adversarial Networks (GANs)
have been deployed for unsupervised DA by generat-
ing realistic from synthetic images to train classiﬁers
(Shrivastava et al., 2017), 3D pose estimators (Bous-
malis et al., 2017b) and grasping algorithms (Bousmalis
et al., 2017a). While constituting a promising approach,
GANs often yield fragile training results. Supervised
DA can lower the need for real annotated data, but
does not abstain from it.
2.1.3 Domain Randomization
Domain Randomization (DR) builds upon the hypoth-
esis that by training a model on rendered views in a
variety of semi-realistic settings (augmented with ran-
dom lighting conditions, backgrounds, saturation, etc.),
it will also generalize to real images. Tobin et al. (2017)
demonstrated the potential of the DR paradigm for
3D shape detection using CNNs. Hinterstoisser et al.
(2017) showed that by training only the head network
of FasterRCNN of Ren et al. (2015) with randomized
synthetic views of a textured 3D model, it also gener-
alizes well to real images. It must be noted, that their
rendering is almost photo-realistic as the textured 3D
models have very high quality. Kehl et al. (2017) pio-
neered an end-to-end CNN, called ’SSD6D’, for 6D ob-
ject detection that uses a moderate DR strategy to uti-
lize synthetic training data. The authors render views of
textured 3D object reconstructions at random poses on
top of MS COCO background images (Lin et al., 2014)
while varying brightness and contrast. This lets the net-
work generalize to real images and enables 6D detection
at 10Hz. Like us, for accurate distance estimation they
rely on Iterative Closest Point (ICP) post-processing
using depth data. In contrast, we do not treat 3D ori-
entation estimation as a classiﬁcation task.
2.2 Training Pose Estimation with SO(3) targets
We describe the diﬃculties of training with ﬁxed SO(3)
parameterizations which will motivate the learning of
view-based representations.
2.2.1 Regression
Since rotations live in a continuous space, it seems nat-
ural to directly regress a ﬁxed SO(3) parameterizations
like quaternions. However, representational constraints
and pose ambiguities can introduce convergence issues
as investigated by Saxena et al. (2009). In practice, di-
rect regression approaches for full 3D object orienta-
tion estimation have not been very successful (Mahen-
dran et al., 2017). Instead Tremblay et al. (2018); Tekin
et al. (2017); Rad and Lepetit (2017) regress local 2D-
3D correspondences and then apply a Perspective-n-
Point (PnP) algorithm to obtain the 6D pose. However,
these approaches can also not deal with pose ambigui-
ties without additional measures (see Sec. 2.2.3).
2.2.2 Classiﬁcation
Classiﬁcation of 3D object orientations requires a dis-
cretization of SO(3). Even rather coarse intervals of

4
Martin Sundermeyer1 et al.
∼5o lead to over 50.000 possible classes. Since each
class appears only sparsely in the training data, this
hinders convergence. In SSD6D (Kehl et al., 2017) the
3D orientation is learned by separately classifying a dis-
cretized viewpoint and in-plane rotation, thus reducing
the complexity to O(n2). However, for non-canonical
views, e.g. if an object is seen from above, a change of
viewpoint can be nearly equivalent to a change of in-
plane rotation which yields ambiguous class combina-
tions. In general, the relation between diﬀerent orienta-
tions is ignored when performing one-hot classiﬁcation.
2.2.3 Symmetries
Symmetries are a severe issue when relying on ﬁxed rep-
resentations of 3D orientations since they cause pose
ambiguities (Fig. 2). If not manually addressed, iden-
tical training images can have diﬀerent orientation la-
bels assigned which can signiﬁcantly disturb the learn-
ing process. In order to cope with ambiguous objects,
most approaches in literature are manually adapted
(Wohlhart and Lepetit, 2015; Hinterstoisser et al., 2012a;
Kehl et al., 2017; Rad and Lepetit, 2017). The strategies
reach from ignoring one axis of rotation (Wohlhart and
Lepetit, 2015; Hinterstoisser et al., 2012a) over adapt-
ing the discretization according to the object (Kehl
et al., 2017) to the training of an extra CNN to pre-
dict symmetries (Rad and Lepetit, 2017). These depict
tedious, manual ways to ﬁlter out object symmetries
(Fig. 2a) in advance, but treating ambiguities due to
self-occlusions (Fig. 2b) and occlusions (Fig. 2c) are
harder to address.
Symmetries do not only aﬀect regression and clas-
siﬁcation methods, but any learning-based algorithm
that discriminates object views solely by ﬁxed SO(3)
representations.
2.3 Learning Representations of 3D orientations
We can also learn indirect pose representations that
relate object views in a low-dimensional space. The de-
scriptor learning can either be self-supervised by the
object views themselves or still rely on ﬁxed SO(3) rep-
resentations.
2.3.1 Descriptor Learning
Wohlhart and Lepetit (2015) introduced a CNN-based
descriptor learning approach using a triplet loss that
minimizes/maximizes the Euclidean distance between
similar/dissimilar object orientations. In addition, the
distance between diﬀerent objects is maximized. Al-
though mixing in synthetic data, the training also relies
(a) Object symmetries
(b) Self-occlusion
induced symmetries
(c) Occlusion
induced symmetries
Fig. 2: Causes of pose ambiguities
on pose-annotated sensor data. The approach is not im-
mune against symmetries since the descriptor is built
using explicit 3D orientations. Thus, the loss can be
dominated by symmetric object views that appear the
same but have opposite orientations which can produce
incorrect average pose predictions.
Balntas et al. (2017) extended this work by enforc-
ing proportionality between descriptor and pose dis-
tances. They acknowledge the problem of object sym-
metries by weighting the pose distance loss with the
depth diﬀerence of the object at the considered poses.
This heuristic increases the accuracy on symmetric ob-
jects with respect to Wohlhart and Lepetit (2015).
Our work is also based on learning descriptors, but
in contrast we train our Augmented Autoencoders (AAEs)
such that the learning process itself is independent of
any ﬁxed SO(3) representation. The loss is solely based
on the appearan

## conclusion
We have proposed a new self-supervised training strat-
egy for Autoencoder architectures that enables robust
3D object orientation estimation on various RGB sen-
sors while training only on synthetic views of a 3D
model. By demanding the Autoencoder to revert ge-
ometric and color input augmentations, we learn rep-
resentations that (1) speciﬁcally encode 3D object ori-
entations, (2) are invariant to a signiﬁcant domain gap
between synthetic and real RGB images, (3) inherently
regard pose ambiguities from symmetric object views.
Around this approach, we created a real-time (42 fps),
RGB-based pipeline for 6D object detection which is es-
pecially suitable when pose-annotated RGB sensor data
is not available.
Acknowledgement
We would like to thank Dr. Ingo Kossyk, Dimitri Henkel
and Max Denninger for helpful discussions. We also
thank the reviewers for their useful comments. This
work has been partly funded by Robert Bosch GmbH,
Corporate Research.

Augmented Autoencoders: Implicit 3D Orientation Learning for 6D Object Detection
15