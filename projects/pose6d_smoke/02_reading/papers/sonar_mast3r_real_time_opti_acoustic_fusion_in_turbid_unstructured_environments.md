# Sonar-MASt3R: Real-Time Opti-Acoustic Fusion in Turbid, Unstructured Environments

> 2026 · id: arxiv:2603.13585 · arXiv: 2603.13585 · pdf: https://arxiv.org/pdf/2603.13585 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
Underwater perception in turbid conditions remains a chal-
lenge for subsea intervention operations, which are important
for a variety of tasks ranging from scientific exploration [1]
to infrastructure construction [2] and maintenance [3]. To-
day, intervention operations predominantly rely on optical
cameras for real-time feedback [4], [5], [6], which provide
operators with sufficient information to assess obstacles and
manipulation targets in the environment. However, optical
cameras lack robustness to poor lighting or visibility condi-
tions, which can slow or even halt operations.
Perception during intervention operations poses particular
challenges because tasks involving disturbing soft sediments,
scraping biofouling, drilling, or excavation can quickly in-
crease turbidity. To support effective manipulation in these
conditions, perception methods must satisfy two key require-
ments: (i) operate at real-time rates to provide operators
with continuous visual feedback, and (ii) provide sufficiently
accurate geometric and color correspondences to enable
reliable object identification.
This work was supported by the Strategic Environmental Research and
Development Program Grant W912HQ24P0024. Amy Phung would like
to acknowledge financial support from the National Science Foundation
Graduate Research Fellowship (No. 2141064), from the National Aeronau-
tics and Space Administration (NASA) through the FINESST program (No.
80NSSC23K1391)
1Applied Ocean Physics and Engineering, Woods Hole Oceanographic
Institution, Deep Submergence Laboratory, Woods Hole, MA, USA
2Massachusetts
Institute
of
Technology,
Cambridge,
MA,
USA
aphung@mit.edu
Supplemental Video: https://youtu.be/LkW0TpIiwBA
The code, dataset, and visualization tools used to generate the figures in
this paper are available online at https://sonar-mast3r.github.io/
Prior research has proposed various methods for opti-
acoustic fusion, which leverage the complementary strengths
of optical and acoustic sensors: optical cameras provide high-
resolution color, texture, and semantic information, while
acoustic sensors deliver robust spatial geometry and depth
information, and remain functional in degraded visibility
conditions [7], [8], [9]. However, existing methods cannot
achieve dense 3D reconstructions in real-time, and few
studies have reported results from applying these methods
in a turbid environment.
In this work, we present Sonar-MASt3R, an “opti-acoustic
eye-in-hand” perception method designed for robust per-
ception in turbid underwater environments. Sonar-MASt3R
uses MASt3R to extract dense correspondences from optical
camera streams in real-time, and fuses them with geometric
information obtained from an acoustic 3D reconstruction to
ensure the reconstruction maintains absolute length scale.
This method enables us to use the high-resolution visual and
geometric cues of cameras for dense reconstruction when
sufficient visual structure is available, while maintaining
robustness to visibility degradation with the sonar data. In
summary, the key contributions of this work are as follows:
• Dense metric-scale 3D reconstruction in real-time
Sonar-MASt3R fuses dense optical correspondences
from MASt3R by using acoustic data to apply an abso-
lute scale correction to the unscaled MASt3R output.
• Adapting the use of optical and acoustic data based
on visibility conditions The method leverages dense
optical features when visibility allows, while maintain-
ing robustness by reverting to the acoustic reconstruc-
tion in degraded conditions.
• Experimental validation in turbid conditions We
present qualitative and quantitative results from using
the method in turbidity levels ranging from <0.5 to 12
NTU.
• New opti-acoustic dataset We release a new opti-
acoustic dataset recorded in a test tank with calibrated
turbidity levels and a cluttered workspace.

## method
Sonar-MASt3R is a real-time, opti-acoustic 3D recon-
struction method that fuses the dense optical features and
correspondences from MASt3R with the geometry and scale
information inferred from range-based acoustic data. We use
an “opti-acoustic eye-in-hand” configuration similar to OA-
SIS [16], and a keyframe-based method built on MASt3R-
SLAM [18] to process the incoming data in real time.
The wrist-mounted optical camera and imaging sonar lever-
ages the manipulator’s dexterity to record an opti-acoustic
dataset spanning the manipulator’s reachable workspace. The
manipulator’s joint angle sensors and forward kinematics
model provides pose information. The overall workflow is
summarized in Figure 1.
A. Acoustic 3D Reconstruction
The process is initiated using the OASIS method described
in [16] to compute an acoustic-only 3D reconstruction. This
method uses a low-profile “sweep” trajectory that moves the
arm minimally from its initial stowed position, making it
safe to use in an unmapped environment. It is optimized for
recording intersecting regions of the sonar data to quickly
resolve the sonar’s elevation angle ambiguity, and tilts the
manipulator’s wrist to maximize the use of the sonar’s wide
field-of-view (FOV). The output of OASIS is a voxel grid,
which can be written as
V (ix, iy, iz) →
(
1 if occupied
0 otherwise
(1)
where ix, iy, iz are the corresponding indices for the voxel
grid given an x,y,z position in the world frame. For this
implementation, a voxel resolution of 5 cm is used.
Notably, using an acoustic-only method allows the trajec-
tory to be executed at a much higher speed than would be
optimal for an opti-acoustic dataset. We do not use camera
data recorded using this trajectory since the manipulator’s
speed results in a significant amount of motion blur. By
design, this trajectory maintains a large standoff distance
between the sensors and the workspace. While this has a
marginal impact on the sonar’s performance, the optical
image quality rapidly degrades with range, particularly in
turbid conditions.
B. Real-time Opti-Acoustic Processing
Once the acoustic 3D reconstruction has been computed,
the incoming camera data is fused with the voxel grid data
using a keyframe-based approach. For each new camera
frame, the reconstruction process can be described as fol-
lows:
1) Compare current frame with previous keyframe using
MASt3R
2) Use acoustic 3D reconstruction to rescale the MASt3R
pointmap output
3) Match current frame pointmap scale to keyframe
4) Determine whether to add current frame as a new
keyframe

Fig. 1: The optical camera intrinsics and extrinsics are used to render a depth image from the acoustic 3D reconstruction,
which is used to correct the scale of the pointmap computed by MASt3R
5) After adding new keyframes, run global optimization
These steps will be detailed in the following subsections.
C. Pair-wise Frame Processing
The basis of this method relies on pair-wise image pro-
cessing, consisting of two steps: MASt3R prediction, which
generates dense pointmaps, features, and confidences from
a pair of optical images, and a sonar-based rescaling step,
which corrects the scale of the MASt3R outputs.
Before processing images with MASt3R, we rescale our
camera’s native 1600x1200 px resolution to 512x384 px
since the maximum dimension of input images in MASt3R
is 512. Although MASt3R is typically used with a pair of un-
rectified images, we also rectify the fisheye camera’s images
before processing them. Since MASt3R’s training dataset
consists of in-air images from regular pinhole cameras,
which have less radial distortion than fisheye cameras, using
unrectified fisheye images with MASt3R results in distorted
pointmaps. However, fisheye cameras are commonly used in
underwater applications due to their wider field of view –
when a regular pinhole camera is placed inside a flat-port
housing, the refractive index difference between the water
and air inside the housing results in a significant decrease in
the cameras’s FOV.
MASt3R Prediction: MASt3R takes in a pair of images
Ii, Ij ∈RH×W ×3, and outputs pointmaps X ∈RH×W ×3,
pointmap confidences C ∈RH×W ×1, d-dimensional fea-
tures D ∈RH×W ×d, and feature confidences Q ∈RH×W ×1
for both images. The forward pass of MASt3R can be written
as:
FM(Ii, Ij) →Xi
i, Xj
i , Ci, Cj, Di, Dj, Qi, Qj
(2)
Note that the pointmap outputs for image i (Xi
i) and j
(Xj
i ) are both specified with respect to image i, hence the
matching subscript.
Rescale MASt3R Output with Acoustic Reconstruc-
tion: While MASt3R can extract 3D structure information
from image pairs, the scale of the output pointmap is
arbitrary and can vary between different pairs of images. To
achieve metric scale, we use the acoustic 3D reconstruction
to correct the MASt3R pointmap scale.
Since the MASt3R pointmap Xi
i consists of 3D positions
of each pixel relative to the camera center, the optical depth
image DO ∈RH×W ×1 can be computed using the norm of
this pointmap
DO = ∥Xi
i∥
(3)
Using the optical camera’s intrinsics Ki (after downsam-
pling to the maximum MASt3R resolution) and pose TW f
(derived from the manipulator’s joint angle sensors and
forward kinematics model), a corresponding depth image is
rendered from the acoustic reconstruction voxel grid V . This
process, denoted as FD(Ki, TW f, V ) entails sampling along
rays defined by the camera’s projection model at intervals
specified by the grid’s voxel size for each pixel in the image.
The corresponding acoustic depth image DA ∈RH×W ×1
can be denoted as
DA = FD(Ki, TW f, V )
(4)
The computed depth images DO and DA are illustrated in
Figure 1.
The relative scale s between DO and DA is estimated
using RANSAC [20], which is a robust estimation technique
with moderate tolerance to outliers. Prior to estimation, we
construct subsets D′
O ⊂DO and D′
A ⊂DA, which are
defined by
Ω′
D =









(u, v) ∈ΩD

DO(u, v) > 0,
DA(u, v) > 0,
DO(u, v), DA(u, v) ∈R,
Cii(u, v) > mean(Cii)









(5)
This subset contains the set of pixels where the optical
and acoustic depth values are both positive and finite, and
correspond to pixel values where the corresponding MASt3R
pointmap confidence is within the upper half of the confi-
dence distribution. Since the optical depth values are inferred
from the MASt3R pointmap, this filtering ensures the points
used for estimating scale contain a sufficient number of
points for sampling while discarding lower-confidence depth
values.

The computed scaling factor sm is applied to the original
pointmap Xi
i to construct the metric-scale pointmap ¯Xi
i
¯Xi
i = smXi
i
(6)
D. Initialization
During initialization, each incoming frame is used as
both Ii and Ij in the pair-wise frame process described in
Section III-C. Although there is no baseline difference when
using the same image as the pair, MASt3R will still compute
an unscaled pointmap Xi
i and corresponding confidence
values Ci
i. We select the first frame satisfying max(Ci
i) > τi,
where τi is a predefined initialization confidence threshold,
and designate it as the initial keyframe k. The metric-
scale pointmap for the keyframe ¯Xk
k is computed using the
corresponding acoustic depth image.
E. Pointmap-Based Scale Refinement
After initialization, each incoming frame f is processed
with the last keyframe k using the process described in
Section III-C. Although the corresponding acoustic depth
image for frame f can be used to compute its metric-scale
pointmap ¯Xf
f , using this scale estimation alone produces
inconsistent results (i.e., multiple views of an object may not
appear aligned or consistently scaled). Therefore, to further
refine the scale after the acoustic rescaling step, we compute
a least squares solution between a subset containing valid
matches from each of the pointmaps ¯Xf′
f ⊂¯Xf
f and ¯Xk′
k ⊂
¯Xk
k. These subsets contain the overlapping components of
the pointmaps where the corresponding pointmap and feature
confidences are above their respective thresholds τc, τq, and
are defined by
Ω′
M =





*
(uf, vf)
(uk, vk)
+
∈M

Ck(uk, vk) > τc,
Cf(uf, vf) > τc,
Q > τq





(7)
where
Q =
q
Qf(uf, vf) · Qk(uk, vk)
The set of matches M, which contains pixel mappings
between the current frame and the last keyframe (uf, vf) 7→
(uk, vk) are identified using the projective data association
step as described in [18]. This pointmap-based scale refine-
ment sp is computed by solving
min
sp
∥¯Xk′
k −sp · Tkf ¯Xf′
f ∥2
(8)
where Tkf is the matrix that describes the transformation
between frames k and f. This matrix is derived from the
manipulator pose data recorded with each frame.
F. Keyframe Selection and Global Optimization
Following the keyframe selection method in [18], a new
keyframe is introduced when the fraction of valid matches
αmatch or unique keyframe pixels αunique falls below a thresh-
old τk. These fractions are defined as
αmatch = |Ω′
M|
H ∗W
(9)
αunique =|unique(Mk)|
H ∗W
(10)
where H and W are the dimen

## experiments
To evaluate our method, we use the sonar data recorded
using the “sweep” trajectory and the optical data recorded
using the object-centric trajectory using the setup described
in Section IV. Since existing opti-acoustic reconstruction
methods cannot create dense reconstructions of complex
workspaces in real-time, we benchmark our results using
optical-only methods to assess reconstruction quality. Re-
construction results from using Sonar-MASt3R, MASt3R-
SLAM, and Metashape on datasets A, C, E, and F (with
turbidity values ranging from <0.5 - 8 NTU) are presented
in Figure 5.

Fig. 5: Optical 3D reconstruction results from (a) Sonar-MASt3R, (b) Metashape, and (c) MASt3R-SLAM using datasets A
(1), C (2), E (3), and F (4) with turbidity values from <0.5 to 8 NTU. A 1-meter grid is included in Sonar-MASt3R’s results
for scale. No scale reference is provided for Metashape or MASt3R-SLAM since these methods do not produce metric-scale
reconstructions.
TABLE II: Measured object sizes in Sonar-MASt3R results for each dataset, in meters.
Cargo
Net
Push
Core
Brick
Shackle
Milk
Crate
Cinder
Block
Mug
Pipe
Stick
Rock
A
0.204
0.241
0.154
0.090
0.261
0.330
0.097
0.067
0.047
0.380
B
0.180
0.248
0.156
0.092
0.263
0.309
0.083
0.070
0.038
0.336
C
0.201
0.273
0.171
0.106
0.262
0.336
0.093
0.066
0.043
0.328
D
0.201
0.275
0.166
0.111
0.291
0.343
0.098
0.073
0.044
0.382
E
0.251
0.258
0.147
0.107
0.309
0.379
0.109
0.065
0.031
0.376
F
0.291
0.247
0.138
0.083
0.346
0.465
0.132
0.080
0.052
0.422
G
0.257
0.327
0.146
0.102
0.293
0.310
0.089
0.086
ND
NM
H
0.211
ND
0.146
0.088
0.311
ND
0.102
ND
ND
ND
Mean
0.224
0.267
0.153
0.097
0.292
0.353
0.100
0.072
0.042
0.371
STD
0.038
0.030
0.011
0.010
0.030
0.055
0.015
0.008
0.007
0.034
Ground-Truth
0.363
0.325
0.202
0.134
0.328
0.395
0.118
0.097
0.038
0.479
GT STD
0.010
0.002
0.001
0.001
0.001
0.001
0.001
0.004
0.001
0.003
% Error
38
18
24
27
11
11
15
25
11
23
*NM = not measurable, ND = not detected, GT STD = Standard deviation for ground-truth measurements

As illustrated by Figure 5, Sonar-MASt3R’s reconstruction
results remain relatively unchanged at turbidity ranges from
<0.5 - 8 NTU. Since the color of the cinder block roughly
matches the color of the suspended sediment, its reconstruc-
tion is the first to degrade with higher turbidity values. Incor-
porating external pose information and the sonar-based scale
corrections also improves the Sonar-MASt3R’s geometric
consistency. While the geometric consistency of Metashape’s
reconstruction is higher than that of Sonar-MASt3R and
MASt3R-SLAM (e.g., features such as the mug handle only
appear once), this outcome is expected, as Metashape applies
a global optimization over the entire dataset rather than
operating incrementally. However, Metashape’s performance
in turbid conditions was the least consistent, yielding only
partial reconstructions at 2.6 and 8 NTU. As an incremental
optimization method that does not incorporate external pose
estimates, MASt3R-SLAM exhibited the lowest geometric
consistency, with several objects multiple times in the re-
construction (e.g., mug, pipe, milk crate, stick).
The MASt3R-SLAM and Sonar-MASt3R reconstructions
are notably less hazy than the one produced by Metashape.
This is likely due to the fact that these keyframe-based
methods decide which frames to include in the 3D recon-
struction based on the number of detected features, and thus
it preferentially selects frames which were recorded at shorter
stand-off distances. Meanwhile, Metashape blends the colors
based on feature matches across all of the images, regardless
of stand-off distance. The histogram equalization process
used for color correction in Sonar-MASt3R produced slightly
clearer colors than the uncorrected outputs of MASt3R-
SLAM, although this improvement was less pronounced than
the haze observed in Metashape’s reconstructions.
At 11.3 NTU (dataset G), MASt3R-SLAM was unable to
produce a 3D reconstruction since there were insufficient
features for camera pose estimation. Metashape produced
a partial reconstruction of this dataset which was missing
the milk crate, cinder block, rock, and stick. At 12.4 NTU
(dataset H), Metashape and MASt3R-SLAM were each un-
able to produce a reconstruction. In these datasets, Sonar-
MASt3R adapted the use of optical and acoustic data based
on the turbidity level, as illustrated in Figure 6. Sonar-
MASt3R used dense optical features to add high-resolution
features the acoustic reconstruction when visibility allowed,
while maintaining robustness in turbid conditions by relying
on sonar data and manipulator pose to maintain metric scale
and preserve spatial relationships between objects.
To quantitatively evaluate the scale of Sonar-MASt3R’s
reconstruction results, Table II reports object measurements
using the reconstruction generated from each of the recorded
datasets. The results highlight both the dimensional stability
of the reconstructions despite the increasing turbidity and
the ability to capture fine-scale structures (e.g., mug handle,
shackle shape, milk crate pattern) smaller than a single voxel
in the acoustic reconstruction. Although the object dimen-
sions are self-consistent across the datasets, the objects’
absolute scale in the reconstruction are consistently biased
smaller than their actual size.
Fig. 6: Sonar-MASt3R reconstruction results with dataset G
(11.3 NTU) (a) and H (12.4 NTU) (b). The optical recon-
struction is overlaid on the meshed acoustic reconstruction,
which provides spatial context for the detected objects.
We tested our implementation on an NVIDIA GeForce
RTX 3070 GPU with an AMD Ryzen 9 5900HX CPU, where
Sonar-MASt3R processed data at 3 FPS. Although MASt3R-
SLAM reports a runtime of 15 FPS, we observed only 4
FPS when using their implementation on this hardware. This
suggests that Sonar-MASt3R’s processing speeds are likely
comparable to MASt3R-SLAM, and that runtime differences
can be attributed to GPU limitations rather than the methods
themselves. For comparison, Metashape required approxi-
mately 25 minutes to reconstruct a dataset of ∼800 frames
using this hardware.

## related_work
Prior research has developed a variety of opti-acoustic per-
ception methods for different applications. Early approaches
primarily relied on geometric formulations, which leverage
the epipolar geometry of the sensors to compute the 3D
position of matching correspondences [10], [11]. While these
approaches have been widely used for extrinsic calibration
with known targets [10], [12], [13], the need for feature
correspondences makes them difficult to use with unknown
arXiv:2603.13585v1  [cs.RO]  13 Mar 2026

workspaces, and results in a very sparse reconstruction. A
contour-based approach proposed in [14], [15] improves
the robustness and reconstruction density of feature-based
methods by fitting a contour to correspondences, but can
only reconstruct discrete objects rather than the overall scene
geometry and still provides insufficient resolution for object
recognition.
The AoNeuS method focused on accurate high-resolution
3D surfaces from opti-acoustic measurements captured over
heavily-restricted baselines, where acquiring full 360 degree
views of objects may not be possible [8]. Rather than
focusing on correspondences, this method uses the raw
optical and acoustic data to compute the optimal geometry
by using a neural surface reconstruction method. While the
AoNeuS method produces a high-resolution reconstruction,
it requires significant computational resources to process and
can take hours to optimize, with typical rendering times of
∼5 minutes per frame depending on resolution and dataset
size [8].
To provide sufficient resolution for object recognition in
real time, the OASIS method proposed a volumetric approach
for reconstructing the workspace geometry using a wrist-
mounted sonar, then projected data from a wrist-mounted
camera on the reconstruction to aid in object recogni-
tion [16]. Although this method achieved sufficient resolution
for object identification in real time, its sole reliance on
acoustic data for reconstructing workspace geometry limited
its ability to reconstruct small objects.
Other than the contour method presented in [15], it is also
worth noting that the aforementioned works do not evaluate
the performance of these methods in turbid water. Recently,
[17] introduced a real-time opti-acoustic 3D reconstruction
method, which identifies regions of interest in camera images
and uses corresponding sonar measurements to estimate
range. Although the presented seawall and piling reconstruc-
tion results are compelling, the authors acknowledge that
the method’s applicability to more complex scenes remains
limited.
Computing
dense
3D
reconstructions
in
complex
workspaces remains an open challenge for existing opti-
acoustic methods. For optical datasets, MASt3R-SLAM
proposed a keyframe-based approach which achieved dense
3D reconstruction in complex environments [18]. The
method uses the reconstruction model MASt3R [19] to
obtain dense correspondences, which are then used to
estimate the camera pose and align sequential frames.
However, because MASt3R was trained on in-air datasets,
MASt3R-SLAM performs poorly under turbid conditions
when it selects a keyframe with no coherent features.
Additionally, as a monocular system, it lacks an inherent
sense of spatial scale and cannot incorporate external
pose estimates provided in real-time. Consequently, its
reconstructions are sometimes distorted since the data is
aligned on a frame-by-frame basis.

## conclusion
In an operational context, Sonar-MASt3R’s automated
ability to adapt its reconstruction strategy based on visibility
conditions makes it a promising approach for use in the
field, particularly during intervention tasks that may generate
transient turbidity plumes. As shown in Figure 6, although
some objects were missing from the reconstruction in very
turbid conditions, the acoustic reconstruction provided spatial
context for coarse geometry between the detected objects.
In the real-time visualization, we found that the projected
data could be used to locate and identify objects even if
they were not fully reconstructed or added to the keyframe
map, which is similar to the approach used in [16]. Future
work can add this image projection method as an automatic
contingency step to enable object identification in conditions
that are inadequate for dense optical 3D reconstruction.
By integrating external pose information and acoustic data,
Sonar-MASt3R is capable of producing 3D reconstruction
results with absolute spatial scaling in real time. Unlike
previous opti-acoustic methods, it can reconstruct centimeter-
scale objects with complex features in addition to the en-
vironment. The geometry of object features such as the
brick holes, shackle loop, and mug handle are reconstructed,
despite the coarse 5 cm acoustic voxel grid resolution.
For the acoustic 3D reconstruction, we re-implemented
the OASIS method [16] to run on the GPU instead of the
CPU, achieving processing rates above 100 FPS even with
a 1 cm voxel grid resolution. However, we still used the
lower voxel grid resolution of 5 cm in our experiments since
the coarser resolution allowed each voxel to be sampled
more frequently, which reduced false negatives. This also
reduced GPU memory requirements, particularly during the

acoustic reconstruction depth rendering process. However,
given that the scale of the objects were consistently smaller
in the reconstruction, we hypothesize that the coarse grid
resolution may have caused the acoustic depth render to bias
the depth estimates closer, which would cause the objects to
appear smaller. Future work could examine the trade-offs of
voxel grid resolution, and assess whether a higher-resolution
depth rendering would improve dimensional accuracy of the
reconstruction.
Sonar-MASt3R currently requires an external source of
pose information, which works well for a fixed-base ma-
nipulator with joint angle sensors. However, future work
on adapting this method to use a SLAM-based approach
(e.g., [7]) could extend its applicability to free-floating plat-
forms with greater pose uncertainty. Future work could also
use a SLAM-based or another uncertainty-based approach to
map dynamic environments, and track moving objects in the
environment or changes made by the manipulator.
Compared to MASt3R-SLAM, we found that Sonar-
MASt3R required using stricter pointmap confidence thresh-
olds during keyframe selection and when adding edges to the
factor graph. In MASt3R-SLAM, low threshold values were
required to maintain tracking without interruptions, since a
few untracked frames could cause the method to lose tracking
and stall. In contrast, Sonar-MASt3R incorporates external
pose and scale estimates from the manipulator and the sonar
data, and thus mapping results are improved by discarding
low-confidence frames with few features.