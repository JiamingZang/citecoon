# 3D Human Pose Perception from Egocentric Stereo Videos

> 2024 · id: W4402726927 · arXiv: 2401.00889 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
While head-mounted devices are becoming more com-
pact, they provide egocentric views with significant self-
occlusions of the device user. Hence, existing methods often
fail to accurately estimate complex 3D poses from egocen-
tric views. In this work, we propose a new transformer-
based framework to improve egocentric stereo 3D human
pose estimation, which leverages the scene information and
temporal context of egocentric stereo videos. Specifically,
we utilize 1) depth features from our 3D scene reconstruc-
tion module with uniformly sampled windows of egocentric
stereo frames, and 2) human joint queries enhanced by tem-
poral features of the video inputs. Our method is able to
accurately estimate human poses even in challenging sce-
narios, such as crouching and sitting. Furthermore, we in-
troduce two new benchmark datasets, i.e., UnrealEgo2 and
UnrealEgo-RW (RealWorld). The proposed datasets offer a
much larger number of egocentric stereo views with a wider
variety of human motions than the existing datasets, al-
lowing comprehensive evaluation of existing and upcoming
methods. Our extensive experiments show that the proposed
approach significantly outperforms previous methods. Un-
realEgo2, UnrealEgo-RW, and trained models are available
on our project page1 and Benchmark Challenge2.
1https://4dqv.mpi-inf.mpg.de/UnrealEgo2/
2https://unrealego.mpi-inf.mpg.de/

## introduction
Egocentric 3D human motion capture using wearable de-
vices has received increased attention recently [1, 11, 22,
31, 37, 38, 40–42, 45, 48, 52, 53].
Different from tra-
ditional vision-based motion capture setups that require a
fixed recording space, egocentric systems allow flexible
motion capture in less constrained situations. Therefore, the
egocentric setups offer various applications, such as motion
analysis and XR technologies (Fig. 1-(g)).
Previous works proposed various egocentric methods to
capture device users. On the one hand, the vast majority of
existing methods—which use a monocular camera—would
fail for complex human poses due to depth ambiguity and
self-occlusion. On the other hand, the methods designed
for stereo devices do not yet realize the full potential of
their stereo settings, especially with the most recent com-
pact eyeglasses-based setups [1, 53]. Specifically, they do
not deliver high 3D reconstruction accuracy across differ-
ent scenarios. Moreover, these approaches do not consider
scene information, which further limits their accuracy.
To address the challenges outlined above, we propose a
new transformer-based framework for egocentric 3D human
motion capture from compact eyeglasses-based devices; see
Fig. 1. The first step of our framework is to estimate 2D
joint heatmaps from egocentric stereo fisheye RGB videos
(Sec. 4.1).
These 2D joint heatmaps are then processed
with human joint queries in our transformer-based 3D mod-
1
arXiv:2401.00889v2  [cs.CV]  15 May 2024

ule to estimate 3D poses. Here, we leverage the scene in-
formation and temporal context of the input videos in the
3D module to improve estimation accuracy. Firstly, we use
uniformly sampled windows of egocentric stereo frames to
reconstruct a 3D background scene using Structure from
Motion (SfM) [33], obtaining scene depth as additional in-
formation for the 3D module (Sec. 4.2 and 4.3). In our
challenging eyeglasses-based setup, however, the 3D scene
and camera poses can not always be estimated due to se-
vere self-occlusion in the egocentric images. This results in
depth maps with zero (invalid) values and undesired compu-
tation of network gradients during training. To mitigate this
issue, we propose to use depth padding masks that prevent
processing such invalid depth values in the 3D module. Ad-
ditionally, we propose video-dependent query augmentation
that enhances the joint queries with the temporal context of
stereo video inputs to effectively capture the temporal rela-
tion of human motions at a joint level (Sec. 4.4).
We also introduce two new benchmark datasets: Un-
realEgo2 and UnrealEgo-RW. UnrealEgo2 is an extended
version of UnrealEgo [1] and the largest eyeglasses-
based synthetic data with various new motions, offer-
ing 2.8× larger data (2.5M images) than the existing
dataset [1]. UnrealEgo-RW is a real-world dataset recorded
with our newly developed device that resembles the virtual
eyeglasses-based setup [1], offering 260k images with var-
ious motions and 3D poses. The proposed datasets make
it possible to evaluate existing and upcoming methods on a
variety of motions, not only in synthetic scenes but also in
real-world cases.
In short, the contributions of this paper are as follows:
• The transformer-based framework for egocentric stereo
3D human pose estimation that accounts for temporal
context in egocentric stereo views.
• 3D pose estimation is enhanced via the utilization of
scene information from our video-based 3D scene recon-
struction module as well as joint queries obtained from
our video-dependent query augmentation policy.
• A new portable device for egocentric stereo view capture
with its specification and two new benchmark datasets:
UnrealEgo2 and UnrealEgo-RW recorded with our de-
vice. The proposed datasets allow for a comprehensive
evaluation of methods for egocentric 3D human pose es-
timation from stereo views.
Our experiments demonstrate that the proposed method out-
performs the previous state-of-the-art approaches by a sub-
stantial margin, i.e., >15% on UnrealEgo [1], ≥40% on Un-
realEgo2, and ≥10% on UnrealEgo-RW (on MPJPE). We
release UnrealEgo2, UnrealEgo-RW, and our trained mod-
els on our project page3 and Benchmark Challenge4 to fos-
ter the area of egocentric 3D vision.
3https://4dqv.mpi-inf.mpg.de/UnrealEgo2/
4https://unrealego.mpi-inf.mpg.de/

## method
We propose a new framework for egocentric stereo 3D hu-
man pose estimation as shown in Fig. 3. Our framework
first estimates the 2D joint heatmaps from egocentric stereo
fisheye videos in our 2D module (Sec. 4.1). The heatmaps
and input videos are then processed in our segmentation
module to obtain 2D human body masks (Sec. 4.2). Next,
we use uniformly sampled windows of input frames and hu-
man body masks to reconstruct 3D scenes (Sec. 4.3). Here,
we render depth maps and depth region masks from the re-
constructed mesh. Finally, our transformer-based 3D mod-
ule processes the joint heatmaps, depth information, and
joint queries to estimate 3D poses (Sec. 4.4). Here, the 3D
module leverages depth padding masks based on the avail-
ability of the depth maps as well as joint queries enhanced
by the stereo video features from the 2D module.
4.1. 2D Pose Estimation
Given egocentric stereo videos with T frames {It
Left, It
Right ∈
RH×W ×3|t = 1, 2, . . . , T}, we use the existing stereo
2D joint heatmap estimator [1] to obtain a sequence of
corresponding 2D heatmaps of 15 joints {Ht
Left, Ht
Right ∈
R
H
4 × W
4 ×15}, including the neck, upper arms, lower arms,
hands, thighs, calves, feet, and balls of the feet.
We
also extract intermediate feature maps {Ft
Left, Ft
Right
∈
R
H
32 × W
32 ×C} where C = 512, which are used later in the
3D module.
3

Transformer Decoder
3D Module (Sec. 4.4)
𝐆𝐑𝐢𝐠𝐡𝐭
𝟏
𝐆𝐋𝐞𝐟𝐭
𝟏
𝐆𝐑𝐢𝐠𝐡𝐭
𝟐
𝐆𝐋𝐞𝐟𝐭
𝟐
𝐆𝐑𝐢𝐠𝐡𝐭
𝑻
𝐆𝐋𝐞𝐟𝐭
𝑻
𝐪𝟏
𝐪𝟐
𝐪𝑻
Regression Head
𝐏𝟏
𝐏𝟐
𝐏𝑻
𝐔𝐑𝐢𝐠𝐡𝐭
𝟏
𝐔𝐋𝐞𝐟𝐭
𝟏
𝐔𝐑𝐢𝐠𝐡𝐭
𝟐
𝐔𝐋𝐞𝐟𝐭
𝟐
𝐔𝐑𝐢𝐠𝐡𝐭
𝑻
𝐔𝐋𝐞𝐟𝐭
𝑻
+
+
+
𝑉𝐃𝐞𝐩𝐭𝐡
𝟏
𝑉𝐃𝐞𝐩𝐭𝐡
𝟐
𝑉𝐃𝐞𝐩𝐭𝐡
𝑻
3D Pose Output
𝐃𝐑𝐢𝐠𝐡𝐭
𝟏
𝐃𝐋𝐞𝐟𝐭
𝟏
𝐃𝐑𝐢𝐠𝐡𝐭
𝟐
𝐃𝐋𝐞𝐟𝐭
𝟐
𝐃𝐑𝐢𝐠𝐡𝐭
𝑻
𝐃𝐋𝐞𝐟𝐭
𝑻
𝐌𝐑𝐢𝐠𝐡𝐭
𝟏
𝐌𝐋𝐞𝐟𝐭
𝟏
𝐌𝐑𝐢𝐠𝐡𝐭
𝟐
𝐌𝐋𝐞𝐟𝐭
𝟐
𝐌𝐑𝐢𝐠𝐡𝐭
𝑻
𝐌𝐋𝐞𝐟𝐭
𝑻
𝐈𝐑𝐢𝐠𝐡𝐭
𝟏
𝐈𝐋𝐞𝐟𝐭
𝟏
𝐈𝐑𝐢𝐠𝐡𝐭
𝑻
𝐈𝐋𝐞𝐟𝐭
𝑻
෡𝐇𝐑𝐢𝐠𝐡𝐭
𝟏
෡𝐇𝐋𝐞𝐟𝐭
𝟏
෡𝐇𝐑𝐢𝐠𝐡𝐭
𝑻
෡𝐇𝐋𝐞𝐟𝐭
𝑻
2D Module
(Sec. 4.1)
𝐈𝐑𝐢𝐠𝐡𝐭
𝟐
𝐈𝐋𝐞𝐟𝐭
𝟐
෡𝐇𝐑𝐢𝐠𝐡𝐭
𝟐
෡𝐇𝐋𝐞𝐟𝐭
𝟐
Human Body 
Segmentation
(Sec. 4.2)
3D Scene 
Reconstruction
(Sec. 4.3)
Stereo Video Input
𝐑𝐑𝐢𝐠𝐡𝐭
𝟏
𝐑𝐋𝐞𝐟𝐭
𝟏
𝐑𝐑𝐢𝐠𝐡𝐭
𝟐
𝐑𝐋𝐞𝐟𝐭
𝟐
𝐑𝐑𝐢𝐠𝐡𝐭
𝑻
𝐑𝐋𝐞𝐟𝐭
𝑻
Depth Map and Depth Region Mask
Human Body Mask
2D Joint Heatmap
※ depth data when no depth value is found
Scaled 3D Scene Mesh
𝐅𝐑𝐢𝐠𝐡𝐭
𝑻
𝐅𝐋𝐞𝐟𝐭
𝑻
𝐅𝐑𝐢𝐠𝐡𝐭
𝟏
𝐅𝐋𝐞𝐟𝐭
𝟏
𝐅𝐑𝐢𝐠𝐡𝐭
𝟐
𝐅𝐋𝐞𝐟𝐭
𝟐
𝐅𝐒𝐭𝐞𝐫𝐞𝐨
Figure 3. Overview of our framework. Our method takes egocentric stereo videos {It
Left, It
Right} as inputs. We first apply the 2D module
to obtain 2D joint heatmaps {Ht
Left, Ht
Right} and video features {Ft
Left, Ft
Right} (Sec. 4.1). The heatmaps are used with input videos to create
human body masks {Mt
Left, Mt
Right} (Sec. 4.2). Next, we use uniformly sampled windows of input frames and human body masks to
reconstruct a 3D scene mesh (Sec. 4.3). From the mesh, we generate depth maps {Dt
Left, Dt
Right} and depth region masks {Rt
Left, Rt
Right}.
Note that this diagram shows an example case of missing depth values for the second input frame. Lastly, the depth data, 2D joint heatmaps,
video features, joint queries qt and the padding masks V t
Depth are processed in the 3D module to estimate 3D poses Pt (Sec. 4.4).
4.2. Human Body Segmentation
To reconstruct 3D scenes from egocentric videos, it is nec-
essary to identify the pixels corresponding to the back-
ground environment.
Therefore, we integrate an exist-
ing segmentation method, i.e., ViT-H SAM model [16],
as our segmentation network FSAM. In this module, we
firstly obtain 2D joint locations from the 2D joint heatmap
{ bHt
Left, bHt
Right}.
Then, we use the input video frames
{It
Left, It
Right} and its corresponding 2D joints to extract a
human body mask {Mt
Left, Mt
Right ∈RH×W ×1}:
Mt
Left = FSAM(It
Left, bHt
Left).
(1)
The same process can be applied to obtain Mt
Right. Note that
we use the SAM model without re-training on ground-truth
human body masks. Instead, we guide the predictions of
SAM using joint positions extracted from the 2D heatmaps.
4.3. 3D Scene Reconstruction
We aim to reconstruct 3D environments from uniformly
sampled windows of input frames {It
Left, It
Right} and hu-
man body masks {Mt
Left, Mt
Right} with a fixed length. The
length is set to 4 seconds (some motion data contains shorter
sequences).
Given these data, we use Metashape [24]
to perform SfM to obtain camera poses and a 3D scene
mesh. Here, as the baseline length between stereo cam-
eras is known, i.e., 12cm, we can obtain the mesh in the
real-world scale.
Next, we render down-sampled depth
maps {Dt
Left, Dt
Right ∈R
H
4 × W
4 ×1} and depth region masks
{Rt
Left, Rt
Right ∈R
H
4 × W
4 ×1} from the reconstructed 3D
scene mesh.
The depth region masks show the regions
where the depth values are obtained from the 3D scene.
This depth information will be used later in the 3D mod-
ule as additional cues for pose estimation. However, there
are some cases where the egocentric RGB videos are largely
occupied by a human body. In such scenarios, the 3D scene
can not be reconstructed or camera poses can not be esti-
mated. This results in missing (invalid) depth values and
undesired computation of network gradients during train-
ing. Therefore, we tackle this issue in our 3D module.
4.4. 3D Pose Estimation
In the 3D module, we aim to estimate a sequence of 3D
poses by considering scene information and the tempo-
ral context of the egocentric stereo videos.
Specifically,
given the 2D joint heatmaps, depth maps, depth region
masks, and T sets of joint queries qt ∈R16× C
2 , we use
a transformer decoder to estimate a sequence of 3D poses
{Pt ∈R16×3|t = 1, 2, . . . , T}. Our pose output is the 3D
pose at the last time step PT .
We follow the existing
4

## experiments
We compare our method with existing stereo-based ego-
centric pose estimation methods [1, 53]. We use the of-
6

Figure 5. Results of our framework and comparison methods on example sequences from UnrealEgo2 (above) and UnrealEgo-RW (below).
Left: MPJPE curves. Right: Outputs of our method at frame 87 and 329 of the sequences, respectively. 3D pose estimation and ground
truth are colored in red and green, respectively.
ficial source code of Akada et al. [1] and re-implement the
framework of Zhao et al. [53] as its source code is not avail-
able. Note that the comparison methods are trained on the
same datasets as our model. Kang et al. [12] (arXiv pre-
print at the time of submission) only shows results of the
pelvis-relative estimation on UnrealEgo. Therefore, we in-
clude them for reference. Furthermore, we are interested
in the performance of the publicly available state-of-the-art
method [1] with temporal inputs. Thus, we modify their
3D module such that it can take as an input a sequence of
stereo 2D keypoint heatmaps with the same time step as
ours, i.e., T = 5. Here, we replace the first and the last fully
connected layers in the encoder, the pose decoder, and the
heatmap reconstruction decoder of their autoencoder-based
3D module [1] by those with T times the size of the orig-
inal hidden dimension. We denote this model as Baseline
and train it with the same training procedure as Akada et
al. [1]. Note that Akada et al. [1], Baseline, and our model
use the same 2D module.
We follow the existing works [1, 11, 37, 38, 40–42, 45,
52, 53] to report Mean Per Joint Position Error (MPJPE)
and Mean Per Joint Position Error with Procrustes Align-
ment [13] (PA-MPJPE). We additionally report 3D Percent-
age of Correct Keypoints (3D PCK) and Area Under the
Curve (AUC) for UnrealEgo2 and UnrealEgo-RW.
Results on Synthetic Datasets. Tables 1 and 2 report the
results with UnrealEgo [1] and UnrealEgo2. Our method
outperforms the existing methods [1, 12, 53] and Baseline
across all metrics by a significant margin, e.g., >15% on
UnrealEgo [1] and ≥40% on UnrealEgo2 (on MPJPE). The
qualitative results on UnrealEgo2 in Fig. 4-(left part) show
that existing methods and Baseline fail to estimate lower
bodies of complex poses with severe self-occlusions, such
as crouching. Even under such challenging scenarios, how-
ever, our approach yields accurate 3D poses. See Fig. 5-
(above part) for a MPJPE curve and visual outputs of our

## related_work
Egocentric 3D Human Motion Capture.
Recent years
witnessed significant innovations in egocentric 3D human
pose estimation. To capture device users, many existing
works use downward-facing cameras and the existing meth-
ods can be categorized into two groups. The first group are
monocular approaches [11, 21, 22, 27, 37, 38, 40, 41, 43,
45, 48, 52]. For example, Wang et al. [43] uses a diffusion-
based [10] motion prior to tackle self-conclusions. Due to
the depth ambiguity, monocular methods often fail to esti-
mate accurate 3D poses. Wang et al. [42] tackled this issue
by projecting depth and 2D pose features into a pre-defined
voxel space. This method requires additional training with
ground-truth depths and human body segmentation; it can-
not easily be extended for multi-view or temporal inputs.
Zhang et al. [51] utilized a diffusion model [10] conditioned
on a 3D scene to generate poses. They require pre-scanned
scene mesh as an input and cannot capture a device user.
The second group, including our work, focuses on the
multi-view (often stereo) setting. Rhodin et al. [31] pro-
posed an optimization approach whereas Cha et al. [3] used
eight cameras to estimate a 3D body and reconstruct a 3D
scene separately. Other works [1, 53] used the multi-branch
autoencoder [37] to the stereo setup. Kang et al. [12] (arXiv
pre-print at the time of submission) leveraged a stereo-
matching mechanism and perspective embedding heatmaps.
In contrast to the existing methods, we propose a new
transformer-based method that effectively utilizes egocen-
tric stereo videos via our video-based 3D scene reconstruc-
tion module and video-dependent query augmentation pol-
icy. Our method considers the scene information without
the supervision of the scene data.
Transformers in 3D Human Pose Estimation from Ex-
ternal Cameras. 3D pose estimation from external cam-
eras has shown significant progress due to the advances in
transformer architectures [39]. Some works [20, 47] pre-
dict 3D human pose and mesh from monocular views. Other
works [5, 18, 19, 28, 29, 36, 46, 49, 54–58] present a 2D-to-
3D lifting module that estimates 3D poses from monocular
2D joints obtained with off-the-shelf 2D joint detectors. Al-
though their lifting modules show impressive results, those
monocular methods cannot be easily applied to our stereo
setting. On the other hand, some works utilize transformers
in multi-view settings. He et al. [9] and Ma et al. [23] aggre-
gate stereo information on epipolar lines of stereo images,
which are difficult to obtain from fisheye images. Recent
work [44] regresses multi-person 3D poses from multi-view
inputs, powered by projective attention and query adapta-
tion. However, no existing works explored the potential of
transformers along with 2D joint heatmaps or explicit scene
information in stereo 3D pose estimation. In this paper, we
propose a transformer-based framework that accounts for
the temporal relation of human motion at a joint level via
2

intermediate 2D joint heatmap and depth maps even with
inaccurate depth values mixed in the framework.
Datasets for Egocentric 3D Human Pose Estimation.
Several works proposed unique setups to create datasets,
using a monocular camera [11, 17, 22, 37, 40, 41, 45, 48]
and forward-facing cameras [11, 14, 17, 22, 26, 48, 50,
51].
There also exist datasets captured with stereo de-
vices [3, 7, 14, 26, 31, 53]. However, they are small [31]
with limited motion types [31, 53], not publicly avail-
able [3, 53], or do not provide ground truth 3D poses of de-
vice users [7, 14, 26]. Recently, Akada et al. [1] introduced
UnrealEgo, a synthetic dataset based on virtual eyeglasses
with two fisheye cameras.
However, they provide only
synthetic images. Meanwhile, more glasses-based stereo
datasets that offer a wider variety of motions or real-world
footage are required nowadays for an extensive evaluation
of existing and upcoming methods. Hence, we introduce
two new benchmark datasets that in their characteristics go
beyond the existing data: UnrealEgo2 and UnrealEgo-RW.
We describe the proposed datasets in the following section.
3. Mobile Device and Datasets
We present two new datasets for egocentric stereo 3D mo-
tion capture: UnrealEgo2 and UnrealEgo-RW; see Fig. 1.
Please watch our supplementary video for visualizations.
UnrealEgo2 Dataset. To create UnrealEgo2 (an extension
of UnrealEgo [1]), we adapt the publicly available setup
with a virtual eyeglasses device [1]. This setup comes with
two downward-facing fisheye cameras attached 12cm apart
from each other on the glasses frames. The camera’s field of
view is 170◦. With this device, we capture 17 realistic 3D
human models [30] animated by the Mixamo [25] dataset in
various 3D environments. We record simple to highly com-
plex motions such as crouching and crawling, for 14 hours.
Overall, UnrealEgo2 offers 15,207 motions and >1.25M
stereo views (2.5M images) as well as depth maps with a
resolution 1024×1024 pixel rendered at 25 frames per sec-
ond. Each frame is annotated with 32 body and 40 hand
joints. Note that UnrealEgo2 is the largest glasses-based
dataset and 2.8× larger than UnrealEgo. Also, it does not
share the same motions with UnrealEgo, providing a larger
motion variety for a comprehensive evaluation.
Design of Our Mobile Device. Evaluation with real-world
datasets plays a pivotal role in computer vision research.
Therefore, we build a new portable device; see Fig. 2. Our
device is based on a helmet with two RIBCAGE RX0 II
cameras [32] and two FUJINON FE185C057HA-1 fisheye
lenses [6]. We placed the cameras 12cm away from each
other and 2cm away from user’s face. We cropped the mar-
gins of the egocentric images to resemble the field of view
of 170◦of the UnrealEgo and UnrealEgo2 setups. Note that
our setup is more compact than EgoCap [31] that placed
cameras 25cm away from user’s face.
Figure 2. Our portable setup to acquire UnrealEgo-RW.
UnrealEgo-RW (Real-World) Dataset. With our device,
we record various motions of 16 identities in a multi-view
motion capture studio (Fig. 1-(d)). We capture simple and
challenging activities, e.g., crawling and dancing, for 1.5
hours. This is in strong contrast to the existing real-world
stereo dataset [53] (not publicly available) that records only
three simple actions, i.e., sitting, standing, and walking.
In total, we obtained 591 motion segments from 16
identities with various textured clothing.
This results in
more than 130k stereo views (260k images) of a resolu-
tion 872×872 pixel rendered at 25 frames per second with
ground-truth 3D poses of 16 joints. Note that UnrealEgo-
RW offers 4.3× larger data with a wider variety of motions
than the publicly available real-world stereo data [31].

## conclusion
In this paper, we proposed a new transformer-based frame-
work that significantly boosts the accuracy of egocentric
stereo 3D human pose estimation. The proposed frame-
work leverages the scene information and temporal context
of egocentric stereo video inputs via our video-based 3D
scene reconstruction module and video-based joint query
augmentation policy. Our extensive experiments on the new
synthetic and real-world datasets with challenging human
motions validate the effectiveness of our approach com-
pared to the existing methods. We hope that our proposed
benchmark datasets and trained models will foster the fur-
ther development of methods for egocentric 3D vision.
Acknowledgment. The work was supported by the ERC
Consolidator Grant 4DReply (770784) and the Nakajima
Foundation. We thank Silicon Studio Corp. for providing
the fisheye plug-in for Unreal Engine.
8