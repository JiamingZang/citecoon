# Learning from Synthetic Humans

> 2017 · id: W2576289912 · arXiv: 1701.01370 · pdf: https://arxiv.org/pdf/1701.01370 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Estimating human pose, shape, and motion from images
and videos are fundamental challenges with many applica-
tions. Recent advances in 2D human pose estimation use
large amounts of manually-labeled training data for learn-
ing convolutional neural networks (CNNs). Such data is
time consuming to acquire and difﬁcult to extend. More-
over, manual labeling of 3D pose, depth and motion is im-
practical.
In this work we present SURREAL (Synthetic
hUmans foR REAL tasks): a new large-scale dataset with
synthetically-generated but realistic images of people ren-
dered from 3D sequences of human motion capture data. We
generate more than 6 million frames together with ground
truth pose, depth maps, and segmentation masks. We show
that CNNs trained on our synthetic dataset allow for ac-
curate human depth estimation and human part segmenta-
tion in real RGB images. Our results and the new dataset
open up new possibilities for advancing person analysis us-
ing cheap and large-scale synthetic data.

## introduction
Convolutional Neural Networks provide signiﬁcant
gains to problems with large amounts of training data. In the
ﬁeld of human analysis, recent datasets [4, 37] now gather
a sufﬁcient number of annotated images to train networks
for 2D human pose estimation [23, 41]. Other tasks such as
accurate estimation of human motion, depth and body-part
segmentation are lagging behind as manual supervision for
such problems at large scale is prohibitively expensive.
Images of people have rich variation in poses, cloth-
ing, hair styles, body shapes, occlusions, viewpoints, mo-
tion blur and other factors. Many of these variations, how-
∗Inria, France
†D´epartement d’informatique de l’ENS, ´Ecole normale sup´erieure,
CNRS, PSL Research University, 75005 Paris, France
‡Max Planck Institute for Intelligent Systems, T¨ubingen, Germany
§Laboratoire Jean Kuntzmann, Grenoble, France
¶Currently at Body Labs Inc., New York, NY. This work was per-
formed while JR was at MPI-IS.
Figure 1. We generate photo-realistic synthetic images and their
corresponding ground truth for learning pixel-wise classiﬁcation
problems: human part segmentation and depth estimation. The
convolutional neural network trained only on synthetic data gen-
eralizes to real images sufﬁciently for both tasks. Real test images
in this ﬁgure are taken from MPII Human Pose dataset [4].
ever, can be synthesized using existing 3D motion capture
(MoCap) data [3, 18] and modern tools for realistic render-
ing. Provided sufﬁcient realism, such an approach would be
highly useful for many tasks as it can generate rich ground
truth in terms of depth, motion, body-part segmentation and
occlusions.
Although synthetic data has been used for many years,
realism has been limited.
In this work we present
SURREAL: a new large-scale dataset with synthetically-
generated but realistic images of people. Images are ren-
dered from 3D sequences of MoCap data. To ensure real-
ism, the synthetic bodies are created using the SMPL body
model [20], whose parameters are ﬁt by the MoSh [21]
method given raw 3D MoCap marker data. We randomly
sample a large variety of viewpoints, clothing and light-
ing. SURREAL contains more than 6 million frames to-
gether with ground truth pose, depth maps, and segmenta-
1
arXiv:1701.01370v3  [cs.CV]  19 Jan 2018

Figure 2. Our pipeline for generating synthetic data. A 3D human body model is posed using motion capture data and a frame is rendered
using a background image, a texture map on the body, lighting and a camera position. These ingredients are randomly sampled to increase
the diversity of the data. We generate RGB images together with 2D/3D poses, surface normals, optical ﬂow, depth images, and body-part
segmentation maps for rendered people.
tion masks. We show that CNNs trained on synthetic data
allow for accurate human depth estimation and human part
segmentation in real RGB images, see Figure 1. Here, we
demonstrate that our dataset, while being synthetic, reaches
the level of realism necessary to support training for mul-
tiple complex tasks. This opens up opportunities for train-
ing deep networks using graphics techniques available now.
SURREAL dataset is publicly available together with the
code to generate synthetic data and to train models for body
part segmentation and depth estimation [1].
The rest of this paper is organized as follows. Section 2
reviews related work. Section 3 presents our approach for
generating realistic synthetic videos of people. In Section 4
we describe our CNN architecture for human body part seg-
mentation and depth estimation. Section 5 reports experi-
ments. We conclude in Section 6.

## method
In this section, we present our approach for human body
part segmentation [5, 25] and human depth estimation [10,
11, 19], which we train with synthetic and/or real data, see
Section 5 for the evaluation.
Our approach builds on the stacked hourglass network
architecture introduced originally for 2D pose estimation
problem [23].
This network involves several repetitions
of contraction followed by expansion layers which have
skip connections to implicitly model spatial relations from
different resolutions that allows bottom-up and top-down
structured prediction. The convolutional layers with resid-
ual connections and 8 ‘hourglass’ modules are stacked on
top of each other, each successive stack taking the previous
stack’s prediction as input. The reader is referred to [23] for
more details. A variant of this network has been used for
scene depth estimation [6]. We choose this architecture be-
cause it can infer pixel-wise output by taking into account
human body structure.
Our network input is a 3-channel RGB image of size
256 × 256 cropped and scaled to ﬁt a human bounding box
using the ground truth. The network output for each stack
has dimensions 64×64×15 in the case of segmentation (14
classes plus the background) and 64×64×20 for depth (19
depth classes plus the background). We use cross-entropy
loss deﬁned on all pixels for both segmentation and depth.
The ﬁnal loss of the network is the sum over 8 stacks. We
train for 50K iterations for synthetic pre-training using the
RMSprop algorithm with mini-batches of size 6 and a learn-
4

ing rate of 10−3. Our data augmentation during training
includes random rotations, scaling and color jittering.
We formulate the problem as pixel-wise classiﬁcation
task for both segmentation and depth. When addressing
segmentation, each pixel is assigned to one of the pre-
deﬁned 14 human parts, namely head, torso, upper legs,
lower legs, upper arms, lower arms, hands, feet (separately
for right and left) or to the background class. Regarding
the depth, we align ground-truth depth maps on the z-axis
by the depth of the pelvis joint, and then quantize depth
values into 19 bins (9 behind and 9 in front of the pelvis).
We set the quantization constant to 45mm to roughly cover
the depth extent of common human poses. The network is
trained to classify each pixel into one of the 19 depth bins
or background. At test time, we ﬁrst upsample feature maps
of each class with bilinear interpolation by a factor of 4 to
output the original resolution. Then, each pixel is assigned
to the class for which the corresponding channel has the
maximum activation.

## experiments
We test our approach on several datasets. First, we eval-
uate the segmentation and depth estimation on the test set of
our synthetic SURREAL dataset. Second, we test the per-
formance of segmentation on real images from the Freiburg
Sitting People dataset [25]. Next, we evaluate segmentation
and depth estimation on real videos from the Human3.6M
dataset [17, 18] with available 3D information. Then, we
qualitatively evaluate our approach on the more challeng-
ing MPII Human Pose dataset [4]. Finally, we experiment
and discuss design choices of the SURREAL dataset.
5.1. Evaluation measures
We use intersection over union (IOU) and pixel accuracy
measures for evaluating the segmentation approach. The ﬁ-
nal measure is the average over 14 human parts as in [25].
Depth estimation is formulated as a classiﬁcation problem,
but we take into account the continuity when we evaluate.
We compute root-mean-squared-error (RMSE) between the
predicted quantized depth value (class) and the ground truth
quantized depth on the human pixels. To interpret the error
in real world coordinates, we multiply it by the quantization
constant (45mm). We also report a scale and translation in-
variant RMSE (st-RMSE) by solving for the best transla-
tion and scaling in z-axis to ﬁt the prediction to the ground
truth. Since inferring depth from RGB is ambiguous, this is
a common technique used in evaluations [11].
5.2. Validation on synthetic images
Train/test split.
To evaluate our methods on synthetic im-
ages, we separate 20% of the synthetic frames for the test set
and train all our networks on the remaining training set. The
split is constructed such that a given CMU MoCap subject is
assigned as either train or test. Whereas some subjects have
a large number of instances, some subjects have unique ac-
Input
Predsegm
GTsegm
Preddepth
GTdepth
Figure 4. Segmentation and depth predictions on synthetic test set.
Input
Real
Synth
Synth+Real
GT
Figure 5. Part segmentation on the Freiburg Sitting People dataset,
training only on FSitting (Real), training only on synthetic im-
ages (Synth), ﬁne-tuning on 2 training subjects from FSitting
(Synth+Real). Fine-tuning helps although only for 200 iterations.
tions, and some actions are very common (walk, run, jump).
Overall, 30 subjects out of 145 are assigned as test. 28 test
subjects cover all common actions, and 2 have unique ac-
tions. Remaining subjects are used for training. Although
our synthetic images have different body shape and appear-
ance than the subject in the originating MoCap sequence,
we still found it appropriate to split by subjects. We separate
a subset of our body shapes, clothing and background im-
ages for the test set. This ensures that our tests are unbiased
with regards to appearance, yet are still representative of all
actions. Table 1 summarizes the number of frames, clips
and MoCap sequences in each split. Clips are the continu-
ous 100-frame sequences where we have the same random
body shape, background, clothing, camera and lighting. A
new random set is picked at every clip. Note that a few
sequences have less than 100 frames.
Results on synthetic test set.
The evaluation is per-
formed on the middle frame of each 100-frame clip on
the aforementioned held-out synthetic test set, totaling in
12,528 images. For segmentation, the IOU and pixel ac-
curacy are 69.13% and 80.61%, respectively. Evaluation
of depth estimation gives 72.9mm and 56.3mm for RMSE
and st-RMSE errors, respectively. Figure 4 shows sample
predictions. For both tasks, the results are mostly accurate
on synthetic test images. However, there exist a few chal-
lenging poses (e.g. crawling), test samples with extreme
close-up views, and ﬁne details of the hands that are caus-
5

Table 2. Parts segmentation results on 4 test subjects of Freiburg
Sitting People dataset. IOU for head, torso and upper legs (aver-
aged over left and right) are presented as well as the mean IOU
and mean pixel accuracy over 14 parts. The means do not include
background class. By adding an upsampling layer, we get the best
results reported on this dataset.
Head
Torso
Legsup
mean
mean
Training data
IOU
IOU
IOU
IOU
Acc.
Real+Pascal[25]
-
-
-
64.10
81.78
Real
58.44
24.92
30.15
28.77
38.02
Synth
73.20
65.55
39.41
40.10
51.88
Synth+Real
72.88
80.76
65.41
59.58
78.14
Synth+Real+up
85.09
87.91
77.00
68.84
83.37
Table 3. Parts segmentation results on Human3.6M. The best result
is obtained by ﬁne-tuning synthetic network with real images. Al-
though the performance of the network trained only with real data
outperforms training only with synthetic, the predictions visually
are worse because of overﬁtting, see Figure 6.
IOU
Accuracy
Training data
fg+bg
fg
fg+bg
fg
Real
49.61
46.32
58.54
55.69
Synth
46.35
42.91
56.51
53.55
Synth+Real
57.07
54.30
67.72
65.53
ing errors. In the following sections, we investigate if simi-
lar conclusions can be made for real images.
5.3. Segmentation on Freiburg Sitting People
Freiburg Sitting People (FSitting) dataset [25] is com-
posed of 200 high resolution (300x300 pixels) front view
images of 6 subjects sitting on a wheel chair. There are 14
human part annotations available. See Figure 5 for sample
test images and corresponding ground truth (GT) annota-
tion. We use the same train/test split as [25], 2 subjects
for training and 4 subjects for test. The amount of data is
limited for training deep networks. We show that our net-
work pre-trained only on synthetic images is already able to
segment human body parts. This shows that the human ren-
derings in the synthetic dataset are representative of the real
images, such that networks trained exclusively on synthetic
data can generalize quite well to real data.
Table 2 summarizes segmentation results on FSitting.
We carry out several experiments to understand the gain
from synthetic pre-training.
For the ‘Real’ baseline, we
train a network from scratch using 2 training subjects. This
network overﬁts as there are few subjects to learn from and
the performance is quite low. Our ‘Synth’ result is obtained
using the network pre-trained on synthetic images with-
out ﬁne-tuning. We get 51.88% pixel accuracy and 40.1%
IOU with this method and clearly outperform training from
real images. Furthermore, ﬁne-tuning (Synth+Real) with 2
training subjects helps signiﬁcantly. See Figure 5 for qual-
itative results. Given the little amount for training in FSit-
ting, the ﬁne-tuning converges after 200 iterations.
In [25], the authors introduce a network that outputs a
high-resolution segmentation after several layers of upcon-
volutions. For a fair comparison, we modify our network
to output full resolution by adding one bilinear upsampling
layer followed by nonlinearity (ReLU) and a convolutional
layer with 3 × 3 ﬁlters that outputs 15 × 300 × 300 in-
stead of 15 × 64 × 64 as explained in Section 4. If we ﬁne-
tune this network (Synth+Real+up) on FSitting, we improve
performance and outperform [25] by a large margin. Note
that [25] trains on the same FSitting training images, but
added around 2,800 Pascal images. Hence they use signiﬁ-
cantly more manual annotation than our method.
5.4. Segmentation and depth on Human3.6M
To evaluate our approach, we need sufﬁcient real data
with ground truth annotations. Such data is expensive to
obtain and currently not available.
For this reason, we
generate nearly perfect ground truth for images recorded
with a calibrated camera and given their MoCap data. Hu-
man3.6M [17, 18] is currently the largest dataset where such
information is available. There are 3.6 million frames from
4 cameras. We use subjects S1, S5, S6, S7, S8 for train-
ing, S9 for validation and S11 for testing as in [35, 42], but
from all 4 cameras. Note that this is different from the ofﬁ-
cial train/test split [18]. Each subject performs each of the
15 actions twice. We use all frames from one of the two
instances of each action for training, and every 64th frame
from all instances for testing. The frames have resolution
1000×1000 pixels, we assume a 256×256 cropped human
bounding box is given to reduce computational complex-
ity. We evaluate the performance of both segmentation and
depth, and compare with the baseline for which we train a
network on real images only.
Segmentation.
Table 3 summarizes the parts segmenta-
tion results on Human3.6M. Note that these are not com-
parable to the results in [16] both because they assume the
background segment is given and our ground truth segmen-
tation data is not part of the ofﬁcial release (see Section 3.2).
We report both the mean over 14 human parts (fg) and the
mean together with the background class (fg+bg). Training
on real images instead of synthetic images increases IOU
by 3.4% and pixel accuracy by 2.14%. This is expected be-
cause the training distribution matches the test distribution
in terms of background, camera position and action cate-
gories (i.e. poses). Furthermore, the amount of real data is
sufﬁcient to perform CNN traini

## related_work
Knowledge transfer from synthetic to real images has
been recently studied with deep neural networks. Dosovit-
skiy et al. [8] learn a CNN for optical ﬂow estimation us-
ing synthetically generated images of rendered 3D moving
chairs. Peng et al. [26] study the effect of different visual
cues such as object/background texture and color when ren-
dering synthetic 3D objects for object detection task. Sim-
ilarly, [40] explores rendering 3D objects to perform view-
point estimation. Fanello et al. [12] render synthetic in-
frared images of hands and faces to predict depth and parts.
Recently, Gaidon et al. [13] have released the Virtual KITTI
dataset with synthetically generated videos of cars to study
multi-object tracking.
Several works focused on creating synthetic images of
human bodies for learning 2D pose estimation [27, 30, 36],
3D pose estimation [7, 9, 14, 24, 35, 39, 44], pedestrian
detection [22, 27, 28], and action recognition [31, 32].
Pishchulin et al. [28] generate synthetic images with a game
engine. In [27], they deform 2D images with a 3D model.
More recently, Rogez and Schmid [35] use an image-based
synthesis engine to augment existing real images. Ghezel-
ghieh et al. [14] render synthetic images with 10 simple
body models with an emphasis on upright people; however,
the main challenge using existing MoCap data for training
is to generalize to poses that are not upright. Human3.6M
dataset [18] presents realistic rendering of people in mixed
reality settings; however, the approach to create these is ex-
pensive.
A similar direction has been explored in [31, 32, 33, 38].
In [31], action recognition is addressed with synthetic hu-
man trajectories from MoCap data. [32, 38] train CNNs
with synthetic depth images. EgoCap [33] creates a dataset
by augmenting egocentric sequences with background.
The closest work to this paper is [7], where the authors
render large-scale synthetic images for predicting 3D pose
with CNNs. Our dataset differs from [7] by having a richer,
per-pixel ground truth, thus allowing to train for pixel-wise
predictions and multi-task scenarios. In addition, we argue
that the realism in our synthetic images is better (see sample
videos in [1]), thus resulting in a smaller gap between fea-
tures learned from synthetic and real images. The method
in [7] heavily relies on real images as input in their train-
ing with domain adaptation. This is not the case for our
synthetic training. Moreover, we render video sequences
which can be used for temporal modeling.
Our dataset presents several differences with existing
synthetic datasets. It is the ﬁrst large-scale person dataset
providing depth, part segmentation and ﬂow ground truth
for synthetic RGB frames. Other existing datasets are used
either for taking RGB image as input and training only for
2D/3D pose, or for taking depth/infrared images as input
and training for depth/parts segmentation. In this paper, we
2

show that photo-realistic renderings of people under large
variations in shape, texture, viewpoint and pose can help
solving pixel-wise human labeling tasks.
3. Data generation
This section presents our SURREAL (Synthetic hUmans
foR REAL tasks) dataset and describes key steps for its
generation (Section 3.1). We also describe how we obtain
ground truth data for real MoCap sequences (Section 3.2).
3.1. Synthetic humans
Our pipeline for generating synthetic data is illustrated in
Figure 2. A human body with a random 3D pose, random
shape and random texture is rendered from a random view-
point for some random lighting and a random background
image. Below we deﬁne what “random” means in all these
cases. Since the data is synthetic, we also generate ground
truth depth maps, optical ﬂow, surface normals, human part
segmentations and joint locations (both 2D and 3D). As a
result, we obtain 6.5 million frames grouped into 67, 582
continuous image sequences. See Table 1 for more statis-
tics, Section 5.2 for the description of the synthetic train/test
split, and Figure 3 for samples from the SURREAL dataset.
Body model.
Synthetic bodies are created using the
SMPL body model [20]. SMPL is a realistic articulated
model of the body created from thousands of high-quality
3D scans, which decomposes body deformations into pose
(kinematic deformations due to skeletal posture) and shape
(body deformations intrinsic to a particular person that
make them different from others). SMPL is compatible with
most animation packages like Blender [2]. SMPL deforma-
tions are modeled as a combination of linear blend skinning
and linear blendshapes deﬁned by principal components of
body shape variation. SMPL pose and shape parameters are
converted to a triangulated mesh using Blender, which then
applies texture, shading and adds a background to generate
the ﬁnal RGB output.
Body shape.
In order to render varied, but realistic, body
shapes we make use of the CAESAR dataset [34], which
was used to train SMPL. To create a body shape, we select
one of the CAESAR subjects at random and approximate
their shape with the ﬁrst 10 SMPL shape principal compo-
nents. Ten shape components explain more than 95% of the
shape variance in CAESAR (at the resolution of our mesh)
and produce quite realistic body shapes.
Body pose.
To generate images of people in realistic
poses, we take motion capture data from the CMU MoCap
database [3]. CMU MoCap contains more than 2000 se-
quences of 23 high-level action categories, resulting in more
than 10 hours of recorded 3D locations of body markers.
It is often challenging to realistically and automatically
retarget MoCap skeleton data to a new model. For this rea-
son we do not use the skeleton data but rather use MoSh [21]
to ﬁt the SMPL parameters that best explain raw 3D Mo-
Cap marker locations. This gives both the 3D shape of the
subject and the articulated pose parameters of SMPL. To in-
crease the diversity, we replace the estimated 3D body shape
with a set of randomly sampled body shapes.
We render each CMU MoCap sequence three times using
different random parameters. Moreover, we divide the se-
quences into clips of 100 frames with 30%, 50% and 70%
overlaps for these three renderings. Every pose of the se-
quence is rendered with consistent parameters (i.e. body
shape, clothing, light, background etc.) within each clip.
Human texture.
We use two types of real scans for the
texture of body models. First, we extract SMPL texture
maps from CAESAR scans, which come with a color tex-
ture per 3D point. These maps vary in skin color and person
identities, however, their quality is often low due to the low
resolution, uniform tight-ﬁtting clothing, and visible mark-
ers placed on the face and the body. Anthropometric mark-
ers are automatically removed from the texture images and
inpainted. To provide more variety, we extract a second set
of textures obtained from 3D scans of subjects with normal
clothing. These scans are registered with 4Cap as in [29].
The texture of real clothing substantially increases the re-
alism of generated images, even though SMPL does not
model 3D deformations of clothes.
20% of our data is rendered with the ﬁrst set (158 CAE-
SAR textures randomly sampled from 4000), and the rest
with the second set (772 clothed textures). To preserve the
anonymity of subjects, we replace all faces in the texture
maps by the average CAESAR face. The skin color of this
average face is corrected to ﬁt the face skin color of the
original texture map. This corrected average face is blended
smoothly with the original map, resulting in a realistic and
anonymized body texture.
Light.
The body is illuminated using Spherical Harmon-
ics with 9 coefﬁcients [15]. The coefﬁcients are randomly
sampled from a uniform distribution between −0.7 and 0.7,
apart from the ambient illumination coefﬁcient (which has
a minimum value of 0.5) and the vertical illumination com-
ponent, which is biased to encourage the illumination from
above. Since Blender does not provide Spherical Harmon-
ics illumination, a spherical harmonic shader for the body
material was implemented in Open Shading Language.
Camera.
The projective camera has a resolution of 320×
240, focal length of 60mm and sensor size of 32mm. To
generate images of the body in a wide range of positions,
we take 100-frame MoCap sub-sequences and, in the ﬁrst
frame, render the body so that the center of the viewport
points to the pelvis of the body, at a random distance (sam-
pled from a normal distribution with 8 meters mean, 1 meter
deviation) with a random yaw angle. The remainder of the
sequence then effectively produces bodies in a range of lo-
cations relative to the static camera.
Background.
We render the person on top of a static
background image. To ensure that the backgrounds are rea-
sonably realistic and do not include other people, we sam-
ple from a subset of LSUN dataset [43] that includes total
3

Figure 3. Sample frames from our SURREAL dataset with a large variety of poses, body sh

## conclusion
In this study, we have shown successful large-scale train-
ing of CNNs from synthetically generated images of people.
8

We have addressed two tasks, namely, human body part seg-
mentation and depth estimation, for which large-scale man-
ual annotation is infeasible. Our generated synthetic dataset
comes with rich pixel-wise ground truth information and
can potentially be used for other tasks than considered here.
Unlike many existing synthetic datasets, the focus of SUR-
REAL is on the realistic rendering of people, which is a
challenging task. In our future work, we plan to integrate
the person into the background in a more realistic way by
taking into account the lighting and the 3D scene layout.
We also plan to augment the data with more challenging
scenarios such as occlusions and multiple people.
Acknowledgements.
This work was supported in part
by the Alexander von Humbolt Foundation, ERC grants
ACTIVIA and ALLEGRO, the MSR-Inria joint lab, and
Google and Facebook Research Awards. We acknowledge
the Human3.6M dataset owners for providing the MoCap
marker data.