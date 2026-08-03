# Consensus-Based Optimization for 3D Human Pose Estimation in Camera Coordinates

> 2022 · id: W2990671151 · arXiv: 1911.09245 · pdf: https://arxiv.org/pdf/1911.09245 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
3D human pose estimation is frequently seen as the task
of estimating 3D poses relative to the root body joint. Alter-
natively, we propose a 3D human pose estimation method
in camera coordinates, which allows effective combination
of 2D annotated data and 3D poses and a straightforward
multi-view generalization. To that end, we cast the problem
as a view frustum space pose estimation, where absolute
depth prediction and joint relative depth estimations are
disentangled. Final 3D predictions are obtained in cam-
era coordinates by the inverse camera projection. Based on
this, we also present a consensus-based optimization algo-
rithm for multi-view predictions from uncalibrated images,
which requires a single monocular training procedure. Al-
though our method is indirectly tied to the training camera
intrinsics, it still converges for cameras with different in-
trinsic parameters, resulting in coherent estimations up to
a scale factor. Our method improves the state of the art on
well known 3D human pose datasets, reducing the predic-
tion error by 32% in the most common benchmark. We also
reported our results in absolute pose position error, achiev-
ing 80 mm for monocular estimations and 51 mm for multi-
view, on average. Source code is available at https://
github.com/dluvizon/3d-pose-consensus.

## introduction
3D human pose estimation is a very active research topic,
mainly due to the several applications that beneﬁt from pre-
cise human poses, such as sports performance analysis, 3D
model ﬁtting, human behavior understanding, among many
others. Despite the recent works on 3D human pose es-
timation, most of the methods in the literature are limited
to the problem of relative pose prediction [8, 49, 61, 4, 2],
where the root body joint is centered at the origin and the re-
maining joints are estimated relative to the center. This lim-
itation hinders the generalization for multi-view scenarios
since predictions are not in the camera coordinates. Con-
trarily, when estimations are relative to a static referential,
predictions can be easily projected from one view to an-
other, as illustrated in Fig. 1.
Figure 1. Absolute 3D human pose estimated from a single image
(top-left) with occlusion and projected into a different view (top-
right). Our multi-view consensus-based approach (bottom) results
in a more precise absolute pose estimation and effectively handles
cases of occlusion.
The methods in the state of the art frequently handle
3D human pose estimation as a regression task, directly
converting the input images to predicted poses in millime-
ters [47, 24]. However, this is a depth learning problem,
because identical distances in pixels can result in different
distances in millimeters. For example, a person close to
the camera with the hand next to the head has a distance
(head to hand in mm) much shorter than a person far from
the camera with her arm extended, although both result in
the same distance in pixels. Consequently, those methods
1
arXiv:1911.09245v3  [cs.CV]  20 Aug 2021

have to learn the intrinsic parameters indirectly. Moreover,
by predicting 3D poses directly in millimeters, the abundant
images with annotated 2D poses in pixels cannot be easily
exploited, since this data has no associated 3D information,
and relative poses predicted from one camera cannot be eas-
ily projected into a different view, making it more difﬁcult
to handle occlusion cases in multi-view scenarios.
In our method, we tackle these limitations by casting the
problem of 3D human pose estimation into a different per-
spective: instead of directly predicting pose in millimeters
relative to the root joint, we predict 3D poses in the view
frustum space, i.e., we predict (u, v) coordinates in the im-
age plane, in pixels, and the absolute depth in millimeters.
We further split depth estimation as a global absolute depth
and joint relative depth estimations. Both 2D human pose
and absolute depth estimation are well known problems in
the literature [3, 11, 9, 22], including absolute depth estima-
tion benchmarks [33, 53], but are usually not correlated. In
our method, we train a feed-forward neural network by ef-
fectively merging in-the-wild 2D data and precise 3D poses,
making the best use of each. Even though our network is
trained only with monocular images, the predictions from
individual views can be merged by the proposed consensus-
based optimization in order to produce multi-view estima-
tions, resulting in an effective way to handle the challeng-
ing cases of occlusions, as demonstrated by a signiﬁcant
improvement in accuracy in our experiments. Although our
training scheme is indirectly tied to the camera intrinsics,
our method has demonstrated a generalization capability to
predict 3D poses up to a scale from a completely different
camera setup, including different intrinsic parameters. This
was evidenced by qualitative and quantitative evaluations.
Considering the exposed limitations of relative 3D hu-
man pose estimation, we aim to ﬁll the gap of current
methods by addressing the more complex problem of abso-
lute 3D human pose estimation, where predictions are per-
formed with respect to a static referential i.e., the camera
position, and not to the person’s root joint. In that direc-
tion, we present our contributions: First, we propose an
absolute 3D human pose estimation method from monoc-
ular cameras which achieves results in the state of the art
when considering similar camera intrinsics at training and
inference time. Second, we propose a consensus-based opti-
mization for multi-view absolute 3D human pose estimation
from uncalibrated images, which requires a single monocu-
lar training procedure. The multi-view estimation approach
is capable of generalizing for different camera setups, re-
sulting in coherent 3D absolute predictions up to a scale
factor. Our method sets the new state-of-the-art results on
the challenging test set from Human3.6M, improving previ-
ous results by 10% with monocular predictions and by 32%
considering multiple views.
The remaining of this paper is divided as follows. In
section 2 we present the related work. Our method for 3D
human pose estimation is explained in section 3 and our
algorithm for consensus-based optimization is detailed in
section 4. The experiments are presented in section 5 and in
section 6 we conclude this paper.

## method
Cam. calib.
Sit. D.
Smoking
Photo
Waiting
Walking
Walk.Dog
Walk.Pair
Avg
PVH-TSP [52]
GT
83.5
94.8
85.8
82.0
114.6
94.9
79.7
87.3
Trumble et al. [51]
GT
61.0
95.0
70.0
62.3
66.2
53.7
52.4
62.5
Pavlakos et al. [38]
GT
97.5
119.9
52.1
42.6
51.9
41.7
39.3
56.8
Tome et al. [50]
GT
95.2
50.2
64.3
52.2
43.9
51.1
45.3
52.8
Kadkho. et al. [18]
GT
78.8
49.8
54.8
46.2
51.1
40.5
41.0
49.1
Iskakov et al. [17]
GT
28.7
21.2
19.4
20.8
22.1
19.7
20.2
20.8
Ours
Estimated
69.5
42.0
44.6
39.6
31.0
40.2
35.3
44.7
Ours
GT
51.5
39.2
38.8
32.4
29.6
38.9
33.2
36.9
Cam1
Cam3
Cam4
Cam2
Cam2
X
−750
−500
−250
0
250
500
750
1000
Y
250
500
750
1000
1250
1500
1750
2000
Z
0
250
500
750
1000
1250
1500
1750
2000
X
−750
−500
−250
0
250
500
750
1000
Y
250
500
750
1000
1250
1500
1750
2000
Z
0
250
500
750
1000
1250
1500
1750
2000
X
−750
−500
−250
0
250
500
750
1000
Y
250
500
750
1000
1250
1500
1750
2000
Z
0
250
500
750
1000
1250
1500
1750
2000
Camera
projection
Merge
Figure 4. On top, the absolute prediction from camera 1 is pro-
jected into camera 2 with considerable errors in occluded joints.
At the bottom, predictions from cameras 1, 3, 4 are projected into
camera 2 and merged, improving the prediction signiﬁcantly.
nario, where we used our model trained on Human3.6M and
and evaluated it on KTH. Due to the high disparity in the
camera intrinsics between both datasets, the absolute depth
predictions stayed in the range observed in Human3.6M,
with an average of 4.040 meters.
The consensus-based
algorithm still converged in the multi-view scenario, even
though the ﬁnal 3D poses are shifted to a smaller size due
to the absolute depth predicted by our method (see Fig. 5).
To correct the scale and shift, we rescaled the predicted
3D poses using the torso size from KTH (the length from
the neck to the hip center) and shifted our predictions to
the KTH poses in the hip center. After this, we computed
the PCP metrics of our predictions, which results in 0.812
and .929 for lower and upper legs, and in 0.620 and 0.804
for lower and upper arms. Additional qualitative results on
KTH are shown in Fig. 6, where the ﬁnal 3D estimations
are also projected to the source images.
Finally, Fig. 4 shows an example of highly occluded
body parts where multiple camera predictions results in a
signiﬁcantly better reconstruction. Note that in this case we
are projecting the estimated absolute 3D pose to a new point
of view, not used during inference. Despite the highly oc-
cluded joints in some views, the resulting absolute pose is
9

Table 3. Results on MPI-INF-3DHP compared to the state of the art. Training data: MPI-INF-3DHP, Human3.6M, and MPII

## experiments
In this section, we present the results of our method on
two well known datasets, as well as a sequence of ablation
studies to provide insights about our approach.
5.1. Datasets
Human3.6M [16] is a large-scale dataset with 3D hu-
man poses collected by a motion capture system (MoCap)
6

and RGB images captured by 4 synchronized cameras. A
total of 15 activities are performed by 11 actors, 5 females
and 6 males, resulting in 3.6 million images. Poses are com-
posed of 23 body joints, from which 17 are used for evalu-
ation as in the previous work [37, 56].
MPI-INF-3DHP [28] is a dataset for 3D human pose esti-
mation captured with a marker-less MoCap system, which
allows outdoor video recording, e.g., TS5 and TS6 from
testing.
A total of 8 activities are performed by 8 dif-
ferent actors in two distinct sequences.
Human poses
are composed of 28 body joints, from which 17 are used
for evaluation. The activities involve complex exercising
poses, which makes this dataset more challenging than Hu-
man3.6M. However, the precision of marker-less motion
capture is visually less precise than ground truth poses
from [16].
Despite having a training set captured by 8
different cameras, test samples are captured by a single
monocular camera.
PennAction [58] is a dataset composed by 2,326 videos
in the wild with annotated 2D poses of people performing
15 different actions. This dataset does not provide 3D pose
annotations, but it is usefull to access the generability of our
method in a qualitative evaluation, since the images are very
challenging for pose estimation.
KTH Multiview Football Dataset II [19] consists of im-
ages from football players with ground truth 2D and 3D
poses centered in the root joint. Partial camera parameters
are given for projecting the 3D poses into the three different
views, however, explicit intrinsic and extrinsic parameters
are not available. This dataset is challenging since the cam-
era setup is very different from the training scenario on both
Human3.6M and MPI-INF-3DHP. Therefore, we used KTH
for zero-shot evaluation.
5.2. Evaluation protocols and metrics
Three evaluation protocols are widely used for Human-
3.6M. In protocol 1, six subjects are used for training and
only one is used for evaluation. Since this protocol uses a
Procrustes alignment between prediction and ground truth,
we do not consider it in our work. In protocol 2, ﬁve sub-
jects (S1, S5, S6, S7, S8) are dedicated for training and
S9 and S11 for evaluation, and evaluation videos are sub-
sampled every 64th frames. The third protocol is the of-
ﬁcial test set (S2, S3, S4), of which ground truth poses are
withheld by the authors and evaluation is performed over all
test frames (almost 1 million images) through a server. In
our experiments, we report our scores in the most challeng-
ing ofﬁcial test set. Additionally, we consider protocol 2
for the ablation studies and for comparison with multi-view
approaches.
The standard metric for Human3.6M is the mean per
joint position error (MPJPE), which measures the average
joint error after centering both predictions and ground truth
poses to the origin. We also evaluated our method consid-
ering the mean of the root joint position error (MRPE) [32],
which measures the average error related to the absolute
pose estimation. This metric is considered only for vali-
dation, since the server does not support this protocol.
For MPI-INF-3DHP, evaluation is performed on a test
set composed of 6 videos/subjects, of which 2 are recorded
in outdoor scenes, resulting in almost 25K frames. The au-
thors of [28] proposed three evaluation metrics: the mean
per joint position error, in millimeters, the 3D Percent-
age of Correct Keypoints (PCK), and the Area Under the
Curve (AUC) for different thresholds on PCK. The stan-
dard threshold for PCK is 150mm [28], which corresponds
nearly to half of the head size. Differently from previous
work [28, 20, 59], we use the real 3D poses to compute the
error instead of the normalized 3D poses, since the last is
not compatible with a constant camera projection. Since
evaluation is performed on monocular images, we use the
available intrinsic camera parameters to recover absolute
poses in millimeters. Finally, we also evaluated our method
on KTH considering the PCP metric from [7].
5.3. Implementation details
During training, we use the elastic net loss (L1+L2) [62]
for both absolute z and relative 3D pose predictions, respec-
tively deﬁned by:
Lz = 1
Ns
Ns
X
i=1
∥zai −ˆzai∥1 + ∥zai −ˆzai∥2
2, and
(15)
Lp = 1
Ns
Ns
X
i=1
∥pi −ˆpi∥1 + ∥pi −ˆpi∥2
2,
(16)
where zai and ˆzai are the ground truth and the estimated
absolute z values, and pi and ˆpi are the ground truth and
the estimated 3D poses. The ﬁnal loss is then represented
by L = Lz + Lp.
Once the ﬁrst part of our network is trained, we compute
the average prediction error d on training, which is used to
train the conﬁdence score network using the mean average
error (MAE). RMSprop and Adam are used for optimiza-
tion, respectively for the ﬁrst and second training processes,
starting with a learning rate of 0.001 and decreased by 0.2
after 150K and 170K iterations. Batches of 24 images are
used. The full training process takes less then two days with
a GTX 1080 Ti GPU. We augmented the training data with
common techniques, such as random rotations (±30◦), re-
scaling (from 0.7 to 1.3), horizontal ﬂipping, color gains
(from 0.9 to 1.1), and artiﬁcial occlusions with rectangu-
lar black boxes. We also added some randomness in the
cropped bounding boxes, on both position and size, in or-
der to make the model more robust against variations in
human detection. Additionally, we augmented the training
7

data in a 50/50 ratio with 2D images from MPII [3], which
becomes an standard data augmentation technique for 3D
human pose estimation.
5.4. Comparison with the state of the art
Human3.6M. In Table 1, we show our results on the test set
from Human3.6M. We provide results of our method con-
sidering monocular predictions and multi-view predictions,
for estimated and ground truth camera calibration. In all the
cases our method obtains state-of-the-art results by a fair
merging, reducing the prediction error by more than 10%
in monocular scenario. In the multi-view setup, our method
achieves 39mm error, reducing errors by more than 32% on
average. In the most challenging activity (Sitting Down),
our method performs better than all previous approaches re-
porting results in the ofﬁcial test set. These results demon-
strate the effectiveness of our method, considering that the
test set from Human3.6M is very challenging and labels are
withheld by the authors.
For a fairer comparison, we also consider results only
from multi-view approaches in Table 2. We present our
scores considering ground truth and estimated camera cal-
ibration, while all previous methods use the available cali-
bration from the dataset. Still, our method obtains 36.9mm
error, which is a strong results, specially considering that
the methods from [17, 50] require multi-view training with
a known calibration setup, while our network is trained with
monocular images. In this comparison, we are not consid-
ering methods that make use of the ground truth absolute
position of the root joint, since in our method we estimate
this information.
MPI-INF-3DHP. Our results on MPI-INF-3DHP are
shown in Table 3. We do not report results considering mul-
tiple views in this dataset, since the testing samples were
captured by a single camera. Contrarily to what is more
common in this dataset, we evaluated our method using
non-normalized 3D poses, otherwise it will not be possi-
ble to perform the inverse camera projection. Nevertheless,
our method achieves results comparable to the state of the
art, even considering other methods using normalized 3D
poses.
5.5. Qualitative results
In Fig. 3 we present some qualitative results of predicted
absolute 3D poses by our method. Not that the distance
from predictions to the images are proportional to the abso-
lute distance in z. In Fig. 7 we show monocular predictions
by our method on the MPI-INF-3DHP dataset, including
challenging outdoor scenes, which are not present in the
training set. Finally, in Fig. 8, we show the results from
our consensus-based optimization approach, from multi-
view predictions on Human3.6M. Finally, in Fig. 9, we
show some generalization results from our method trained
Figure 3. Absolute 3D pose predictions from monocular single im-
ages by our method.
on Human3.6, considering predictions on challenging im-
ages from Penn Action dataset.
5.6. Ablation studies
In this part, we present additional experiments to provide
insights about our method and our design choices.
Network architecture. We evaluated three different net-
work architectures as presented in Table 4. An off-the-shelf
ResNet performed 62.2mm and 53.7mm, respectively when
cut at blocks 4 and 5. The proposed ResNet-U improves on
ResNet block 5 by 3.2mm

## related_work
In this section, we review the methods most related to our
work, giving special attention to monocular (relative and
absolute) and multi-view 3D human pose estimation. We
recommend the survey in [44] for readers seeking for a more
detailed review.
2.1. Monocular relative 3D human pose estimation
In the last decade, monocular 3D human pose estima-
tion has been a very active research topic in the commu-
nity [1, 60, 48, 24, 15]. Many recent works have proposed to
directly predict relative 3D poses from images [47, 46, 37],
which requires the model to learn a complex projection
from 2D pixels to millimeters in three dimensions. Another
drawback is their limitation to beneﬁt from the abundant 2D
data, since manually annotated images have no associated
3D information.
A common approach to directly use 2D data during train-
ing is to ﬁrst learn a 2D pose estimator, than lift 3D poses
from 2D estimations [23, 40, 54, 49, 27, 8]. However, lift-
ing 3D from 2D points only is an ill-deﬁned problem since
no visual cues are available, frequently resulting in ambi-
guity and, consequently, limited precision. Other methods
assume that the absolute location of the root joint is pro-
vided during inference [59, 25], so the inverse projection
from pixels to millimeters can be performed. In our ap-
proach, this assumptions is not made, since we estimate the
3D pose in absolute coordinates. The only additional in-
formation we need are the intrinsic parameters for monocu-
lar prediction, which is often given by the manufacturer or
could be estimated by standard tools. In addition and differ-
ently from [25], our approach allows combining predictions
from multiple views of the scene, resulting in more precise
estimations.
Contrarily to the previous work, we are able to train our
method simultaneously with 3D and 2D annotated data in an
effective way, since one part of our prediction is performed
in the image plane and completely independent from 3D in-
formation. Moreover, estimating the ﬁrst two coordinates in
pixels in the image plane is a better deﬁned problem than es-
timating ﬂoating 3D positions directly in millimeters. These
advantages translate into higher accuracy for our method.
2.2. Monocular absolute 3D human pose estimation
Contrarily to relative estimation, in absolute pose pre-
diction the 3D coordinates of the human body are predicted
with respect to the camera or in the view frustum space. A
2

simple approach is to infer the distance to the camera con-
sidering a normalized or constant body size [61, 30], which
is an information that may not be available and difﬁcult to
be estimated [12]. Inspired by the many works on depth
estimation, Nie et al. [35] predict the depth of body joints
individually. The drawback of this method is that it suffers
to capture the human body structure, since errors in the es-
timated depth for individual joints can degenerate the ﬁnal
pose.
More recently, multi-person absolute pose estimation
methods were proposed [32, 29].
In [32], the absolute
distance from the person to the camera is predicted based
on the area of the cropped 2D bounding box. However,
it is known from the literature on absolute depth estima-
tion [10, 9] that not only the size of objects are important,
but also their positions in the image is an informative cue to
predict its depth. For example, a person in the bottom of an
image is more likely to be closer to the camera than a per-
son on the top of the same image. Differently, in [29], the
authors optimized the person absolute distance based on the
initial bone lengths, estimated from the ﬁrst 10 frames of a
video sequence, and on the re-projection of the 3D pose into
the 2D body joint locations. Besides this approach relies on
video sequences, it also requires the camera parameters.
In our approach, we combine three different information
to predict the distance of the root joint w.r.t. the camera po-
sition: the size of the bounding box (including its ratio), the
target position in the image, and deep convolutional features
that provide additional visual cues.
2.3. Multi-view pose estimation and camera cali-
bration
For the challenging cases of occlusion or clutter back-
ground, multiple views can be decisive to disambiguate un-
certain positions of body joints (see Fig. 1). To handle this,
several works have proposed multi-view solutions for 3D
human pose estimation [4, 2, 6, 5, 14], mostly exploring
the classical concept of pictorial structures from multi-view
images. Deep neural networks have been used to estimate
relative 3D poses from a set of 2D predictions from differ-
ent views [42, 38, 36]. As an example, Pavlakos et al. [38]
proposed to collect 3D poses from 2D multi-view image,
which are used to learn a second model to perform 3D es-
timations. Since these methods estimate 3D from multiple
2D images, they often require both intrinsic and extrinsic
parameters.
In order to estimate the full camera calibration parame-
ters, Micusik and Pajdla [31] proposed to use a human body
seen at different positions in the image. The main limita-
tion of this approach is the fact that it assumes that all poses
as nearly vertical and parallel to each other. Considering
multiple views of the same person, Rhodin et al. [42] pro-
pose to estimate 3D poses from each individual view and
to estimate the extrinsic camera calibration, assuming that
the intrinsic parameters are provided as input. More re-
cently, Iskakov et al. [17] proposed a learnable triangulation
of 3D poses, considering multiple fully calibrated views
during training. Despite the impressive results achieved in
this work, the network model is training considering a pre-
deﬁned camera positioning, which could result in a strong
overﬁting in the experimental setup. Differently, our model
is trained without priors about the camera positions and the
proposed multi-view optimization algorithm is not directly
tied to a speciﬁc camera setup.
From the recent literature, we can notice that current
multi-view approaches are still completely dependent on the
camera intrinsic parameters and often require a complete
calibration setup, which can be prohibitive in some circum-
stances. Available methods are also limited to the inference
of 3D from multiple 2D predictions, requiring multi-view
datasets for training. Alternatively, we propose to predict
absolute 3D poses from each individual view, which has
two important advantages over previous methods. First, it
allows us to easily combine predictions from multiple cali-
brated cameras, while requiring a single monocular training
procedure. Second, we are able to estimate camera calibra-
tion, both intrinsic and extrinsic, from multi-view images,
by a consensus-based optimization without retraining the
model. The strength of our approach is evidenced by its
strong results, even when considering unknown and uncali-
brated cameras.
3. Proposed method for 3D human pose esti-
mation
One of the goals of our method is to predict 3D human
poses in absolute coordinates with respect to the camera po-
sition. For this, we believe that the most effective approach
is to predict each body joint in image pixel coordinates and
in absolute depth, orthogonal to the image plane, in mil-
limeters. Then, the predicted pixel coordinates and depth
can be projected to the world coordinates, considering a pin-
hole camera model.
We further split the problem as relative 3D pose esti-
mation and absolute depth estimation. The motivation for
this comes from the idea that a well cropped bounding box
around the person is better for predicting its pose than the
full image frame, since a better resolution could be attained
and the person scale is automatically handled by the im-
age crop, although small variations in the bounding boxes
during training result in better robustness. Additionally, by
providing a separated loss on relative depth for each joint
helps the network to learn the human body structure, which
would be more difﬁcult to learn directly from absolute co-
ordinates due to position shift.
Recent works on depth estimation have demonstrated
that neural networks rely on both pictorial cues and geome-
3

try information to predict depth [10]. For the speciﬁc prob-
lem of 3D human pose estimation, the structure of the hu-
man body is also an important domain knowledge to be ex-
plored. Considering our motivations and the exposed chal-
lenges, we propose to predict 3D poses relative to a cropped
region centered at the person, which eases the network to
encode the human body structure, and absolute depth from
combined local pictorial cues and global position and size
of the cropped region.
Legend:
Image input
(W×H×3)
+
+
U-block (2×)
ResNet
block.4f
 Bounding box    
(4×1)
Global
Avg. Pooling
3D pose
regression
MaxPooling
UpScaling
Residual block
Transp. Conv.
Supervision
Fully-connected
×
+
Invese
Camera
Projection
Figure 2. Proposed ResNet-U architecture. A given input image
and the corresponding bounding box

## conclusion
In this paper, we have proposed a new method for the
problem of predicting 3D human poses in absolute coordi-
nates and a new algorithm for multi-view predictions opti-
mization. We show that, by casting the problem into a new
perspective, we can beneﬁt from training with 2D and 3D
data indistinguishably, while performing 3D predictions in
a more effective way. These improvements boost monoc-
ular 3D pose estimation signiﬁcantly. As another conse-
quence of the absolute prediction, we show that multi-view
10

Figure 7. 3D pose predictions from monocular single images on MPI-INF-3DHP dataset, including indoor and outdoor scenes.
Figure 8. 3D pose predictions from our consensus-based optimization algorithm, considering multi-view on Human3.6M. Final 3D poses
are projected into the different views (a,b,c,d) and shown in perspective (e,f).
Figure 9.
Generalization of our method for 3D pose estimation on unseen dataset (PennAction), including outdoor scenes in different
contexts.
estimations can be easily performed from multiple absolute
monocular estimations, resulting in much higher precision
than previous methods in the literature, even when consid-
ering multiple uncalibrated images.
11