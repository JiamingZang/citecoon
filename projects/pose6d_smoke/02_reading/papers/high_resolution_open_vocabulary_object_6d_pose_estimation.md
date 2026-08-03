# High-resolution open-vocabulary object 6D pose estimation

> 2024 · id: W4400023965 · arXiv: 2406.16384 · pdf: https://arxiv.org/pdf/2406.16384 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
T
HE estimation of the 6D pose of an object requires the
prediction of its 3D rotation matrix and 3D translation
vector with reference to the camera. Accurate object 6D
pose estimation is a fundamental phase in a wide range of
applications, such as augmented reality [1], robot grasping [2],
and autonomous driving [3]. Data-driven methods [4], [5] can
achieve reliable pose estimation, but require expensive object
annotations. This problem is mitigated by the development
of techniques to generate large-scale synthetic datasets [6],
[7]. Another line of research instead defines the unseen-object
setting [8], which assumes no overlap between the set of
training and test objects [7], [9]. However, most methods for
the unseen-object setting are model-based, as they require
a CAD model of the object at test time [5], [10]–[12], as
detailed in the first group of Tab. I. Recent works changed
this assumption, and instead require a video sequence of the
object at test time [9], [13], from which a set of reference
views is extracted (second group in Tab. I). These methods use
(Corresponding author: Jaime Corsetti)
Jaime Corsetti is with Fondazione Bruno Kessler (FBK), 38123 Trento,
Italy, and also with the Department of Information Engineering and Computer
Science, University of Trento, 38122 Trento, Italy (e-mail: jcorsetti@fbk.eu).
Davide Boscaini, Francesco Giuliari, and Fabio Poiesi are with Fondazione
Bruno Kessler (FBK), 38123 Trento, Italy (e-mail: dboscaini@fbk.eu; fgiu-
liari@fbk.eu; poiesi@fbk.eu).
Changjae Oh is with the School of Electronic Engineering and Computer
Science, Queen Mary University of London, E1 4NS London, United King-
dom (e-mail: c.oh@qmul.ac.uk).
Andrea Cavallaro is with the Idiap Research Institute, CH-1920 Martigny,
Switzerland, and also with the ´Ecole polytechnique f´ed´erale de Lausanne
(EPFL), CH-1015 Lausanne, Switzerland (e-mail: a.cavallaro@idiap.ch).
Structure-from-Motion (SfM) [14] to reconstruct the object 3D
model from the reference views [9], [13].
To eliminate the need for the object models or multiple
views, in our previous work, Oryon [15], we showed that the
object can be described with a textual prompt, thus defining
the open-vocabulary object 6D pose estimation setting. This
setting assumes the availability of two RGBD scenes, along
with a natural language description of the object of interest,
provided by the user. The object 6D pose in a scene is
estimated with respect to the object in the other scene.
In this work, we present Horyon (High-resolution Oryon),
a VLM-based architecture for open-vocabulary object 6D pose
estimation that generalises to unseen objects. Horyon uses a
VLM to extract visual and textual representations from the
input scene pair and the natural language description (i.e.,
the prompt). The two representations are fused using cross-
attention layers that enable information exchange between
each scene patch with each prompt token, without losing
key details contained in the object description. The resulting
feature maps are upsampled and fused with the high-resolution
feature maps from a visual encoder. Horyon estimates the
object pose in the query scene with respect to the anchor
scene by segmentation and pixel-level view matching, and
subsequently backprojecting and registering the 3D matches.
This work significantly extends our previous work [15] in
several aspects. Instead of processing the whole scene, we
identify and localise the object of interest in both scenes with
GroundingDino [16] to obtain a bounding box. This leads to
a higher resolution feature map without losing generalisation
to unseen objects. Moreover, we remove the need for the
guidance backbone and instead extract the guidance features
directly from the VLM, reducing the training time and the
number of parameters to 50% and 37% of the original method,
respectively. To validate our choices, we test our method on
an extended version of the Oryon benchmark, which includes
two popular datasets for pose estimation, i.e., Linemod [17]
and YCB-Video [18], in addition to REAL275 [19] and
Toyota-Light [20]. Linemod features scenes with high clutter,
occlusions and objects that are small and unusual. YCB-
Video presents a large variety of poses, and objects that
often belong to similar categories (e.g., boxes and cans). We
provide extended ablation studies to justify each choice, and
also evaluate the effect of the prompt on the final results. In
summary, our main contributions are:
• We provide a comprehensive analysis of recent ap-
proaches for estimating the 6D pose of unseen objects,
detailing their requirements and operational conditions.
arXiv:2406.16384v2  [cs.CV]  11 Jul 2024

2
TABLE I
COMPARISON OF THE DATA REQUIREMENTS OF HORYON WITH EXAMPLES OF STATE-OF-THE-ART METHODS FOR UNSEEN-OBJECT 6D POSE
ESTIMATION. WE CLASSIFY THE METHODS BASED ON: INPUT: THE TYPE OF INPUT DATA, TYPICALLY RGB OR RGBD; REFERENCE: ADDITIONAL
DATA USED TO IDENTIFY THE UNSEEN OBJECT AT TEST TIME; POSE: WHETHER THE METHOD IS CAPABLE OF ESTIMATING THE 6D POSE OR IS LIMITED
TO THE ROTATION COMPONENT. METHODS THAT CAN ESTIMATE THE TRANSLATION COMPONENT UP TO A SCALE ARE IDENTIFIED BY R,˜t.; OBJECT
PREPROCESSING: EVENTUAL PROCESS REQUIRED AT TEST TIME TO ACQUIRE INFORMATION ABOUT THE OBJECT OF INTEREST; LOCALISATION:
EVENTUAL EXTERNAL MODULES USED TO LOCALISE THE OBJECT, TYPICALLY A SEGMENTOR OR A DETECTOR; ZERO-SHOT: TRUE IF THE
LOCALISATION MODULE WAS NOT SPECIFICALLY TRAINED FOR THE TEST DATASET.

## method
Crop
Mask
REAL275
Toyota-Light
Linemod
YCB-Video
Average
AR
ADD
mIoU
AR
ADD mIoU
AR
ADD
mIoU
AR
ADD
mIoU
AR
ADD
mIoU
Sparse-view
1 PoseDiffusion [26]
Oracle
-
9.2
0.8
-
7.8
1.2
-
10.8
1.4
-
7.5
0.8
-
8.8
1.1
-
2
GDino
-
9.5
0.8
-
8.1
1.6
-
7.7
1.0
-
8.5
1.1
-
8.5
1.1
-
3 RelPose++ [25]
Oracle
-
22.8
11.9
-
30.9
11.6
-
14.6
9.4
-
15.1
10.8
-
20.8
10.9
-
4
GDino
-
23.1
12.8
-
30.5
11.6
-
15.1
10.8
-
15.5
10.6
-
21.1
11.5
-
Crop-free
5 ObjectMatch [54]
-
Oracle
9.7
4.5
100.0
3.5
1.6
100.0
14.2
13.8
100.0
3.7
2.8
100.0
7.8
5.7
100.0
6
-
Oryon
9.2
4.2
66.5
2.9
1.1
68.1
7.5
6.8
30.1
1.6
1.3
39.2
5.3
3.4
51.0
7 SIFT [55]
-
Oracle
34.1
16.4
100.0
30.3
14.1
100.0
17.6
10.0
100.0
18.5
12.8
100.0
25.1
13.3
100.0
8
-
Oryon
24.4
12.8
66.5
27.2
9.9
68.1
6.4
2.6
30.1
4.7
2.1
39.2
15.7
6.9
51.0
9 LatentFusion [27]
-
Oracle
22.9
9.8
100.0
28.2
10.2
100.0
14.5
7.1
100.0
18.0
10.2
100.0
20.9
9.3
100.0
10
-
Oryon
13.7
3.8
66.5
23.1
5.0
68.1
4.8
1.0
30.1
6.7
3.5
39.2
12.1
3.3
51.0
11 Oryon [15]
-
Oracle
46.5
34.9
100.0
34.1
22.9
100.0
25.3
20.4
100.0
19.4
12.8
100.0
31.3
22.7
100.0
12
-
Oryon
32.2
24.3
66.5
30.3
20.9
68.1
12.2
10.2
30.1
8.6
1.6
39.2
20.8
14.3
51.0
Crop-based
13
ObjectMatch† [54]
Oracle
Oracle
20.5
11.1
100.0
8.0
4.1
100.0
12.2
11.1
100.0
6.0
3.7
100.0
11.7
7.5
100.0
14
GDino CATSeg
21.5
11.6
84.5
7.3
4.0
81.6
13.8
11.7
54.4
6.0
3.4
71.7
12.2
7.7
73.0
15
GDino
Ours
21.0
11.0
81.3
8.2
4.3
82.1
13.6
11.8
67.6
5.9
3.4
52.0
12.2
7.6
70.7
16
SIFT† [55]
Oracle
Oracle
38.8
21.6
100.0
32.4
16.5
100.0
18.7
10.8
100.0
19.3
13.9
100.0
27.3
15.7
100.0
17
GDino CATSeg
32.9
15.2
84.5
29.5
14.8
81.6
9.4
5.0
54.4
11.3
6.1
71.7
20.8
10.3
73.0
18
GDino
Ours
33.5
18.1
81.3
29.9
14.6
82.1
14.6
8.5
67.6
10.1
6.5
52.0
22.0
11.9
70.7
19
LatentFusion† [27]
Oracle
Oracle
22.6
9.6
100.0
28.2
10.2
100.0
14.5
6.4
100.0
17.5
10.2
100.0
20.7
9.1
100.0
20
GDino CATSeg
18.6
6.0
84.5
26.6
9.8
81.6
9.3
4.8
54.4
9.3
5.1
71.7
15.9
6.4
73.0
21
GDino
Ours
19.8
8.2
81.3
26.0
10.3
82.1
10.3
3.9
67.6
11.0
8.9
52.0
16.8
7.8
70.7
22
Horyon
Oracle
Oracle
57.7
49.8
100.0
37.4
28.4
100.0
34.4
27.6
100.0
28.6
22.6
100.0
39.5
32.1
100.0
23
GDino CATSeg
47.4
39.6
84.5
32.7
23.5
81.6
19.6
17.4
54.4
17.9
9.3
71.7
29.4
22.5
73.0
24
GDino
Ours
57.9
51.6
81.3
33.0
25.1
82.1
22.0
20.4
67.6
20.6
12.3
52.0
33.4
27.3
70.7
25 ∆score
+25.5 +27.3 +14.8 +3.0
+4.2
+14.0 +9.8 +10.2 +37.5 +12.0 +10.7 +12.8 +12.6 +13.0 +19.7
masks and detection boxes (i.e., row 24 vs row 12). While
all the changes are positive, the behaviour is different if each
dataset is considered separately. REAL275 shows the largest
improvements in pose estimation, with an AR increment
of 25.5. The increment in ADD(S) is even higher (+27.3),
showing that Horyon can produce a much higher ratio of very
accurate poses. On the other hand, TOYL exhibits the smaller
improvement, as AR and ADD(S) only improve by 3.0 and
4.2 respectively. This dataset is less affected by the usage of
a detector, as it presents a single object for each scene. It is
also the dataset with the largest change in light and point of
view across scenes: this suggests that architectural changes
can be made to address this specific setting and provide a
larger improvement. LM and YCBV both exhibit consistent
improvements in AR (9.8 and 12.0 respectively), albeit their
performance is still low compared the REAL275 and TOYL.
We attribute this fact to the clear higher difficult of these
datasets compared to the original ones used in Oryon, which
is due to high clutter, object occlusion and object similarity.
F. Qualitative results
In Fig. 3 we report some qualitative result on the pose
predicted by Horyon, compared to the ones predicted by
Oryon [15], SIFT† [55] and the ground-truth pose. Oryon’s
results are reported using its predicted mask, while results for
SIFT† and Horyon are reported using Horyon’s predicted mask
and detections from GroundingDino.
Fig. 3(a) show results on REAL275 [19]. In this example
Horyon obtains fairly accurate poses, albeit the black camera
presents a small rotation error, which causes it to be not
completely aligned to the ground truth. SIFT† and Oryon
obtain correct localisations, but they present clearly visible
rotation and translation errors.
On Figs. 3(b), we can observe that Horyon is quite accu-
rate on TOYL. SIFT† similarly obtains good performances,
with only small errors in translation. Instead, Oryon fails
completely at localising the object, which results in a large
translation error. We observe that the high variation in light-
ning conditions makes particularly difficult to obtain correct
matches, especially when the images are represented with a
low-resolution feature map as in Oryon.
Figs. 3(c) show results on LM. Horyon retrieves a pose
with a small rotation error, which is larger in the prediction
of SIFT†. In this case Oryon fails, likely due to a wrong
localisation of the blue iron. This example is quite difficult
due to the high variation in viewpoint between the two scenes.
Figs. 3(d) present results on YCBV. we can observe that
Oryon and SIFT† both fail in estimating the pose of the
water jug: the first presents a large translation error, while
the second shows a wrong rotation. In this case, Horyon
predicts a pose which is mostly aligned, but one of the rotation
components is wrong due to the partial symmetry of the object.
This suggest that, while our method benefits from the high-
resolution feature map, it still has difficulties in performing

9
Anchor
Ground truth
Oryon [15]
SIFT† [55]
Ours
(a) Prompt: Black lens camera
(b) Prompt: Green plastic bottle
(c) Prompt: Light blue iron
(d) Prompt: Blue plastic water jug
Fig. 3.
Sample pose results from REAL275 [19] (a), Toyota-Light [20] (b), Linemod [17] (c) and YCB-Video [18] (d). All the results use crop from
GroundingDino [16] and segmentation mask predicted by Horyon. We show the object model coloured by mapping its 3D coordinates to the RGB space.
Query images are darkened to highlight the object poses.
TABLE III
WE SHOW THE RESULTS OBTAINED BY ABLATION THE ARCHITECTURAL COMPONENTS OF ORYON, WITH THE BASELINE AT ROW 8 TO BETTER COMPARE
WITH THE OTHER RESULTS. ALL THE RESULTS ARE COMPUTED WITH GROUNDINGDINO AS DETECTOR PRIOR AND OUR PREDICTED MASK AS
SEGMENTATION PRIOR. ALL METRICS ARE THE HIGHER THE BETTER.
Components
REAL275
Toyota-Light
Linemod
YCB-Video
Average
Crop Loss Hyp. Our ϕD
Fusion Guidance
VLM
AR ADD mIoU
AR ADD mIoU
AR ADD mIoU
AR ADD mIoU
AR ADD mIoU
1
✓
Ours
DINO
GDino
35.8
22.2
61.2 35.4
27.3
83.9
5.7
2.9
15.9
6.7
4.5
32.6 20.9
14.2
48.4
2
✓
✓
Ours
DINO
GDino
43.5
30.8
80.8 28.6
15.9
82.3 15.6
14.2
51.4 13.6
8.8
67.6 25.3
17.4
70.5
3
✓
✓
Ours
DINO
GDino
26.2
12.6
64.8 30.4
19.6
82.2
6.6
4.7
19.4
8.0
2.7
36.5 17.8
9.9
50.7
4
✓
✓
✓
-
DINO
GDino
51.2
39.0
74.8 31.2
19.2
77.9 19.9
18.2
48.0 13.7
5.7
55.6 29.0
20.5
64.1
5
✓
✓
✓
Oryon
DINO
GDino
49.2
41.8
79.2 32.9
24.4
80.9 21.5
19.8
51.3 18.9
9.4
60.4 30.6
23.8
67.9
6
✓
✓
Ours
DINO
GDino
50.5
41.2
81.8 32.7
24.4
82.5 21.9
20.1
53.0 20.4
13.2
67.6 31.4
24.7
71.6
7
✓
✓
✓
Ours
-
GDino
48.2
32.6
80.3 32.3
23.1
82.3 21.5
20.1
52.2 20.6
12.8
67.6 30.6
22.2
70.6
8
✓
✓
✓
Ours
DINO
GDino
57.9
51.6
81.3 33.0
25.1
82.1 22.0
20.4
67.6 20.6
12.3
52.0 33.4
27.3
70.7
9
✓
✓
✓
Ours
Swin-B
GDino
49.1
46.2
81.3 33.2
24.2
82.3 21.9
20.3
51.9 21.0
13.8
68.5 31.3
26.1
71.0
10
✓
✓
✓
Ours
Swin-B
CLIP
37.3
19.7
72.0 26.9
13.4
69.1 16.7
14.2
53.8 11.4
4.9
54.5 23.1
13.1
62.3
11
✓
✓
✓
Ours
Swin-B
ALIGN 40.0
21.6
77.8 26.5
12.8
71.1 16.2
13.3
53.7 11.1
4.7
55.6 23.4
13.1
64.6
matches in small object regions, such as the handle in the
water jug.
G. Ablation study
We report in Tab. III an ablation study on the components
of Horyon, with the baseline at row 8.
What is the effect of cropping? The effect of the crop
is strictly related to the change in loss hyperparameters, as
using the crop raises the number of matches due to the
higher resolution. Therefore, in rows 1 to 3 we examine how
the addition of the crop and the change in hyperparameters
influence each other. In row 1 we do not use any crop and
keep the original loss hyperparameters, resulting in average
AR similar to Oryon (20.9 vs 20.8) and much worse than our
baseline of 33.4 AR. Only using the crop (row 2) improves
the performance, but still results in a significant gap with
respect to our baseline (25.3 vs 33.4 AR). In row 3 we observe
that only updating the loss hyperparameters leads to a worse
performance than Oryon (17.8 vs 20.8 in AR), thus motivating
our choice. The most significant change in the loss is in the
dimension of the excluding kernel of the negative loss, which
restricts the pool of negative candidates. Without using the

10
TABLE IV
WE REPORT THE RESULTS OBTAINED BY CHANGING THE PROMPTS AT TEST TIME WITH AN ALTERNATIVE VERSION, WHICH CAN HAVE AN INCORRECT
DESCRIPTION (MISLEADING), SHOW ONLY THE OBJECT NAME (GENERIC), OR SHOW ONLY THE OBJECT DESCRIPTION WITHOUT THE NAME (NO NAME).
CROP PRIORS CAN BE ORACLE OR PREDICTED BY GROUNDINGDINO [16] SEGMEN

## experiments
A. Experimental setup
During training, the weights of the image and text encoders
ϕV , ϕT are frozen, we only update the weights of the fusion
and decoder modules. We train our model using the Adam
optimiser [46] for 20 epochs with learning rate 10−4, weight
decay 5·10−4, and a cosine annealing scheduler [47] to lower
the learning rate to 10−5. We randomly augment training data
by applying horizontal flipping, vertical flipping, and colour
jittering. The output resolution of FA, FQ is 192 × 192, and
their feature dimension is F = 32. Loss weights are set as
λP = λN = 0.5. We set the positive and negative margins in
the loss as µP = 0.2 and µN = 0.9, and the excluding distance
for hardest negatives as τ = 20. At test time we set the
maximum feature distance to identify a match as µT = 0.25.
We limit the number of matches to C = 2000. We implement
Horyon with PyTorch Lightning [48]. We set the batch size
to 8 and train on four Nvidia V100 GPUs. In this standard
setting, a training requires about 12 hours.
B. Datasets
We train on ShapeNet6D [6], the same synthetic dataset
used by Oryon [15]. For evaluation, we introduce a novel

6
benchmark that extends the one proposed in Oryon. This com-
prises four real-world datasets: REAL275 [19] and Toyota-
Light [20], which are also part of the original benchmark,
as well as two additional datasets, Linemod [17] and YCB-
Video [18], featuring more severe occlusions, clutter and
object variety. To comply with our relative pose estimation
setting, we form image pairs (A, Q) by randomly sampling
A, Q from their image distribution while ensuring that they
are captured from different scenes. We extract 2K image
pairs from each test dataset. To accommodate our open-
vocabulary setting, we provide natural language descriptions
for each object across the four test datasets. In the following
paragraphs, we detail the training and testing datasets.
ShapeNet6D (SN6D for short) [6] is a large-scale synthetic
dataset comprising a diverse collection of scenes generated by
rendering ShapeNet [49] objects in various poses against ran-
dom backgrounds. SN6D adhere to our zero-shot assumption,
as it does not contain any object instances present in the test
datasets. We provide natural language descriptions for each
object of SN6D by leveraging ShapeNetSem [50], a subset
of ShapeNet that includes additional metadata associated with
the 3D models, to extract both the object name and a set of
synonyms. To generate an object description, we randomly
select either the object name or one of the provided synonyms
(e.g., possible synonyms for “television” are “tv”, “telly”, and
“television receiver”). This augmentation strategy increases
the object description variability and reduces overfitting. By
following this procedure, we collect a set of 20K data tuples
(A, Q, T) to form our training set. Note that T contains only
semantic information and does not include additional details
such as the object colour, material, or physical attributes.
REAL275 [19] features 18 objects spanning six different
categories, arranged in realistic configurations within diverse
indoor scenarios (e.g., on tables, floors). Its challenges are the
presence of multiple instances of the same object categories
and a wide variety of viewpoints captured in the scenes.
Toyota-Light (TOYL for short) [20] contains scenes where
one of 21 different objects is randomly positioned on various
fabric types. The images are captured under challenging light-
ing conditions, which is particularly relevant in our setting.
As we process pairs of images, significant lighting differences
between them pose a major challenge for image matching.
Linemod [17] (LM for short) comprises 15 different objects
arranged on a table in various configurations, along with other
objects acting as clutter. It is challenging because the objects
are often small, poorly textured with mostly uniform colours,
and less conventional compared to those in REAL275 (e.g.,
plastic toy figures of an ape and a cat).
YCB-Video [18] (YCBV for short) includes 22 household
objects, mainly boxes or cans, arranged in 12 different scenes
with distractor objects in the background. Many objects share
similar geometries, making photometric information crucial
for accurate identification and pose estimation. It also contains
occlusions, including objects stacked atop one another.
C. Evaluation metrics
We evaluate the poses using Average Recall [8] (AR) and
ADD(S)-0.1d (abbreviated as ADD) [51]. AR measures pose
error using three metrics: VSD (Visible Surface Discrepancy),
MSSD (Maximum Symmetry-aware Surface Distance), and
MSPD (Maximum Symmetry-Aware Projection Distance). For
each metric, AR computes the average recall over a set of
thresholds. For VSD and MSSD, the thresholds range from
5% to 50% of the object’s diameter, while for MSPD, the
thresholds range from 5% to 50% of the image size. The
final AR score is the average of these individual recalls. ADD
measures pose error as the average of the pairwise distances
between the 3D model points transformed according to the
ground-truth and predicted poses. It then computes the recall
of pose errors that are smaller than 10% of the object’s
diameter [51]. Both AR and ADD are designed to be robust to
object symmetries. Compared to AR, ADD’s more restrictive
threshold makes it more effective in measuring highly accurate
poses. As the quality of the masks influences the matches,
we also evaluate the segmentation by mean Intersection-over-
Union (mIoU) [52], [53] across the image pairs (A, Q).
D. Comparison procedure
We consider three groups of methods for comparison. As
crop-free methods, we report the results of ObjectMatch [54],
LatentFusion [27] and a pipeline built from SIFT [55]
and PointDSC [45]. Additionally, in this group we report
Oryon’s [15] results. As crop-based methods, we report Ho-
ryon along with the results from ObjectMatch, LatentFusion,
and SIFT+PointDSC obtained by using a detector to crop
the objects. We refer as these versions of the baselines as
ObjectMatch†, LatentFusion† and SIFT† respectively. Further-
more, as sparse-view methods we report results with PoseDif-
fusion [26] and RelPose++ [25]. These are RGB-only methods
that require an object detector, but no segmentation mask. To
ensure a fair comparison, our main focus is on crop-based
methods that use RGBD data (i.e., the crop-based group), as
presented in rows 13-21 of Tab. II.
Oryon [15] serves as our primary baseline for comparison,
being the only existing method for open-vocabulary object 6D
pose estimation. In contrast to approaches that use detectors
to crop the input images around the object of interest, Oryon
processes entire images without requiring any cropping. We
evaluate Oryon’s performance using the official checkpoints
made available through its repository.
ObjectMatch [54] was proposed to tackle point cloud regis-
tration of scenes with low overlap. ObjectMatch is based on
SuperGlue [56] to estimate the matches, and on a custom pose
estimator. To compare with it, the mask is used to filter the
keypoints obtained by the first step (i.e., keypoint estimation
with SuperGlue), and the remaining matches are forwarded
to the pose estimator model. When evaluating with the crop
(ObjectMatch†), we first crop the object according to the
predicted box, and then follow the same procedure as above.
We use ObjectMatch’s model trained on ScanNet [57] from
the official repository. In Oryon [15], results with ObjectMatch
were reported by using the mask to crop the image, and
subsequently run the method on the resulting crop.
SIFT [55] is used to extract keypoints and descriptors from
each image pair, from which matches are computed through

7
descriptor similarity. We use the mask to filter the matches, and
subsequently backproject them to the 3D space. The final pose
is obtained by registering the points with PointDSC. When
evaluating with the crop (SIFT†), we first crop the object of
interest according to the predicted bounding box, and then
follow the same procedure as above.
LatentFusion [27] is a method for RGBD-based pose es-
timation of unseen objects. Given a set of RGBD support
views with associated masks and poses, LatentFusion first
builds a latent representation of the object, by aggregating the
features of each view. Subsequently, given a query view of
the same object, the optimal pose is obtained by optimising a
differentiable renderer with the latent object representation as
input. To compare with LatentFusion we use A to build the
representation, and Q as query view. Although LatentFusion
has been designed to use 8-16 references, the ablation studies
shows that it retains a good performance even with a single ref-
erence view, as in our setting. With the crop (LatentFusion†),
we fed the cropped images of A and Q.
For sparse-view methods (rows 1-4) we report results with
ground-truth detections and the ones obtained from Ground-
ingDino [16] (GDino). Crop-free m

## related_work
In Tab. I we report examples of methods for unseen-object
6D pose estimation, along with their assumptions. We classify
them in model-based (first group), model-free (second group),
relative pose estimation methods (third group) and open-
vocabulary methods (last group). In the following paragraphs,
we describe the assumptions and discuss some example meth-
ods for each of the last three groups, as they have the most
similar assumptions to Horyon. We also discuss the usage of
localisation modules for pose estimation, to contextualise our
choice of an open-vocabulary detector to crop the object.
Model-free pose estimation. Recent research has focused
on developing methods for pose estimation of unseen objects
without requiring their 3D models during inference, as these
3D models might not always be available. Model-free methods
only require a video sequence captured from multiple view-
points around the object of interest. The video sequence is
then used to reconstruct an approximate 3D model of the
object through structure-from-motion (SfM) techniques [14].
Exemplary methods include OnePose++ [9] and Gen6D [13].
OnePose++ employs a graph neural network based on at-
tention [28] to establish correspondences between the input
image and the 3D reconstruction of the unseen object. Gen6D
leverages three distinct modules for localisation, viewpoint
estimation, and pose refinement. Both methods require a video
sequence of the unseen object during inference, assuming the
physical availability of the object. Additionally, a preprocess-
ing procedure is needed to perform the 3D reconstruction
of the object, typically involving the execution of an SfM-
based algorithm on the video sequence. This procedure can
be cumbersome and challenging for users without technical
expertise. A related approach, FS6D [6], instead relies on a
sparse set of reference images annotated with their respective
camera poses. In contrast, Horyon does not require annotated
images or complex preprocessing procedures; the user only
needs to provide a natural language description of the object.
Relative pose estimation approaches estimate the relative
pose of an object with respect to one or more reference
views [24]–[26]. They are independent on the object’s 3D
model. NOPE [23] estimates the rotation of an object in a
scene given a single reference view, but the usage of RGB
information only does not allow it to estimate the translation
component of the pose. Similarly, RelPose [24] estimates the
relative rotation of an object using as few as three reference
views, but it requires that these views come from the same
scene. RelPose++ [25] enhances its predecessor by introduc-
ing a reference frame, enabling the estimation of both the
translation and rotation components of the pose using as few
as two RGB views of the same scene. However, RelPose++
only estimates the pose up to a scale factor; the translation
component retrieved is scaled under the assumption that the
first camera has q unitary distance from the object. Moreover,
RelPose++ requires the images to be approximately centred
on the object of interest and does not handle the presence of
multiple objects in the same scene. Similarly to RelPose++,
PoseDiffusion [26] addresses camera pose estimation up to

3
Fig. 1.
The main modules of our proposed method, Horyon. (a) Overview of a processing branch: Horyon first crops the object of interest from the scene
given a textual prompt, and subsequently extracts visual and textual features with DINO and BERT, respectively from the image crop and the prompt. The
fusion module ϕT V outputs a multimodal representation of the scene, which is upsampled by a decoder ϕD. At this stage, skip connections from the image
encoder ϕV are used to enrich the final representation. The output features F are used to obtain the object segmentation mask M. (b) Optimisation procedure.
FA, FQ are optimised by a hardest contrastive loss which uses ground-truth matches as supervision, while the segmentation is supervised by a Dice loss.
(c) Test procedure. The predicted masks are used to filter FA, FQ, and matches are obtained by nearest neighbor. Finally, the matches are backprojected in
3D, and a registration algorithm is used to retrieve the final pose TA→Q.
a scale factor in the translation component. It allows for the
estimation of camera poses using 2-8 views of the object by
enforcing consistency between pairwise matches through ge-
ometric constraints. However, like RelPose++, PoseDiffusion
also requires object-centric images and is therefore limited to
working with a single object per image. In contrast, Horyon
can estimate an absolute value for the translation component
of the pose and does not assume the views are captured from
the same scene.
Language models for pose estimation. So far, the use
of VLMs for pose estimation tasks has been limited. An
early approach [29] adopted a grounding network to extend
the capabilities of a grasping robot in a category-level pose
estimation scenario. Subsequent works similarly used textual
prompts for instance-level [30] or category-level [31] pose
estimation, but their generalisation capability is limited to
the training categories. Recently, Oryon [15] has shown an
effective approach based on joint image matching and seg-
mentation in order to estimate the pose of a common object
given two scenes. Instead of relying on an object model or a
set of reference views, the object of interest is described with a
textual prompt, and localised by segmentation. The reference
scene and the usage of depth information allows Oryon to
estimate the relative 6D pose between the two scenes, even for
objects unseen at test time. However, the limited resolution of
Oryon’s feature map limits its ability to estimating accurate
poses. In this work, we address this limitation by proposing
a set of key modifications, which include a higher-resolution
feature map for fine-grained feature matching, and an updated
VLM to better exploit the textual prompt.
Object localisation for pose estimation. Adopting external
modules to localise the object of interest within input images
is standard practice in the pose estimation community [5],
[11], [32]. This is often necessary to deal with highly occluded
datasets [33], and is common both in the classic setting [4],
[5] and in the unseen-object setting [9], [22]. Crops are
obtained by training a detector on the specific objects (e.g.,
YOLOv5 [34] in OnePose and OnePose++, FCOS [35] in
ZebraPose [5]) or by using ground-truth bounding boxes [6],
[25], [26]. Notable exceptions are methods that train the detec-
tor contextually to the rest of the pipeline, such as Gen6D [13]
and OSOP [12]. For non-generalisable methods, adopting
supervised detectors [34]–[36] to obtain a bounding box is
reasonable, as they assume to have access to the object model
at test time. Therefore using them to train a detector does
not lead to loss of generalisation. Recently, CNOS [37] has
been proposed to provide accurate segmentation masks given
a CAD model of the object, without training on the specific
object instances. This method can be naturally paired with
model-based methods for unseen-object pose estimation [7],
[10], [12], as CNOS shares their assumptions (i.e., object 3D
model available at test time, but no specific training on it).
Horyon retains the open-vocabulary assumptions, as we use
an open-vocabulary detector to locate and crop the object.

4
Transp. 
Conv.
+
Group. 
Conv.
[·|·]
2D Conv
2D Conv
[·|·]
Transp. 
Conv.
+
Group. 
Conv.
[·|·]
Transp. 
Conv.
+
Group. 
Conv.
2D Conv
Deformable 
Self-Attention
Q, K, V
Q, K, V
Self-Attention
Image-to-Text Cross-Attention
Q
K, V
Text-to-Image Cross-Attention
K, V
Q
N Layers
Fig. 2.
(a) Overview of a layer of the fusion module ϕT V . Note that the left-side output of the module is not used in the last layer. (b) Architecture of the
decoder ϕD. E1,E2,E3: visual features obtained from ϕV ; Transp.Conv.: transposed 2D convolution; Group.Conv.: group composed by two blocks, each
contains a 2D convolution, a group normalisation and a ReLU; [·|·] denotes feature concatenation.
III. PROPOSED APPROACH: HORYON
A. Overview
Let two RGBD scenes, an anchor A and a query Q, depict
two scenes with a common object. Each scene is represented
by the RGB channels RGBA (RGBQ) and a depth map DA
(DQ). The intrinsic camera parameters used to capture both
A and Q are available. For each pair (A, Q) a textual prompt
T describing the common object O is provided. The object
present at test time was not observed at training time. The
objective is to estimate the 6D pose of the object in Q with
respect to the reference provided by A.
Fig. 1(a) shows the proposed Horyon architecture. Horyon
consists of four main modules: an open-vocabolary detector,
the vision and text encoders, the fusion module, and the de-
coder. The open-vocabulary detector identifies the object O in
both scenes. The two pre-trained networks: the vision e

## conclusion
We presented Horyon, an approach that significantly im-
proves upon previous open-vocabulary 6D pose estimation
methods, by increasing the feature map resolution and pro-
viding more accurate matches to perform registration. Our
experiments show that Horyon obtains good performances
also in scenarios with unusual objects (Linemod) and mild
occlusions (YCB-Video). The ablation studies show that the
token-wise representation provided by the updated VLM,
together with the new fusion strategy, greatly benefit Horyon.
The improvement is consistent in terms of generalisation
capabilities and robustness to prompt noise as well.
Horyon’s limitations are the need for depth maps and
intrinsic camera parameters to perform registration. Such data
requirements could be relaxed by exploring monocular depth
estimation methods such as DepthAnything [59] on each
RGB image. While Horyon is more robust to suboptimal
prompts than Oryon, the resulting drop in performance is still
significant. Moreover, the variety of prompt usable at test time
is limited by the training data, which provides prompts without
descriptions. To enrich the prompts at training time, LLMs or
image captioners could be used on image samples to provide
prompts that also include a description of the object’s colour
and physical attributes.
Acknowledgements. This work was supported by the Euro-
pean Union’s Horizon Europe research and innovation pro-
gramme under grant agreement No 101058589 (AI-PRISM),
and it made use of time on the Tier 2 HPC facility JADE2,
funded by EPSRC (EP/T022205/1).