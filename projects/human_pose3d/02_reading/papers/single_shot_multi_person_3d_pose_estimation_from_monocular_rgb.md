# Single-Shot Multi-person 3D Pose Estimation from Monocular RGB

> 2018 · id: W2888934629 · 来源: web-agent
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

Single-Shot Multi-Person 3D Pose Estimation From Monocular RGB
Dushyant Mehta[1,2], Oleksandr Sotnychenko[1,2], Franziska Mueller[1,2],
Weipeng Xu[1,2], Srinath Sridhar[3], Gerard Pons-Moll[1,2], Christian Theobalt[1,2]
[1] MPI For Informatics
[2] Saarland Informatics Campus
[3] Stanford University
Abstract
We propose a new single-shot method for multi-
person 3D pose estimation in general scenes from a
monocular RGB camera.
Our approach uses novel
occlusion-robust pose-maps (ORPM) which enable full
body pose inference even under strong partial occlu-
sions by other people and objects in the scene. ORPM
outputs a ﬁxed number of maps which encode the 3D
joint locations of all people in the scene.
Body part
associations [8] allow us to infer 3D pose for an ar-
bitrary number of people without explicit bounding box
prediction. To train our approach we introduce MuCo-
3DHP, the ﬁrst large scale training data set showing
real images of sophisticated multi-person interactions
and occlusions. We synthesize a large corpus of multi-
person images by compositing images of individual peo-
ple (with ground truth from mutli-view performance
capture). We evaluate our method on our new chal-
lenging 3D annotated multi-person test set MuPoTs-
3D where we achieve state-of-the-art performance. To
further stimulate research in multi-person 3D pose esti-
mation, we will make our new datasets, and associated
code publicly available for research purposes.
1. Introduction
Single-person pose estimation, both 2D and 3D,
from monocular RGB input is a challenging and widely
studied problem in vision [4, 3, 33, 34, 7, 11, 28, 37]. It
has many applications, e.g., in activity recognition and
content creation for graphics. While methods for 2D
multi-person pose estimation exist [43, 17, 8, 37], most
3D pose estimation methods are restricted to a sin-
gle un-occluded subject. Natural human activities take
place with multiple people in cluttered scenes hence ex-
hibiting not only self-occlusions of the body, but also
strong inter-person occlusions or occlusions by objects.
This work was funded by the ERC Starting Grant project
CapReal (335545). We would also like to thank Hyeongwoo Kim
and Eldar Insafutdinov for their assistance.
This makes the under-constrained problem of inferring
3D pose (of all subjects) from monocular RGB input
even harder and leads to drastic failure of existing sin-
gle person 3D pose estimation methods.
Recent work approaches this more general 3D multi-
person pose estimation problem by decomposing it into
multiple single-person instances [31, 49], often with sig-
niﬁcant redundancy in the decomposition [49].
The
single person predictions are post-processed to ﬁlter,
reﬁne and fuse the predictions into a coherent esti-
mate. Bottom-up joint multi-person reasoning remains
largely unsolved, and the multi-person 3D pose estima-
tion lacks appropriate performance benchmarks.
We propose a new single shot CNN-based method to
estimate multi-person 3D pose in general scenes from
monocular input. We call our method single shot since
it reasons about all people in a scene jointly in a sin-
gle forward pass, and does not require explicit bound-
ing box proposals by a separate algorithm as a pre-
processing step [49, 40].
The latter may fail under
strong occlusions and may be expensive to compute
in dense multi-person scenes. Our fully-convolutional
method jointly infers 2D and 3D joint locations using
our new occlusion-robust pose-map (ORPM) formula-
tion. ORPM enables multi-person 3D pose estimates
under strong (self-)occlusions by incorporating redun-
dancy in the encoding, while using a ﬁxed number of
outputs regardless of the number of people in the scene.
Our subsequent hierarchical read-out strategy starts
with a base pose estimate, and is able to reﬁne the
estimate based on which joints of a person are visible,
leading to robust 3D pose results.
To train our CNN we introduce a new multi-person
3D pose data set MuCo-3DHP. While there are sev-
eral single-person datasets with 3D joint annotations,
there are no annotated multi-person datasets contain-
ing large corpora of real video recordings of human–
human interaction with large person and background
diversity.
Important advances in this direction have
been made by Joo et al. [24] using a multi-camera stu-
dio setup but background diversity remains limited.
1
arXiv:1712.03453v3  [cs.CV]  28 Aug 2018

Figure 1.
Qualitative results of our approach shown on MPII 2D [3] dataset (blue), as well as our new MuPoTS-3D
evaluation set (green). Our pose estimation approach works for general scenes, handling occlusions by objects or other
people. Note that our 3D pose predictions are root-relative, and scaled and overlaid only for visualization.
Some prior work creates 3D annotated multi-person
images by using 2D pose data augmented with 3D poses
from motion capture datasets [49], or by ﬁnding 3D
consistency in 2D part annotations from multi-view
images recorded in a studio [53].
To create training
data with much larger diversity in person appearance,
camera view, occlusion and background, we transform
the MPI-INF-3DHP single-person dataset [33] into the
ﬁrst multi-person set that shows images of real peo-
ple in complex scenes.
MuCo-3DHP is created by
compositing multiple 2D person images with ground-
truth 3D pose from multi-view marker-less motion cap-
ture.
Background augmentation and shading-aware
foreground augmentation of person appearance enable
further data diversity. To validate the generalizability
of our approach to real scenes, and since there are only
very few annotated multi-person test sets [12] show-
ing more than two people, we contribute a new multi-
person 3D test set, MuPoTS-3D. It features indoor
and outdoor scenes, challenging occlusions and inter-
actions, varying backgrounds, more than two persons,
and ground truth from commercial marker-less motion
capture. All datasets will be made publicly available.
In summary, we contribute:
• A CNN-based single-shot multi-person pose esti-
mation method based on a novel multi-person 3D
pose-map formulation to jointly predict 2D and
3D joint locations of all persons in the scene. Our
method is tailored for scenes with occlusion by ob-
jects or other people.
• The ﬁrst multi-person dataset of real person im-
ages with 3D ground truth that contains complex
inter-person occlusions, motion, and background
diversity.
• A real in-the-wild test set for evaluating multi-
person 3D pose estimation methods that contains
diverse scenes, challenging multi-person interac-
tions, occlusions, and motion.
Our method achieves state-of-the-art performance
on challenging multi-person scenes where single-person
methods completely fail.
Although designed for the
much harder multi-person task, it performs competi-
tively on single-person test data.
2. Related Work
We focus on most directly related work estimating
the pose of multiple people in 2D or a single person in
3D from monocular RGB input. [50] provide a more
comprehensive review.
Multi-Person 2D Pose Estimation: A common
approach for multi-person 2D pose estimation is to ﬁrst
detect single persons and then 2D pose [44, 13, 55, 19,
40]. Unfortunately, these methods fail when the detec-
tors fail—a likely scenario with multiple persons and
strong occlusions. Hence, a body of work ﬁrst localizes
the joints of each person with CNN-based detectors
and then ﬁnd the correct association between joints
and subjects in a post-processing step [43, 16, 36, 8].
Single-Person 3D Pose Estimation:
Existing
monocular single-person 3D pose methods show good
performance on standard datasets [18, 51, 59, 41, 28,
27].
However, since many methods train a discrimi-
native predictor for 3D poses [5], they often do not
generalize well to natural scenes with varied poses, ap-
pearances, backgrounds and occlusions.
This is due
to the fact that most 3D datasets are restricted to
indoor setups with limited backgrounds and appear-
ance. The advent of large real world image datasets
with 2D annotations made 2D pose estimation in the
wild remarkably accurate.
However, annotating im-
ages with 3D pose is much harder with many recent
work focusing on leveraging 2D image datasets for 3D
human pose estimation or multi-view settings [24]. Ad-
ditional annotations to these 2D image datasets al-
low some degree of 3D reasoning, either through body
joint depth ordering constraints [42] or dense shape
correspondences [14].
Some works split the problem
in two: ﬁrst estimate 2D joints and then lift them to
3D [58, 54, 9, 66, 35, 69, 1, 52, 20, 32, 6, 26, 38, 57, 2],
e.g., by database matching, neural network regression,
or ﬁtting the SMPL body model [30]. Some works in-
tegrate SMPL within the CNN to exploit 3D and 2D
annotations in an end-to-end fashion [39, 25, 42].
Other work leverages the features learned by a 2D
pose estimation CNN for 3D pose estimation. For ex-
ample, [60] learn to merge features from a 2D and 3D
joint prediction network. Another approach is to train

Figure 2. Examples from our MuCo-3DHP dataset, cre-
ated through compositing MPI-INF-3DHP [33] data. (Top)
composited examples without appearance augmentation,
(bottom) with BG and clothing augmentation.
The last
two columns show rotation and scale augmentation, and
truncation with the frame boundary.
a network with separate 2D and 3D losses for the diﬀer-
ent data sources [46, 68, 56, 65, 31]. Some approaches
jointly reason about 2D and 3D pose with multi-stage
belief maps [62].
The advantage of such methods is
that they can be trained end to end. A simpler yet
very eﬀective approach is to reﬁne a network trained
for 2D pose estimation for the task of 3D pose estima-
tion [34, 33]. A major limitation of methods that rely
on 2D joint detections directly or on bounding boxes
is that they easily fail under body occlusion or with
incorrect 2D detections, both of which are common in
multi-person scenes. In contrast, our approach is more
robust to occlusions since a base 3D body pose esti-
mate is available even under signiﬁcant occlusion. [34]
showed that 3D joint prediction works best when the
receptive ﬁeld is centered around the joint of interest.
We build upon this insight to reﬁne the base body pose
where 2D joint detections are available.
Multi-Person 3D Pose Estimation:
To our
knowledge, only [49] tackle multi-person 3D pose esti-
mation from single images.1 They ﬁrst identify bound-
ing boxes likely to contain a person using [47]. Instead
of a direct regression to pose, the bounding boxes are
classiﬁed into a set of K-poses similar to [45]. These
poses are scored by a classiﬁer and reﬁned using a re-
gressor. The method implicitly reasons using bounding
boxes and produces multiple proposals per subject that
need to be accumulated and fused. However, perfor-
mance of their method under large person-person oc-
clusions is unclear. In contrast, our approach produces
multi-person 2D joint locations and 3D pose maps in
a single shot, from which the 3D pose can be inferred
even under severe person-person occlusion.
3D Pose Datasets: Existing pose datasets are ei-
ther for a single person in 3D [18, 51, 63, 64, 33] or
multi-person with only 2D pose annotations [3, 29]. Of
1A second approach [67] was published in the review period.
the two exceptions, the MARCOnI dataset [12] features
5 sequences but contains only 2 persons simultaneously,
and there are no close interactions. The other is the
Panoptic dataset [24] which has a limited capture vol-
ume, pose and background diversity.
There is work
on generating synthetic images [48, 10] from mocap
data, however the resulting images are not plausible.
We choose to leverage the person segmentation masks
available in MPI-INF-3DHP [33] to generate annotated
multi-person 3D pose images of real people at scale
through compositing using the available segmentation
masks.
Furthermore, we captured a 3D benchmark
dataset featuring multiple closely interacting persons
which was annotated by a video-based multi-camera
motion capture system.
3. Multi-Person Dataset
Generating data by combining in-the-wild multi-
person 2D pose data [3, 29] and multi-person multi-
view motion capture for 3D annotation would be a
straightforward extension of previous (single-person)
approaches [33].
However, multi-person 3D motion
capture under strong occlusions and interactions is
challenging even for commercial systems, often requir-
ing manual pose correction constraining 3D accuracy.
Hence, we merely employ purely multi-view marker-less
motion capture to create the 20 sequences of MuPoTs-
3D, the ﬁrst expressive in-the-wild multi-person 3D
pose benchmark.
For the much larger training set
MuCo-3DHP, we resort to a new compositing and aug-
mentation scheme that leverages the single-person im-
age data of real people in MPI-INF-3DHP[33] to com-
posite an arbitrary number of multi-person interaction
images under user control, with 3D pose annotations.
3.1. MuCo-3DHP: Compositing-Based Training Set
The MPI-INF-3DHP [33] single-person 3D pose
dataset provides marker-less motion capture based an-
notations for real images of 8 subjects, each captured
with 2 clothing sets, using 14 cameras at diﬀerent ele-
vations.
We build upon these person segmentation
masks to create per-camera composites with 1 to 4
subjects, with frames randomly selected from the 8×2
sequences available per camera. Since we have ground-
truth 3D skeleton pose for each video subject in the
same space, we can composite in a 3D-aware manner
resulting in correct depth ordering and overlap of sub-
jects. We refer to this composited training set as the
Multiperson Composited 3D Human Pose dataset
(see Fig. 2 for examples).
The compositing process
results in plausible images covering a range of simu-
lated inter-person overlap and activity scenarios. Fur-
thermore, user-control over the desired pose and occlu-

Figure 3. Examples from our MuPoTS-3D evaluation set.
Ground truth 3D pose reference and joint occlusion annota-
tions are available for up to 3 subjects in the scene. The set
covers a variety of scene settings, activities and clothing.
sion distribution, and foreground/background augmen-
tation using the masks provided with MPI-INF-3DHP
is possible (see supplementary document for more de-
tails). Even though the synthesized composites may
not simulate all the nuances of human-human interac-
tion fully, we observe that our approach trained on this
data generalizes well to real world scenes in the test set.
3.2. MuPoTS-3D: Diverse Multi-Person 3D Test Set
We also present a new ﬁlmed (not composited)
multi-person test set comprising 20 general real world
scenes with ground-truth 3D pose for up to three sub-
jects obtained with a multi-view marker-less motion
capture system [61]. Additionally, per joint occlusion
annotations are available.
The set covers 5 indoor
and 15 outdoor settings, with trees, oﬃce buildings,
road, people, vehicles, and other stationary and mov-
ing distractors in the background. Some of the outdoor
footage also has challenging elements like drastic illu-
mination changes, and lens ﬂare. The indoor sequences
use 2048 × 2048px footage at 30fps, and outdoor se-
quences use 1920 × 1080px GoPro footage at 60fps.
The test set consists of >8000 frames, split among the
20 sequences, with 8 subjects, in a variety of clothing
styles, poses, interactions, and activities. Notably, the
test sequences do not resemble the training data, and
include real interaction scenarios. We call our new test
set Multiperson Pose Test Set in 3D (MuPoTS-3D).
Evaluation Metric:
We use the robust 3DPCK
evaluation metric proposed in [33]. It treats a joint’s
prediction as correct if it lies within a 15cm ball cen-
tered at the ground-truth joint location, and is evalu-
ated for the common minimum set of 14 joints marked
in green in Fig. 5. We report the 3DPCK numbers per
sequence, averaged over the subjects for which GT ref-
erence is available, and additionally report the perfor-
mance breakdown for occluded and un-occluded joints.
The relative robustness of 3DPCK over MPJPE[18] is
also useful to oﬀset the eﬀect of jitter that arises in all
non-synthetic annotations, including ours.
For com-
pleteness, we also report the MPJPE error for predic-
tions matched to an annotated subject.
4. Method
At the core of our approach is a novel formulation
which allows us to estimate the pose of multiple peo-
ple in a scene even under strong occlusions with a sin-
gle forward pass of a fully convolutional network. Our
method builds upon the location-maps formulation [34]
that links 3D pose inference more strongly to image ev-
idence by inferring 3D joint positions at the respective
2D joint pixel locations.We ﬁrst recap the location-map
formulation before describing our approach.
Location-Maps [34]:
A location-map is a joint
speciﬁc feature channel storing the 3D coordinate x, y,
or z at the joint 2D pixel location. For every joint, three
location-maps, as well as a 2D pixel location heatmap
are estimated. The latter encodes the 2D pixel location
of the joint as a conﬁdence map in the image plane.
The 3D position of a joint can be read out from its
location-map at the 2D pixel location of the joint, as
shown in Fig. 4.
For an image of size W × H, 3n
location-maps of size W/k × H/k are used to store the
3D location of all n joints, where k is a down-sampling
factor. During training, the L2 loss between the ground
truth and the estimated location-map is minimized in
the area around the joint’s 2D pixel location. Although
this simple location-map formulation enables full 3D
pose inference, it has several shortcomings. First, it
assumes that all joints of a person are fully visible, and
breaks down under partial occlusion, which is common
in general scenes. Second, eﬃcient extension to multi-
ple people is not straightforward. Introducing separate
location-maps per person requires dynamically chang-
ing the number of outputs.
Occlusion-Robust Pose-Maps (ORPMs):
We
propose a novel occlusion-robust formulation that has
a ﬁxed number of outputs regardless of the number
of people in the scene, while enabling pose read-outs
for strongly occluded people. Our key insight is the
incorporation of redundancy into the location-maps.
We represent the body by decomposing it into torso,
four limbs, and head (see Fig. 5). Our occlusion-robust
pose-maps (ORPMs) support multiple levels of redun-
dancy: (1) they allow the read-out of the complete base
pose P ∈R3×n at one of the torso joint locations (neck
or pelvis), (2) the base pose (which may not capture
the full extent of articulation) can be further reﬁned by
reading out the head and individual limb poses where
2D detections are available, and (3) the complete limb

pose can be read out at any 2D joint location of that
limb. Together, these ensure that a complete and as
articulate as possible pose estimate is available even
in the presence of heavy occlusions of the body (see
Fig. 5, and Fig. 2,3 in the supplementary document).
In addition, the redundancy in ORPMs allows to en-
code the pose of multiple partially overlapping persons
without loss of information, thus removing the need for
a variable number of output channels. See Fig. 4.
Na¨ıve Redundancy:
The na¨ıve approach to in-
troduce redundancy by allowing full pose read-out at
all body joint locations breaks down for interacting and
overlapping people, leading to supervision and read-out
conﬂicts in all location-map channels.
Our selective
introduction of redundancy restricts these conﬂicts to
pose-map channels of similar limbs, i.e., wrist of one
person in the proximity of a knee of another person
cannot cause read-out conﬂicts because their pose is
encoded in their respective pose-maps. If the complete
pose was encoded at each joint location, there would be
conﬂicts for each pair of proximate joints across people.
We now formally deﬁne ORPMs (Sec. 4.1) and explain
the inference process (Sec. 4.2).
4.1. Formulation
Given a monocular RGB image I of size W × H,
we seek to estimate the 3D pose P = {Pi}m
i=1 for
each of the m persons in the image.
Here, Pi ∈
R3×n describes the 3D locations of the n
=
17
body joints of person i.
The body joint locations
are expressed relative to the parent joints as indi-
cated in Fig. 5 and converted to pelvis-relative loca-
tions for evaluation.
We ﬁrst decompose the body
into pelvis, neck, head, and a set of limbs:
L =
{{shoulders, elbows, wrists}, {hips, knees, ankles} | s ∈
{right, left}} . The 3D locations of the joints are then
encoded in the occlusion-robust pose-maps denoted by
M = {Mj}n
j=1, where Mj ∈RW ×H×3. In contrast to
simple location-maps, the ORPM Mj stores the 3D
location of joint j not only at this joint’s 2D pixel
location (u, v)j but at a set of 2D locations ρ(j) =
{(u, v)neck, (u, v)pelvis} ∪{(u, v)k}k∈limb(j), where:
limb(j) =





l,
if ∃l ∈L with j ∈l
{head},
if j = head
∅,
otherwise
.
(1)
Note that—since joint j of all persons i is encoded in
Mj—it can happen that read-out locations coincide
for diﬀerent people, i.e., ρi1(j) ∩ρi2(j)̸ = ∅. In this
case, Mj contains information about the person closer
to the camera at the overlapping locations. However,
due to our built-in redundancy in the ORPMs, a pose
Figure 4.
Multiple levels of selective redundancy in our
Occlusion-robust Pose-map (ORPM) formulation. VNect
Location-maps [34] (left) only support readout at a single
pixel location per joint type. ORPMs (middle) allow the
complete body pose to be read out at torso joint pixel lo-
cations (neck, pelvis). Further, each individual limb’s pose
can be read out at all 2D joint pixel locations of the re-
spective limb. This translates to read-out of each joint’s
location being possible at multiple pixel locations in the
joint’s location map.
The example at the bottom shows
how 3D locations of multiple people are encoded into the
same map per joint and no additional channels are required.
estimate for the partially occluded person can still be
obtained at other available read-out locations.
To estimate where the pose-maps can be read out,
we make use of 2D joint heatmaps H = {Hj
∈
RW ×H}n
j=1 predicted by our network. Additionally, we
estimate part aﬃnity ﬁelds A = {Aj ∈RW ×H×2}n
j=1
which represent a 2D vector ﬁeld pointing from a joint
of type j to its parent [8]. This facilitates association
of 2D detections in the heatmaps (and hence read-out
locations for the ORPMs) to person identities and en-
ables per-person read-outs when multiple people are
present. Note that we predict a ﬁxed number of maps
(n heatmaps, 3n pose-maps, and 2n part aﬃnity ﬁelds)
in a single forward pass irrespective of the number of
persons in the scene, jointly encoding the 2D and 3D
pose for all subjects, i.e., our network is single-shot.
4.2. Pose Inference
Read-out of 3D pose of multiple people from
ORPMs starts with inference of 2D joint locations
P2D = {P2Di}m
i=1 with P2Di = {(u, v)i
j}n
j=1 and joint
detection conﬁdences C2D = {C2Di ∈Rn}m
i=1 for each
person i in the image. Explicit 2D joint-to-person asso-
ciation is done with the predicted heatmaps H and part
aﬃnity ﬁelds A using the approach of Cao et al. [8].

Figure 5. Example of the choice of read-out pixel location for right elbow pose under various scenarios. First the complete
body pose is read out at one of the torso locations. a.) If the limb extremity is un-occluded, the pose for the entire limb is
read out at the extremity (wrist), b.) If the limb extremity is occluded, the pose for the limb is read out at the joint location
further up in the joint hierarchy (elbow), c.) If the entire limb is occluded, we retain the base pose read out at one of the
torso locations (neck), d.) Read-out locations indicated for inter-person interaction, e.) If two joints of the same type (right
wrist here) overlap or are in close proximity, limb pose read-out is done at a safer isolated joint further up in the hierarchy.
Next, we use the 2D joint locations P2D and the joint
detection conﬁdences C2D in conjunction with ORPMs
M to infer the 3D pose of all persons in the scene.
Read-Out Process: By virtue of the ORPMs we
can read out 3D joint locations at select multiple pixel
locations as described above.
We deﬁne extremity
joints:
the wrists, the ankles, and the head.
The
neck and pelvis 2D detections are usually reliable, these
joints are most often not occluded and lie in the middle
of the body. Therefore, we start reading the full base
pose at the neck location. If the neck is invalid (as
deﬁned below) then the full pose is read at the pelvis
instead. If both of these joints are invalid, we consider
this person as not visible in the scene and we do not
predict the person’s pose. While robust, full poses read
at the pelvis and neck tend to be closer to the average
pose in the training data.
Hence, for each limb, we
continue by reading out the limb pose at the extremity
joint. Note again that the complete limb pose can be
accessed at any of that limb’s 2D joint locations. If
the extremity joint is valid, the limb pose replaces the
corresponding elements of the base pose. If the extrem-
ity joint is invalid however, we walk up the kinematic
chain and check the other joints of this limb for valid-
ity. If all joints of the limb are invalid, the base pose
cannot be further reﬁned. This read-out procedure is
illustrated in Fig. 5 and algorithmically described in
the supplementary document.
2D Joint Validation: We declare a 2D joint loca-
tion P2Dj
i = (u, v)i
j of person i as valid read-out loca-
tion iﬀ(1) it is un-occluded, i.e., has conﬁdence value
higher than a threshold tC, and (2) it is suﬃciently far
(≥tD) away from all read-out locations of joint j of
other individuals:
valid(P2Dj
i) ⇔C2Dj
i > tC ∧||a −P2Dj
i||2 ≥tD
∀¯i = [1:m], ¯i̸ = i. ∀a ∈ρ¯i(j).
(2)
Our ORPM formulation together with the occlusion-
aware inference strategy with limb reﬁnement enables
us to obtain accurate poses even for strongly occluded
body parts while exploiting all available information if
individual limbs are visible. We validate our perfor-
mance on occluded joints and the importance of limb
reﬁnement on our new test set (see Sec. 5).
4.3. Network and Training Details
Our network has ResNet-50 [15] as the core, after
which we split it into two—a 2DPose+Aﬃnity stream
and a 3DPose stream. The core network and the ﬁrst
branch are trained on MS-COCO [29] and the sec-
ond branch is trained with MPI-INF-3DHP or MuCo-
3DHP as per the scenario. Training and architectural
speciﬁcs are in the supplementary document.
The
2DPose+Aﬃnity
stream
predicts
the
2D
heatmaps HCOCO for the MS-COCO body joint set,
and part aﬃnity ﬁelds ACOCO. The 3DPose stream
predicts 3D ORPMs MMP I as well as 2D heatmaps
HMP I for the MPI-INF-3DHP [33] joint set, which has
some overlap with the MS-COCO joint set. For pose
read-out locations as described previously, we restrict
ourselves to the common minimum joint set between
the two, indicated by the circles in Fig. 5.
Loss:
The 2D heatmaps HCOCO and HMP I are
trained with per-pixel L2 loss comparing the predic-
tions to the reference which has unit peak Gaussians
with a limited support at the ground truth 2D joint lo-
cations, as is common. The part aﬃnity ﬁelds ACOCO
are similarly trained with a per-pixel L2 loss, using
the framework made available by Cao et al. [8]. While
training ORPMs with our MuCo-3DHP, per joint type
j, for all subjects i in the scene, a per-pixel L2 loss
is enforced in the neighborhood of all possible read-
out locations ρi(j). The loss is weighted by a limited
support Gaussian centered at the read-out location.

Table 1.
Sequence-wise evaluation of our method and LCR-net[49] on multi-person 3D pose test set MuPoTS-3D. We
report both (a) the overall accuracy (3DPCK), and (b) accuracy only for person annotations matched to a prediction
TS1 TS2 TS3 TS4 TS5 TS6 TS7 TS8 TS9 TS10 TS11 TS12 TS13 TS14 TS15 TS16 TS17 TS18 TS19 TS20 Total
a.) LCR-net 67.7 49.8 53.4 59.1 67.5 22.8 43.7 49.9 31.1 78.1
50.2
51.0 51.6 49.3 56.2 66.5 65.2
62.9
66.1
59.1
53.8
Ours
81.0 59.9 64.4 62.8 68.0 30.3 65.0 59.2 64.1 83.9 67.2 68.3 60.6 56.5 69.9 79.4 79.6 66.1 66.3 63.5
65.0
b.) LCR-net
69.1 67.3 54.6 61.7 74.5 25.2 48.4 63.3 69.0 78.1 53.8 52.2 60.5 60.9
59.1
70.5
76.0
70.0 77.1 81.4
62.4
Ours
81.0 64.3 64.6 63.7 73.8 30.3 65.1 60.7 64.1 83.9 71.5 69.6 69.0 69.6 71.1 82.9 79.6 72.2 76.2 85.9
69.8
5. Results and Discussion
The main goal of our method is multi-person 3D
pose estimation in general scenes, which exhibits spe-
ciﬁc and more diﬃcult challenges than single-person
pose estimation.
However, we validate the useful-
ness of our ORPM formulation on the single-person
pose estimation task as well.
To validate our ap-
proach, we perform extensive experiments on our pro-
posed multi-person test set MuPoTS-3D, as well as two
publicly available single-person benchmarks, namely
Human3.6m [18] and MPI-INF-3DHP [33].
Fig. 1
presents qualitative results of our method showcasing
the ability to handle complex in-the-wild scenes with
strong inter-person occlusion. An extensive collection
of qualitative results is provided in the supplementary
document and video.
5.1. Comparison with Prior Art
For sequences with strong occlusion, we obtain
much better results than the state-of-the-art. For un-
occluded sequences our results are comparable to meth-
ods designed for single person.
We outperform the
only other multi-person method (LCR-net [49]) quan-
titatively and qualitatively on both single-person and
multi-person tasks. For fairness of comparison, in all
evaluations, we re-target the predictions from LCR-
net [49] to a skeleton with bone-lengths matching the
ground truth.
Multi-Person Pose Performance:
We use our
proposed MuPoTS-3D (see Sec. 3.2) to evaluate multi-
person 3D pose performance in general scenes for our
approach and LCR-net [49]. In addition, we evaluate
VNect [34] on images cropped with the ground truth
bounding box around the subject. We evaluate for all
subjects that have 3D pose annotations available. If an
annotated subject is missed by our method, or by LCR-
net, we consider all of its joints to be incorrect in the
3DPCK metric. Table 1(a) reports the 3DPCK metric
for all 20 sequences when taking all available annota-
tions into account. Our method performs signiﬁcantly
better than LCR-net for most sequences, while being
comparable for a few, yielding an overall improved per-
formance of 65.0 3DPCK vs 53.8 3DPCK for LCR-net.
We provide a joint-wise breakdown of the overall accu-
racy in the supplementary document.
Overall, our approach detects 93% of the annotated
subjects, whereas LCR-net was successful for 86%.
This is an additional indicator of performance. Even
ignoring the undetected annotated subjects, our ap-
proach outperforms LCR-net in terms of 3D pose error
(69.8 vs 62.4 3DPCK, and 132.5 vs 146 mm MPJPE).
VNect is evaluated on ground truth crops of the sub-
jects, and therefore it operates at a 100% detection
rate. In contrast, we do not use ground truth crops,
and missed detections by our method count as all joints
wrong. Despite this our method achieves better accu-
racy (65.0 vs 61.1 3DPCK, 30.1 vs 27.6 AUC).
Single-Person Pose Performance:
On the
MPI-INF-3DHP dataset (see Table 2) we compare
our method trained on MPI-INF-3DHP (single-person)
and MuCo-3DHP (multi-person) to three single-person
methods—VNect, Zhou et al. [68], Mehta et al. [33]—
and LCR-net as the only other multi-person ap-
proach.
Our method trained on multi-person data
(73.4 3DPCK) performs marginally worse than our
single-person version (75.2 3DPCK) due to the eﬀec-
tive loss in network capacity when training on harder
data. Nevertheless, both our versions consistently out-
perform Zhou et al. (69.2 3DPCK) and LCR-net (59.7
3DPCK) over all metrics. LCR-net predictions have a
tendency to be conservative about the extent of artic-
ulation of limbs as shown in Fig. 6. In comparison to
VNect (76.6 3DPCK) and Mehta et al. (75.7 3DPCK),
our method (75.2 3DPCK) achieves an average accu-
racy that is on par. Our approach outperforms existing
methods by ≈3-4 3DPCK for activities which exhibit
signiﬁcant self- and object-occlusion like Sit on Chair
and Crouch/Reach.
For the full activity-wise break-
down, see the supplemental document.
For detailed comparisons on Human3.6m [18], re-
fer to the supplementary document.
Our approach
at 69.6mm MPJPE performs ≈17mm better than
LCR-net [49] (87.7mm), and outperforms the VNect
location-map [34] (80.5mm) formulation by ≈10mm.
Our results are comparable to the recent state-of-the-
art results of Pavlakos et al. [41] (67.1mm), Martinez
et al. [32] (62.9mm), Zhou et al. [68] (64.9mm), Mehta
et al. [33] (68.6mm) and Tekin et al. [60] (70.81mm),
and better than the recent results from Nie et al. [38]
(79.5mm) and Tome et al. [62] (88.39mm).

Figure 6. Qualitative comparison of LCR-net [49] and our
method. LCR-net output is limited in the extent of artic-
ulation of limbs, tending towards neutral poses. LCR-net
also has more detection failures under signiﬁcant occlusion.
Occlusion evaluation:
To demonstrate the oc-
clusion robustness of our method, we create synthetic
random occlusions on the MPI-INF-3DHP test set.
The synthetic occlusions cover about 14% of the joints.
Our single-person and multi-person variants both out-
perform VNect for occluded joints (62.8 vs 64.0 vs 53.2
3DPCK) by a large margin, and are comparable for
un-occluded joints (67.0 vs 71.0 vs 69.4 3DPCK). Note
again that the single-person variant is trained on simi-
lar data as VNect and the occlusion robustness is inher-
ent to the formulation. See the supplemental document
for a more detailed breakdown by test sequence.
We use the per-joint occlusion annotations from
MuPoTS-3D to further assess LCR-net and our ap-
proach under occlusion.
Considering both self- and
inter-personal occlusions, ≈23.7 % of the joints of all
subjects are occluded. Our method is more robust than
LCR-net on both occluded (48.7 vs 42 3DPCK) and
un-occluded (70.0 vs 57.5 3DPCK) joints.
5.2. Ablative Analysis
We validate the improvement provided by our limb
reﬁnement strategy on MPI-INF-3DHP test set. We
empirically found that the base pose read out at one
of the torso joints tends towards the mean pose of the
training data. Hence, for poses with signiﬁcant artic-
ulation of the limbs, the base pose does not provide
an accurate pose estimate, especially for the end eﬀec-
tors. On the other side, the limb poses read out further
down in the kinematic chain, i.e., closer to the extrem-
ities, include more detailed articulation information for
that limb.
Our proposed read-out process exploits
this fact, signiﬁcantly improving overall pose quality
by limb reﬁnement when limbs are available. Table 2
shows that the beneﬁt of the full read-out is consis-
tent over all metrics and valid for our method indepen-
dent of whether it is trained on single-person (MPI-
INF-3DHP) or multi-person data (MuCu-3DHP), with
a ≈10 3DPCK advantage over torso read-out.
See
Fig. 2,3 in the supplementary document.
Table 2. Comparison of results on MPI-INF-3DHP [33] test
set. We report the Percentage of Correct Keypoints measure
in 3D (@150mm) for select activities, and the total 3DPCK
and the Area Under the Curve for all activities. Complete
activity-wise breakdown in the supplementary document
Sit
Crouch
Total
Method
PCK
PCK
PCK AUC
VNect [34]
74.7
72.9
76.6
40.4
LCR-net [49]
58.5
69.4
59.7
27.6
Zhou et al.[68]
60.7
71.4
69.2
32.5
Mehta et al.[33]
74.8
73.7
75.7
39.3
Our Single-Person (Torso)
69.1
68.7
65.6
32.6
Our Single-Person (Full)
77.8
77.5
75.2
37.8
Our Multi-Person (Torso)
64.6
65.8
63.6
31.1
Our Multi-Person (Full)
75.9
73.9
73.4
36.2
6. Limitations and Future Work
As discussed in Section 4, we handle overlapping
joints of the same type by only supervising the one clos-
est to the camera. However, when joints of the same
type are in close proximity (but not overlapping) the
ground-truth ORPM for those may transition sharply
from one person to the other, which are hard to regress
and may lead to inaccurate predictions. One possible
way to alleviate the issue is to increase the resolution
of the output maps. Another source of failures is when
2D joints are mis-predicted or mis-associated. Further-
more, we have shown accurate root-relative 3D pose
estimation, but estimating the relative sizes of people
is challenging and remains an open problem for future
work. While the compositing based MuCo-3DHP cov-
ers many plausible scenarios, further investigation into
capturing/generating true person-person interactions
at scale would be an important next step.
7. Conclusion
Multi-person 3D pose estimation from monocular
RGB is a challenging problem which has not been fully
addressed by previous work. Experiments on single-
person and multi-person benchmarks show that our
method, relying on a novel occlusion robust pose for-
mulation (ORPM), works well to estimate the 3D pose
even under strong inter-person occlusions and human–
human interactions better than previous approaches.
The method has been trained on our new multi-person
dataset (MuCo-3DHP) synthesized at scale from ex-
isting single-person images with 3D pose annotations.
Our method trained on this dataset generalizes well to
real world scenes shown in our MuPOTS-3D evalua-
tion set. We hope further investigation into monocular
multi-person pose estimation would be spurred by the
proposed training and evaluation data sets.

Supplementary Document:
Single-Shot Multi-Person 3D Pose
Estimation From Monocular RGB
1. Read-out Process
An algorithmic description of the read-out process
is provided in Alg. 1.
Algorithm 1 3D Pose Inference
1: Given: P2D, C2D, M
2: for all i ∈(1..m) do
3:
if C2D
i
[k] > thresh, k ∈{pelvis, neck} then
4:
Person i is detected
5:
for all joints j ∈(1..n) do
6:
rloc = P2D
i
[k]
7:
Pi[:, j] = ReadLocMap(j, rloc)
8:
for
all
limbs
l
∈
{arml, armr, legl, legr, head} do
9:
for
j
=
getExtremity(l); j
/∈
{pelvis, neck}; j = parent(j) do
10:
if isValidReadoutLoc(i, j) then
11:
refineLimb(l, P2D
i
[j])
12:
break
13:
else
14:
No person detected
15: function getExtremity(limb l)
16:
if l = legs then return ankles
17:
else
18:
if l = arms then return wrists
19:
else return head
20: function ReadLocMap(joint j, 2DLocation rloc)
21:
rloc = rloc/locMap scale factor
22:
return Mj[rloc]
23: function refineLimb(limb l, 2DLocation rloc)
24:
for all joints b ∈limb l do
25:
Pi[:, b]=ReadLocMap(b, rloc)
26: function isValidReadoutLoc(person i, joint j)
27:
if (C2D
i
[j] > 0) then
28:
return isIsolated(i,j)
29:
else
30:
return 0
31: function isIsolated(person i, joint j)
32:
isol = 1
33:
for all persons ¯i ∈(1..m),¯i̸ = i do
34:
for all 2DLocations a ∈ρ¯i(j) do
35:
if ||a −P2D
i
[j]||2 < isoThresh then
36:
isol = 0
37:
break
38:
return isol
Figure 1. The network architecture with 2DPose+Aﬃnity
branch predicting the 2D heatmaps HCOCO and part aﬃn-
ity maps ACOCO with a spatial resolution of (W/8, H/8),
and 3DPose branch predicting 2D heatmaps HMP I and
ORPMs MMP I with a spatial resolution of (W/4, H/4),
for an input image with resolution (W, H).
2. Network Details
2.1. Architecture
A visualization if our network architecture using the
web-based visualization tool Netscope can be found
at: http://ethereon.github.io/netscope/#/gist/
069a592125c78fbdd6eb11fd45306fa0.
2.2. Data
We use 12 out of the 14 available camera viewpoints
(using only 1 of the 3 available top down views) in MPI-
INF-3DHP [33] training set, and create 400k composite
frames of MuCo-3DHP, of which half are without ap-
pearance augmentation. For training, we crop around
the subject closest to the camera, and apply rotation,
scale, and bounding-box jitter augmentation. Since the
data was originally captured in a relatively restricted
space, the likelihood of there being multiple people vis-
ible in the crop around the main person is high. The
combination of scale augmentation, bounding-b