# Deep Kinematic Pose Regression

> 2016 · id: W2522527348 · arXiv: 1609.05317 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
Estimating the pose of objects is important for understanding the behavior of the
object and relevant high level tasks, e.g., facial point localization for expression
recognition, human pose estimation for action recognition. It is a fundamental
problem in computer vision and has been heavily studied for decades. Yet, it
remains challenging, especially when object pose and appearance is complex,
e.g., human pose estimation from single view RGB images.
There is a vast range of deﬁnitions for object pose. In the simple case, the pose
just refers to the global viewpoint of rigid objects, such as car [42] or head [19].
But more often, the pose refers to a set of semantically important points on the
object (rigid or non-rigid). The points could be landmarks that can be easily
distinguished from their appearances, e.g., eyes or nose on human face [16], and
wings or tail on bird [38]. The points could further be the physical joints that
deﬁnes the geometry of complex articulated objects, such as human hand [41,21]
and human body [17,40,31].
⋆Corresponding author.
arXiv:1609.05317v1  [cs.CV]  17 Sep 2016

2
Zhou et al.
Fig. 1. Illustration of our framework. The input image undergoes a convolutional neu-
tral network and a fully connected layer to output model motion parameters (global
potision and rotation angles). The kinematic layer maps the motion parameters to
joints. The joints are connected to ground truth joints to compute the joint loss that
drives the network training.
Arguably, the articulated object pose estimation is the most challenging.
Such object pose is usually very high dimensional and inherently structured.
How to eﬀectively represent the pose and perform structure-preserving learning
is hard and have been heavily studied. Some approaches represent the object
pose in a non-parametric way (as a number of points) and directly learn the
pose from data [28,27,5]. The inherent structure is implicitly learnt and mod-
eled from data. Many other approaches use a low dimensional representation
by using dimensionality reduction techniques such as PCA [12,21], sparse cod-
ing [34,39,40] or auto-encoder [30]. The structure information is embedded in
the low dimensional space. Yet, such embedding is mostly linear and cannot well
preserve the complex articulated structural constraints.
In this work, we propose to directly incorporate the articulated object model
into the deep neutral network learning, which is the dominant approach for ob-
ject pose estimation nowadays, for hand [32,29,21,22,41,8] or human body[33,35,20,10,17,
Our motivation is simple and intuitive. The kinematic model of such objects is
well known as prior knowledge, such as the object bone lengths, bone connections
and deﬁnition of joint rotations. From such knowledge, it is feasible to deﬁne a
continuous and diﬀerentiable kinematic function with respect to the model mo-
tion parameters, which are the rotation angles. The kinematic function can be
readily put into a neutral network as a special layer. The standard gradient de-
scent based optimization can be performed in the same way for network training.
The learning framework is exempliﬁed in Fig. 1. In this way, the learning fully
respects the model geometry and preserves the structural constraints. Such end-
to-end learning is better than the previous approaches that rely on a separate
post-processing step to recover the object geometry [32,40].
This idea is ﬁrstly proposed in the recent work [41] for depth based hand pose
estimation and is shown working well. However, estimating 3D structure from
depth is a simple problem by nature. It is still unclear how well the idea can be
generalized to other objects and RGB images. In this work, we apply the idea to
more problems (a toy example and human pose estimation) and for the ﬁrst time
show that the idea works successfully on diﬀerent articulated pose estimation
problems and inputs, indicating that the idea works in general. Especially, for

Deep Kinematic Pose Regression
3
the challenging 3D human pose estimation from single view RGB images, we
present state-of-the-art results on the Human3.6M dataset [13].
2

## experiments
Fig. 4. Illustration of the toy problem. The input images are synthesized and binary.
Top: Motion parameter and joint representation of a simple object with 3 motion
parameters. Bottom: Example input images for 3 objects with diﬀerent complexity
levels. They have 6, 8, and 10 motion parameters, respectively.
The work in [41] applies the kinematic pose regression approach for depth
based 3D hand pose estimation and has shown good results. To verify the gen-
erality of the idea, we apply this approach for two more diﬀerent problems. The
ﬁrst is a toy example for simple 2D articulated object on synthesized binary
image. The second is 3D human pose estimation from single RGB images, which
is very challenging.
4.1
A Toy Problem
In the toy problem, the object is 2D. The image is synthesized and binary. As
shown in Fig. 4 top, the input image is generated from a 3 dimensional motion
parameter Θ = {x, y, θ}, where x, y is the image coordinate (normalized between
0−1) of the root joint, and θ indicates the angle between the each bone and the
vertical line.
We use a 5 layer convolutional neutral network. The network structure and
hyper-parameters are the same as [41]. The input image resolution is 128×128.
The bone length is ﬁxed as 45 pixels. We randomly synthesize 16k samples for
training and 1k samples for testing. Each model is trained for 50 epoches.
As described in Fig. 3, we perform our direct joint, kinematic joint and
direct parameter on this task. The joint location for direct parameter is
computed by the kinematic layer as a post process in testing. It turns out all the
3 methods achieve low joint errors in this simple case. The mean joint errors for
direct joint, kinematic Joint, direct parameter are 5.1 pixels, 4.9 pixels,
and 4.8 pixels, respectively. direct joint is the worst, probably because the task

Deep Kinematic Pose Regression
9
Fig. 5. Experimental results on mean joint locations error(Left) and mean angle er-
ror(Right) with respect to model complexity. It shows when as kinematic model be-
coming complex, our approach is stable in both metric.
is easy for all the setting and these two require to learn more parameters. When
we evaluate the average length of the two bones for direct joint regression,
we ﬁnd it has a standard deviation of 5.3 pixels (11.8% of the bone length 45
pixels), indicating that the geometry constraint is badly violated.
Since it is hard to claim any other signiﬁcant diﬀerence between the 3 method
in such a simple case, we gradually increase the model complexity. Global ori-
entation and more joint angles are added to the kinematic model. For each level
of complexity, we add one more bone with one rotational angle on each distal
bone. Example input image are illustrated in Fig. 4 bottom.
The joint location errors and angle errors with respect to the model com-
plexity are shown in Fig. 5. Note that for direct joint regression, the angles
are directly computed from the triangle. The results show that the task become
more diﬃcult for all methods. Direct parameter gets high joint location errors,
probably because a low motion parameter error does not necessarily implies a
low joint error. It is intuitive that it always get best performance on joint angle,
since it is the desired learning target. Direct joint regression also has large
error on its recovered joint angles, and the average length of each bone becomes
more unstable. It shows that geometry structure is not easy to learn. Using a
generative kinematic joint layer keeps a decent accuracy on both metric among
all model complexity. This is important for complex objects in real applications,
such as human body.
4.2
3D Human Pose Regression
We test our method on the problem of full 3D human pose estimation from single
view RGB images. Following
[17], the 3D coordinate of joints is represented
by its oﬀset to a root joint. We use Human 3.6M dataset [13]. Following the
standard protocol in
[13,17,39], we deﬁne J = 17 joints on the human body.
The dataset contains millions of frames of RGB images. They are captured over

10
Zhou et al.
Fig. 6. Illustration of Human Model. It contains 17 joints and 27 motion parameters.
See text for the detail kinematic structure.
7 subjects performing 15 actions from 4 diﬀerent camera views. Each frame
is accurately annotated by a MoCap system. We treat the 4 cameras of the
same subject separately. The training and testing data partition follows previous
works [13,17,40]. All frames from 5 subjects(S1, S5, S6, S7, S8) are used for
training. The remaining 2 subjects(S9, S11) are for testing.
Our kinematic human model is illustrated in Fig. 6. It deﬁnes 17 joints with
27 motion parameters. The pelvis is set as the root joint. Upside it is the neck,
which can roll and yaw among the root. Torso is deﬁned as the mid point of neck
and pelvis. It has no motion parameter. Pelvis and neck orientation determine
the positions of shoulders and hips by a ﬁxed bone transform. Each shoulder/hip
has full 3 rotational angles, and elbow/knee has 1 rotational angle. Neck also
has 3 rotational angles for nose and head orientation. Note that there can be
additional rotation angles on the model, for example shoulders can rotate among
neck within a subtle degree and elbows can roll itself. Our rule of thumb is to
simulate real human structure and keep the model simple.
We found that the ground truth 3D joints in the dataset has strictly the
same length for each bone across all the frames on the same subject. Also, the
lengths of the same bone across the 7 subjects are very close. Therefore, in our
human model, the bone lengths are simply set as the average bone lengths of
the 7 subjects. In addition, every subject is assigned a global scale. The scale is
computed from the sum bone lengths divided by the average sum bone length.
It is a ﬁxed constant for each subject during training. During testing, we assume
the subject scale is unknown and simply set it as 1. In practical scenarios, the
subject scale can be estimated by a calibrating pre processing and then ﬁxed.
Following [17,30], we assume the bounding box for the subject in known. The
input images are resized to 224 × 224. Note that it is important not to change
the aspect ratio for the kinematic based method, we use border padding to keep
the real aspect ratio. The training target is also normalized by the bounding
box size. Since our method is not action-dependent, we train our model using
all the data from the 15 actions. By contrast, previous methods [13,18,40] use

Deep Kinematic Pose Regression
11
Directions

## related_work
The literature on pose estimation is comprehensive. We review previous work
from two perspectives that are mostly related to our work: object pose represen-
tation and deep learning based human pose estimation.
2.1
Pose Representation
An object pose consists of a number of related points. The key for pose repre-
sentation is how to represent the mutual relationship or structural constraints
between these points. There are a few diﬀerent previous approaches.
Pictorial Structure Model Pictorial structure model [7] is one of the most
popular methods in early age. It represents joints as vertexes and joint relations
as edges in a non-circular graph. Pose estimation is formulated as inference
problems on the graph and solved with certain optimization algorithms. Its ex-
tensions [15,36,24] achieve promising results in 2D human estimation, and has
been extended to 3D human pose [2]. The main drawback is that the inference
algorithm on the graph is usually complex and slow.
Linear Dictionary A widely-used method is to denote the structural points
as a linear combination of templates or basis [34,39,40,16]. [16] represent 3D face
landmarks by a linear combination of shape bases [23] and expression bases [4].
It learns the shape, expression coeﬃcients and camera view parameters alter-
natively.
[34] express 3D human pose by an over-complex dictionary with a
sparse prior, and solve the sparse coding problem with alternating direction
method. [39] assign individual camera view parameters for each pose template.
The sparse representation is then relaxed to be a convex problem that can be
solved eﬃciently.
Linear Feature Embedding Some approaches learn a low dimensional
embedding [12,21,13,30] from the high dimensional pose. [12] applies PCA to
the labeled 3D points of human pose. The pose estimation is then performed
in the new orthogonal space. The similar idea is applied to 3D hand pose esti-
mation [21]. It uses PCA to project the 3D hand joints to a lower space as a
physical constraint prior for hand.
[30] extend the linear PCA projector to a
multi-layer anto-encoder. The decoder part is ﬁne-tuned jointly with a convolu-
tional neural network in an end-to-end manner. A common drawback in above
linear representations is that the complex object pose is usually on a non-linear
manifold in the high dimensional space that cannot be easily captured by a linear
representation.
Implicit Representation by Retrieval Many approaches [6,18,37] store
massive examples in a database and perform pose estimation as retrieval, there-
fore avoiding the diﬃcult pose representation problem. [6] uses a nearest neigh-
bors search of local shape descriptors. [18] proposes a max-margin structured

4
Zhou et al.
learning framework to jointly embed the image and pose into the same space,
and then estimates the pose of a new image by nearest neighbor search in this
space.
[37] builds an image database with 3D and 2D annotations, and uses
a KD-tree to retrieve 3D pose whose 2D projection is similar to the input im-
age. The performance of these approaches highly depends on the quality of the
database. The eﬃciency of nearest neighbor search could be an issue when the
database is large.
Explicit Geometric Model The most aggressive and thorough representa-
tion is to use an explicit and generative geometric model, including the motion
and shape parameters of the object [26,3]. Estimating the parameters of the
model from the input image(s) is performed by heavy optimization algorithms.
Such methods are rarely used in a learning based manner. The work in [41]
ﬁrstly uses a generative kinematic model for hand pose estimation in the deep
learning framework. Inspire by this work, we extend the idea to more object
pose estimation problems and diﬀerent inputs, showing its general applicability,
especially for the challenging problem of 3D human pose estimation from single
view RGB images.
2.2
Deep Learning on Human Pose Estimation
The human pose estimation problem has been signiﬁcantly advanced using deep
learning since the pioneer deep pose work [33]. All current leading methods
are based on deep neutral networks.
[35] shows that using 2D heat maps as
intermediate supervision can dramatically improve the 2D human part detection
results. [20] use an hourglass shaped network to capture both bottom-up and
top-down cues for accurate pose detection.
[10] shows that directly using a
deep residual network (152 layers) [9] is suﬃcient for high performance part
detection. To adopt these fully-convolutional based heat map regression method
for 3D pose estimation, an additional model ﬁtting step is used [40] as a post
processing. Other approaches directly regress the 2D human pose [33,5] or 3D
human pose [17,30,31]. These detection or regression based approaches ignore
the prior knowledge of the human model and does not guarantee to preserve the
object structure. They sometimes output geometrically invalid poses.
To our best knowledge, for the ﬁrst time we show that integrating a kinematic
object model into deep learning achieves state-of-the-art results in 3D human
pose estimation from single view RGB images.
3
Deep Kinematic Pose Estimation
3.1
Kinematic Model
An articulated object is modeled as a kinematic model. A kinematic model is
composed of several bones and joints. A bone is a segment of a ﬁxed length,
and a joint is the end point of a bone. One bone meets at another at a joint,
forming a tree structure. Bones can rotate among a conjunct joint. Without

Deep Kinematic Pose Regression
5
Fig. 2. A sample 2D kinematic model. It has 3 and 4 joints. The joint location is
calculated by multiplying a series of transformation matrices.
loss generality, one joint is considered as the root joint (For example, wrist for
human hand and pelvis for human body). The root deﬁnes the global position
and global orientation of the object.
For a kinematic model of J joints, it has J −1 bones. Let {li}J−1
i=1 be the
collection of bone lengths, they are ﬁxed for a speciﬁc subject and provided as
prior knowledge. For diﬀerent subjects, we assume they only diﬀer in a global
scale, i.e. ∀i, l′
i = s×li. The scale is also provided as prior knowledge, e.g. through
a calibration process.
Let the rotation angle of the i-th joint be θi, the motion parameter Θ includes
the global position p, global orientation o, and all the rotation angles, Θ =
{p, o} ∪{θi}J
i=1. The forward kinematic function is a mapping from motion
parameter space to joint location space.
F : {Θ} →Y
(1)
where Y is the coordinate for all joints, Y ∈R3×J for 3D object and Y ∈R2×J
for 2D object.
The kinematic function is deﬁned on a kinematic tree. An example is shown
in Fig. 2. Each joint is associated with a local coordinate transformation deﬁned
in the motion parameter, including a rotation from its rotation angles and a
translation from its out-coming bones. The ﬁnal coordinate of a joint is obtained
by multiplying a series of transformation matrices along the path from the root
joint to itself. Generally, the global position of joint u is
pu = (
Y
v∈P a(u)
Rot(θv) × Trans(lv))O⊤
(2)
where Pa(u) is the set of its parents nodes at the kinematic tree, and O is the ori-
gin in homogenous coordinate, i.e., O = [0, 0, 1]⊤for 2D and O = [0, 0, 0, 1]⊤for
3D. For 3D kinematic model, each rotation is assigned with one of the {X, Y, Z}
axis, and at each joint there can be multiple rotations. The direction of transla-
tion is deﬁned in the canonical local coordinate frame where the motion param-
eters are all zeros.

6
Zhou et al.
In
[41], individual bounds for each angle can be set as additional prior
knowledge for the objects. It is feasible for human hand since all the joints have
at most 2 rotation angles and their physical meaning is clear. However, in the
case of human body, angle constraint are not individual, it is conditioned on
pose [1] and hard to formulate. We leave it as future work to explore more
eﬃcient and expressive constraints.
As shown in Fig. 2, the forward kinematic function is continuous with respect
to the motion parameter. It is thus diﬀerentiable. As each parameter occurs in
one matrix, this allows easy implementation of back-propagation. We simply re-
place the corresponding rotational matrix by its derivation matrix and keep other
items unchanged. The kinematic model can be easily put in a neural network as
a layer for gradient descent-based optimization.
3.2
Deep Learning with a Kinematic Layer
We discuss our proposed approach and the other two baseline methods to learn
the pose of an articulated object. They are illustrated in Fig. 3. All three methods
share the same basic convolutional neutral network and only diﬀers in their
ending parts, which is parameter-free. Therefore, we can make fair comparison
between the three methods.
Now we elaborate on them. The ﬁrst method is a baseline. It directly es-
timates the joint locations by a convolutional neural network, using Euclidean
Loss on the joints. It is called dir

## conclusion
Eating
Greeting Phoning
Photo
Posing
Purchases
LinKDE [13]
132.71
183.55
132.37
164.39
162.12
205.94
150.61
171.31
Li et al [17]
-
148.79
104.01
127.17
-
189.08
-
-
Li et al [18]
-
136.88
96.94
124.74
-
168.68
-
-
Tekin et al [30]
-
129.06
91.43
121.68
-
162.17
-
-
Tekin et al [31]
132.71
158.52
87.95
126.83
118.37
185.02
114.69
107.61
Zhou et al [40]
87.36
109.31
87.05
103.16
116.18
143.32
106.88
99.78
Ours(Direct)
106.38
104.68
104.28
107.80
115.44
114.05
103.80
109.03
Ours(ModelFit)
109.75
110.47
113.98
112.17
123.66
122.82
121.27
117.98
Ours(Kinematic)
91.83
102.41
96.95
98.75
113.35
125.22
90.04
93.84
Sitting
SittingDown Smoking Waiting WalkDog Walking WalkPair
Average
LinKDE [13]
151.57
243.03
162.14
170.69
177.13
96.60
127.88
162.14
Li et al [17]
-
-
-
-
146.59
77.60
-
-
Li et al [18]
-
-
-
-
132.17
69.97
-
-
Tekin et al [30]
-
-
-
-
130.53
65.75
-
-
Tekin et al [31]
136.15
205.65
118.21
146.66
128.11
65.86
77.21
125.28
Zhou et al [40]
124.52
199.23
107.42
118.09
114.23
79.39
97.70
113.01
Ours(Direct)
125.87
149.15
112.64
105.37
113.69
98.19
110.17
112.03
Ours(ModelFit)
137.29
157.44
136.85
110.57
128.16
102.25
114.61
121.28
Ours(Kinematic)
132.16
158.97
106.91
94.41
126.04
79.02
98.96
107.26
Table 1. Results of Human3.6M Dataset. The numbers are mean Euclidean dis-
tance(mm) between the ground-truth 3D joints and the estimations of diﬀerent meth-
ods.
data for each action individually, as their local feature, retrieval database or pose
dictionary may prefer more concrete templates.
We use the 50-layer Residual Network [9] that is pre-trained on ImageNet [25]
as our initial model. It is then ﬁne-tuned on our task. Totally available training
data for the 5 subjects is about 1.5 million images. They are highly similar and
redundant. We randomly sample 800k frames for training. No data augmenta-
tion is used. We train our network for 70 epoches, with base learning rate 0.003
(dropped to 0.0003 after 50 epochs), batch size 52 (on 2 GPUs), weight decay
0.0002 and momentum 0.9. Batch-normalization [11] is used. Our implementa-
tion is based on Caﬀe [14].
The experimental results are shown in Table 1. The results for comparison
methods [13,17,18,30,30,31,40] are from their published papers. Thanks to the
powerful Residual Network [9], our direct joint regression base line is already
the state-of-the-art. Since we used additional training data from ImageNet, com-
paring our results to previous works is unfair, and the superior performance of
our approach is not the contribution of this work. We include the previous works’
results in Table 1 just as references.
Kinematic joint achieves the best average accuracy among all methods,
demonstrating that embedding a kinematic layer in the network is eﬀective.
Qualitative results are shown in Table 2, including some typical failure cases for
direct joint include ﬂipping the left and right leg when the person is back to
the camera(Row 1) and abnormal bone length(Row 2,3).
Despite direct joint regression achieve a decent accuracy for 3D joint loca-
tion, we can further apply a kinematic model ﬁtting step, as described in the

12
Zhou et al.
Fig. 7. Training curve of direct motion parameter regression. Although the training
loss keeps dropping, the testing loss remains high.
previous sections. The model ﬁtting is based on gradient-descent for each frame.
The results is shown in Table. 1 as ours(Fit), it turns out to be worse than di-
rect joint, indicating such post-preprocessing is sub-optimal if the initial poses
do not have valid structural information.
We also tried direct parameter regression on this dataset. The training
target for motion parameter is obtained in the same way as described above,
by gradient descent. However, as shown in Fig. 7, the testing error keeps high.
Indicating direct parameter regression does not work on this task. There could
be two reasons: many joints have full 3 rotational angles, this can easily cause
ambiguous angle target, for example, if the elbow or knee is straight, the roll
angle for shoulder or hip can be arbitrary. Secondly, learning 3D rotational angles
is more obscure than learning 3D joint oﬀsets. It is even hard for human to
annotate the 3D rotational angles from an RGB image. Thus it may require
more data or more time to train.
5