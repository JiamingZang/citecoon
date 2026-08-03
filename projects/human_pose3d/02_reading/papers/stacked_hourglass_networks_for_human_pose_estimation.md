# Stacked Hourglass Networks for Human Pose Estimation

> 2016 · id: W2307770531 · arXiv: 1603.06937 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
A key step toward understanding people in images and video is accurate pose
estimation. Given a single RGB image, we wish to determine the precise pixel
location of important keypoints of the body. Achieving an understanding of a
person’s posture and limb articulation is useful for higher level tasks like ac-
tion recognition, and also serves as a fundamental tool in ﬁelds such as human-
computer interaction and animation.
arXiv:1603.06937v2  [cs.CV]  26 Jul 2016

2
Newell et al.
As a well established problem in vision, pose estimation has plagued re-
searchers with a variety of formidable challenges over the years. A good pose
estimation system must be robust to occlusion and severe deformation, success-
ful on rare and novel poses, and invariant to changes in appearance due to factors
like clothing and lighting. Early work tackles such diﬃculties using robust im-
age features and sophisticated structured prediction [1–9]: the former is used
to produce local interpretations, whereas the latter is used to infer a globally
consistent pose.
This conventional pipeline, however, has been greatly reshaped by convolu-
tional neural networks (ConvNets) [10–14], a main driver behind an explosive
rise in performance across many computer vision tasks. Recent pose estimation
systems [15–20] have universally adopted ConvNets as their main building block,
largely replacing hand-crafted features and graphical models; this strategy has
yielded drastic improvements on standard benchmarks [1,21,22].
We continue along this trajectory and introduce a novel “stacked hourglass”
network design for predicting human pose. The network captures and consoli-
dates information across all scales of the image. We refer to the design as an
hourglass based on our visualization of the steps of pooling and subsequent up-
sampling used to get the ﬁnal output of the network. Like many convolutional
approaches that produce pixel-wise outputs, the hourglass network pools down
to a very low resolution, then upsamples and combines features across multiple
resolutions [15,23]. On the other hand, the hourglass diﬀers from prior designs
primarily in its more symmetric topology.
We expand on a single hourglass by consecutively placing multiple hourglass
modules together end-to-end. This allows for repeated bottom-up, top-down in-
ference across scales. In conjunction with the use of intermediate supervision,
repeated bidirectional inference is critical to the network’s ﬁnal performance.
The ﬁnal network architecture achieves a signiﬁcant improvement on the state-
of-the-art for two standard pose estimation benchmarks (FLIC [1] and MPII
Human Pose [21]). On MPII there is over a 2% average accuracy improvement
across all joints, with as much as a 4-5% improvement on more diﬃcult joints
like the knees and ankles. 1
2

## method
99.0
97.0
Table 1. FLIC results (PCK@0.2)
Fig. 7. PCKh comparison on MPII
Head Shoulder Elbow Wrist Hip Knee Ankle Total
Tompson et al. [16], CVPR’15
96.1
91.9
83.9
77.8 80.9 72.3
64.8
82.0
Carreira et al. [19], CVPR’16
95.7
91.7
81.7
72.4 82.8 73.2
66.4
81.3
Pishchulin et al. [17], CVPR’16 94.1
90.2
83.4
77.3 82.6 75.7
68.6
82.4
Hu et al. [27], CVPR’16
95.0
91.6
83.0
76.6 81.9 74.5
69.5
82.4
Wei et al. [18], CVPR’16
97.8
95.0
88.7
84.0 88.4 82.8
79.4
88.5

## experiments
Evaluation is done using the standard Percentage of Correct Keypoints (PCK)
metric which reports the percentage of detections that fall within a normalized
distance of the ground truth. For FLIC, distance is normalized by torso size, and
for MPII, by a fraction of the head size (referred to as PCKh).
FLIC: Results can be seen in Figure 6 and Table 1. Our results on FLIC are
very competitive reaching 99% PCK@0.2 accuracy on the elbow, and 97% on
the wrist. It is important to note that these results are observer-centric, which
is consistent with how others have evaluated their output on FLIC.

10
Newell et al.
Fig. 8.
Comparison of validation accuracy as training progresses. The accuracy is
averaged across the wrists, elbows, knees, and ankles. The diﬀerent network designs
are illustrated on the right, the circle is used to indicate where a loss is applied
MPII: We achieve state-of-the-art results across all joints on the MPII Hu-
man Pose dataset. All numbers can be seen in Table 2 along with PCK curves in
Figure 7. On diﬃcult joints like the wrist, elbows, knees, and ankles we improve
upon the most recent state-of-the-art results by an average of 3.5% (PCKh@0.5)
with an average error rate of 12.8% down from 16.3%. The ﬁnal elbow accuracy
is 91.2% and wrist accuracy is 87.1%. Example predictions made by the network
on MPII can be seen in Figure 5.
4.2
Ablation Experiments
We explore two main design choices in this work: the eﬀect of stacking hour-
glass modules together, and the impact of intermediate supervision. These are
not mutually independent as we are limited in how we can apply intermediate
supervision depending on the overall architectural design. Applied separately,
each has a positive impact on performance, and together we see a further im-
provements to training speed and in the end, ﬁnal pose estimation performance.
We look at the rate of training of a few diﬀerent network designs. The results
of which can be seen in Figure 8 which shows average accuracy on the valida-
tion set as training progresses. The accuracy metric considers all joints excluding
those associated with the head and torso to allow for easier diﬀerentiation across
experiments.
First, to explore the eﬀect of the stacked hourglass design we must demon-
strate that the change in performance is a function of the architecture shape and
not attributed to an increase in capacity with a larger, deeper network. To make
this comparison, we work from a baseline network consisting of eight hourglass
modules stacked together. Each hourglass has a single residual module at each
resolution as in Figure 3. We can shuﬄe these layers around for various network
arrangements. A decrease in the number of hourglasses would result in an in-
crease in the capacity of each hourglass. For example, a corresponding network
could stack four hourglasses and have two consecutive residual modules at each

Stacked Hourglass Networks for Human Pose Estimation
11
Fig. 9. Left: Example validation images illustrating the change in predictions from
an intermediate stage (second hourglass) (left) to ﬁnal predictions (eighth hourglass)
(right). Right: Validation accuracy at intermediate stages of the network compared
across diﬀerent stacking arrangements.
resolution (or two hourglasses and four residual modules). This is illustrated in
Figure 9. All networks share the same number of parameters and layers, though
a slight diﬀerence is introduced when more intermediate supervision is applied.
To see the eﬀect of these choices we ﬁrst compare a two-stacked network
with four residual modules at each stage in the hourglass, and a single hour-
glass but with eight residual modules instead. In Figure 8 these are referred
to as HG-Stacked and HG respectively. A modest improvement in training can
be seen when using the stacked design despite having approximately the same
number of layers and parameters. Next, we consider the impact of intermediate
supervision. For the two-stack network we follow the procedure described in the
paper to apply supervision. Applying this same idea with a single hourglass is
nontrivial since higher order global features are present only at lower resolutions,
and the features across scales are not combined until late in the pipeline. We
explore applying supervision at various points in the network, for example either
before or after pooling and at various resolutions. The best performing method
is shown as HG-Int in Figure 8 with intermediate supervision applied after up-
sampling at the next two highest resolutions before the ﬁnal output resolution.
This supervision does oﬀer an improvement to performance, but not enough to
surpass the improvement when stacking is included (HG-Stacked-Int).
In Figure 9 we compare the validation accuracy of 2-, 4-, and 8-stack mod-
els that share approximately the same number of parameters, and include the
accuracy of their intermediate predictions. There is a modest improvement in
ﬁnal performance for each successive increase in stacking from 87.4% to 87.8% to
88.1%. The eﬀect is more notable at intermediate stages. For example, halfway
through each network the corresponding accuracies of the intermediate predic-
tions are: 84.6%, 86.5%, and 87.1%. Note that the accuracy halfway through the
8-stack network is just short of the ﬁnal accuracy of the 2-stack network.
It is interesting to observe the mistakes made early and corrected later on
by the network. A few examples are visualized in Figure 9. Common mistakes
show up like a mix up of other people’s joints, or misattribution of left and

12
Newell et al.
Fig. 10. The diﬀerence made by a slight translation and change of scale of the input
image. The network determines who to generate an annotation for based on the central
ﬁgure. The scaling and shift right of the input image is enough for the network to switch
its predictions.
right. For the running ﬁgure, it is apparent from the ﬁnal heatmap that the
decision between left and right is still a bit ambiguous for the network. Given
the appearance of the image, the confusion is justiﬁed. One case worth noting is
the middle example where the network initially activates on the visible wrists in
the image. Upon further processing the heatmap does not activate at all on the
original locations, instead choosing a reasonable position for the occluded wrist.
5
Further Analysis
5.1
Multiple People
The issue of coherence becomes especially important when there are multiple
people in an image. The network has to decide who to annotate, but there
are limited options for communicating who exactly deserves the annotation.
For the purposes of this work, the only signal provided is the centering and
scaling of the target person trusting that the input will be clear enough to parse.
Unfortunately, this occasionally leads to ambiguous situations when people are
very close together or even overlapping as seen in Figure 10. Since we are training
a system to generate pose predictions for a single person, the ideal output in
an ambiguous situation would demonstrate a commitment to the joints of just
one ﬁgure. Even if the predictions are lower quality, this would show a deeper
understanding of the task at hand. Estimating a location for the wrist with a
disregard for whom the wrist may belong is not desired behavior from a pose
estimation system.
The results in Figure 10 are from an MPII test image. The network must
produce predictions for both the boy and girl, and to do so, their respective center
and scale annotations are provided. Using those values to crop input images
for the network result in the ﬁrst and third images of the ﬁgure. The center
annotations for the two dancers are oﬀby just 26 pixels in a 720x1280 image.
Qualitatively, the most perceptible diﬀerence between the two input images is
the change in scale. This diﬀerence is suﬃcient for the network to change its
estimate entirely and predict the annotations for the correct ﬁgure.

Stacked Hourglass Networks for Human Pose Estimation
13
Fig. 11. Left: PCKh curves on validation comparing performance when exclusively
considering joints that are visible (or not). Right: Precision recall curves showing the
accuracy of predicting whether an annotation is present for a joint when thresholding
on either the mean or max activation of a heatmap.
A more comprehensive management of annotations for multiple people is
out of the scope of this work. Many of the system’s failure cases are a result of
confusing the joints of multiple people, but it is promising that in many examples
with severe overlap of ﬁgures the network will appropriately pick out a single
ﬁgure to annotate.
5.2
Occlusion
Occlusion performance can be diﬃcult to assess as it often falls into two distinct
categories. The ﬁrst consists of cases where a joint is not visible but its position
is apparent given the context of the image. MPII generally provides ground
truth locations for these joints, and an additional annotation indicates their

## related_work
With the introduction of “DeepPose” by Toshev et al. [24], research on human
pose estimation began the shift from classic approaches [1–9] to deep networks.
Toshev et al. use their network to directly regress the x,y coordinates of joints.
The work by Tompson et al. [15] instead generates heatmaps by running an
image through multiple resolution banks in parallel to simultaneously capture
features at a variety of scales. Our network design largely builds oﬀof their work,
exploring how to capture information across scales and adapting their method
for combining features across diﬀerent resolutions.
1 Code is available at http://www-personal.umich.edu/~alnewell/pose

Stacked Hourglass Networks for Human Pose Estimation
3
Fig. 2. Example output produced by our network. On the left we see the ﬁnal pose
estimate provided by the max activations across each heatmap. On the right we show
sample heatmaps. (From left to right: neck, left elbow, left wrist, right knee, right
ankle)
A critical feature of the method proposed by Tompson et al. [15] is the joint
use of a ConvNet and a graphical model. Their graphical model learns typical
spatial relationships between joints. Others have recently tackled this in similar
ways [17,20,25] with variations on how to approach unary score generation and
pairwise comparison of adjacent joints. Chen et al. [25] cluster detections into
typical orientations so that when their classiﬁer makes predictions additional
information is available indicating the likely location of a neighboring joint. We
achieve superior performance without the use of a graphical model or any explicit
modeling of the human body.
There are several examples of methods making successive predictions for pose
estimation. Carreira et al. [19] use what they refer to as Iterative Error Feedback.
A set of predictions is included with the input, and each pass through the network
further reﬁnes these predictions. Their method requires multi-stage training and
the weights are shared across each iteration. Wei et al. [18] build on the work
of multi-stage pose machines [26] but now with the use of ConvNets for feature
extraction. Given our use of intermediate supervision, our work is similar in spirit
to these methods, but our building block (the hourglass module) is diﬀerent. Hu
& Ramanan [27] have an architecture more similar to ours that can also be used
for multiple stages of predictions, but their model ties weights in the bottom-up
and top-down portions of computation as well as across iterations.
Tompson et al. build on their work in [15] with a cascade to reﬁne predic-
tions. This serves to increase eﬃcency and reduce memory usage of their method
while improving localization performance in the high precision range [16]. One
consideration is that for many failure cases a reﬁnement of position within a
local window would not oﬀer much improvement since error cases often con-
sist of either occluded or misattributed limbs. For both situations, any further
evaluation at a local scale will not improve the prediction.
There are variations to the pose estimation problem which include the use
of additional features such as depth or motion cues. [28–30] Also, there is the
more challenging task of simultaneous annotation of multiple people [17,31]. In
addition, there is work like that of Oliveira et al. [32] that performs human part
segmentation based on fully convolutional networks [23]. Our work focuses solely
on the task of keypoint localization of a single person’s pose from an RGB image.

4
Newell et al.
Fig. 3. An illustration of a single “hourglass” module. Each box in the ﬁgure corre-
sponds to a residual module as seen in Figure 4. The number of features is consistent
across the whole hourglass.
Our hourglass module before stacking is closely connected to fully convolu-
tional networks [23] and other designs that process spatial information at mul-
tiple scales for dense prediction [15, 33–41]. Xie et al. [33] give a summary of
typical architectures. Our hourglass module diﬀers from these designs mainly
in its more symmetric distribution of capacity between bottom-up processing
(from high resolutions to low resolutions) and top-down processing (from low
resolutions to high resolutions). For example, fully convolutional networks [23]
and holistically-nested architectures [33] are both heavy in bottom-up process-
ing but light in their top-down processing, which consists only of a (weighted)
merging of predictions across multiple scales. Fully convolutional networks are
also trained in multiple stages.
The hourglass module before stacking is also related to conv-deconv and
encoder-decoder architectures [42–45]. Noh et al. [42] use the conv-deconv ar-
chitecture to do semantic segmentation, Rematas et al. [44] use it to predict re-
ﬂectance maps of objects. Zhao et al. [43] develop a uniﬁed framework for super-
vised, unsupervised and semi-supervised learning by adding a reconstruction loss.
Yang et al. [46] employ an encoder-decoder architecture without skip connections
for image generation. Rasmus et al. [47] propose a denoising auto-encoder with
special, “modulated” skip connections for unsupervised/semi-supervised feature
learning. The symmetric topology of these networks is similar, but the nature of
the operations is quite diﬀerent in that we do not use unpooling or deconv layers.
Instead, we rely on simple nearest neighbor upsampling and skip connections for
top-down processing. Another major diﬀerence of our work is that we perform
repeated bottom-up, top-down inference by stacking multiple hourglasses.
3
Network Architecture
3.1
Hourglass Design
The design of the hourglass is motivated by the need to capture information at
every scale. While local evidence is essential for identifying features like faces and

Stacked Hourglass Networks for Human Pose Estimation
5
hands, a ﬁnal pose estimate requires a coherent understanding of the full body.
The person’s orientation, the arrangement of their limbs, and the relationships
of adjacent joints are among the many cues that are best recognized at diﬀerent
scales in the image. The hourglass is a simple, minimal design that has the
capacity to capture all of these features and bring them together to output
pixel-wise predictions.
The network must have some mechanism to eﬀectively process and consoli-
date features across scales. Some approaches tackle this with the use of separate
pipelines that process the image independently at multiple resolutions and com-
bine features later on in the network [15,18]. Instead, we choose to use a single
pipeline with skip layers to preserve spatial information at each resolution. The
network reaches its lowest resolution at 4x4 pixels allowing smaller spatial ﬁlters
to be applied that compare features across the entire space of the image.
The hourglass is set up as follows: Convolutional and max pooling layers are
used to process features down to a very low resolution. At each max pooling
step, the network branches oﬀand applies more convolutions at the original
pre-pooled resolution. After reaching the lowest resolution, the network begins
the top-down sequence of upsampling and combination of features across scales.
To bring together information across two adjacent resolutions, we follow the
process described by Tompson et al. [15] and do nearest neighbor upsampling
of the lower resolution followed by an elementwise addition of the two sets of
features. The topology of the hourglass is symmetric, so for every layer present
on the way down there is a corresponding layer going up.
After reaching the output resolution of the network, two consecutive rounds
of 1x1 convolutions are applied to produce the ﬁnal network predictions. The
output of the network is a set of heatmaps where for a given heatmap the network
predicts the probability of a joint’s presence at each and every pixel. The full
module (excluding the ﬁnal 1x1 layers) is illustrated in Figure 3.
3.2
Layer Implementation
While maintaining the overall hourglass shape, there is still some ﬂexibility in the
speciﬁc implementation of layers. Diﬀerent choices can have a moderate impact
on the ﬁnal performance and training of the network. We explore several options
for layer design in our network. Recent work has shown the value of reduction
steps with 1x1 convolutions, as well as the beneﬁts of using consecutive smaller
ﬁlters to capture a larger spatial context. [12,14] For example, one can replace
a 5x5 ﬁlter with two separate 3x3 ﬁlters. We tested our overall network design,
swapping in diﬀerent layer modules based oﬀof these insights. We experienced
an increase in network performance after switching from standard convolutional
layers with large ﬁlters and no reduction steps to newer methods like the residual
learning modules presented by He et al. [14] and “Inception”-based designs [12].
After the initial performance improvement with these types of designs, various
additional explorations and modi

## conclusion
We demonstrate the eﬀectiveness of a stacked hourglass network for producing
human pose estimates. The network handles a diverse and challenging set of
poses with a simple mechanism for reevaluation and assessment of initial predic-
tions. Intermediate supervision is critical for training the network, working best
in the context of stacked hourglass modules. There still exist diﬃcult cases not
handled perfectly by the network, but overall our system shows robust perfor-
mance to a variety of challenges including heavy occlusion and multiple people
in close proximity.