# Deep Learning-Based Object Pose Estimation: A Comprehensive Survey

> 2025 · id: W4396914081 · arXiv: 2405.07801 · pdf: https://arxiv.org/pdf/2405.07801 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Object pose estimation is a fundamental computer vision problem with broad applications in augmented reality and robotics. 
Over the past decade, deep learning models, due to their superior accuracy and robustness, have increasingly supplanted 
conventional algorithms reliant on engineered point pair features. Nevertheless, several challenges persist in contemporary 
methods, including their dependency on labeled training data, model compactness, robustness under challenging conditions, 
and their ability to generalize to novel unseen objects. A recent survey discussing the progress made on different aspects of this 
area, outstanding challenges, and promising future directions, is missing. To ﬁll this gap, we discuss the recent advances in deep 
learning-based object pose estimation, covering all three formulations of the problem, i.e., instance-level, category-level, and 
unseen (including both instance-unseen and category-unseen cases) object pose estimation. Our survey also covers multiple 
input data modalities, degrees-of-freedom of output poses, object properties, and downstream tasks, providing the readers 
with a holistic understanding of this ﬁeld. Additionally, it discusses training paradigms of different domains, inference modes, 
application areas, evaluation metrics, and benchmark datasets, as well as reports the performance of current state-of-the-art 
methods on these benchmarks, thereby facilitating the readers in selecting the most suitable method for their application. 
Finally, the survey identiﬁes key challenges, reviews the prevailing trends along with their pros and cons, and identiﬁes 
promising directions for future research. We cover the literature up to our submission date and will continue to follow the 
latest works at https://github.com/CNJianLiu/Awesome-Object-Pose-Estimation.
Keywords Object pose estimation · Deep learning · Comprehensive survey · 3D computer vision
Communicated by Svetlana Lazebnik.
B 
jianliu@hnu.edu.cn
Wei Sun
wei_sun@hnu.edu.cn
Hui Yang
huiyang@hnu.edu.cn
Zhiwen Zeng
zingaltern@hnu.edu.cn
Chongpei Liu
chongpei56@hnu.edu.cn
Jin Zheng
zheng.jin@csu.edu.cn
Xingyu Liu
liuxy21@mails.tsinghua.edu.cn
Hossein Rahmani
h.rahmani@lancaster.ac.uk
Nicu Sebe
sebe@disi.unitn.it
Ajmal Mian
ajmal.mian@uwa.edu.au
1
School of Artiﬁcial Intelligence and Robotics, Hunan
University, Changsha 410082, Hunan, China
2
School of Architecture and Art, Central South University,
Changsha 410083, Hunan, China
3
Department of Automation, Tsinghua University, Beijing
100084, China
4
School of Computing and Communications, Lancaster
University, Lancaster LA1 4YW, United Kingdom
5
Department of Information Engineering and Computer
Science, University of Trento, Trento 38123, Italy
6
Department of Computer Science, The University of Western
Australia, Perth WA 6009, Australia
0123456789().: V,-vol 
123
B 
    Jian Liu

   81 
Page 2 of 45
International Journal of Computer Vision          (2026) 134:81

## introduction
Object pose estimation is a fundamental computer vision
problem that aims to estimate the pose of an object in a given
image relative to the camera that captured the image. Object
pose estimation is a crucial technology for augmented real-
ity (Liu et al., 2022f; He et al., 2022a; Wen et al., 2024),
robotic manipulation (Liu et al., 2023d, 2024a), hand-object
interaction (Lin et al., 2023b; Rezazadeh et al., 2023), etc.
Depending on the application needs, the object pose is esti-
mated up to varying degrees of freedom (DoF) such as
3DoF that only includes 3D rotation, 6DoF that additionally
includes 3D translation, or 9DoF which includes estimating
the 3D size of the object besides the 3D rotation and 3D
translation.
In the pre-deep learning era, many hand-crafted feature-
based approaches such as SIFT (Lowe, 2004), FPFH (Rusu
et al., 2009), VFH (Rusu et al., 2010), and Point Pair Features
(PPF) (Drost et al., 2010; Choi & Christensen, 2012; Choi
et al., 2013; Birdal & Ilic, 2015) were designed for object
pose estimation. However, these methods exhibit deﬁciencies
in accuracy and robustness when confronted with complex
scenes (Xiang et al., 2017; Wang et al., 2019a). These tra-
ditional methods have now been supplanted by data driven
deep learning-based approaches that harness the power of
deep neural networks to learn high-dimensional feature rep-
resentations from data, leading to improved accuracy and
robustness to handle complex environments.
Deeplearning-basedmethodscanbedividedintoinstance-
level, category-level, and unseen (including both instance-
unseen and category-unseen cases) object pose estimation
according to the problem formulation. Fig. 1 shows a com-
parison of the three methods. Early methods were mainly
instance-level (Wang et al., 2019a; Peng et al., 2019; He et
al., 2020; Li et al., 2019; Zakharov et al., 2019), trained to
estimate the pose of speciﬁc object instances. Instance-level
methods can be further divided into correspondence-based,
template-based, voting-based, and regression-based meth-
ods. Since instance-level methods are trained on instance-
speciﬁc data, they can estimate pose with high precision for
the given object instances. However, their generalization per-
formance is poor because they are meant to be applied only
to the instances on which they are trained. Moreover, many
instance-level methods (He et al., 2020, 2021) require CAD
models of the objects. Recognizing these limitations, Wang et
al. (2019b) proposed the ﬁrst category-level object pose and
size estimation method. They generalize to intra-class unseen
objects without necessitating retraining and employing CAD
models during inference. Subsequent category-level meth-
ods (Lin et al., 2022c; Di et al., 2022; Zheng et al., 2023;
Liu et al., 2023c, 2025b) can be divided into shape prior-
based and shape prior-free methods. While improving the
generalization ability within a category, these category-level
Fig. 1 Comparison of instance-level, category-level, and unseen
(including both instance-unseen and category-unseen cases) object pose
estimation methods. Instance-level methods can only estimate the pose
of speciﬁc object instances on which they are trained. Category-level
methods can generalize to unseen instances within a known category,
rather than being limited to the speciﬁc training instances. In contrast,
unseen object pose estimation methods aim for stronger generalization,
handling both novel instances and entirely new object categories not
encountered during training
methods still need to collect and label extensive training data
for each object category. Moreover, these methods cannot
generalize to unseen object categories. To this end, some
unseen object pose estimation methods have been recently
proposed (Liu et al., 2022f; Labbé et al., 2022; Nguyen et
al., 2024b; Lin et al., 2024a; Wen et al., 2024), which can
be further classiﬁed into CAD model-based and manual ref-
erence view-based methods. These methods further enhance
the generalization of object pose estimation, i.e., they can
be generalized to unseen objects without retraining. Never-
theless, they still need to obtain the object CAD model or
annotate a few reference images of the object.
Although signiﬁcant progress has been made in the area
of object pose estimation, several challenges persist in cur-
rent methods, such as the reliance on labeled training data,
difﬁcultyingeneralizingtonovelunseenobjects,modelcom-
pactness, and robustness in challenging scenarios. To enable
readers to swiftly grasp the current state-of-the-art (SOTA)
in object pose estimation and facilitate further research in
this direction, it is crucial to provide a thorough review of
all the relevant problem formulations. A close examination
of the existing academic literature reveals a signiﬁcant gap
when reviewing the various problem formulations in object
pose estimation. Current prevailing reviews (Hoque et al.,
2021; Marullo et al., 2023; Fan et al., 2022b; Du & Wang,
2021; Guan et al., 2024) tend to exhibit a narrow focus,
either conﬁned to particular input modalities (Marullo et al.,
2023; Fan et al., 2022b) or tethered to speciﬁc application
domains (Du & Wang, 2021; Guan et al., 2024). Further-
more, these reviews predominantly scrutinize instance-level
and category-level methods, thus neglecting the exploration
of the most practical problem formulation in the domain
which is unseen object pose estimation. This hinders readers
123

International Journal of Computer Vision          (2026) 134:81 
Page 3 of 45
   81 
Fig. 2 A taxonomy of this survey. Firstly, we review the datasets
and evaluation metrics used to evaluate object pose estimation. Next,
we review the deep learning-based methods by dividing them into
three categories: instance-level, category-level, and unseen object
pose estimation. Instance-level methods can be further classiﬁed into
correspondence-based, template-based, voting-based, and regression-
based methods. Category-level methods can be further divided into
shape prior-based and shape prior-free methods. Unseen object pose
estimation methods can be further classiﬁed into CAD model-based
and manual reference view-based methods
from gaining a comprehensive understanding of the area. For
instance, Fan et al. (2022b) provided valuable insights into
RGB image-based object pose estimation. However, their
focus is limited to a singular modality, hindering readers
from comprehensively understanding methods across var-
ious input modalities. Conversely, Du and Wang (2021)
exclusively examined object pose estimation within the con-
text of the robotic grasping task, which limits the readers to
understand object pose estimation only from the perspective
of a single speciﬁc application.
To address the above problems, we present here a com-
prehensive survey of recent advancements in deep learning-
based methods for object pose estimation. Our survey
encompasses all problem formulations, including instance-
level, category-level, and unseen object pose estimation,
aiming to provide readers with a holistic understanding of
this ﬁeld. Additionally, we discuss different domain train-
ing paradigms, application areas, evaluation metrics, and
benchmark datasets, as well as report the performance of
state-of-the-art methods on these benchmarks, aiding readers
in selecting suitable methods for their applications. Further-
more, we also highlight prevailing trends and discuss their
strengths and weaknesses, as well as identify key challenges
and promising avenues for future research. The taxonomy of
this survey is shown in Fig. 2. Our main contributions and
highlights are as follows:
• We present a comprehensive survey of deep learning-
based object pose estimation methods. This is the ﬁrst
survey that covers all three problem formulations in
the domain, including instance-level, category-level, and
unseen object pose estimation.
• Our survey covers popular input data modalities (RGB
images, depth images, RGBD images), the different
degrees of freedom (3DoF, 6DoF, 9DoF) in output poses,
object properties (rigid, articulated) for the task of pose
estimation as well as tracking. It is crucial to cover all
these aspects in a single survey to give a complete pic-
ture to readers, an aspect overlooked by existing surveys
which only cover a few of these aspects.
• We discuss different domain training paradigms, infer-
ence modes, application areas, evaluation metrics, and
benchmark datasets as well as report the performance
of existing SOTA methods on these benchmarks to help
readers choose the most appropriate ones for deployment
in their application.
• We highlight popular trends in the evolution of object
pose estimation techniques over the past decade and dis-
cuss their strengths and weaknesses. We also identify key
challenges that are still outstanding in object pose esti-
mation along with promising research directions to guide
future efforts.
The rest of this article is organi

## method
6DoF
rigid
reﬁnement
source
two-stage, S+P
general
-
-
97.0
123

   81 
Page 12 of 45
International Journal of Computer Vision          (2026) 134:81 
bounding box corners. Finally, they used the PnP algo-
rithm (Fischler & Bolles, 1981) to estimate the object pose.
Additionally, they employed a classiﬁer to determine the
pose range in real-time, addressing the issue of ambigu-
ity in symmetric objects. Tekin et al. (2018) proposed a
CNN network inspired by YOLO (Redmon et al., 2016)
to integrate object detection and pose estimation, directly
predicting the locations of the projected vertices of the 3D
object bounding box. Unlike Rad and Lepetit (2017) and
Tekin et al. (2018), Pavlakos et al. (2017) predicted the 2D
projections of predeﬁned semantic keypoints. Doosti et al.
(2020) introduced a compact model comprising two adap-
tive graph convolutional neural networks (GCNNs) (Kipf &
Welling, 2016), collaborating to estimate object and hand
poses. To further enhance the robustness of object pose esti-
mation, Song et al. (2020) employed a hybrid intermediate
representation to convey geometric details in the input image,
encompassing keypoints, edge vectors, and symmetry corre-
spondences. Liu et al. (2021a) proposed a multi-directional
feature pyramid network along with a method that calculates
object pose estimation conﬁdence by incorporating spatial
and plane information. Hu et al. (2021) introduced a single-
stage hierarchical end-to-end trainable network to address
pose estimation challenges associated with scale variations
in aerospace objects. In a recent development, Lian and Ling
(2023) increased the number of predeﬁned 3D keypoints to
enhance the establishment of correspondences. Moreover,
they devised a hierarchical binary encoding approach for
localizing keypoints, enabling gradual reﬁnement of corre-
spondences and transforming correspondence regression into
a more efﬁcient classiﬁcation task. To estimate transparent
object pose, Chang et al. (2021) used a 3D bounding box pre-
diction network and multi-view geometry. Their method ﬁrst
detects 2D projections of 3D bounding box vertices, and then
reconstructs 3D points based on the multi-view detected 2D
projections incorporating camera motion data. Additionally,
they introduced a generalized pose deﬁnition to address pose
ambiguity for symmetric objects. To enhance the efﬁciency
of pose estimation networks, Guo et al. (2023) integrated
knowledge distillation into object pose estimation by dis-
tilling the teacher’s distribution of local predictions into the
student network. Liu et al. (2023b) argued that differentiable
PnP strategies conﬂict with the averaging nature of the PnP
problem, resulting in gradients that may encourage the net-
work to degrade the accuracy of individual correspondences.
To mitigate this, they introduced a linear covariance loss,
which can be used for both sparse and dense correspondence-
based methods.
To mitigate vulnerability caused by large occlusions, Criv-
ellaro et al. (2017) used several control points to represent
each object part. Then, they predicted the 2D projections
of these control points to calculate the object pose. Some
researchers solved the occlusion problem by predicting
keypoints using small patches. Oberweger et al. (2018) pro-
cessed each patch separately to generate heatmaps and then
aggregated the results to achieve precise and reliable pre-
dictions. Additionally, they offered a straightforward but
efﬁcient strategy to resolve ambiguities between patches
and heatmaps during training. Hu et al. (2019) unveiled a
segmentation-driven pose estimation framework in which
every visible object part offers a local pose prediction
through 2D keypoint locations. Furthermore, Huang et al.
(2021) conceptualized 2D keypoint locations as probabilis-
tic distributions within the loss function and designed a
conﬁdence-based network.
Reducing the reliance on annotated real-world data is also
an important task. Some methods exploit geometric con-
sistency as additional information to alleviate the need for
annotation. Zhao et al. (2020) employed image pairs with
object annotations and relative transformation between view-
points to automatically identify objects’ 3D keypoints that
are geometrically and visually consistent. In addition, Yang
et al. (2021) used a keypoint consistency regularization for
dual-scale images with a labeled 2D bounding box. Using
semi-supervised learning, Liu et al. (2021b) developed a uni-
ﬁed framework for estimating 3D hand and object poses.
They constructed a joint learning framework that conducts
explicit contextual reasoning between hand and object rep-
resentations. To generate pseudo labels in semi-supervised
learning, they utilized the spatial-temporal consistency found
in large-scale hand-object videos as a constraint. Synthetic
data is also a way to solve the annotation problem. Georgakis
et al. (2019) reduced the need for expensive 3DoF pose anno-
tations by selecting keypoints and maintaining viewpoint and
modality invariance in RGB images and CAD model render-
ings. Sock et al. (2020) utilized self-supervision to minimize
the gap between synthetic and real data and enforced photo-
metric consistency across different object views to ﬁne-tune
the model. Further, Zhang et al. (2021) utilized the invari-
ance of geometry relations between keypoints across real
and synthetic domains to accomplish domain adaptation.
Thalhammer et al. (2021) introduced a specialized feature
pyramid network to compute multi-scale features, enabling
the simultaneous generation of pose hypotheses across vari-
ous feature map resolutions.
Overall, sparse correspondence-based methods can esti-
mate object pose efﬁciently. However, relying on only a few
control points can lead to sub-optimal accuracy.
3.1.2 Dense Correspondence Methods
Dense correspondence-based methods aim to establish dense
2D-3D or 3D-3D correspondences. They utilize a sig-
niﬁcantly larger number of correspondences compared to
sparse correspondence-based methods. This enables them
to achieve higher accuracy and handle occlusions more
123

International Journal of Computer Vision          (2026) 134:81 
Page 13 of 45
   81 
effectively. Speciﬁcally, for the RGB image, they lever-
age every pixel or multiple patches to generate pixel-wise
correspondences, while for the point cloud, they use the
entire point cloud to ﬁnd point-wise correspondences. Li
et al. (2019) argued for the differentiation between rotation
and translation, proposing the coordinates-based disentan-
gled pose network. This network separates pose estima-
tion into distinct predictions for rotation and translation.
Zakharov et al. (2019) introduced the dense multi-class 2D-
3D correspondence-based object pose detector and a tailored
deep learning-based reﬁnement process. In addition, Cai and
Reid (2020) proposed a technique to automatically identify
and match image landmarks consistently across different
views, aiming to enhance the process of learning 2D-3D
mapping. Wang et al. (2021a) developed a pose estimation
pipeline guided by reconstruction, capitalizing on geomet-
ric consistency. Further, Shugurov et al. (2021) built upon
Zakharov et al. (2019) by developing a uniﬁed deep net-
work capable of accommodating multiple image modalities
(such as RGB and Depth) and integrating a differentiable
rendering-based pose reﬁnement method. Su et al. (2022)
introduced a discrete descriptor realized by hierarchical
binary grouping, capable of densely representing the object
surface. As a result, this method can predict ﬁne-grained cor-
respondences. Chen et al. (2022a) introduced a probabilistic
PnP (Fischler & Bolles, 1981) layer designed for general
end-to-end pose estimation. This layer generates a pose dis-
tribution on the SE(3) manifold. On the other hand, Xu et al.
(2022) argued that encoding pose-sensitive local features and
modeling the statistical distribution of inlier poses are cru-
cial for accurate and robust 6DoF pose estimation. Inspired
by PPF (Drost et al., 2010), they exploited pose-sensitive
information carried by each pair of oriented points and an
ensemble of redundant pose predictions to achieve robust
performance on severe inter-object occlusion and systematic
noises in scene point clouds.
Some methods recover object poses by establishing 3D-
3D correspondences. Huang et al. (2022) used an RGB image
to predict 3D object coordinates in the camera frustum,
thus establishing 3D-3D correspondences. Further, Jiang et
al. (2023) introduced a center-based decoupled framework,
leveraging bird’s eye and front views for object center voting.
They utilized feature similarity between the center-aligned
object and the object CAD model to establish correspon-
dencesforSingularValueDecomposition(SVD)-based(Besl
& McKay, 1992) rotation estimation. More recently, Lin et
al. (2024d) utilized an RGBD image as input and employed
point-to-surface matching to estimate the object surface
corresp

## conclusion
In
general,
the
aforementioned
correspondence-based methods exhibit robustness to occlu-
sion since they use local correspondences to predict object
pose. However, these methods may encounter challenges
when handling objects that lack salient shape features or tex-
ture.
3.2 Template-Based Methods
By leveraging global information from the image, template-
based methods can effectively address the challenges posed
by texture-less objects. Template-based methods involve
identifying the most similar template from a set of tem-
plates labeled with ground-truth object poses. They can
be categorized into RGB-based template (Sec. 3.2.1) and
point cloud-based template (Sec. 3.2.2) methods. These two
methods are illustrated in Fig. 5. The characteristics and per-
formance of some representative methods are shown in Table
1.
3.2.1 RGB-Based Template Methods
When the input is an RGB image, the templates comprise 2D
projections extracted from object CAD models, with annota-
tions of ground-truth poses. This process reformulates object
pose estimation as an image retrieval task. As a seminal
contribution, Sundermeyer et al. (2018) achieved 3D rota-
tion estimation through a variant of denoising autoencoder,
which learns an implicit representation of object rotation. If
depth is available, it can be used for pose reﬁnement. Liu
et al. (2019a) developed a CNN akin to an autoencoder to
reconstruct arbitrary scenes featuring the target object and
extract the object area. In addition, Zhang et al. (2020b) uti-
lized an object detector and a keypoint extractor to simplify
the template search process. Papaioannidis et al. (2020) sug-
gested that estimating object poses in synthetic images is
more straightforward. Therefore, they employed a generative
adversarial network to convert real images into synthetic ones
while preserving the object pose. Li and Ji (2020) utilized a
new pose representation (i.e., 3D location ﬁeld) to guide an
auto-encoder to distill pose-related features, thereby enhanc-
ing the handling of pose ambiguity. Stevšiˇc and Hilliges
(2020) proposed a spatial attention mechanism to identify
and utilize spatial details for pose reﬁnement. Different from
the above methods, Deng et al. (2021) addressed the 6DoF
object pose tracking problem within the Rao-Blackwellized
particle ﬁltering (Doucet et al., 2001) framework. They ﬁnely
discretized the rotation space and trained an autoencoder net-
work to build a codebook of feature embeddings for these
discretized rotations. This method efﬁciently estimates the
3D translation along with the full distribution over the 3D
rotation.
RGB cameras are widely used as visual sensors, yet they
struggle to capture sufﬁcient information under poor lighting
conditions. This results in poor pose estimation performance.
3.2.2 Point Cloud-Based Template Methods
With the popularity of consumer-grade 3D cameras, point
cloud-based methods take full advantage of their ability to
adapt to poor illumination and capture geometric informa-
tion.Whendealingwithapointcloud,thetemplatecomprises
theobjectCADmodelincanonicalpose.Notably,weclassify
the methods that directly regress the relative pose between
the object CAD model and the observed point cloud as
template-