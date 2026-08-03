# A survey on deep 3D human pose estimation

> 2024 · id: W4404703236 · pdf: https://link.springer.com/content/pdf/10.1007/s10462-024-11019-3.pdf · 来源: pdf_url
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

Accepted: 6 November 2024 / Published online: 25 November 2024
© The Author(s) 2024
	
 Kan Li
likan@bit.edu.cn
Rama Bastola Neupane
rama@bit.edu.cn
Tesfaye Fenta Boka
tfenta@bit.edu.cn
1	
School of Computer Science and Technology, Beijing Institute of Technology, Beijing, China
A survey on deep 3D human pose estimation
Rama Bastola Neupane1 · Kan Li1 · Tesfaye Fenta Boka1
Artificial Intelligence Review (2025) 58:24
https://doi.org/10.1007/s10462-024-11019-3
Abstract
3D Human Pose Estimation (3D-HPE) is a highly active and evolving research area in 
computer vision with numerous applications such as extended reality, action recognition, 
and video surveillance. The field has significantly advanced with deep learning, public da­
tasets, and enhanced computational power, addressing challenges like depth ambiguity, oc­
clusion, and data scarcity. Researchers confront scenario-specific issues such as ill-posed 
problems in monocular setups, cross-view aggregation with camera synchronizations in 
multi-view systems, and inter-person occlusion in multi-person scenarios. This survey 
comprehensively reviews contemporary strategies covering a technological spectrum in­
cluding Convolutional Neural Networks, Graph Convolutional Networks, Transformers, 
and their combinations employed to address these challenges. It includes scenarios such 
as monocular and multi-view setups, single and multi-person cases, as well as image and 
video inputs. The survey explores various solution paradigms, including single-stage vs 
2D-to-3D lifting, absolute vs relative keypoints, pixel vs voxel vs Neural Radiance Field 
spaces, and deterministic, probabilistic, or diffusion-based strategies, along with top-down 
vs bottom-up approaches. It examines advanced learning techniques beyond supervised 
methods and data augmentation for diverse pose datasets. It analyzes the performance of 
recent methods on benchmark datasets for different scenarios. Challenges are categorized 
into common and scenario-specific issues, and future research directions are proposed 
to foster further advancements in the field. Additionally, key sections are summarized in 
tables or visual formats for quick understanding. This survey is a valuable resource and 
a solid reference for researchers in the dynamic landscape of 3D human pose estimation.
Keywords  3D human pose estimation · Single-person · Multi-person · Data 
augmentation · Learning techniques · Scenario-specific challenges
1 3

R. B. Neupane et al.
1  Introduction
3D Human Pose Estimation (3D-HPE) is one of the most active and dynamic fields in 
computer vision aiming to predict 3D coordinates of articulated human joints from images 
or video frames (Zhang et al. 2023a; Shuai et al. 2023; Zhang et al. 2023b). When a 3D 
scene with a person is projected onto a 2D image, both the positional and depth informa­
tion of all joints are lost. The ultimate aim of 3D pose estimation is to accurately determine 
their 2D positions in the images and their depth within the camera’s coordinate system 
(Ci et al. 2022). Applications of 3D-HPE are diverse, spanning human-computer interac­
tion (Huo et al. 2023), action recognition(Rajasegaran et al. 2023), extended reality (XR) 
(Zhang 2012), autonomous vehicles (Lamas et al. 2022), video surveillance (Matsukawa et 
al. 2020), gaming (Tuyls et al. 2021), and medical fields (Liu et al. 2023). The performance 
of these applications relies heavily on the precision of 3D-HPE, making it a vital area of 
study in computer vision. The advancements in deep learning techniques, availability of 
large-scale 3D pose datasets such as Human3.6M (Ionescu et al. 2014), and computational 
resources have accelerated research in this area, leading to significant contributions from 
academia and industry.
The 3D-HPE problem depends on the number of sensors capturing the images and indi­
viduals in the scene, as both affect estimation performance. Based on the number of cam­
eras, input data can be either monocular (single camera) or multi-view (multiple cameras), 
hence the study of 3D-HPE can be categorized into four scenarios: Monocular images, 
Monocular videos, Multi-view images, and Multi-view videos. These scenarios can further 
involve single-person or multi-person pose estimation, leading to eight distinct scenarios. 
Consequently, each scenario presents unique challenges alongside common issues such as 
depth ambiguity, occlusion, inadequate data, changes in appearance (such as clothing and 
lighting), and complexity of the articulated human body. Capturing an RGB image is much 
easier and more affordable in uncontrolled environments, however, depth ambiguity and 
self-occlusion are major issues in monocular images (Ci et al. 2022; Xu et al. 2022). Monoc­
ular video requires methods to extract and utilize temporal information (Zheng et al. 2021; 
Honari et al. 2023) from frame-to-frame relationships for better pose estimation. Multi-view 
Images are better at handling self-occlusion and depth ambiguity through triangulation or 
similar techniques (Bartol et al. 2022; Wan et al. 2023), however, they require precise cam­
era calibration and synchronization (Shuai et al. 2023).
Moreover, we cannot directly extend the methods designed for single-person to multi-
persons due to new challenges in multi-person scenarios such as intra-person occlusion and 
requirements of camera-centric coordinates (Cheng et al. 2023). Researchers have devel­
oped various methods to address these challenges, often leveraging deep learning architec­
tures such as convolutional neural networks (CNNs) (Li et al. 2020b), graph convolution 
networks (GCN) (Kipf and Welling 2017), transformers (Vaswani et al. 2017), and their 
combinations. Moreover, online/offline data augmentation techniques and learning tech­
niques beyond supervision have been utilized to handle the data scarcity problem mak­
ing the model more adaptive to natural scenarios. Continuous advancements in this field 
will likely lead to more accurate and efficient models, benefiting various technological and 
scientific domains. This survey provides a comprehensive overview of the current state 
of 3D-HPE techniques, highlighting common and scenario-specific challenges, and future 
research directions.
1 3
24 
Page 2 of 53

A survey on deep 3D human pose estimation
1.1  Previous survey
The early review work (Sarafianos et al. 2016) offers a taxonomy of methods depending on 
input types such as images or image sequences and single or multiple views scenarios. It 
covers both traditional and a few deep learning-based approaches. The study also introduces 
a novel synthetic dataset, SynPose300, and evaluates several state-of-the-art techniques on 
this dataset. The review work (Gamra and Akhloufi 2021) categorizes 2D single-person 
pipelines into regression-based and detection-based methods and 3D approaches into one-
stage and two-stage methodologies. It also examines multi-person pipelines through top-
down and bottom-up approaches. The survey (Desmarais et al. 2021) reviews 3D-HPE 
techniques based on the network backbone architecture and input types. The performance of 
those methods is compared based on their accuracy, speed, and robustness. The work (Wang 
et al. 2021) reviews existing deep learning approaches for 3D-HPE across a spectrum of 
input types, number of individuals, and direct 3D estimation versus lifting from 2D to 3D 
pose. The review work (Liu et al. 2022) focuses on monocular 2D and 3D HPE. The work 
(Yan et al. 2022) reviews the 3D HPE from classical methods to deep learning methods and 
includes learning methods. The survey (Tian et al. 2023) centers on categorizing monocular 
3D human mesh recovery techniques according to their design paradigms, reconstruction 
detail, and application contexts. The review (Zheng et al. 2023) examines contemporary 
deep learning approaches for 2D and 3D pose estimation and compares methods based on 
their input data and inference procedures.
The field of 3D-HPE is advancing quickly, requiring frequent reviews of new methods 
and trends. Survey papers are essential for understanding the progress in this domain, as 
they offer comprehensive overviews of techniques available at the time of their publication. 
This work aims to deliver a thorough and well-structured review of the latest developments 
in 3D-HPE, building on previous surveys and providing valuable insights into recent tech­
nological advancements.
1.2  Scope of the study
Most survey papers highlight common challenges and suggest future directions in the field. 
While data scarcity and generalization issues are frequently mentioned among these chal­
lenges, there is often little focus on the strategies used by existing techniques to address 
these problems. Our review addresses this gap by exploring and organizing solutions such 
as advanced learning algorithms beyond supervised learning and online/offline pose data 
augmentation techniques. This structured presentation aims to provide readers with a clear 
and practical understanding of how these strategies can mitigate the issues of data scarcity 
and generalization.
With the advancement of machine learning algorithms, problem-solving approaches 
have shifted from CNN-based methods to hybrid algorithms such as graph-transformers, 
solution spaces from pixel-based to voxel/ NeRF (Neural Radiance Fields), determinis­
tic to probabilistic multi-hypotheses and recently diffusion-based approaches. We focus on 
these latest trends and perspectives, thoroughly reviewing recent papers from top journals 
and conferences in computer vision, covering the period from 2020 to March 2024. A few 
relevant studies beyond this period are also included. Our review includes comprehensive 
summaries of key sections, presented with tabular or visual illustrations for clarity.
1 3
Page 3 of 53 
24

R. B. Neupane et al.
Additionally, we have extensively studied existing problems, categorizing them into com­
mon challenges and scenario-specific issues, and providing future directions with actionable 
recommendations for research. We believe these insights will significantly benefit readers 
and help them make rapid progress in the outlined areas. Figure 1 illustrates the focus area 
of this survey, which centers on estimating 3D human pose from RGB images. The main 
contributions of this survey are summarized as follows:
1.	 Unlike previous works, we begin by examining various recent problem-solving strate­
gies (Section 2) and then review cutting-edge deep learning approaches for different 
3D-HPE scenarios (Sections 3 and 4). This approach provides a comprehensive under­
standing of the rapidly evolving field of 3D-HPE.
2.	 We specifically explore learning strategies beyond traditional supervision, including 
semi-, weakly-, unsupervised, and self-supervised approaches. Additionally, we review 
pose data augmentation techniques into offline and online categories. These approaches 
are highly regarded by researchers for addressing the common challenges of data inad­
equacy and diversity in 3D-HPE.
3.	 We present the performance of state-of-the-art methods across benchmark datasets, 
highlighting comparisons of different deep learning algorithms across various sce­
narios. We uniquely categorize the challenges into common and scenario-specific 
issues, offering future directions with actionable recommendations.The rest of this 
survey is organized as follows: Sect. 2 categorizes 3D-HPE strategies based on prob­
lem-solving stages, joint keypoint coordinates, multi-person paradigms, probabilistic/
diffusion-based approaches, and solution spaces. Sections 3 and 4 summarize state-
of-the-art techniques for single-person and multi-person scenarios, respectively. Sec­
tion 5 describes learning techniques beyond supervised learning, while Sect. 6 presents 
benchmark datasets, online/offline pose data augmentation techniques and evaluation 
metrics. Section 7 discusses the performance of various categories of 3D HPE mod­
els on benchmark datasets. Section 8 categorizes challenges in the field of 3D-HPE 
into common and scenario-specific challenges and suggests potential future directions. 
Finally, Sect. 9 concludes the study.
Fig. 1  Taxonomic landscape of 3D human pose estimation survey
 
1 3
24 
Page 4 of 53

A survey on deep 3D human pose estimation
2  3D-HPE problem-solving strategies
This section examines the problem-solving strategies utilized by current methods for 
3D-HPE. These methods are analyzed based on several factors: the number of stages in the 
solution process, which includes end-to-end single-stage approaches or two-stage methods 
involving 2D-to-3D lifting; the coordinate system for keypoint estimation, whether absolute 
or relative; multi-person paradigms that address scenarios with multiple individuals; deter­
ministic or probabilistic and diffusion-based approaches; the problem-solving space, which 
can involve pixel space, voxel space, or NeRF; and emerging trends and needs. Before 
delving into the technical specifics, Table 1 provides brief definitions of key terms to better 
understand the concept.
2.1  Problem-solving stage
Based on problem-solving stages, the existing 3D-HPE methods are categorized into sin­
gle-stage or end-to-end and two-stage or 2D to 3D lifting paradigms. End-to-end learning 
approache aims to directly estimate 3D poses from images using methods such as coor­
dinate regression, nearest neighbor matching between images and poses, or classification 
over a set of pose classes. In the two-stage approach, 2D pose-aware features or directly the 
2D human pose are utilized as intermediate inputs to estimate 3D coordinates. The field of 
2D-HPE has made significant advancements due to the availability of large-scale datasets 
and advancements in deep learning techniques. Building on these developments, research­
ers have applied state-of-the-art 2D-HPE networks, such as Stack Hourglass (Newell et al. 
2016), Cascaded Pyramid Network (CPN) (Chen et al. 2018), and High-Resolution Net­
work (HRNet)(Sun et al. 2019), to recover a 3D human pose. The two-stage strategy miti­
gates the risk of overfitting to small 3D datasets and enables the utilization of existing 2D 
pose datasets such as MPII (Andriluka et al. 2014) and COCO (Lin et al. 2014). Additional 
image information such as texture and semantic details can aid in accurately predicting 3D 
poses from 2D poses (Zhou et al. 2024a).
2.1.1  Single-stage or End-to-end
Umar et al. (2020) propose to utilize multi-view data to ensure spatial consistency without 
costly 3D pose annotations for 3D-HPE from monocular images. LCR-Net++ (Rogez et 
al. 2020) and UniPose+ (Artacho and Savakis 2022) present end-to-end method for 2D 
and 3D pose estimation. These methods incorporate depth regression directly into the pose 
estimation network. Wang et al. (2021b) introduce a Multi-view Pose Transformer model 
for multi-person 3D pose estimation. This model uses a direct regression method with a 
hierarchical joint query embedding scheme and projective attention mechanism. Despite its 
superior performance and speed, it struggles with data scarcity and varying camera setups. 
Reddy et al. (2021) address this issue with TesseTrack, which simultaneously addresses 3D 
joint reconstruction and person association in space and time using a spatio-temporal for­
mulation in a common voxelized feature space. Honari et al. (2023) employ contrastive self-
supervised learning for single-stage human pose estimation from RGB video. The research 
work (Luvizon et al. 2023) addresses scalability issues that employ a scalable sequential 
1 3
Page 5 of 53 
24

R. B. Neupane et al.
Key terms
Brief concepts
Convolutional Neural 
Networks (CNNs)
Mostly used to extract features from 2D 
images or video frames to predict the 2D 
location of body keypoints or estimate depth 
maps, which are then used to infer 3D poses
Data Augmentation
Involves artificially increasing the size and 
diversity of the dataset by applying transfor­
mations to images, such as rotation, flipping, 
scaling, or altering keypoint positions. This 
technique enhances the ability of the model 
to generalize, enabling pose estimation mod­
els to effectively manage different human 
poses, body shapes, and camera perspectives
Diffusion Models
These are generative models that synthesize 
human poses by progressively refining ran­
dom noise into a plausible human skeleton. 
These models aim to predict missing pose 
data or generate diverse pose samples
Graph Convolutional 
Networks (GCNs)
Useful for modeling the skeleton as a graph 
where joints are nodes, and bones are edges, 
learning how the spatial configuration of 
these joints evolves. GCNs are particularly 
effective in inferring the 3D pose from 2D 
pose keypoints by learning joint relationships
Hybrid Approach
In 3D-HPE, a hybrid approach combines 
different methods (e.g., CNNs, GCNs, or 
transformers) to better capture both local and 
global relationships between body joints. 
Top-down and bottom-up approaches are 
also combined to improve the performance 
of multi-person pose estimation
Neural Radiance 
Field (NeRF)
NeRF is a neural rendering technique to 
model and synthesize realistic 3D scenes 
from 2D images. It works by learning a 
volumetric scene representation where each 
point in 3D space is associated with color 
and density. In 3D-HPE, NeRF can be used 
to represent a human body’s volumetric 
scene in 3D by learning the appearance and 
geometry from a set of 2D images
Transformer
In 3D-HPE, transformers can model long-
range dependencies between body joints. 
They use self-attention mechanisms to focus 
on how joints relate to each other spatially 
and temporally across video frames
Voxel
In 3D-HPE, voxels represent the 3D volume 
of the human body or parts of the body on 
a grid of 3D units called volumetric pixels 
(voxels). Each voxel contains information 
about whether it is occupied by part of the 
human body and may carry appearance 
information such as color or texture
Table 1  Glossary of key 
terminologies
 
1 3
24 
Page 6 of 53

A survey on deep 3D human pose estimation
pyramid network for regressing the pose at multiple scales using a sequential coarse-to-fine 
approach.
Pros and cons of single-stage paradigm: 2D pose estimates tend to be quite noisy in 
real-world applications, which can significantly and irreversibly affect the subsequent 3D 
pose estimation step (Zhang et al. 2023c). Single-stage methods, which directly estimate 
3D poses from 2D images without relying on intermediate 2D estimators, avoid this issue. 
However, the focus of convolutional operations on local features and the lack of intermedi­
ate processing make these methods computationally intensive (Chen et al. 2024) and hence 
unsuitable for real-time processing. Additionally, single-stage methods require 3D annota­
tions for training so the scarcity of large-scale labeled datasets in natural environments can 
lead to overfitting and poor generalization of the pose estimator. The absence of intermedi­
ate supervision also makes these methods susceptible to variations in background and light­
ing, and the complexity of features makes the learning process for a single model extremely 
difficult (Zhang et al. 2023b). Moreover, directly regressing from an image to a 3D human 
pose is a highly nonlinear problem, leading to a large estimation model, a vast solution 
search space, and a high likelihood of sub-optimal solutions. Therefore, lifting the predicted 
2D human pose to a 3D pose has emerged as a viable direction within the two-stage methods 
in this field (Nie et al. 2023).
2.1.2  2D-to-3D lifting
Depending on the type of intermediate input used for 3D coordinate estimation, two-stage 
methods for 3D human pose estimation can be classified into 2D Pose-Aware Feature-Based 
and 2D Pose-Based lifting methods. 2D Pose-Aware Feature-Based Lifting methods utilize 
2D pose-aware features derived from the detected 2D human pose, such as heatmaps, part 
affinity fields, or other keypoints descriptors. These features capture the spatial and con­
textual information from the 2D pose, which is then used to estimate the 3D coordinates 
(Pavlakos et al. 2017; Wang et al. 2020; Liu et al. 2020a; Kundu et al. 2020a; Remelli et al. 
2020; Benzine et al. 2020; Wu and Xiao 2020; Tu et al. 2020). The work (Liu et al. 2021) 
uses 2D poses as input and employs an attention-based temporal convolutional neural net­
work to estimate 3D poses. Zhan et al. (2022) transform the input from 2D-pixel space to 
3D rays in a normalized coordinate system, mitigating variations due to changes in camera 
intrinsic parameters and pitch angle.
Some research works (Wandt et al. 2021; Cheng et al. 2021a; Wehrbein et al. 2021; 
Kundu et al. 2022; Usman et al. 2022) incorporate 2D pose confidence scores and heatmaps 
to estimate 3D poses. Such methods use high-confidence joints to correct the positions of 
low-confidence, uncertain joints, enhancing overall prediction accuracy. Kim et al. (2024) 
utilize the confidence levels of joints in 2D poses to create pseudo-labels. The model gener­
ates 3D pseudo ground-truths by calculating a weighted average for each joint based on the 
confidence of the 2D poses, which are then used for self-supervised learning.
2D Pose-Based Lifting methods (Zhou et al. 2017; Pavllo et al. 2019; Liu et al. 2020b; 
Shuai et al. 2023) directly use the 2D human pose, consisting of the detected keypoints or 
joints in 2D space, as the intermediate input. The spatial configuration of these 2D joints 
is leveraged to infer the corresponding 3D positions, aiming to reconstruct the 3D pose 
from the 2D joint locations. The research work (Xu et al. 2020) refines 2D poses using 
perspective projection and estimates 3D poses by applying kinematic constraints. Zhang et 
1 3
Page 7 of 53 
24

R. B. Neupane et al.
al. (2020) leverage pose geometry, including poses and viewpoints, to lift 2D poses into 3D 
poses. AdaFuse (Zhang et al. 2021) utilizes a 2D pose method to generate heatmaps for each 
view and fuse them using epipolar geometry prior to estimating the 3D pose.
Pros and cons of 2D-to-3D lifting: The task of estimating 2D human poses from images 
is a well-solved problem (Nie et al. 2023), providing a clear visualization of joint positions 
and aligning well spatially with the 3D pose, making it robust against image distortions such 
as varying backgrounds, clothing, and colors, given sufficient ground-truth (GT) data. This 
approach can significantly reduce uncertainty and semantic loss (Lee et al. 2023). Addition­
ally, there are many 2D labels available, especially in-the-wild, because labeling 2D poses 
is easier and cheaper than capturing 3D ground truth (Nie et al. 2023). The widespread 
availability of 2D human pose detectors and the lightweight nature of 2D skeleton repre­
sentations have made lifting-based methods prevalent in 3D pose estimation (Zhao et al. 
2023b). However, the 2D-to-3D pipeline involves several variable factors, such as the num­
ber of views, video sequence length, and whether camera calibration is used (Shuai et al. 
2023). Moreover, the 2D-to-3D approach inherits errors from the 2D model and increases 
computational complexity by sequentially calculating depth in a second step, rather than 
estimating depth concurrently.
2.2  Joint keypoint coordinates
Existing 3D-HPE methods predict 3D poses either relative to the root body joints or the 
camera coordinates. Based on the reference coordinate system, these methods are classified 
into person-centric and camera-centric. Some models estimate keypoints in both person-
centric and camera-centric coordinates to compare their performance with state-of-the-art 
technologies and the datasets used for training. For instance, TesseTrack (Reddy et al. 2021) 
calculates the root-centered MPJPE metric on the Human3.6M dataset and the non-root-
centered MPJPE on the CMU Panoptic dataset. It is relevant to estimate 3D pose in relative 
body coordinates for a single person, however, it requires absolute coordinate for multi-
person from monocular images and global coordinate is meaningful for multi-person pose 
estimation from multiple camera views.
2.2.1  Person-centric
Person-centric, also known as relative coordinates or root-centric methods predict 3D coor­
dinates for keypoints relative to a central body point, such as the pelvis, with this root joint 
centered at the origin and other joints estimated in relation to it. This approach is primarily 
useful for single-person pose estimation using monocular images. Various methods for esti­
mating pose in root-centric coordinates for single-person 3D-HPE with monocular images 
are described in works (Zhang et al. 2023a; Ci et al. 2022; Zheng et al. 2021; Bartol et al. 
2022; Cheng et al. 2021a; Wehrbein et al. 2021; Usman et al. 2022). Additionally, some 
studies (Honari et al. 2023; Liu et al. 2021; Lee et al. 2023; Choi et al. 2021a; Kundu et al. 
2021) focus on video input to estimate 3D pose in person-centric coordinates. However, 
person-centric methods are less suitable for scenarios involving multiple people or multiple 
views because they lose the location of persons in the scene and do not know where to place 
them.
1 3
24 
Page 8 of 53

A survey on deep 3D human pose estimation
2.2.2  Camera-centric
Camera-centric or absolute coordinates methods predict 3D keypoints relative to the camera 
as a reference point. The distance between the camera and the person defines the person’s 
size and keypoints, which is crucial in multi-person 3D HPE involving multi-view image 
or video input. For example, a person farther from the camera appears shorter than a person 
closer to the camera, even if the distant person is taller. Person-centric coordinates limit 
generalization for multi-view setups because relative poses predicted from one camera can­
not be easily projected into a different view, complicating occlusion handling in multi-view 
scenarios. In contrast, when estimations are made relative to a static reference, predictions 
can be easily projected from one view to another (Luvizon et al. 2022). For multiple camera 
scenarios, it is relevant to estimate 3D pose in a global or world coordinate system. Esti­
mating absolute 3D poses is more advantageous than root-relative 3D poses in real-world 
applications. For instance, in an unmanned store, it is essential to detect the merchandise 
picked up by a customer, which relies on accurate hand localization in the world coordinate 
system (Zhan et al. 2022). Similarly, applications such as augmented reality require camera-
centric human body keypoints for re-localizing a person in the world coordinate system. The 
research works (Cheng et al. 2023; Wu and Xiao 2020; Cheng et al. 2021a; Luvizon et al. 
2022; Wei et al. 2022; Cheng et al. 2021b; Zou and Tang 2021; Jin et al. 2022; Wang et al. 
2022) use a monocular image or video input to find human keypoints in camera coordinates. 
Most multi-view research works (Zhang et al. 2021; Dong et al. 2022; Gholami et al. 2022) 
estimate human keypoints with respect to world coordinates, enabling articulation across 
multiple views.
2.3  Multi-person paradigms
The problem-solving paradigms used in 3D multi-person pose estimation (3D-MPPE) can be 
divided into top-down, bottom-up, and hybrid approaches as shown in Fig. 2. The top-down 
paradigm involves first identifying each person using a bounding box and then detecting 
their keypoints. In contrast, the bottom-up paradigm generates heatmaps for all individuals 
at once and subsequently assigns keypoints to each person. Generally, top-down methods 
tend to be more accurate, while bottom-up methods are usually faster. The hybrid approach 
integrates aspects of both top-down and bottom-up methods to leverage their respective 
advantages. Additionally, some research tackles the multi-person 3D-HPE problem using 
a single-stage approach, differing from the typical two-stage solutions of top-down and 
bottom-up paradigms.
2.3.1  Top-down paradigm
Top-down methods first identify bounding boxes likely to contain a person and then per­
form single-person HPE for each detected individual. Existing methods (Reddy et al. 2021; 
Cheng et al. 2021a; Dong et al. 2022; Han et al. 2022; Guo et al. 2021; Lin and Lee 2021; 
Chen et al. 2020; Wu et al. 2021) utilize this top-down approach to estimate 3D poses in 
multi-person scenarios. Generally, top-down approaches offer superior accuracy in pose 
estimation and are ideal for applications requiring high precision (Wu and Xiao 2020). 
However, these methods heavily depend on the accuracy of the people detector (Han et al. 
1 3
Page 9 of 53 
24

R. B. Neupane et al.
2022) and often fail to detect individuals who are mostly occluded. Additionally, they do not 
scale well in crowded scenes, as their computational complexity increases with the number 
of people, resulting in slower performance and not suitable for real-time processing. The 
research work (Benzine et al. 2020) employs bounding box detection without depending on 
the number of persons in the images.
2.3.2  Bottom-up paradigm
Bottom-up approaches do not require human detection and can estimate the poses of mul­
tiple people simultaneously. Typically, these methods involve generating heatmaps fol­
lowed by post-processing steps to assemble joint detections into complete human skeletons. 
Zhou et al. (2021, 2023) present a bottom-up method for instant-aware human body pars­
ing, where semantic and geometry-rich human joints are regressed as pixel embeddings, 
enabling efficient grouping of fine-grained semantics through joint association. Existing 
methods (Wang et al. 2021b; Zhang et al. 2023c; Tu et al. 2020; Benzine et al. 2021; Fab­
bri et al. 2020) adhere to the bottom-up paradigm for multi-person 3D-HPE and are more 
efficient than top-down methods. Unlike top-down approaches, bottom-up methods produce 
multi-person joint locations in a single pass, allowing for 3D pose inference even under 
significant occlusion (Benzine et al. 2020). However, most current bottom-up methods pro­
cess all individuals at a uniform scale, making them sensitive to variations in scale among 
multiple people and less capable of detecting joints of smaller individuals.
2.3.3  Hybrid/single-stage
Top-down methods are generally effective in considering the scale of people within a scene 
but are susceptible to detection errors due to occlusion within a person and require more 
resources as the number of people increases in crowded scenes. On the other hand, bottom-
Fig. 2  Multi-person paradigms: top-down, bottom-up, and hybrid approaches
 
1 3
24 
Page 10 of 53

A survey on deep 3D human pose estimation
up methods are usually efficient and do not rely on the number of people in the scene but 
tend to make errors with small-scale individuals. Cheng et al. (2021b, 2023) propose com­
bining top-down and bottom-up networks to take advantage of both approaches. In their 
integrated network, the top-down network estimates the joints of all individuals in an image 
patch, while the bottom-up network uses human detection based on normalized heatmaps, 
enhancing the model’s robustness to scale variations.
The authors of the papers (Jin et al. 2022; Wang et al. 2022) argue that current multi-
person methods using either two-stage top-down or bottom-up approaches face issues with 
redundant computations and high computational costs, resulting in inadequate efficiency 
for real-time processing. Jin et al. (2022)propose a single-stage solution with a decoupled 
regression model for multi-person 3D human pose estimation using regression maps. Wang 
et al. (2022) introduce a single-stage distribution-aware method that enhances body joint 
estimation recursively.
2.4  Deterministic vs. probabilistic vs. diffusion-based
Existing research on 3D-HPE can be categorized into deterministic and probabilistic 
approaches based on the number of solutions they provide. Deterministic approaches gen­
erate a single, definitive 3D pose for each image, making them practical for real-world 
applications. Probabilistic approaches model the 2D-to-3D conversion as a probability dis­
tribution and produce a set of possible solutions for each image, accommodating uncer­
tainty and ambiguity in the lifting process. To make these probabilistic approaches practical, 
aggregation techniques combine multiple hypotheses into a single, higher-quality 3D pose. 
Recently, diffusion models have been popular in image generation tasks and have also been 
explored for 3D pose estimation. These models are based on the concept of gradually trans­
forming simple data distributions such as Gaussian noise into more complex distributions 
that represent the final 3D pose. The generic flow of these techniques is illustrated in Fig. 3.
2.4.1  Deterministic approach
Most of the existing 3D-HPE methods (Honari et al. 2023; Artacho and Savakis 2022; Nie et 
al. 2023; Liu et al. 2020a; Wang et al. 2020; Zhao et al. 2023b; Luvizon et al. 2022; Tang et al. 
2023) follow a deterministic approach, providing a definite, single 3D pose for each frame. 
Deterministic methods are favored because they produce consistent and straightforward 
results, making them practical for various applications. However, accurately reconstructing 
the correct 3D pose from 2D joint detections is challenging due to depth ambiguities and 
occluded body parts, leading to multiple feasible solutions. Approaches based on determin­
ism often overlook these ambiguities by assuming a single solution exists, which can result 
in less satisfactory results.
2.4.2  Probabilistic approach
Probabilistic methods aim to model the uncertainty in pose estimation mostly from 2D 
detectors. Bertoni et al. (2019) tackles the ambiguity in the localization by estimating con­
fidence intervals using a loss function derived from the Laplace distribution. Rogez et al. 
(2020); Wehrbein et al. (2021) propose a probabilistic model to generate multiple 3D pose 
1 3
Page 11 of 53 
24

R. B. Neupane et al.
hypotheses to create a series of pose suggestions for the localization process in pose detec­
tion. Han et al. (2022) proposed uncertainty learning to tackle uncertainties stemming from 
noisy labeling, imprecise person detection, and occlusion by modeling uncertainties in joint 
locations using probability. Wandt et al. (2022) employ normalizing flows to establish a 
prior distribution of poses, facilitating the identification of the most probable up-to-scale 3D 
pose from random projections. Research in (Li et al. 2023b, 2022) proposes a methodology 
involving initial one-to-many mapping followed by a many-to-one mapping through mul­
tiple intermediate hypotheses. Bartol et al. (2022) propose a method for triangulating human 
poses that generates multiple 3D pose hypotheses by triangulating random subsets of views 
for each joint. Lee et al. (2023) utilize a posture entropy-based probabilistic model, Jiang 
et al. (2023) utilize probability distributions to represent camera poses whereas MHCanon­
Net (Kim et al. 2024) generates multiple hypotheses for pose and rotation within the spatial 
domain while capturing temporal correlations of features across multiple frames. A probabi­
listic 3D-HPE framework can be enhanced by integrating dynamic parameters and heuristic 
functions, such as joint or motion constraints. A recent study by (Sun et al. 2024) exempli­
fies this approach by incorporating proprioceptive feedback into their probabilistic model.
2.4.3  Diffusion-based approach
Diffusion models such as (Gong et al. 2023) start from an initial noisy pose estimate and 
iteratively refine it over time through a learned diffusion process. These methods often 
involve a learned reverse process where noise is removed step-by-step to recover the true 
pose. Shan et al. (2023) propose a diffusion-based method that is compatible with exist­
ing 3D-HPE methods as a backbone and allows for a customizable number of hypotheses 
during inference. Holmquist and Wandt (2023) introduce a conditional diffusion model to 
predict multiple hypotheses as the gradual transformation of a vector representing 3D joint 
Fig. 3  Generic flow of deterministic, probabilistic, and diffusion-based approaches
 
1 3
24 
Page 12 of 53

A survey on deep 3D human pose estimation
coordinates into a Gaussian distribution. The denoising phase relies on joint-wise heatmaps 
generated by a 2D joint detector that uses an embedding transformer. This embedding trans­
former non-linearly embeds all samples for each joint into a single vector, preserving their 
multi-modality. Cai et al. (2024a) utilize a disentanglement approach using hierarchical 
information to explicitly model pose priors in the forward diffusion process. In the reverse 
process, they use a denoiser consisting of a spatial transformer related to the hierarchy and 
a temporal transformer.
2.5  Problem solving space
Existing methods are examined based on the problem-solving space where the actual 3D 
pose estimation task is performed. Many methods for 3D-HPE operate in pixel space, while 
a few research efforts have proposed models that function in 3D space using discretized 
voxels. Recently, some studies have leveraged NeRF for various tasks related to 3D human 
modeling.
2.5.1  Pixel-based approach
Most existing methods (Honari et al. 2023; Cheng et al. 2023; Kim et al. 2024; Chen et 
al. 2024; Nie et al. 2023; Lee et al. 2023; Zhao et al. 2023b; Chen et al. 2022; Tang et al. 
2023; Gong et al. 2023; Yu et al. 2023) perform the task 3D-HPE within the pixel space. 
These approaches typically begin by extracting image features related to human pose and 
representing them in a 2D space. The subsequent pose estimation task is then performed on 
these 2D feature maps. These pixel-based methods are computationally efficient, but they 
face significant challenges due to the highly non-linear process of regressing 3D coordinates 
from 2D feature space.
2.5.2  Voxel-based approach
Volumetric supervision in 3D pose estimation offers detailed ground truth data for each 
voxel in 3D space, providing richer information. This approach directly regresses 3D coor­
dinates by transforming them into predicting confidence estimates for each voxel (Pavlakos 
et al. 2017). Additionally, directly regressing joint coordinates becomes impractical in multi-
person environments when the number of people is unknown beforehand, making volumet­
ric heatmaps a natural choice for bottom-up multi-person 3D-HPE (Fabbri et al. 2020). Tu 
et al. (2020); Reddy et al. (2021); Zhang et al. (2023c) propose voxel-based approach for 
estimating the 3D poses of multiple people from multiple camera views, operating directly 
in 3D voxel space by integrating all camera views. In this technique, the integrated 3D rep­
resentation effectively eliminates the need for associating 2D poses across different views, 
which is beneficial for handling occlusions. However, the accuracy is highly dependent on 
the resolution of the volumetric heatmap, with precision linked to the size of each voxel, 
and high-resolution heatmaps require significant memory, leading to scalability problems. 
Consequently, these methods are either accurate but slow or fast but inaccurate, and a new 
model must be trained for each specific speed-accuracy trade-off (Luvizon et al. 2023).
1 3
Page 13 of 53 
24

R. B. Neupane et al.
2.5.3  NeRF-based approach
Recently, Neural Radiance Field (NeRF)(Mildenhall et al. 2020) has gained prominence for 
generating novel views by synthesizing scenes from multiple images. Researchers propose 
to leverage the NeRF for 3D human modeling in various tasks such as animation, avatar, 
and portrait synthesis. Some recent works, such as A-NeRF (Su et al. 2