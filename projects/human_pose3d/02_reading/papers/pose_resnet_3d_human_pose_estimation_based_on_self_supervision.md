# Pose ResNet: 3D Human Pose Estimation Based on Self-Supervision

> 2023 · id: W4324056836 · pdf: https://www.mdpi.com/1424-8220/23/6/3057/pdf?version=1678697810 · 来源: web-agent
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

Sensors 2023, 23, 3057. https://doi.org/10.3390/s23063057 
www.mdpi.com/journal/sensors 
Article 
Pose ResNet: 3D Human Pose Estimation Based  
on Self-Supervision 
Wenxia Bao 1, Zhongyu Ma 1, Dong Liang 1, Xianjun Yang 2,* and Tao Niu 1 
1 School of Electronics and Information Engineering, Anhui University, Hefei 230601, China 
2 Hefei Institutes of Physical Science, Chinese Academy of Sciences, Hefei 230031, China 
* Correspondence: xjyang@iim.ac.cn 
Abstract: The accurate estimation of a 3D human pose is of great importance in many fields, such 
as human–computer interaction, motion recognition and automatic driving. In view of the difficulty 
of obtaining 3D ground truth labels for a dataset of 3D pose estimation techniques, we take 2D im-
ages as the research object in this paper, and propose a self-supervised 3D pose estimation model 
called Pose ResNet. ResNet50 is used as the basic network for extract features. First, a convolutional 
block attention module (CBAM) was introduced to refine selection of significant pixels. Then, a wa-
terfall atrous spatial pooling (WASP) module is used to capture multi-scale contextual information 
from the extracted features to increase the receptive field. Finally, the features are input into a de-
convolution network to acquire the volume heat map, which is later processed by a soft argmax 
function to obtain the coordinates of the joints. In addition to the two learning strategies of transfer 
learning and synthetic occlusion, a self-supervised training method is also used in this model, in 
which the 3D labels are constructed by the epipolar geometry transformation to supervise the train-
ing of the network. Without the need for 3D ground truths for the dataset, accurate estimation of 
the 3D human pose can be realized from a single 2D image. The results show that the mean per joint 
position error (MPJPE) is 74.6 mm without the need for 3D ground truth labels. Compared with 
other approaches, the proposed method achieves better results. 
Keywords: 3D human pose estimation; epipolar geometry; self-supervised learning; transfer learn-
ing; synthetic occlusion 
 
1. Introduction 
It has always been the goal of human society to allow machines to perceive and ob-
tain information from the outside world through vision like human do. With the rapid 
development of human–computer interaction and computer vision technology, the issue 
of how to make machines understand and analyze human actions and behaviors correctly 
has become important. In this context, the task of human pose estimation was developed 
[1]. The 3D estimation of a human pose involves estimating the 3D positions of human 
joints from input images or videos. Human pose estimation is widely used in many fields, 
such as sports science, rehabilitation training, video surveillance and behavioral recogni-
tion [2,3]. 
In practical applications based on 3D human pose estimation, human motion infor-
mation and joint positions can be obtained by wearing sensors and other devices, and the 
joint positions can be calculated by collecting depth maps through Microsoft Kinect [4]. 
However, the inconvenience of this operation and the high cost of the devices limit its 
range of applications. In contrast, ordinary cameras are common in daily life, and are used 
in monitoring equipment, digital photography and mobile phones. Hence, the estimation 
of a human pose based on ordinary RGB images has great research significance and 
Citation: Bao, W.; Ma, Z.; Liang, D.;  
Yang, X.; Niu, T. Pose ResNet: 3D  
Human Pose Estimation Based on  
Self-Supervision. Sensors 2023, 23, 
3057. https://doi.org/10.3390/ 
s23063057 
Academic Editor: Stefanos Kollias 
Received: 11 February 2023 
Revised: 8 March 2023 
Accepted: 10 March 2023 
Published: 12 March 2023 
 
Copyright: © 2023 by the authors. Li-
censee MDPI, Basel, Switzerland. 
This article is an open access article 
distributed under the terms and con-
ditions of the Creative Commons At-
tribution (CC BY) license (https://cre-
ativecommons.org/licenses/by/4.0/). 

Sensors 2023, 23, 3057 
2 of 20 
 
 
application value, and is receiving increasing attention from researchers in industry and 
academia. 
At present, the mainstream 3D human pose estimation algorithms can be divided 
into traditional and deep learning methods. In the early days, traditional human pose es-
timation methods generally included a detector of human body parts that was based on a 
graph structure and a deformable part model, and described the spatial relationships be-
tween the body parts through a graph model. The extracted features were mainly set man-
ually, such as HOG, SIFT and other features [2]; these approaches could not make full use 
of the image features, and were susceptible to image appearance. Moreover, the part 
model had a single structure, which imposed limitations on the possible application sce-
narios. 
With the advent of the era of big data, deep learning approaches have been rapidly 
developed for the field of computer vision, and research on human pose estimation has 
shifted from traditional methods to deep learning methods. Deep-learning-based pose es-
timation methods mainly use convolutional neural networks (CNNs) to extract human 
pose features from images, thus avoiding the need for the manual design of image fea-
tures. In 2014, Li et al. [5] first showed that deep neural networks could achieve reasonable 
accuracy in 3D human pose estimation from a single image. Their framework included 
joint regression networks and body part detectors. Although many approaches had pre-
viously been proposed to try to predict a 3D pose directly from an image, Chen et al. [6] 
explored a simple approach to 3D human pose estimation by performing 2D pose estima-
tion followed by 3D exemplar matching. 
Although existing methods have achieved remarkable results, there are still signifi-
cant challenges in regard to 3D human pose estimation at the application level. Since the 
most commonly used 3D datasets were captured by motion capture systems (MOCAPs) 
in controlled laboratory environments, the cost of 3D ground truth labels is high, and the 
background of the image is single. Network models trained on these datasets cannot easily 
be extended to outdoor and complex environments with significant occlusion of body 
parts. However, data without 3D labels are easily available, so the question arises as to 
whether we can train a network model using data without 3D ground truth labels. To 
solve this problem, we construct a network model for 3D human pose estimation in this 
paper, and use a self-supervised training method to estimate 3D human pose accurately 
from images without 3D labels. 
We present a 3D human pose estimation network model called Pose ResNet, which 
integrates the WASP and CBAM modules into the base network of ResNet50. We use epi-
polar geometry to construct 3D labels to realize self-supervision, hence solving the prob-
lem of obtaining 3D ground truth labels for datasets for 3D pose estimation technology. 
Our model is pre-trained on a 2D dataset and then undergoes further self-supervised 
training on a 3D dataset without 3D labels after migrating the trained model parameters, 
and synthetic occlusion is used to enhance the training images. Without the need for 3D 
ground truth labels, the accurate estimation of 3D human pose is realized from 2D images.  
The contributions of this study are as follows: 
(1) A network model called Pose ResNet is designed for 3D human pose estimation. This 
model is based on ResNet50 and WASP, and a CBAM attention mechanism is intro-
duced to increase the receptive field and select the important pixels in a fine-grained 
way. 
(2) This model uses the strategies of transfer learning and data enhancement for training, 
which not only improve the accuracy of joint detection but also enhance the general-
ization performance of the model. Transfer learning involves a transfer from 2D to 
3D data, while data enhancement refers to the synthetic occlusion of images in the 
3D dataset. 
(3) A set of 3D labels constructed through an epipolar geometric transformation between 
multiple views are used to train this model. Without the need for 3D ground truth 
labels, this model realizes true self-supervised training. 

Sensors 2023, 23, 3057 
3 of 20 
 
 
The rest of this paper is organized as follows: In Section 2, related work on 3D human 
pose estimation is described. In Section 3, the proposed model and method are introduced. 
In Section 4, the datasets and training process of the model are introduced. In Section 5, 
we explain the evaluation index used, and present experimental results, an analysis and 
a discussion. In Section 6, we summarize our results and discuss prospects for future 
work. 
2. Related Work 
In this paper, a single-stage method is used to estimate 3D pose directly from an im-
age, and self-supervised method is used in training. This section will give an overview of 
the deep learning method of human pose estimation from two different perspectives: the 
number of model stages and the supervision mode. 
2.1. Single- and Two-Stage Models 
In recent years, CNNs have made significant achievements in the field of image pro-
cessing, and deep learning technology has been used to meet the need for pose estimation 
in practical applications more effectively. Three-dimensional human pose estimation 
methods based on deep learning are mainly divided into two categories based on the 
number of stages: single-stage methods, which train the neural network to regress joint 
locations directly, and two-stage methods, which decompose the 3D pose estimation task 
into two independent stages, involving the estimation of 2D poses and then lifts them into 
3D space. 
Many papers have adopted single-stage methods, such as [7–13], in which CNNs 
were used to estimate the coordinates of 3D joints directly from images. Li et al. [5] com-
bined a regression network with a body part detection network, and for the first time 
proved that a deep neural network could directly obtain a 3D human pose from a single 
image. The estimation accuracy was significantly improved compared with traditional 
methods. Tekin et al. [8] trained a self-encoder–decoder by combining traditional CNNs 
for supervised learning with auto-encoders for structural learning and achieved good re-
sults. Apart from that, Pavlakos et al. [14] described the problem of 3D human pose esti-
mation as 3D keypoint localization in a discretized space around the subject. The ad-
vantage of a single-stage method is that the whole network model can achieve end-to-end 
training, although the requirements for the network structure are higher. 
Compared to single-stage methods, which directly regress 3D coordinates from im-
ages, two-stage methods can also achieve good results. Zhou et al. [15] proposed a transfer 
learning method that used a mixture of 2D and 3D labels in a deep neural network based 
on a two-stage cascaded structure. Their network was augmented with a state-of-the-art 
2D pose estimation sub-network with a 3D deep regression sub-network. In addition, an 
advanced 2D pose estimation module can be directly used to obtain 2D poses. By input-
ting 2D poses into a 3D pose estimation module, 3D poses can be obtained [16–18]. Mar-
tinez et al. [16] built a system that could predict 3D joint positions given 2D joint positions, 
and suggested directions for future work that could further advance the state of the art in 
3D human pose estimation. Compared with single-stage methods, this approach benefits 
from the maturity of the current 2D pose estimation technology [19–21], and reduces the 
learning pressure on the model. 
2.2. Weakly Supervised and Self-Supervised Methods 
The successful application of deep learning in computer vision relies heavily on large 
amounts of fully labeled data, although labeled data with high confidence are sometimes 
difficult to obtain. Due to the high cost of acquiring 3D annotations for 3D pose estimation 
datasets, weakly supervised and self-supervised methods have become research hotspots 
in recent years. 

Sensors 2023, 23, 3057 
4 of 20 
 
 
Many researchers [22–26] have explored methods of human pose estimation based 
on weak and self-supervision. Compared with fully supervised methods [8–12,27] in 
which regression networks are used to estimate 3D joint coordinates directly from images, 
self-supervision and weakly supervised methods can solve the difficulty of obtaining 3D 
ground truth labels to a certain extent. Pavlakos et al. used [26] a pictorial structure model 
to obtain global pose configurations from key point heatmaps of multi-view images. Nev-
ertheless, their method needed full camera calibration and a keypoint detector to produce 
the 2D heatmaps. Rhodin et al. [24] used multi-view consistency constraints to supervise 
a network. They needed small amounts of 3D ground truth data to avoid degenerate so-
lutions in which the pose collapses to a single location. Thus, a lack of in-the-wild 3D 
ground truth data represent a limiting factor for this method. Drover et al. [23] used an 
adversarial framework to impose a priori on 3D structures, learning only from their ran-
dom 2D projections. Their method did not require correspondence between the 2D and 
3D points to construct explicit 3D priors, but still requires the use of 3D labels. Zhou et al. 
[15] mixed 2D data with 3D indoor data to form a batch. For the 3D data, a Euclidean loss 
function was used, while for the 2D data, a weak supervision loss was proposed based on 
the 2D annotations and prior knowledge of the human skeleton. Their approach, while 
enhancing the applicability of outdoor images, still requires 2D and 3D labels. Wandt et 
al. [28] trained a generative adversarial network that consisted of three parts: a pose and 
camera estimation network, a critic network and a reprojection network. They used an 
additional camera estimation network and a reprojection layer that projected the esti-
mated 3D pose back to 2D. Kocabas et al. [29] showed that even without any 3D ground 
truth data or a knowledge of camera external paramenters multi-view images could be 
leveraged to achieve self-supervision. Hua et al. [30] proposed a coarse-to-fine approach 
for cross-view 3D human pose estimation. Specifically, they first use triangulation to lift 
the 2D detections to coarse 3D poses, and then use a refined model to obtain accurate 
results. Although their method is more accurate, they need to input multi-view images in 
both the training stage and the estimation stage, while our method only needs to input a 
monocular image in the estimation stage to estimate the 3D pose. Gong et al. [31] proposed 
a new self-supervised method: PoseTriplet. Their model consists of three parts: pose esti-
mator, imitator and illusionist. The pose estimator transforms an input 2D pose sequence 
to a low-fidelity 3D output, which is then enhanced by the imitator that enforces physical 
constraints. The refined 3D poses are subsequently fed to the hallucinator to produce even 
more diverse data. Although their method does not rely on any given 3D data, their net-
work uses 2D pose sequence data as input and cannot achieve direct estimation of 3D pose 
from 2D images. 
Existing 3D human pose estimation methods usually require 3D labels or camera in-
ternal and external parameters and rely heavily on consistency or adversarial loss to gen-
erate supervised signals, which are weak and can affect the performance of the model. 
Unlike the above existing methods, the supervised signal of our method is a strongly su-
pervised signal, which is directly supervised with pseudo-labels calculated by an epipolar 
geometric formula, without using 3D labels or external parameters of the camera, and the 
trained network model is capable of 3D human pose estimation from monocular images. 
3. Models and Methods 
As a 3D human pose estimation model based on self-supervision, Pose ResNet is a 
dual-branch structure consisting of upper and lower branches. As shown in Figure 1, the 
upper branch is composed of a 2D Pose CNN and an epipolar geometric module, and the 
lower branch is composed of a 3D Pose CNN. When the 2D Pose CNN of the upper branch 
has output the 2D human body pose, the 3D label is constructed by using the epipolar 
geometric module to supervise the training of the 3D Pose CNN of the lower branch, thus 
realizing self-supervision. 

Sensors 2023, 23, 3057 
5 of 20 
 
 
 
Figure 1. Structure of Pose ResNet. 
3.1. Pose ResNet 
As shown in Figure 1, the 2D Pose CNN in the upper branch is composed of a feature 
extraction module and key joint detection module, while the 3D Pose CNN in the lower 
branch is also composed of a feature extraction module and key joint detection module. 
The feature extraction modules of both branches are the same, and ResNet50 is used as 
the basic network to extract features. The ResNet network is simple and practical, and has 
been widely used in various feature extraction. When the number of layers of deep learn-
ing network is deeper, theoretically the network performance will be stronger, but after 
the CNN network reaches a certain depth, and then deeper, the network performance will 
not improve, but it will lead to a slower convergence of the network, and the accuracy rate 
decreases with it. Additionally, the skip connection of the ResNet network can solve this 
problem well and alleviate the gradient disappearance to some extent. The CBAM is in-
troduced before Layer 1 and after Layer 4 of ResNet50 for the fine-grained selection of 
important pixels. The WASP module is added after the ResNet-CBAM, which is applied 
to the extracted features to increase the receptive field and capture the multi-scale context 
information of the image. Finally, a deconvolution network is connected after the WASP 
module, and the features are applied to the deconvolution network to obtain the volumet-
ric heatmap 𝐻. 
The key joint detection module of the 2D Pose CNN obtains the 2D pose 𝑈 by ap-
plying a soft argmax function to the x- and y-dimensions of the volumetric heatmap 𝐻. 
The key joint detection module of the 3D Pose CNN directly obtains the predicted 3D pose 
𝑉 by applying a soft argmax function to the x-, y- and z-dimensions of the volumetric 
heatmap 𝐻. 
The self-supervised learning of this model involves mining and constructing the su-
pervision information from the data without manual annotations, and training the net-
work based on the supervision information. Self-supervision in this case is achieved by 
the epipolar geometry module of the upper branch. As shown in Figure 1, the 2D Pose 
CNN in the upper branch outputs the multi-view 2D pose, and the epipolar geometry 
module then constructs the 3D pose and caches it as a 3D label 𝑉෠. The constructed 3D 
labels are used for self-supervised training of the 3D Pose CNN in the lower branch. To 
some extent, this solves the difficulty of obtaining 3D ground truth labels for 3D human 
pose datasets. 

Sensors 2023, 23, 3057 
6 of 20 
 
 
3.2. CBAM 
The attention mechanism module operates similarly to the selective visual attention 
mechanism of humans. The aim is to select information that is more important to the cur-
rent target task from all of the information that is available, thus effectively suppressing 
useless information and greatly improving the efficiency and accuracy of visual infor-
mation processing. In the estimation of a human pose, due to the presence of complex 
image backgrounds, the spatial location information of the human pose features in the 
image plays an important role in accurately identifying the joints. Attention mechanisms 
can be used to suppress redundant background information and select the important pix-
els in a fine-grained way, thus making the network pay more attention to the joint infor-
mation of human body. In this study, a CBAM [32] that combines channel and spatial 
attention is applied to the output feature map of the convolution network to improve the 
estimation performance of the model. 
In this paper, the CBAM attention mechanism, an attention mechanism module that 
combines spatial and channel information, is introduced before Layer 1 and after Layer 4 
of the basic ResNet50 network [32]. The structure of ResNet50 after the introduction of the 
CBAM module is shown in Figure 2. 
 
Figure 2. Structure diagram of ResNet50-CBAM. 
As shown in Figure 3, the CBAM module contains two submodules cascaded in se-
quence: a channel attention (CA) submodule and a spatial attention (SA) submodule. As 
shown in Figure 3a, the input feature map is first passed through the CA module to get 
the initial weighted results, and then through the SA module to get the final weighted 
results. 
 
Figure 3. Structure of the CBAM module. 

Sensors 2023, 23, 3057 
7 of 20 
 
 
The CA submodule weights the channel dimensions of the input feature map, its 
structure is shown in Figure 3b. We assume that the size of input feature map 𝐹 is 
ℎ× 𝑤× 𝑐 where 𝑤 is the width of the feature map, ℎ is the height and 𝑐 is the number 
of channels. The CA module can be expressed by the mathematical equation  
𝑀஼ሺ𝐹ሻ= 𝜎ሼ𝑀𝐿𝑃ሾ𝐴𝑣𝑔𝑃𝑜𝑜𝑙ሺ𝐹ሻሿ+ 𝑀𝐿𝑃ሾ𝑀𝑎𝑥𝑃𝑜𝑜𝑙ሺ𝐹ሻሿሽ 
(1)
where AvgPool and MaxPool represent average pooling and max pooling, respectively, 
which give two sets of feature vectors of size 1 × 1 × 𝑐. MLP represents a learnable mul-
tilayer perceptron with one hidden layer, which can output a set of learned feature vec-
tors. σ is the sigmoid activation function used to map each point of the feature vector to 
between zero and one, and 𝑀ௌሺ𝐹ሻ are the weights of the different channels. 
The SA submodule weights the spatial positions of the input feature map, its struc-
ture is shown in Figure 3c, and can be expressed by the mathematical equation 
𝑀ௌሺ𝐹ሻ= 𝜎ሼ𝑓଻×଻ሾ𝐴𝑣𝑔𝑃𝑜𝑜𝑙ሺ𝑀஼ሻ; 𝑀𝑎𝑥𝑃𝑜𝑜𝑙ሺ𝑀஼ሻሿሽ 
(2)
where 𝑓଻×଻ represents a convolution kernel with size 7 × 7, which is the spatial feature 
extractor of the target. The feature map is max pooled and average pooled along the chan-
nel dimension. The SA submodule uses the sigmoid activation function after the 7 × 7 
convolution to map the value of each pixel in the feature map to a probability value be-
tween zero and one.  
3.3. WASP Module 
In order to capture the multi-scale context information of the image, expand the re-
ceptive field and make the detection of joint position more accurate, the WASP module 
[33] is added to the model. The receptive field refers to the area size of the input layer 
corresponding to an element deciding the output result of a certain layer. When the input 
image is passed through the different convolutional layers of the network, the structure 
of the image that can be seen is different. After passing through a shallow convolution 
layer, each pixel represents only one feature extraction of the local information of the orig-
inal image, with rich detail but little context information, meaning that the receptive field 
is small. In contrast, after passing through a deep convolutional layer, the range of the 
original image can be seen is larger, but some details are missed, meaning that the recep-
tive field is large. 
Hence, occlusion is a difficult problem when estimating the 3D pose of the human 
body. Expanding the receptive field enables the network to learn the relationships be-
tween the joints of the body using context information, which is effective for the inference 
of occluded joints. However, the differentiation between local information of the joints in 
the background is weak, which can easily cause local confusion. Expanding the receptive 
field makes the range of the visible image larger and enhances the differentiation of the 
local information on the joints. 
Before applying the WASP module, we first introduce atrous convolution, which is 
a way of increasing the receptive field. Atrous convolution means injecting atrous into the 
standard convolution to increase the receptive field. As shown in Figure 4, Figure 4a is the 
schematic diagram of standard convolution and Figure 4b is the schematic diagram of 
atrous convolution. Compared with the standard convolution, atrous convolution has an 
extra parameter called dilation rate, which refers to the interval number of kernels. The 
specific meaning is to fill 0 in the middle of the convolution kernel, and the number of 
filled 0s is the dilation rate −1. In this case, we set the kernels to 3, and the dilation rate to 
2 (a dilation rate of 1 is the standard convolution). 

Sensors 2023, 23, 3057 
8 of 20 
 
 
 
 
(a) 
(b) 
Figure 4. (a) Standard convolution with a 3 × 3 kernel. (b) Atrous convolution with a 3 × 3 kernel. 
The WASP module samples the input in parallel with empty convolutions of differ-
ent sampling rates and combines the results together. The number of channels is then re-
duced to the expected value through 1 × 1 convolution, as shown in Figure 5, which is 
equivalent to capturing the context information of the image in multiple proportions. The 
addition of the WASP module means the network can make better use of global infor-
mation, rather than paying too much attention to a small proportion of the features. In 
addition to enriching the level of detail, it also expands the receptive field to ensure that 
important information is not ignored when making decisions. 
 
Figure 5. Structure of the WASP module. 
4. Model Training 
4.1. Dataset 
At present, the lack of large outdoor datasets and datasets with special types of poses 
(such as falling, rolling, etc.) constitutes the main bottleneck to the development of 3D 
pose estimation models. This phenomenon mainly arises because 3D datasets are built 
with MOCAP systems, which are suitable for indoor environments but require complex 
devices with multiple sensors and tights that are impractical for outdoor use. In this paper, 
we propose a solution to this problem by using two large public datasets, namely an 

Sensors 2023, 23, 3057 
9 of 20 
 
 
outdoor dataset called MPII [34] and an indoor 3D dataset called Human3.6M [35]. The 
model is pre-trained on the former and then undergoes self-supervised training on the 
latter. 
MPII [34], a dataset used for evaluating 2D human pose estimation, contains about 
25,000 labeled images with 16 annotated joints. These images are extracted from YouTube 
videos. Overall, this dataset covers 410 kinds of human behaviors, most of which are cap-
tured in outdoor environments with complex backgrounds, as shown in Figure 6, which 
shows some images of this dataset. 
 
Figure 6. Images from the MPII dataset. 
The Human3.6M dataset [35] is the largest and most widely used dataset for 3D hu-
man pose estimation, and contains 3.6 million RGB images from four different perspec-
tives captured by MOCAP systems in indoor environments. The dataset includes 11 sub-
jects in 15 action scenarios, such as talking, eating, exercising, greeting, etc., as shown in 
Figure 7a. The dataset includes annotations for 17 joints of the human body, and the to-
pology of the joints is shown in Figure 7b. Although 11 subjects participated in the dataset, 
3D annotations are available for only seven (S1, S3, S5, S7, S8, S9, S11). In general, subjects 
S1, S3, S5, S7 and S8 are used as the training set, while S9 and S11 are used as the testing 
set. To reduce data redundancy, key frames are extracted from the videos. In the training 
set, one frame is extracted from every 5 frames, while in the testing set, one frame is ex-
tracted from every 64 frames. After preprocessing to remove redundancies, there are 
51,765 samples in the training set and 8606 in the testing set. 
The MPII dataset was used for pre-training, and the Human3.6M dataset was then 
used for self-supervised training. In the MPII dataset, the skeleton topology has 16 joints 
connected according to certain relationships, while in the Human3.6M dataset, the skele-
ton topology has 17 joints. The skeleton topology finally output by the network model has 
16 joints, and joint 7 (spine) in Figure 7b is abandoned. 

Sensors 2023, 23, 3057 
10 of 20 
 
 
 
(a) 
 
(b) 
Figure 7. (a) Images from the Human 3.6M dataset; (b) skeleton topology in the Human3.6M da-
taset. 
4.2. Data Augmentation 
At the training stage, synthetic occlusion [36] was used to enhance the images in the 
Human3.6M dataset. In the dataset without occlusion, all targets are clearly visible, and 
the trained network may achieve relatively high accuracy. However, the generalization 
ability of the model is poor, the accuracy may be low for occluded objects and some targets 
may not even be recognized. Synthetic occlusion was therefore used to enhance the train-
ing images so that the network could make better use the of global information, rather 
than paying too much attention to a small part of the features, in order to improve the 
robustness of the model. 
Synthetic occlusion was applied using the Pacal VOC dataset. Segmented objects ex-
tracted from the Pacal VOC dataset were filtered to remove people and objects marked as 
difficult or truncated, and the remaining 2638 objects were pasted into random positions 
in the Human3.6M dataset with a certain probability (Pocc) to synthesize training images 

Sensors 2023, 23, 3057 
11 of 20 
 
 
with random occlusion [36]. During the experiment, the probability of occlusion (Pocc) was 
set to 0.5, and the degree of occlusion was between 0% and 70%. Examples of synthesized 
occluded images are shown in Figure 8, where the original image is shown on the left and 
the occluded images are shown on the right. The TV, bus, dog, bird, etc., in Figure 8 are 
objects extracted from the Pacal VOC dataset. 
 
Figure 8. Occluded images from the Human3.6M dataset. 
4.3. Pre-Training of 2D Pose CNN 
At the training stage of the 3D human pose estimation model, the 2D Pose CNN was 
first pre-trained on the MPII dataset, so that the network could achieve accurate estima-
tion of the 2D human pose. The trained parameters were then transferred to the 3D Pose 
CNN, and the Human3.6M dataset was used for self-supervised training. The data used 
for training were synthetically occluded. The details of the training process are shown in 
Figure 9. 
 
Figure 9. Training process of Pose ResNet. 
Pre-training is one application of transfer learning. After pre-training, the model can 
achieve an accurate estimation of the 2D pose, which makes the 3D labels that are 

Sensors 2023, 23, 3057 
12 of 20 
 
 
constructed based on the 2D pose more accurate, thus improving the accuracy rate of joint 
detection. Furthermore, the environments of the images in the 2D datasets are mostly out-
doors with complex backgrounds, while the 3D datasets are mostly acquired in indoor 
closed environments by MOCAP systems with monotonous backgrounds. Hence, better 
results can be obtained by pre-training on the MPII dataset and then carrying out self-
supervised training on the Human3.6M dataset. 
4.4. Self-Supervised Training of 3D Pose CNN 
In self-supervised training, the supervised information is not manually annotated, 
but it is automatically constructed by algorithms from unsupervised data. The use of self-
supervised learning techniques can reduce the dependence of network models on data, 
which solves the problem of the difficulty of acquiring 3D annotations. The details of the 
implementation of self-supervised learning in this paper are shown in Figure 1. The 3D 
labels are calculated from the 2D pose of the multi-views output from the upper branch 
on the model by the epipolar geometric formulation. 
For self-supervised training, we assume that the number of views is two. The differ-
ent views images 𝐼௜ and 𝐼௜ାଵ with resolution 1000 × 1000 from the Human 3.6M dataset 
are used as input to both the upper and lower branches of the model. The 2D Pose CNN 
in the upper branch directly outputs the multi-view 2D poses 𝑈௜ and 𝑈௜ାଵ. The epipolar 
geometric transformation of 𝑈௜ and 𝑈௜ାଵ is then carried out to obtain the 3D pose 𝑉෠ in 
the global coordinate frame, and this is cached as the 3D label. The 3D Pose CNN in the 
lower branch outputs the 3D human pose 𝑉 predicted from input images. It then calcu-
lates the loss between 𝑉 and 𝑉෠. To calculate the loss, we project 𝑉 onto the correspond-
ing camera space, and then minimize 𝑠𝑚𝑜𝑜𝑡ℎ௅ଵ൫𝑉−𝑉෠൯ to train the lower branch, where 
𝑠𝑚𝑜𝑜𝑡ℎ௅ଵሺ𝑥ሻ= ൜0.5𝑥ଶ              𝑖𝑓|𝑥| < 1
|𝑥| −0.5          𝑜𝑡ℎ𝑒𝑟𝑤𝑖𝑠𝑒
 
(3)
We can substitute 𝑥= 𝑉−𝑉෠ into Equation (3) to calculate the loss, where 𝑉 is the 
3D pose predicted by the lower branch in camera space and 𝑉෠ is the 3D label obtained 
from epipolar geometry in the upper branch. 
Epipolar geometry describes the internal photographic relationship between the two 
views, which is independent of the external scene environment, and is only related to the 
internal parameters of the camera and the relative pose between the two views. The upper 
branch of the model uses epipolar geometry to obtain the 3D labels 𝑉෠, using the following 
steps: 
(a) Let the 2D coordinate of the 𝑗 joint in the 𝑖 image be 𝑈௜,௝=ൣ𝑥௜,௝, 𝑦௜,௝൧ and its 3D 
coordinate be 𝑉௝=ൣ𝑋௝, 𝑌௝, 𝑍௝൧. We can then describe the relation between them by as-
suming a pinhole image projection model. The following formulas are obtained from 
the pinhole image projection model: 
𝑈௜,௝𝐹𝑈௜ାଵ,௝= 0 
(4)
𝐸= 𝐾்𝐹𝐾 
(5)
൥
𝑥௜,௝
𝑦௜,௝
𝑤௜,௝
൩= 𝐾ሾ𝑅|𝑅𝑇ሿ൦
𝑋௝
𝑌௝
𝑍௝
1
൪, 𝐾= ൥
𝑓௫
0
𝑐௫
0
𝑓௬
𝑐௬
0
0
1
൩, 𝑇= ቎
𝑇௫
𝑇௬
𝑇௭
቏ 
(6)
where 𝜔௜,௝ is the depth of the 𝑗 joint in the 𝑖 camera’s image with respect to the 
reference frame of the camera, 𝐾 encodes the intrinsic parameters of the camera (𝑓௫ 
and 𝑓௬ are the focal length, 𝑐௫ and 𝑐௬ are the principal points), and R and T are the 
extrinsic parameters of rotation and translation of the camera. 
(b) Without using the external parameters of the camera, we can assume that the first 
camera is located at the center of the coordinate system, meaning that R for this 

Sensors 2023, 23, 3057 
13 of 20 
 
 
camera is constant. The corresponding joints in 𝑈௜ and 𝑈௜ାଵ satisfy Equation (4). By 
substituting 𝑈௜ and 𝑈௜ାଵ into Equation (4), the fundamental matrix F can be ob-
tained. 
(c) By substituting K and F into Equation (5), we obtain the essential matrix E. By using 
SVD to decompose E, we get four possible solutions to R. 
(d) By substituting the 2D coordinates into the epipolar geometry triangulation in Equa-
tion (6), we obtain the 3D coordinates of the corresponding joints by polynomial tri-
angulation, and cache them as 3D labels, 𝑉෠. 
5. Experiments 
5.1. Evaluation Indicators 
The percentage of correct keypoints (PCK) [34] is the most widely used evaluation 
index for 2D pose estimation. PCK is defined as the proportion of correctly estimated 
joints, when the normalized distance between the predicted coordinates of the joint and 
the ground truth coordinates is less than a pre-set threshold, the estimation of the joint is 
considered correct. Otherwise, the estimation is considered incorrect. The formula for cal-
culating PCK is shown in Equation (7), where i represents the number of joints, 𝑑௜ repre-
sents the Euclidean distance between the predicted value and the real value of the 𝑖 joint, 
d is the scale factor of the human body and T is the threshold value. The normalized dis-
tance is the Euclidean distance between the predicted value of the joint and its correspond-
ing ground truth normalized by the human scale factor. The normalization of the MPII 
dataset uses the current human head diameter as the scale factor, and the PCK threshold 
set to 0.5 in the paper means that the Euclidean distance between the detected point and 
its corresponding ground truth ≤ 0.5*head diameter is considered when the estimation of 
the joint is correct. For different datasets, the normalized scale factors may be different. 
1
i
i
i
d
T
d
PCK


≤




= 

δ
 
(7)
MPJPE [35] is the most widely used evaluation index for 3D pose estimation. It is 
obtained by calculating the average value of the Euclidean distance between the estimated 
coordinates and the real coordinates of all the joints. The formula is shown in Equation 
(8), where 
( )
, ( )
f
f s
m
i is the predicted value of the 𝑖 joint of bone s in the 𝑓 frame, 
( )
, ( )
f
gt s
m
i is the real value of the 𝑖 joint of the bone s in the 𝑓 frame, 𝑁௦ is the number of 
joint points of bone s and “||.||2” means the L2 norm, that is, the Euclidean distance. 
Before evaluation, the estimated and real coordinates of the root joint were aligned via 
rigid body transformations such as translation and rotation. 
(
)
(
)
(
)
,
,
1
2
1
,
( )
( )
S
N
f
f
MPJPE
f s
gt s
i
S
E
f s
m
i
m
i
N
=
=
−

 
(8)
5.2. Experimental Environment and Parameter Selection 
The experimental environment was a Linux system, and the experiment was carried 
out on an NVIDIA 2070 (8G) graphics card. The network models were built in Pytorch, 
and the ADAM optimizer was used to optimize the model. The initial learning rate was 
set to 0.001, the batch size was 16 and the number of iterations (epochs) was 130. 
5.3. Analysis of Experimental Results 
In this paper, we present a self-supervised 3D human pose estimation model. An 
example of the estimation results on the Human3.6M dataset is shown in Figure 10. From 

Sensors 2023, 23, 3057 
14 of 20 
 
 
left to right of Figure 10, there are the original input images, the skeleton graphs of the 2D 
pose and the skeleton graphs of the 3D pose. The epoch-loss plot of the model training 
process is shown in Figure 11. 
 
Figure 10. Examples of estimation results on the Human3.6M dataset (from left to right, the columns 
show the original image, 2D pose, and 3D pose). 
 
Figure 11. Epoch-loss plot of the model training process. 
To validate the accuracy of the 3D human pose estimation model based on self-su-
pervision in this paper, comparative experiments were carried out from the following four 
aspects. 
5.3.1. Comparison with Fully Supervised Methods 
In order to demonstrate the advantages of the proposed method, it was compared 
with other existing fully supervised methods. A fully supervised method is one that uses 
the ground truth labels of the dataset during training. Table 1 shows a comparison of the 
experimental results from our method with other fully supervised methods on the Hu-
man3.6M dataset, and the best of the evaluation metrics are highlighted for better reading. 
Compared with the others, our method has a simpler structure, fewer parameters, a 
smaller MPJPE and a higher accuracy of 3D pose estimation. Although Table 1 shows that 

Sensors 2023, 23, 3057 
15 of 20 
 
 
the MPJPE of Zheng et al.’s methods [37,38] are smaller than the MPJPE of our fully su-
pervised method, their methods require input video information. Furthermore, although 
our self-supervised method is not excellent compared with the other fully supervised 
methods, it does not require any 3D ground truth labels, and the 3D human pose estima-
tion is realized based on 2D image data. 
5.3.2. Comparison with Weakly Supervised and Self-Supervised Methods 
Due to the high cost of 3D annotation, it is very difficult to obtain strong supervised 
information in many cases, and weakly supervised and self-supervised methods [28,29,39-
42 ] have therefore become very popular in recent years. There is no clear boundary be-
tween self-supervision and weak supervision, both of which aim to construct some super-
vised information to obtain pseudo-labels from unlabeled data. Table 2 compares the ex-
perimental results of our method with those of other weakly and self-supervised methods 
on Human3.6M data. Although Table 2 shows that the method of Lqbal et al. has a mini-
mum error of 69.1 mm, their model uses multiple views for prediction, meaning that mul-
tiple images of different views need to be input fo