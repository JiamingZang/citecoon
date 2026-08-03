# Dual-stream Spatio-Temporal GCN-Transformer Network for 3D Human Pose Estimation

> 2026 · id: W7155247101 · arXiv: 2604.17688 · pdf: https://arxiv.org/pdf/2604.17688 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
3D human pose estimation is a classic and important research direction in the field of computer vision. In recent years, 
Transformer-based methods have made significant progress in lifting 2D to 3D human pose estimation. However, these 
methods primarily focus on modeling global temporal and spatial relationships, neglecting local skeletal relationships and 
the information interaction between different channels. Therefore, we have proposed a novel method—the Dual-stream 
Spatio-temporal GCN-Transformer Network (MixTGFormer). This method models the spatial and temporal relationships 
of human skeletons simultaneously through two parallel channels, achieving effective fusion of global and local features. 
The core of MixTGFormer is composed of stacked Mixformers. Specifically, the Mixformer includes the Mixformer 
Block and the Squeeze-and-Excitation Layer ( SE Layer). It first extracts and fuses various information of human 
skeletons through two parallel Mixformer Blocks with different modes. Then, it further supplements the fused information 
through the SE Layer. The Mixformer Block integrates Graph Convolutional Networks (GCN) into the Transformer, 
enhancing both local and global information utilization. Additionally, we further implement its temporal and spatial forms 
to extract both spatial and temporal relationships. We extensively evaluated our model on two benchmark datasets 
(Human3.6M and MPI-INF-3DHP). The experimental results showed that, compared to other methods, our MixTGFormer 
achieved state-of-the-art results, with P1 errors of 37.6mm and 15.7mm on these datasets, respectively. 
 
 
Keywords: 3D human pose estimation; transformer; graph convolution

## introduction
3D human pose estimation is a hot topic in current human pose estimation research, aiming to accurately 
estimate the 3D coordinates of human keypoints from images or videos. Research in this field not only has 
theoretical significance but also shows great potential in various practical applications, such as motion 
analysis [1], virtual reality [2], augmented reality [3], and activity recognition [4,5]. 
 
* Corresponding author. 
E-mail address: xiangjian@zust.edu.cn. 

2 
Author name / Procedia Economics and Finance 00 (2012) 000–000 
Current mainstream research methods for 3D human pose estimation include direct estimation [6,7] and 
2D-to-3D lifting [8,9]. With the widespread application of 2D human pose detectors [10,11,12], the 2D-to-3D 
lifting approach now dominates 3D human pose estimation. From a technical perspective, 3D human pose 
estimation can be further divided into multi-view methods [13,14,15] and monocular methods [8,16]. Due to 
the widespread use of monocular RGB cameras in real-world scenarios, monocular methods have become the 
mainstream of current research. However, the depth uncertainty of monocular methods makes 3D pose 
estimation highly challenging. Since human joints are composed of local spatial and temporal dependencies, 
the spatial and temporal information carried by human motion can be utilized to address these challenges and 
improve the accuracy of 3D pose estimation [17,18,19]. 
For the 2D-to-3D pose estimation problem, two mainstream advanced models have emerged in recent years: 
Transformer-based models [20] and Graph Convolutional Network (GCN)-based models [21]. Transformers, 
initially successful in the field of natural language processing (NLP) [22], have also been applied to computer 
vision tasks, including human pose estimation. They excel at capturing long-range dependencies and have a 
powerful self-attention mechanism, which aligns well with the discrete nature of human joint representations 
and the long-range temporal dependency modeling required in skeleton sequences. Many studies have focused 
on this, such as Poseformer [8] and MixSTE [23]. Although Transformer-based human pose estimation has 
achieved good results, the global attention mechanism[24] it employs lacks sufficient attention to local spatial 
node relationships, which leads to the neglect of relationships between local spatial nodes. Additionally, since 
2D pose sequences are flattened and input into the model, it is difficult to intuitively design the model based 
on the pose structure for tracing back local joint features. To address this issue, some researchers have begun 
to explore the use of Graph Convolutional Networks, which is a deep learning method based on graph-
structured data that learns node representations by aggregating features from local neighborhoods. GCNs are 
adept at handling local dependencies, and since human skeletons can be represented as graph-structured data, 
GCN models can explicitly preserve the structure of 2D and 3D human poses during the convolutional 
propagation process. However, GCNs are not a perfect solution either; their limited number of layers restricts 
their ability to perceive long-range and global information. Therefore, combining GCN and Transformer to 
establish a unified architecture is highly effective, as it allows the network to simultaneously capture local and 
global dependencies. 
To this end, we propose a novel method for 3D human pose estimation, called MixTGFormer, which can 
effectively handles both global and local information in the spatio-temporal dimension. Compared to existing 
fusion models, the core of MixTGFormer lies in its novel fusion backbone network, Mixformer. The core of 
Mixformer is the Mixformer Block, which effectively combines Transformer and GCN. Specifically, it 
adaptively fuses the features of Transformer and GCN, enabling the module to effectively integrate local 
skeletal relationships and global spatial relationships. Furthermore, we designed two forms of this module: the 
Spatial Mixformer Block and theTemporal Mixformer Block. This not only balances the local and 
comprehensive representation of human poses but also achieves simultaneous consideration of temporal and 
spatial dimensions. Additionally, we introduced the Squeeze-and-Excitation module (SE Layer) [25] into 
Mixformer. This module explicitly models the interdependencies between convolutional feature channels, 
further enhancing the model's representational capability. Through these efforts, we have improved the 
model's comprehensive feature extraction ability and increased the accuracy of 3D human pose estimation. In 
summary, our main contributions are as follows: 
1. 
We propose a novel module, Mixformer Block, with both temporal and spatial forms. They aggregate 
the features of Transformer and GCN in the spatio-temporal dimension in a simple and effective 
manner. 

 
Author name / Procedia Economics and Finance 00 (2012) 000–000 
3 
2. 
We introduce the Squeeze-and-Excitation module and combine it with the Mixformer Block to form 
Mixformer, which further improves the model's performance by learning dependencies between 
different channels. 
3. 
The MixTGFormer, built on Mixformer, outperforms other state-of-the-art methods on the 
Human3.6M and MPI-INF-3DHP datasets, achieving the best performance.

## method
3.1. Overall Architecture 
In this section, we will comprehensively introduce the proposed method, which we call the Two-stream 
Mixed Temporal and Spatial Transformer (MixTGFormer). The goal of MixTGFormer is to lift 2D keypoint 
sequences to corresponding 3D pose sequences. It effectively combines MHSA and GCN to achieve good 
fusion and extraction of spatio-temporal information from the input and then lifts the output. The overall 
architecture of the model is shown in Fig. 1. The input of the model is a 2D input sequence with confidence 

 
Author name / Procedia Economics and Finance 00 (2012) 000–000 
5 
scores X ∈ℝ𝑇×𝐽×3, where T and J represent the number of frames and joints, respectively. First, the input is 
projected into d-dimensional features F0 ∈ℝ𝑇×𝐽×𝑑, and then learnable spatial position encoding Ppos
s
∈
ℝ1×𝐽×𝑑 is added. Subsequently, we use stacked Mixformers to compute Fi ∈ℝ𝑇×𝐽×𝑑(𝑖= 1, … , 𝑁) to capture 
the underlying 3D structure of the skeleton sequence, where N represents the network depth. Finally, we use a 
linear layer with a tanh activation function to map FN to a higher dimension to compute the motion 
representation M ∈ℝ𝑇×𝐽×𝑑′, and estimate the human 3D pose P̂ ∈ℝ𝑇×𝐽×3 through a regression head. 
Fig. 1. Top: MixTGFormer model structure; (a) Overall architecture of Mixformer; (b) Spatial Mixformer Block; (c) Temporal 
Mixformer Block. The input tokens are the local joints of the human body and the frames of the pose sequence. 
When performing the 2D-to-3D lifting task, losses may occur due to occlusion, detection failure, and errors. 
The loss terms include position loss (𝐿3𝐷) and acceleration (𝐿△𝐴) loss, which are defined as: 
𝐿3𝐷= ∑∑∥P̂𝑡,𝑗−𝑃𝑡,𝑗∥
𝐽
𝑗=1
𝑇
𝑡=1
,    𝐿△𝐴= ∑∑∥△Â𝑡,𝑗−△𝐴𝑡,𝑗∥
𝐽
𝑗=1
𝑇
𝑡=2
,
(1) 
where Â𝑡=P̂𝑡−P̂𝑡−1，△𝐴𝑡= 𝑃𝑡−𝑃𝑡−1. Therefore, the total loss for 3D human pose estimation is: 

6 
Author name / Procedia Economics and Finance 00 (2012) 000–000 
𝐿= 𝐿3𝐷+ 𝜆△𝐴𝐿△𝐴+ 𝐿2D,
(2) 
where𝜆△𝐴 is a constant coefficient used to balance position accuracy and motion smoothness, and 𝐿2D is the 
loss generated by the 2D human pose detector predicting 2D poses. 
Next, we first introduce the architectural design of the core part of MixTGFormer, Mixformer and then 
elaborate on the basic module composition of Mixformer. 
3.2. Mixformer 
Mixformer’s core consists of two parts: the Spatio-Temporal Mixformer Block and the Squeeze-and-
Excitation Layer (SE Layer, a self-attention mechanism). These two parts are sequentially connected, forming 
the overall structure (Fig. 1(a)). First, we constructed a dual-stream architecture containing two different 
forms of Mixformer Blocks for fusing the spatial and temporal information of the input. Specifically, we 
stacked the Mixformer Blocks in reverse order to form two parallel computational branches. Each branch has 
two forms of modules, temporal and spatial, so the spatio-temporal information is fused for the first time 
within the branch. Both branches model the keypoints by combining spatio-temporal information, but due to 
the different construction orders of the two branches, they have different emphases on spatio-temporal 
modeling. Subsequently, we fused the features extracted from the two streams using adaptive fusion, where 
the fusion is defined as: 
𝐹𝑖= 𝛼𝑆𝑇
𝑖◦𝐹𝑆𝑇
𝑖−1 + 𝛼𝑇𝑆
𝑖
◦𝐹𝑇𝑆
𝑖−1,    𝑖𝜖1, … , 𝑁,
(3) 
where 𝐹𝑖 represents the feature embedding at depth 𝑖, ◦ represents element-wise operations, 𝐹𝑆𝑇
𝑖−1 ,𝐹𝑇𝑆
𝑖−1 
represent the feature extraction performed by the Mixformer Block in the spatial-temporal and temporal-
spatial order at depth 𝑖−1, N represents repeating the module fusion N times, and the adaptive fusion 
weights 𝛼𝑇𝑆
𝑖,𝛼𝑇𝑆
𝑖 are defined as: 
𝛼𝑆𝑇
𝑖, 𝛼𝑇𝑆
𝑖
= softmax (𝑊∙𝐶𝑜𝑛𝑐𝑎𝑡(𝐹𝑆𝑇
𝑖−1, 𝐹𝑇𝑆
𝑖−1)) ,
(4) 
where W represents a learnable linear transformation. 
After adaptive fusion, we use the SE layer for further computation to adaptively adjust the importance of 
each channel's features and improve the fusion of spatial and channel information, further enhancing the 
model's representational capability. 
In summary, the Mixformer Block combines GCN and MHSA to achieve comprehensive modeling of local 
and global features of human skeletons, while the SE Layer further enhances the model's ability to model 
dependencies between channels. The combination of these two enables MixTGFormer to simultaneously 
capture global and local information of human poses in the spatial and temporal dimensions, significantly 
improving the accuracy of 3D human pose estimation. 
3.2.1. Mixformer Block 
The Mixformer Block is the core part of the model. To efficiently capture the global and local 
dependencies of human poses in different spatio-temporal dimensions, we combined GCN and multi-head 
attention mechanisms to form a novel backbone network to enhance the model's feature extraction and 
expressive capabilities. The Mixformer Block is designed in two modes: Spatial Mixformer Block and 
Temporal Mixformer Block, which mainly handle spatial information and temporal information, respectively. 
Their specific structures are shown in Fig. 1(b) and Fig. 1(c). 
Spatial Mixformer Block. This module adopts a parallel processing structure of Spatial Multi-Head Self-
Attention (S-MHSA) and Spatial GCN (S-GCN), treating individual joints as tokens to capture relationships 
between joints within a frame. Among them, S-MHSA is defined as: 

 
Author name / Procedia Economics and Finance 00 (2012) 000–000 
7 
S −MHSA(𝑄𝑠, 𝐾𝑠, 𝑉𝑠) = 𝐶𝑜𝑛𝑐𝑎𝑡(ℎ𝑒𝑎𝑑𝑖, … , ℎ𝑒𝑎𝑑ℎ)𝑊𝑠
(𝑂),
ℎ𝑒𝑎𝑑𝑖= 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝑄𝑠
(𝑖)(𝐾𝑠
(𝑖))
𝑇/√𝑑𝑘) 𝑉𝑠
(𝑖),
(5)
 
where 𝑊𝑠
(𝑂) is the projection parameter matrix, ℎ is the number of parallel attention heads, and 𝑑𝑘 is the 
feature direction of 𝐾𝑠. To compute the query matrix 𝑄𝑠, key matrix 𝐾𝑠, and value matrix 𝑉𝑠, we have 
𝑄𝑠
𝑖= 𝐹𝑠𝑊𝑠
(𝑄,𝑖), 𝐾𝑠
𝑖= 𝐹𝑠𝑊𝑠
(𝐾,𝑖), 𝑉𝑠
𝑖= 𝐹𝑠𝑊𝑠
(𝑉,𝑖),
(6) 
where 𝐹𝑠∈ℝ𝐵𝑇×𝐽×𝑑 is the spatial feature, 𝑊𝑠
(𝑄,𝑖), 𝑊𝑠
(𝐾,𝑖), 𝑊𝑠
(𝑉,𝑖) are projection matrices, and 𝐵 is the batch 
size. S-GCN is defined as: 
𝐺𝐶𝑁(𝐹(𝑖)) = 𝜎(𝐹(𝑖) + 𝑁𝑜𝑟𝑚(𝐷̃ −1/2𝐴̃𝐷̃ −1/2𝐹(𝑖)𝑊1 + 𝐹(𝑖)𝑊2)) ,
(7) 
where 𝐴̃ = 𝐴+ 𝐼𝑁 represents the adjacency matrix with self-connections added, 𝐼𝑁 represents the identity 
matrix, 𝐷̃𝑖𝑖= ∑𝑗𝐴̃𝑗𝑗 is defined as the sum of elements along the diagonal of 𝐴̃, 𝑊1 and 𝑊2 represent the 
trainable weight matrices specific to each layer, and 𝜎 is the activation function. 
Next, we adaptively fuse S-MHSA and S-GCN and add residual connections to capture the global and 
local spatial dependencies between joints. Then, the fused result is input into a multilayer perceptron (MLP), 
followed by LayerNorm and residual connection operations. 
Temporal Mixformer Block. Similar to the Spatial Mixformer Block, this module has a similar structural 
flow, but the difference lies in the choice of MHSA and GCN. Specifically, the Temporal Mixformer Block 
uses Temporal Multi-Head Self-Attention (T-MHSA) and Temporal GCN (T-GCN) as components, so this 
module treats each frame as a token to capture relationships between consecutive frames. Among them, T-
MHSA can be similarly expressed as: 
T −MHSA(𝑄𝑇, 𝐾𝑇, 𝑉𝑇) = 𝐶𝑜𝑛𝑐𝑎𝑡(ℎ𝑒𝑎𝑑𝑖, … , ℎ𝑒𝑎𝑑ℎ)𝑊𝑇
(𝑂),
ℎ𝑒𝑎𝑑𝑖= 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝑄𝑇
(𝑖)(𝐾𝑇
(𝑖))
𝑇/√𝑑𝑘) 𝑉𝑇
(𝑖),
(8)
 
where 𝑄𝑇,𝐾𝑇, and 𝑉𝑇 are computed similarly to formula (6). For T-GCN, it differs from S-GCN in their 
adjacency matrices and input features. Additionally, S-GCN uses 𝑆𝑖𝑚(𝐹𝑇
𝑡𝑖，𝐹𝑇
𝑡𝑗) = （𝐹𝑇
𝑡𝑖）
𝑇𝐹𝑇
𝑡𝑗 to calculate 
the similarity between individual joints in different time ranges. Subsequently, similar to the Spatial 
Mixformer Block, it also adaptively fuses S-MHSA and S-GCN and adds residual connections, followed by 
LayerNorm and residual connection operations. 
3.2.2. Squeeze-and-Excitation Layer 
The SE Layer can be considered a self-attention mechanism, where the features of the convolutional layer 
are reweighted based on the average of all features in that layer, suppressing or emphasizing specific features 
by multiplying the corresponding features by appropriate scalars. Using it after the Mixformer Block further 
compensates for the deficiencies of GCN in global information modeling and MHSA in ignoring local 
structural dependencies, while reducing the error accumulation of human end joints. 
The SE Layer uses the adaptively fused Mixformer Block as input. First, in the Squeeze stage, the number 
of time frames T and the number of joints J are pooled to compress the information in these two dimensions. 
The Squeeze process is defined as: 
𝑥𝑠𝑞𝑢𝑒𝑒𝑧𝑒𝑑= 1/𝑇× 𝐽∑∑𝑥𝑏,𝑡,𝑗,𝑐
𝐽
𝑗=1
𝑇
𝑡=1
,
(9) 
where 𝑥𝑏,𝑡,𝑗,𝑐 is the element of tensor x in the input dimension. Then, in the Excitation stage, we use a 
small MLP consisting of two fully connected layers (FC). The first fully connected layer (using ReLU 

8 
Author name / Procedia Economics and Finance 00 (2012) 000–000 
activation function) performs dimensionality reduction, and then the second fully connected layer restores it. 
Finally, the Sigmoid activation function ensures that the obtained weights are between [0,1]. The Excitation 
stage can be exp

## experiments
4.1. Datasets and Evaluation Metrics 
We comprehensively validated the proposed model (MixTGFormer) on two large-scale 3D human pose 
estimation datasets (Human3.6M [44] and MPI-INF-3DHP [45]). 
The Human3.6M dataset is the most commonly used dataset in 3D human pose estimation. It includes 3.6 
million video frames of 11 professional subjects performing 15 different daily activities, captured by 4 
cameras from different perspectives. To ensure fair evaluation, we followed the evaluation method of most 
previous works, using the data of subjects 1, 5, 6, 7, and 8 for model training and the data of subjects 9 and 11 
for testing. We selected two metrics to evaluate the model: MPJPE and P-MPJPE. MPJPE (called P1) 
calculates the mean per joint position error in millimeters between the estimated pose and the actual pose after 
aligning the root node (sacrum). P-MPJPE (called P2) requires the actual pose and the estimated pose to be 
aligned through rigid transformation to further calculate the loss. Fig. 3 shows the results of our model on the 
P1 metric. 

 
Author name / Procedia Economics and Finance 00 (2012) 000–000 
9 
Fig. 3. Comparison with other 3D human pose estimation methods on the Human3.6M dataset. MPJPE represents the mean (per) joint 
position error (the lower the better), and Param represents the number of parameters. 
The MPI-INF-3DHP dataset is another large-scale dataset commonly used in 3D human pose estimation, 
with three different settings: green screen, non-green screen, and outdoor environment. Following the 
evaluation method of previous works, we adopted MPJPE, the percentage of correct keypoints within 150mm 
(PCK), and the area under the curve (AUC) as evaluation metrics. 
4.2. Implementation Details 
We constructed two models with different layer configurations: MixTGFormer-s and MixTGFormer, to 
adapt to different application requirements. The specific parameters are shown in Table 1. 
Table 1. Details of the two versions of the MixTGFormer model. L: Number of layers. T: Number.

## related_work
2.1. 3D Human Pose Estimation 
3D human pose estimation is a classic and important problem in the field of computer vision, with decades 
of research history [26]. In the early stages [27,28,29], this work relied almost entirely on handcrafted features 
and geometric constraints as means to predict 3D human poses. With the rapid development of deep learning, 
deep learning has now become the primary method for 3D human pose estimation [30]. This problem can be 
classified from different perspectives, such as based on input data and estimation methods. 
From the perspective of input data, the input data can be divided into multi-view and monocular views. 
While multi-view methods can provide richer spatial information, they require the simultaneous use of 
multiple cameras from different angles, which is not highly feasible in practical application scenarios. 
Monocular methods, although lacking depth information, are simpler in data collection, have lower hardware 
costs, and are easier to deploy and use. Additionally, the rapid development of computer vision technology 
has largely compensated for the depth issue, making monocular methods the mainstream approach. This study 
also uses monocular input. 
From the perspective of estimation methods, 3D pose estimation can be divided into direct estimation and 
2D-to-3D lifting. Direct estimation directly estimates 3D poses from images using convolutional networks 
[31], characterized by simplicity but lower accuracy. For example, Pavlakos et al. [7] used voxel likelihood to 
represent the confidence of joint positions in 3D space and inferred joint details through 3D heatmaps, but the 
model showed sensitivity to irrelevant factors. The 2D-to-3D lifting method first uses a 2D human pose 
detector to extract 2D poses, and then an independent method lifts the estimated 2D human poses to 3D 
human poses. This method relies on effective 2D pose detectors, and researchers mainly focus on the lifting 
from 2D to 3D poses. This strategy typically yields higher data accuracy. The methods used in the pose lifting 
stage include Temporal Convolutional Networks (TCN) [32,33], GCN, and Transformer. Currently, the most 
commonly used methods are based on Transformer and GCN. Additionally, some researchers have recently 
mixed these methods and achieved excellent results. In this paper, our work is also based on the 2D-to-3D 
lifting method, with mixed improvements using Transformer and GCN. 
2.2. Transformer-based Methods 
Transformer was first proposed by Vaswani et al. [20] and demonstrated outstanding performance in 
natural language processing (NLP). It later entered the field of 3D human pose estimation and also achieved 
good results. Poseformer [8] was the first work to completely use Transformer as the backbone network in 3D 
human pose estimation. It achieved prediction by modeling spatial and temporal information, significantly 
outperforming previous CNN-based methods. PoseformerV2 [18] expanded the receptive field by utilizing 
compact representations of lengthy skeleton sequences in the frequency domain and improved robustness to 
sudden movements in noisy data. The proposed method effectively fused temporal and frequency domain 
features. MHFormer [34] incorporated multi-hypothesis spatio-temporal feature hierarchies into the model, 
independently and mutually processing multiple hypothesis information of body joints in an end-to-end 

4 
Author name / Procedia Economics and Finance 00 (2012) 000–000 
manner, and averaging the target 3D poses. P-STMO [35] proposed masked pose modeling, applying masked 
joint modeling to 3D human pose estimation through self-supervised learning. STCFormer [36] partitioned 
the input joint features into two partitions, separately performing spatial and temporal attention, and used 
Multi-Head Self-Attention (MHSA) to encapsulate spatial and temporal context in parallel. MotionBERT [19] 
proposed a unified perspective, learning general representations of human motion from large-scale, diverse 
data, and then completing various human-centered downstream video tasks in a unified paradigm. These 
methods use different approaches to combine the temporal and spatial relationships of human skeleton points 
from different perspectives, but the aggregation and collaborative use of these key information is still 
insufficient, which affects the further approximation of real human poses. 
2.3. GCN-based Methods 
Due to their powerful dynamic relationship capture capabilities, Graph Convolutional Networks are widely 
used in skeleton-based action recognition. Due to their high computational efficiency and similar task types, 
they have also been extensively applied in 3D human pose estimation in recent years. SemGCN [37] 
introduced a semantic graph convolutional network, which uses a stacked structure as a non-local module. 
This architectural choice helps learn the weights between adjacent nodes, thereby enhancing the connections 
between 2D joints. Graph Stacked Hourglass Networks [38] proposed a graph-stacked hourglass network 
model to learn human skeleton representations at different scales. GLA-GCN [39] proposed a global-local 
learning architecture that leverages global spatio-temporal representations and local joint representations in 
GCN-based models for pose estimation. Although GCN-based methods have a lighter memory load, there is 
still a certain gap in performance compared to Transformer-based methods. 
2.4. Fusion Methods 
Hybrid models that combine the advantages of Transformer and Graph Convolution have recently attracted 
much attention, and these integrated methods have produced state-of-the-art results on many public datasets. 
GraFormer [40] replaced the multi-layer perceptron of Transformer with learnable GCN layers to form the 
GraAttention module, while ChebGConv [41] modeled implicit connection relationships between non-
adjacent joints. DiffPose [42] interleaved GCN layers with self-attention layers as a diffusion model, which 
can capture spatial features between joints based on human skeletons. MotionAGFormer [43] combined the 
advantages of both, adaptively fusing features extracted from Transformer and Graph Convolutional 
Networks, achieving a comprehensive and balanced representation of human motion. However, although 
these methods have achieved excellent results, similar to GCN-based methods, they are strong in learning 
spatial information of a single pose but relatively weak in learning temporal correlations between different 
frames.

## conclusion
In this paper, we proposed MixTGFormer, a novel Transformer and GCN-based 3D human pose 
estimation model. It adopts a dual-stream fusion mechanism in both the backbone network and core 
components, which enhances the model's understanding and feature-capture capabilities of global and local 
spatio-temporal relationships of human skeletons. Additionally, we introduced the Squeeze-and-Excitation 
Layer to further enhance the model's ability to address the neglect of specific modeling features. Extensive 
experimental evaluations demonstrated the effectiveness of our method on Human3.6M and MPI-INF-3DHP, 
with results surpassing current state-of-the-art algorithms. 
Although our model has achieved excellent performance on multiple benchmark datasets, there are still 
some potential directions for improvement worth exploring, including further optimizing the model structure 
to reduce computational complexity or expanding the model's application scenarios, such as real-time pose 
tracking and multi-view pose estimation. 

14 
Author name / Procedia Economics and Finance 00 (2012) 000–000 
Funding 
This work was supported by the Zhejiang Provincial Key Research and Development Project(No. 
2025C02045). 
CRediT authorship contribution statement 
Jiawen Duan: Writing – original draft, Validation, Visualization, Investigation. Jian Xiang: Writing – 
review and editing, Supervision, Methodology. Zhiqiang Li: Data curation, Formal analysis. Linlin Xue: 
Investigation, Methodology. 
Declaration of competing interest 
The authors declare that they have no known competing financial interests or personal relationships that 
could have appeared to influence the work reported in this paper. 
Data availability 
Data will be made available on request.