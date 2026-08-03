# A deep-learning based gait classification and anomaly detection framework for healthcare surveillance

> 2026 · id: W7169500617 · pdf: https://www.nature.com/articles/s41598-026-62788-6_reference.pdf · 来源: pdf_url
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

ARTICLE IN PRESS
Article in Press
A deep-learning based gait classification and 
anomaly detection framework for healthcare 
surveillance
Scientific Reports
Received: 15 March 2026
Accepted: 14 July 2026
Cite this article as: Alfridi M.F. A deep-
learning based gait classification 
and anomaly detection framework 
for healthcare surveillance. Sci Rep 
(2026). https://doi.org/10.1038/
s41598-026-62788-6
Mohammad F. Alfridi
We are providing an unedited version of this manuscript to give early access to its 
findings. Before final publication, the manuscript will undergo further editing. Please 
note there may be errors present which affect the content, and all legal disclaimers 
apply.
If this paper is publishing under a Transparent Peer Review model then Peer 
Review reports will publish with the final article.
https://doi.org/10.1038/s41598-026-62788-6
© The Author(s) 2026. Open Access This article is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International 
License, which permits any non-commercial use, sharing, distribution and reproduction in any medium or format, as long as you give appropriate credit 
to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if you modified the licensed material. You do 
not have permission under this licence to share adapted material derived from this article or parts of it. The images or other third party material in this 
article are included in the article’s Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the 
article’s Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain 
permission directly from the copyright holder. To view a copy of this licence, visit http://creativecommons.org/licenses/by-nc-nd/4.0/.

A Deep-Learning Based Gait Classification and Anomaly Detection Framework for 
Healthcare Surveillance
Mohammad F Alfridi 1*
1*Department of Health Informatics, College of Applied Medical Sciences, Qassim University, Buraydah 51452, Saudi 
Arabia; moh.alfridi@qu.edu.sa, mfalharbi.2024@gmail.com
*Correspondence: mfalharbi.2024@gmail.com
Abstract
Gait classification and anomaly detection are non-intrusive approaches that can support healthcare surveillance and 
clinical gait analysis by identifying abnormal walking patterns. Despite recent advancements, existing methods remain 
limited by model complexity, high computational requirements, and privacy concerns. This study proposes a unified 
framework that combines three complementary components: (i) transformer-based temporal modeling to capture both 
short-term and long-term gait dynamics, (ii) lightweight architectures for efficient deployment on edge devices, and 
(iii) federated learning (FL) for privacy-preserving distributed training without requiring raw data sharing. 
Experiments were conducted on a balanced subset of 60,000 images from the Gait Detection Processed dataset using 
a controlled federated learning environment designed as a proof-of-concept evaluation rather than a large-scale 
deployment setting. The dataset consisted of three categories: background/non-gait, normal gait, and abnormal gait. 
Vision Transformer (ViT), ConvLSTM, and MobileViT architectures were evaluated for three-class gait classification 
and anomaly detection under a federated learning setting. Among the evaluated models, MobileViT-Large achieved 
the highest performance with 97.2% accuracy, 96.8% precision, 97.5% recall, and 97.1% F1-score, although it 
required higher computational resources and showed greater overfitting tendencies. MobileViT-Small achieved the 
best balance between efficiency and performance with 94.0% accuracy, making it more suitable for edge deployment. 
SHAP-based analysis further showed that the models focused on meaningful gait regions, such as torso and limb 
movements. This proposed framework provides a comparative benchmark of recurrent and transformer-based 
architectures within a privacy-preserving framework for healthcare monitoring applications.
Keywords: Gait classification; Gait anomaly detection; Transformers; Lightweight models; Federated learning; Privacy-
preserving biometrics; Explainable AI
1.
Introduction
Walking enables passive and long-term monitoring, making it suitable for healthcare and surveillance applications 
[1]. Gait classification and anomaly detection are studied, specifically the analysis of gait patterns to distinguish 
normal gait from abnormal gait characteristics rather than identity-based person recognition [2]. In clinical contexts, 
gait anomaly detection can assist in identifying movement disorders, whereas in surveillance applications it can 
support the detection of irregular walking patterns without requiring active user cooperation [3]. However, gait 
analysis remains challenging because it is influenced by both intrinsic and extrinsic factors. Intrinsic factors include 
body morphology, age, and health conditions, whereas extrinsic factors include clothing variations, carried objects, 
occlusions, and environmental conditions. These sources of variability make the development of robust and 
generalizable models for real-world deployment a challenging task.
In literature, gait recognition is generally known as gait based cross subject identity verification or person recognition. 
The current research, however, doesn't conduct identity verification, subject re-identification or biometric 
authentication. There it employs the term gait classification, as a 3-class silhouette based classification task between 
background/non-gait class, normal gait class and abnormal gait class. Hence, the emphasis is on the detection of gait 
anomaly specific to healthcare and the classification of gait patterns based on surveillance rather than identifying the 
biometric identity of the person [4].
Traditional approaches of gait analysis used handcrafted description methods such as silhouette descriptors and Gait 
Energy Images. While silhouette based methods work well in controlled conditions, as recognised from previous 
research [5] they are not robust to variations in appearance, clothes, occlusion, and the environment. Human pose 
estimation methods and those based on the kinematics with modelling of joint trajectory will give richer structural 
information, but they need precision localization of body-joints and in this case, the computational complexity may 
ACCEPTED MANUSCRIPT
ARTICLE IN PRESS
ARTICLE IN PRESS
ARTICLE IN PRESS

be higher [6]. Convolutional, recurrent and graph-based neural networks developed from data using deep learning to 
learn the spatiotemporal gait features directly [7]. These models have achieved great success but still have some 
drawbacks, such as limited ability to model long-term time dependencies, being prone to occlusion, and lacking in 
knowledge in resource-limited settings [8].
There are three principally identified research gaps from recent literature on gait analysis, temporal modeling and 
privacy-preserving learning that motivate this proposed investigation. The first is that CNN-based and RNN-based 
approaches can model local temporal dynamics well, but are unable to model gait dependencies which are far apart in 
time. Second, a large number of high-performing architectures still are cost intensive and not suitable for resource 
constrained edge environments. Finally, there is a lack of investigation of privacy preserving distributed learning 
strategies when it comes to healthcare oriented gait analysis in the face of growing attention to sensitive visual data.
First, CNN and RNN based models are efficient to learn local forms of temporal dynamics but are unable to learn the 
long range gait pattern [8]. Transformers offer a viable alternative by combining to model long range as well as short 
range dependencies in a self-attentive fashion. Second, most of the state-of-the-art designs are still computationally 
intensive and can only be used in latency- or memory-constrained edge devices [9]. This inspires the development of 
lightweight architectures that provide flexible and balanced accuracy and efficiency. Third, centralization of training 
data makes privacy in surveillance and healthcare even more critical. FL, which consists of training models in a 
distributed manner without exchanging uncoded data [10], can handle this threat. However, the potential for its use in 
conjunction with highly developed temporal models, for gait recognition, has yet to be explored.
Addressing challenges in gait classification and anomaly detection, this research proposes three complementary 
modeling techniques: temporal modeling, lightweight architecture design and federated learning. In order to capture 
gait dependencies both at shorter and longer range, as well as to model richer gait dynamics, temporal transformer is 
used. Lightweight designs, and even joint design of the model and selection of the features enhance the performance 
for deployment on the edge. Federated learning allows privacy protected learning among decentralized clients without 
releasing sensitive data.
This study makes four major contributions. The proposed experiments are intended as proof-of-concept evaluations 
for healthcare surveillance and clinical gait monitoring applications rather than identity-based recognition systems. 
First, it presents a comparative analysis of recurrent- and transformer-based federated architectures. The analysis 
evaluates their performance in terms of convergence, generalization, and computational efficiency. Second, it 
compares MobileViT-Large and MobileViT-Small models. The results show that smaller models generalize better 
under heterogeneous client distributions, making them more suitable for edge deployment. Third, SHAP-based 
explainability analysis demonstrates that the models rely on meaningful gait features, such as torso posture and limb 
movement, rather than background information. Finally, a controlled ablation study examines the effects of 
architecture type, model scale, and temporal modeling strategy. This provides a clearer understanding of how to design 
accurate, interpretable, and privacy-preserving gait recognition systems.
2. Literature Review
This section presents a structured review of recent literature related to gait classification and anomaly detection, 
focusing on three interconnected themes: (i) temporal modeling approaches for learning gait dynamics, (ii) lightweight 
architectures for efficient deployment in resource-constrained environments, and (iii) privacy-preserving learning 
strategies for healthcare and surveillance applications. This organization provides clearer context for understanding 
existing limitations and motivates the proposed framework.
2.1 Gait Classification, Anomaly Detection, and Temporal Modeling
The classical gait recognition approaches exploited the handcrafted descriptors, such as Gait Energy Images and 
silhouettes [11-12]. These methods performed in controlled environments were not successful with occlusion, 
change of clothing or noise. Learned structural knowledge- Model-based approaches to the study of joint 
trajectories and body landmarks that were based on instruction [13] were computationally expensive and required 
precise landmark detection. Earlier deep learning suggested CNNs and RNNs to learn gait features directly from 
spatiotemporal data that gave better recognition accuracy but could not learn a dependency over a longer range of 
motions [14]. Self-supervised vision transformers have been recently explored for gait recognition, demonstrating 
more expressive and generalizable sequential modelling across diverse real-world gait datasets [15]. Multi-scale 
ACCEPTED MANUSCRIPT
ARTICLE IN PRESS
ARTICLE IN PRESS
ARTICLE IN PRESS

context-aware networks and Siamese vision transformers are advances, but the dynamics still remain mostly 
insufficient and utilized ineffectively using both short and long-term information [16].
2.2 Lightweight and Efficient Gait Recognition
Many state of the art gait models have low complexity due the high accuracy, thus impeding their use in real-world 
applications or edge deployments [17]. In order to overcome this, pruning, quantization, and knowledge distillation 
have been used for wider vision applications and lightweight CNNs and fusion based network has been proposed for 
gait [18-19]. While some specialized methods like dilated temporal extractors [20] or part-wise attention mechanisms 
boost spatiotemporal clues selectively. They are usually based on big backbone or with the downside of lower accuracy 
[21]. Hence, more integrated designs with light weight are required that can be optimized in feature extraction and 
architecture together for efficient deployment in the edge.
2.3 Privacy-Preserving and Federated Learning in Biometrics
In centralized systems like surveillance and healthcare, recognizing gaits could pose significant privacy issues as raw 
visual data can reveal sensitive behavior and personally identifiable information. With such traditional methods as 
anonymization, encryption, and differential privacy, only partial protection can be achieved, impacting either the 
scalable performance of the data, or predictive power. Besides, the techniques of person de-identification for 
surveillance systems have grown significant for privacy-preserving as a method that preserves the feature concerning 
tasks, while suppressing information concerning identity [22]. Still, there might be some trade-offs between privacy 
protection and recognition accuracy in these methods. It has been recently emphasized that de-identification is an 
important trend for privacy preserving surveillance systems, since less information regarding the identity can be 
disclosed while still being able to analyze all the surveilled data [23]. Unlike methods for de-identification, avoiding 
central storage and transmission of raw gait data is a way to mitigate privacy concerns in the present work done by 
federated learning. FL has the ability to support learning in a distributed manner without the exchange of raw data; it 
has been applied successfully in biometric tasks like face recognition, medical image processing, and activity 
recognition of human beings [24]. FedGait was the first training benchmark for gait recognition based on a client-
distributed manner [25]. However, FL with advanced temporal models and lightweight architectures are still little 
explored, and issues such as non-IID data, communication overhead, or cross-view heterogeneity are yet to be studied.
2.4 Related Works
Recent research in gait recognition has highlighted several challenges associated with deep learning and multimodal 
approaches. Hybrid methods integrating GANs, CNNs, and LSTMs with silhouette extraction achieved high recognition 
accuracy (97.11%) and demonstrated robustness to appearance variations such as carrying conditions and clothing 
changes; however, cross-dataset generalization was not investigated [11]. Lightweight residual CNN-based methods have 
also been explored for resource-constrained environments, although their performance remains highly dataset-dependent 
[12]. Similarly, CNN- and RNN-based approaches have shown promising applications in rehabilitation gait analysis but 
are limited by data scarcity and high computational demands [13].
Recent self-supervised approaches such as DINO with Vision Transformers (ViT) reduced dependence on labeled data 
but remained sensitive to illumination variations [14]. Residual and graph-based deep architectures improved robustness 
against occlusion and appearance changes; however, they still suffered from limitations such as insufficient evaluation 
under challenging conditions and view-angle sensitivity [16]. Multimodal frameworks combining IMU and vision data 
demonstrated improved robustness but increased system complexity [17].
Furthermore, optimization-based and conventional methods have also been investigated. Deep learning fusion with 
feature optimization achieved improved classification performance but lacked real-time validation [18]. Other 
approaches based on traditional machine learning, pose estimation, and handcrafted representations reduced 
computational complexity but remained sensitive to factors such as viewpoint, footwear, and environmental variations 
[20–21]. A comparative summary of these studies is presented in Table 1.
Table 1: Comparative summary of previous gait-analysis studies showing datasets, tasks, methods, findings, and 
limitations used to identify existing research gaps.
References
Dataset
Task
Methods Used
Performance / 
Findings
Limitations
ACCEPTED MANUSCRIPT
ARTICLE IN PRESS
ARTICLE IN PRESS
ARTICLE IN PRESS

Burges et al., 
2024 [11]
CASIA-B
Gait 
Recognition
GAN, CNN, 
LSTM, silhouette 
extraction
Achieved 97.11% 
accuracy; robust 
to bags and 
clothing 
variations
Cross-dataset 
generalization 
not evaluated
Yi et al., 
2023 [12]
CASIA-B
Gait 
Recognition
Residual CNN
Efficient for low-
power devices
Strong dataset 
dependency
Tassignon et 
al., 2021 [13]
Multiple 
datasets 
(Survey)
Rehabilitation 
gait analysis
CNN, RNN 
survey
Highlights 
rehabilitation 
applications
Data scarcity 
and 
computational 
demands
Pinčić et al., 
2022 [14]
CASIA-B
Gait 
Recognition
DINO, Vision 
Transformer
Reduced labeling 
requirements 
through self-
supervised 
learning
Sensitive to 
illumination 
variations
Shopon et al., 
2021 [16]
CASIA-B
Gait 
Recognition
Residual Graph 
CNN
Robust to 
occlusion
Limited testing 
under extreme 
conditions
Marín-
Jiménez et 
al., 2021 [17]
Multimodal 
gait dataset
Gait 
Recognition
IMU + Vision, 
multimodal 
framework
Strong 
multimodal 
robustness
Increased system 
complexity
Jahangir et 
al., 2023 [18]
Benchmark 
gait dataset
Gait 
Recognition
Deep learning 
fusion + PSO
Achieved 94.14% 
accuracy
Real-time 
capability not 
evaluated
Galasso, 
2024 [20]
OpenPose 
dataset
Human pose-
based gait 
analysis
OpenPose + ML
Low 
computational 
cost
View-angle 
sensitivity
Parashar et 
al., 2023 [21]
GEI/CFPI 
benchmark 
dataset
Gait 
Recognition
DCNN, GEI, CFPI
Reduced 
computational 
cost
Sensitive to 
external factors 
(footwear, 
mood, 
environmental 
changes)
Nithyakani & 
Ukrit, 2024 
[22]
Kinect-
based 
dataset
Authentication
Deep capsule 
network
Secure 
authentication 
framework
Lower accuracy 
under complex 
settings
Although there has been significant advances in gait analysis research, there are a number of questions which still have 
not been answered. Prior techniques often focus on optimizing one of these dimensions without addressing the interplay 
among multiple dimensions, e.g., accuracy, efficiency, or privacy. Moreover, most of the studies lack consideration of 
federated learning settings and deployment constraints and address person identification tasks in controlled scenarios, 
with providing limited consideration to healthcare-oriented gait classification. The limitations are yet to be resolved, 
which drives the proposed framework.
ACCEPTED MANUSCRIPT
ARTICLE IN PRESS
ARTICLE IN PRESS
ARTICLE IN PRESS

3.
Materials and Methods
3.1 Dataset Description
The 
data 
used 
in 
this 
study 
were 
publicly 
collected 
from 
the 
Kaggle 
website 
(https://www.kaggle.com/datasets/rajakali/gait-detection-processed-data) and are part of a larger Gait Detection 
Processed Data collection. The data were binary silhouettes for various walking patterns. Silhouette representations 
are sensitive to appearance variance and segmentation quality, but were chosen because silhouette information is less 
visually identifiable than other information, less demanding in terms of computational resources and fits well with the 
healthcare surveillance context and federated edge-learning context. In this work, a balanced sub-set of classes which 
included three Background/Non-gait (bg), Normal gait (cl) and Abnormal gait (nm) were selected. About 14,000 
images were provided in each class as training set and 3,000 images for validation and test sets. The subset was chosen 
based on the three following criteria: (i) only feasible data to train the model with available hardware resources were 
considered; (ii) data were chosen based on class balance to avoid bias in the training process; and (iii) experiments 
were performed with reproducible data to allow for comparable experiments. Compared to existing benchmark 
datasets, including CASIA-B and OU-ISIR, basic gait recognition research is mostly dedicated to identity based 
person recognition and cross view evaluation. However, the selected Gait Detection Processed Data data was viewed 
as being more suited for this proof-of-concept study due to the fact that it offers categorized silhouette samples for 
background/non-gait, normal gait, and abnormal gait data, which directly agree with the healthcare oriented gait 
classification and anomaly detection goals of the current study. Figure 1, left, shows the representative silhouettes of 
the three classes selected. The selected data returned silhouette based images samples ordered by gait classes; 
participant-level data (number of subjects, demo data and recording per subject) are unknown and not included in the 
original source data. Accordingly, here we look into image-based gait classification and anomaly detection instead of 
subject-wise longitudinal analysis. This restriction is recognized since the diversity and repetitions of the participants 
will affect the generalization of the deployment scenarios.
Figure 1: Representative Silhouette Samples for Background, Normal, and Abnormal Gait Classes
The dataset in this study is only from a single mode where that is a silhouette of the 3D model, so it is not very strange 
to the different modes and perspectives. Given the computational constraints and the differences between the CASIA-
B and OU-ISIR data sets, the standard data sets were not evaluated externally in the present proof of concept. Hence, 
it is advisable to study the results of gait recognition methods, which are based on the benchmark databases, with 
cautiousness. The evaluation will be further expanded to include the benchmark datasets with multiple views from the 
same subject, namely CASIA-B and OU-ISIR, to test the robustness, cross-view invariance and out-of-dataset 
generalization of models. In addition, a cross-view evaluation protocol (e.g., training on one view and testing on 
another view) will be adopted to assess the performances of the models under the real deployment environment.
3.2 Preprocessing and Stratified Data Splitting
Preprocessing pipeline included all the images were converted to RGB color space and resized to 224 × 224 pixels, with 
contrast-adjustment of 1.2 and slight denoising in a median filter of size 3 pixels. Since the original inputs are binary 
silhouette images, no color-based information was introduced in RGB conversion, which was only applied to match 3 
channel input format required by pretrained ViT, MobileViT. Neither an explicit silhouette reconstruction nor a 
morphological gap-filling/ contour completion procedure was used. As a consequence the minor discontinuities and 
ACCEPTED MANUSCRIPT
ARTICLE IN PRESS
ARTICLE IN PRESS
ARTICLE IN PRESS

imperfect body outline found in some silhouette samples are inherent to the original data set and are not a part of the 
preprocessing process.
Stratified sampling was done based on images to divide the datasets into 70 percent as a training set, 15 percent as 
validation set, and 15 percent as test set while maintaining equal proportions of classes. Full walking sequences or split 
by participant level data was unavailable in the chosen data set. In the current proof of principle study, there are a number 
of visually similar samples that might be present in more than one partition, which is a drawback of this study and might 
affect the generalization performance. The number of images used for the training subset was 41,999, while the number 
of images for the validation and test subsets were approximately 9,000 images. This partition method helped in training 
the models, tuning the hyper parameters and in evaluating the performance of the trained models without the risk of 
overfitting and increase in the level of generalization.
The dataset was balanced for the three classes (bg, cl, nm) for training, validation and test sets. The size of the training 
set was around 14,000 images per class, while the validation and test sets were around 3,000 images per class. This 
balanced distribution ensured unbiased model training and evaluation with equal representation of all categories. This 
balance is crucial for minimizing class bias and enhancing the generalization capability in gait classification applications.
3.3 Framework Architecture
The suggested framework as shown in Figure 2 combines the three elements, namely: (i) sequence modeling with 
temporal transformers, (ii) computational efficiency with lightweight optimization, and (iii) privacy-preserving training 
with FL.
Figure 2: End-to-end workflow of the proposed gait recognition system.
ACCEPTED MANUSCRIPT
ARTICLE IN PRESS
ARTICLE IN PRESS
ARTICLE IN PRESS

The pipeline starts with data pre-processing and enhancement, and then temporal modeling is done by using the multi-head 
attention to learn short-term and long-term dependencies. Lightweight optimization uses systematic pruning and mixed-
precision quantization to minimize FLOPs and latency. FL enables the training to be decentralized with local updates and global 
aggregation hence retaining data privacy. The last steps of the framework are classification and interpretability analysis.
3.4 Theoretical Foundations and Model Selection
The suggested framework is established on the recent accomplishments in deep learning, sequence modeling, and 
explainability. All the components were selected to solve certain issues in gait recognition: temporal modeling, efficiency, 
and interpretability.
3.4.1
Vision Transformer (ViT)
The ViT [25] has the advantage of being modular and highly scaled with the depth of encoder layers, embedding 
dimension, and attention heads being configured flexibly based on task complexity and computing constraints. Lack 
of convolutional inductive bias allows ViT to learn representation of features directly through data, and thus it is 
especially useful when there is a sufficient number of samples or augmented learning methods when being trained. In 
addition, the self-attention mechanism allows the adaptive weighting of features, i.e., salient regions of the image 
using a mechanism that assigns more weight to the final prediction. This general thinking allows power to scale, 
perspective and backdrop confusion. To achieve maximum optimization, dropout, weight regularization, layer 
normalization, and residual connections are used to ensure constant gradient flow and minimize the chances of 
overfitting, respectively. In general, ViT represents a single framework that is capable of integrating representation 
learning, contextual modeling, and interpretability, which is why it is best applied to high-level vision problems, 
including recognition, classification, and temporal or medical image analysis.
3.4.2
Patch Embedding Function
The ViT processes an input image 𝑋∈𝑅𝐻×𝑊×𝐶 by decomposing it into a sequence of non-overlapping patches of 
fixed spatial resolution 𝑃 × 𝑃. The total number of patches is𝑁= 𝐻𝑊
𝑃2 . Each patch is flattened and mapped into a D-
dimensional latent space using a learnable linear projection defined as
 𝑧𝑝= 𝑓patch 𝑥𝑝= 𝑥𝑝𝐸 (1)
where 𝑥𝑝∈𝑅𝑃𝟚𝐶 denotes the flattened patch and 𝐸∈𝑅(𝑃𝟚𝐶)×𝐷 is the patch embedding matrix. This step converts the 
image into a sequence representation compatible with Transformer-based processing.
3.4.3
Class Token and Positional Encoding Function
To facilitate global image-level representation, a learnable classification token 𝑧cls is prepended to the patch 
embedding sequence. Since Transformers lack intrinsic spatial inductive bias, positional information is incorporated 
through learnable positional embeddings 𝐸pos. The resulting input sequence is expressed as
 𝑍0 = 𝑓pos [𝑧cls;𝑧1;…;𝑧𝑁] = [𝑧cls;𝑧1;…;𝑧𝑁] + 𝐸pos (2)
3.4.4
Multi-Head Self-Attention (MHSA) Function
The core representation learning mechanism of ViT is the multi-head self-attention module, which captures long-
range dependencies across image patches. For an input sequence Z, the query, key, and value matrices are computed 
as
 𝑄= 𝑍𝑊𝑄, 𝐾= 𝑍𝑊𝐾, 𝑉= 𝑍𝑊𝑉 (3)
Self-attention is then calculated using the scaled dot-product formulation:
ACCEPTED MANUSCRIPT
ARTICLE IN PRESS
ARTICLE IN PRESS
ARTICLE IN PRESS

 𝑓attn(𝑄,𝐾,𝑉) = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥𝑄𝐾⊤
𝑑𝑘
𝑉 (4)
Multiple attention heads operate in parallel, and their outputs are concatenated and linearly projected to enhance 
feature expressiveness and contextual understanding.
3.4.5
Feed-Forward Network (FFN) Function
Each Transformer encoder block incorporates a position-wise feed-forward network that applies non-linear 
transformations independently to each token. The FFN is defined as
 𝑓ffn(𝑥) = 𝐺𝐸𝐿𝑈(𝑥𝑊1 + 𝑏1)𝑊2 + 𝑏2 (5)
where GELU is employed as the activation function to improve learning stability and representational capacity.
Residual Connection and Layer Normalization Function To stabilize optimization and enable deep network stacking, 
residual connections and layer normalization are applied after both the MHSA and FFN modules. For the 𝑙-th encoder 
layer, these operations are expressed as
 𝑍′
𝑙= 𝑍𝑙―1 + 𝑓attn 𝐿𝑁(𝑍𝑙―1) , (6)
 𝑍𝑙= 𝑍′
𝑙+ 𝑓ffn 𝐿𝑁𝑍′
𝑙
 (7)
3.4.6
Classification Head Function
After passing through 𝐿 stacked Transformer encoder layers, the final representation of the classification token 𝑧𝐿cls is 
extracted as a global descriptor of the input image. A task-specific linear classification head followed by a softmax 
activation produces the final prediction:
 𝑦= 𝑓cls(𝑧𝐿cls) = 𝑊head 𝑧𝐿cls (8)
3.4.7
MobileViT
MobileViT [26] is a promising bridge between convolutional neural networks and Transformers, incorporating 
lightweight self-attention. Although convolutional layers maintain very high inductive bias in learning local features, 
the Transformer module learns much better the global context without having to pay the high cost of full-image 
attention. The hybrid architecture greatly decreases the number of parameters and FLOPs, and MobileViT is very 
suitable to real-time inference in edge devices. Also, the selective use of attention enhances robustness to spatial 
differences as well as complicated patterns which presents a trade off between accuracy and efficiency. Therefore, 
MobileViT can be used more successfully in vision tasks that are resource-constrained, including gait recognition, 
mobile healthcare imaging, and embedded surveillance systems.
3.4.8
Local Feature Extraction (Convolutional Encoding) Function
MobileViT begins by extracting local spatial features from the input image 𝑋∈𝑅𝐻×𝑊×𝐶using lightweight 
convolutional layers. A standard convolution operation is defined as
 𝐹local = 𝑓conv(𝑋) = 𝑋∗𝑊conv + 𝑏conv (9)
where ∗ denotes convolution, 𝑊conv represents convolutional filters, and 𝐹local captures fine-grained local patterns 
such as edges, textures, and spatial continuity.
3.4.9
Patch Formation and Linear Projection Function
ACCEPTED MANUSCRIPT
ARTICLE IN PRESS
ARTICLE IN PRESS
ARTICLE IN PRESS

The extracted feature map 𝐹local is partitioned into small non-overlapping patches of size 𝑃 × 𝑃. These patches are 
flattened and linearly projected into a lower-dimensional embedding space:
 𝑧𝑝= 𝑓patch 𝑓𝑝= 𝑓𝑝𝐸 (10)
where 𝑓𝑝∈𝑅𝑃𝟚𝐶′′ denotes a flattened feature patch and 𝐸∈𝑅(𝑃𝟚𝐶′)×𝐷 is the learnable projection matrix. This step 
converts convolutional features into tokenized representations suitable for Transformer-based modeling.
3.4.10 Lightweight Transformer Encoding Function
Unlike ViT, MobileViT applies Transformer encoding only to local feature patches, significantly reducing 
computational complexity. For the token sequence 𝑍, queries, keys, and values are computed as
 𝑄= 𝑍𝑊𝑄, 𝐾= 𝑍𝑊𝐾, 𝑉= 𝑍𝑊𝑉 (11)
The self-attention mechanism is defined as
 𝑓attn(𝑄,𝐾,𝑉) = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥𝑄𝐾⊤
𝑑𝑘
𝑉 (12)
allowing MobileViT to model global contextual relationships within localized regions while maintaining efficiency.
3.4.11 Feature Fusion Function
The Transformer-encoded features are reshaped back to their spatial form and fused with the original convolutional 
features through channel-wise concatenation and convolution:
 𝐹fusion = 𝑓fusion [𝐹local,𝐹global] (13)
where 𝐹global represents Transformer-enhanced features. This fusion mechanism combines local spatial precision with 
global contextual awareness.
3.4.12 Residual Learning and Normalization Function
To ensure stable training and efficient gradient flow, MobileViT employs residual connections and normalization 
layers across both convolutional and Transformer components:
 𝐹out = 𝐹in +𝑓𝐿𝑁(𝐹in) (14)
This design enables deeper architectures without significant performance degradation.
3.4.13
Explainable AI with SHAP
Deep neural networks are occasionally accused of being black boxes, thereby limiting their use in sensitive fields such 
as healthcare and surveillance. To address this problem, the proposed framework will use Shapley Additive 
Explanations (SHAP), a game-theoretic model that quantifies the value of input features in a model. SHAP can not 
only provide local explanations (per-sample attributions), but also global (aggregate importance), making it possible 
to visualize silhouette regions that drive classification. In the present framework, SHAP highlights the silhouette 
regions that contribute most strongly to frame-level classification decisions, such as torso posture and limb-related 
spatial features, rather than directly modeling temporally evolving gait dynamics. Interpretability not only enhances 
trust but also demonstrates that the acquired features are semantically related to human-comprehensible gait dynamics.
3.5 Integrated Framework
ACCEPTED MANUSCRIPT
ARTICLE IN PRESS
ARTICLE IN PRESS
ARTICLE IN PRESS

These factors are integrated synergistically to exploit complementary benefits. The representation of spatial detail uses 
CNN-based preprocessing modules (such as MobileViT and ViT embeddings), and the representation of sequential 
dependencies uses temporal models, such as ConvLSTM and transformers. Transformers have long-range modeling 
capabilities, and ConvLSTM has an inductive bias toward motion continuity. MobileViT is a lightweight architecture 
that can be deployed on edge devices, and SHAP-based explainability can help address the performance-
interpretability gap. Such a mix of integration will produce a scalable, privacy-conscious gait recognition system that 
erodes the accuracy, efficiency, and transparency of its real-world application.
3.6Federated Learning Strategy
Deep learning models can be trained in a privacy-conscious, scalable manner using the FL approach, and 
collaboratively across distributed systems. FL is particularly applicable to sensitive application areas such as 
healthcare monitoring, surveillance, and edge-based gait classification because it enables distributed model learning 
without requiring centralized storage of sensitive gait data. Although federated optimization introduces the following 
challenges, including communication overhead and data heterogeneity, the FedAvg-based framework is a powerful 
and efficient solution. Besides enabling the efficient deployment of lightweight architectures, such as MobileViT, and 
time-constrained models, such as ConvLSTM, FL can also guarantee high performance and robust data privacy 
controls.
As shown in Figure 3, aggregated distributed training architecture using federated learning (FL) is proposed to classify 
gaits of each client on layer of privacy concerns. In this architecture, gait raw data are not sent to the cloud and thus, 
local models are trained by silhouette-based gait data stored at the edge devices or institutional nodes. Encrypted or 
anonymized model updates, such as model weights or gradients, are periodically sent to the center on aggregation 
server for secure global model fusion and shared model optimization. Thereafter, the new world level parameters are 
again transmitted to local clients who are participating in this round of training. The proposed framework does not 
require training of a separate model for each individual subject, rather it uses a shared global model with optional 
client level adaptation. This architecture is helpful for expanding usability and deployment in clinical and public 
settings with lots of individuals.
ACCEPTED MANUSCRIPT
ARTICLE IN PRESS
ARTICLE IN PRESS
ARTICLE IN PRESS

Figure 3. Federated Learning–Based Distributed Training Framework for Privacy-Preserving Gait 
Recognition
3.7 Experimental Setup
All experiments were conducted using TensorFlow/Keras on a GPU-enabled computational platform. Data 
augmentation techniques, including rotation, flipping, and shifting, together with batch normalization, were applied 
to improve model generalization. The federated learning setup employed multiple distributed clients with iterative 
communication rounds and centralized aggregation. The selected experimental configuration was designed to provide 
a balance between computational feasibility and reproducibility and is summarized in Table 2.
The current federated learning configuration represents a controlled proof-of-concept evaluation rather than a large-
scale deployment scenario. Although the selected setup allows effective analysis of model behavior within a 
distributed environment, larger client populations and more extensive communication rounds may provide further 
insights into scalability and heterogeneous learning conditions. Future work can therefore investigate larger 
communication settings, non-IID client distributions, and communication-related measures such as bandwidth 
utilization and aggregation latency to enable a more comprehensive evaluation of federated scalability.
Table 2: Model parameters, model size, and communication overhead under federated learning configurations.
Model
Params (M)
Model Size (MB)
Comm per Round 
(MB)
Total Comm (MB)
MobileViT-Small  
4.2
15.8
0.85
25.5
Global model 
Personalization
Client-adapted Model
Local Model 1
Dataset 1
Local training 
Global model 
Personalization
Client-adapted Model
Local Model 2
Dataset 1
Local training 
Global model 
Personalization
Client-adapted Model
Local Model 
K -1
Dataset 
K -1
Local training 
Global model 
Personalization
Client-adapted Model
Local Model 
K
Dataset K
Local training 
Server
Global aggregation 
4
Global model 
distribution  
5
Client local model 
upload 
3
Global model distribution  
Client selection 
1
Client 1
Client 2
Client K -1
Client K 
Client local 
model upload  
2
6
User quality evaluation 
Double deep Q-learning 
+
Non - target 
class
Knowledge 
Distillation 
Model 
interpolation 
Local data 
storage 
. . .
ACCEPTED MANUSCRIPT
ARTICLE IN PRESS
ARTICLE IN PRESS
ARTICLE IN PRESS

MobileViT-Large  
7.5
32.4
1.60
48.5
ViT
12.2
56.7
2.80
78.5
4.
Results
Experiments were done in a FL environment; i.e. local training exists in distributed clients and only model updates are sent to a 
central server. The approach maintains privacy and exploits data heterogeneity but creates problems, including the heterogeneity 
of clients and communication overhead.
Figure 4: ConvLSTM Input Output comparison
Figure 4 shows qualitative results from ConvLSTM, where input silhouettes are compared with ground truth and 
generated outputs across two sequences (T and V). The generated silhouettes closely resemble the ground truth 
across varying poses, demonstrating that ConvLSTM effective