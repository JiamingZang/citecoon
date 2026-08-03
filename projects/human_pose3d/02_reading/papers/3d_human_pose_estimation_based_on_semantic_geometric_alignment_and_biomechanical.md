# 3D Human Pose Estimation Based on Semantic-Geometric Alignment and Biomechanical Priors Using UWB MIMO Radar

> 2026 · id: W7169661113 · pdf: https://link.springer.com/content/pdf/10.1007/978-981-92-3520-9_44.pdf · 来源: pdf_url
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

3D Human Pose Estimation Based 
on Semantic-Geometric Alignment 
and Biomechanical Priors Using UWB MIMO 
Radar 
Yongkun Songenvelope symbol
, Ke Zhang 
, Xian Liu 
, Qingrong Yang 
, 
and Tianxing Yan 
Changsha University of Science and Technology, Changsha 410114, China 
songyk1118@163.com 
Abstract. Ultra-Wideband Multiple-Input Multiple-Output (UWB MIMO) radar 
shows great potential in non-contact 3D human pose estimation due to its pene-
tration, privacy protection and all-weather capabilities. However, varying Radar 
Cross Section (RCS) across human body parts causes severe spatial density inho-
mogeneity in radar point clouds. Existing methods using uniform sampling and 
global feature aggregation often lose critical features of distal limbs, damage 
point cloud spatial topology, and lead to semantic-geometric misalignment. Addi-
tionally, the lack of biomechanical constraints results in insufﬁcient anatomical 
rationality and practical usability of estimated poses. 
To address these challenges, we propose an end-to-end Semantic-Geometric 
Alignment Network (SGA-Net). It integrates three core designs: density-aware 
spatial (DAS) encoder, Cross-Attention Bridge (CAB) module, and data-driven 
biomechanical loss function, achieving accurate semantic-geometric feature align-
ment and robust 3D pose estimation. Experimental results show that our method 
achieves a Mean Per Joint Position Error (MPJPE) of 29.95 mm, signiﬁcantly out-
performing existing baselines. This work provides a feasible technical paradigm 
for highly robust edge-side behavior monitoring in privacy-sensitive scenarios. 
Keywords: UWB MIMO radar cdot3D human pose estimation (3D HPE) cdot
semantic-geometric alignment cdotcross-attention cdotbiomechanical constraints 
1 
Introduction 
3D HPE serves as the core technical support for cutting-edge applications such as human-
computer interaction and medical rehabilitation. Traditional optical sensors suffer from 
severe performance degradation under occlusion and low-light conditions, and pose 
signiﬁcant privacy leakage risks [1]. 
To break through these limitations, non-contact perception technology based on 
Radio Frequency (RF) signals has developed rapidly. Early WiFi-based methods are 
limited by bandwidth and cannot achieve high-precision reconstruction, while UWB
© The Author(s) 2027 
D.-S. Huang et al. (Eds.): ICIC 2026, LNCS 16672, pp. 556–567, 2027. 
https://doi.org/10.1007/978-981-92-3520-9_44 

3D HPE with UWB MIMO Radar via SGA-Net
557
MIMO radar has become the mainstream technology in this ﬁeld due to its strong pen-
etration, all-weather operation and inherent privacy protection advantages [2, 3]. How-
ever, radar point clouds have inherent defects of high sparsity, non-uniformity and high 
noise. Moreover, signiﬁcant variations in RCS across different human body parts result 
in extremely weak returns from distal limbs, which severely restricts the accuracy of 
pose estimation [4]. 
Most existing methods directly migrate backbone networks designed for dense point 
clouds, and suffer from two core limitations: First, global pooling discards local topolog-
ical relationships and fails to establish explicit alignment between geometric point clouds 
and semantic joints, leading to semantic-geometric misalignment and severe localization 
drift of distal limbs [5, 6]. Second, they lack effective long-range temporal modeling and 
physical constraints, and are prone to temporal jitter and biomechanically implausible 
poses [7]. 
To address the above challenges, this paper proposes an end-to-end SGA-Net frame-
work: It adopts a density-aware spatial encoder to perform dynamic resampling and pre-
serve critical features of distal limbs; designs a Cross-Attention Bridge (CAB) module 
to achieve accurate semantic alignment via learnable joint queries; employs MotionAG-
Former to model long-range motion dynamics [8]; and ﬁnally constructs a data-driven 
biomechanical loss function to suppress noise and abnormal poses at the physical level. 
The main contributions are summarized as follows: 
1. A novel end-to-end framework SGA-Net is proposed, which is speciﬁcally tailored to 
handle the high sparsity and multipath noise of 3D point clouds obtained from UWB 
MIMO radar. 
2. A CAB module is designed to effectively mitigate the semantic-geometric misalign-
ment caused by traditional global pooling via an active query mechanism, which helps 
improve the localization accuracy of limb joints. 
3. A data-driven BioMechanical Loss is proposed to replace rigid hard-coded con-
straints, which helps ensure the anatomical rationality of human poses and alleviate 
motion jitter. 
2 
Background and Related Work 
UWB MIMO radar is an important technology for non-contact 3D human pose esti-
mation. However, its point clouds suffer from inherent issues including high sparsity, 
density inhomogeneity and multipath noise. Moreover, the extremely weak returns from 
distal limbs caused by signiﬁcant RCS variations across human body parts severely limit 
the accuracy of pose estimation. 
Most existing RF-based perception methods directly adopt backbone networks 
designed for dense point clouds, leading to signiﬁcant performance bottlenecks when 
processing sparse and non-uniform radar data [4]. Traditional point cloud learning and 
feature alignment methods tend to lose critical features of distal limbs and suffer from 
severe semantic-geometric misalignment, which causes signiﬁcant localization drift of 
distal joints [10]. Meanwhile, most methods only adopt single-frame inference or sim-
ple temporal smoothing, lacking effective modeling of long-range motion dynamics 
and being prone to temporal jitter. Conventional biomechanical constraints mostly rely

558
Y. Song et al.
on rigid hard-coded rules with poor generalization ability, which cannot guarantee the 
anatomical plausibility and motion smoothness of human poses [7]. 
3 
Proposed Method 
3.1 
Overall Model Architecture 
Aiming at the physical characteristics of UWB MIMO radar point clouds such as sparsity, 
disorder, and high noise, we propose an end-to-end SGA-Net. Given an input radar point 
cloud sequence containing T frames P\ in {\mathbb {R}}^{T\times N\times 3}, where N is the number of points per 
frame. The core goal of SGA-Net is to accurately predict the corresponding 3D human 
skeleton sequence \w idehat {y}\in {\mathbb {R}}^{T\times J\times 3}, where J is the total number of human joints, and J= 16
in this paper. 
Fig. 1. SGA-Net overall architecture diagram. 
As shown in Fig. 1, the overall forward propagation of SGA-Net consists of three 
core stages: First, the DAS encoder performs dynamic resampling and multi-scale fea-
ture extraction on point clouds frame by frame, outputting feature maps that integrate 
local geometric details and global scene features. Second, the CAB module actively 
retrieves local point cloud features via learnable joint queries, aligning unordered geo-
metric features into ordered joint semantic features. Finally, the spatio-temporal graph 
Transformer models the long-range dependencies of joint features in both temporal and 
topological dimensions, achieving accurate pose coordinate prediction and outputting 
the ﬁnal results. During the training phase, a data-driven biomechanical loss function is 
adopted to perform end-to-end joint optimization of the entire network. 
3.2 
DAS Encoder 
The DAS encoder addresses the limb feature loss problem caused by uniform FPS. It 
consists of density-aware resampling and multi-scale feature extraction: it prioritizes 
sampling in sparse regions via a three-step density-aware strategy to preserve critical 
limb information, then adopts an improved PointNet++ [10] backbone with MSG strategy 
to extract multi-scale features, and ﬁnally outputs local and global feature maps for 
subsequent modules.

3D HPE with UWB MIMO Radar via SGA-Net
559
3.3 
Cross-Attention Alignment and Spatio-Temporal Decoding 
In this section, we propose the CAB module, whose structure is illustrated in Fig. 2. It 
achieves accurate alignment from unordered local point cloud features to human joint 
semantic features via learnable joint queries and multi-head cross-attention mechanism, 
and injects global features in a residual manner to preserve pose context. Subsequently, 
MotionAGFormer is adopted to model joint topological constraints in the spatial dimen-
sion and capture long-range motion dependencies in the temporal dimension, ﬁnally 
outputting smooth 3D pose coordinates.
Fig. 2. Details of the cross-attention bridge module. 
3.4 
Data-Driven BioMechanical Loss 
To address the generation of physically implausible poses and radar-induced temporal 
jitter, this paper proposes a composite loss function integrating dynamic hard sample 
mining and biomechanical priors, whose structure is illustrated in Fig. 3. It enhances 
the learning of distal joints via a weighted MPJPE loss with historical error memory, 
introduces training-set statistics-based bone length constraints and second-order acceler-
ation penalties to ensure the kinematic validity of poses, and adopts a dynamic warm-up 
strategy to balance the optimization process of reconstruction accuracy and physical 
constraints.

560
Y. Song et al.
Fig. 3. Design of the data-driven BioMechanical Loss. 
4 
Experiments 
4.1 
Experimental Validation 
This chapter introduces the core experimental settings, including dataset construction, 
training environment, and hyperparameter conﬁguration, and veriﬁes the performance 
advantages of SGA-Net through multiple experiments. 
We construct a 3D human pose estimation dataset based on UWB MIMO radar. The 
data acquisition system consists of a 12-transmitter 8-receiver UWB MIMO radar and a 
3D motion capture system. The radar adopts the FMCW signal scheme with a bandwidth 
of 1 GHz and an operating frequency band of 2.5 GHz–3.5 GHz. The motion capture 
system outputs the 3D coordinates of 16 key human joints as pose ground truth. The 
photograph of the radar prototype and the array layout are illustrated in Fig. 4. 
(a)MIMO Radar System
(b)MIMO Virtual Array 
Fig. 4. Photograph and Array Layout of the MIMO Radar Prototype. 
During data collection, 8 volunteers are recruited, covering 5 categories of daily 
activities. The dataset is randomly split into training and test sets at a ratio of 4:1. In the 
data preprocessing stage, raw radar data are sequentially processed with MTI processing, 
3D imaging, normalization, isosurface extraction, and FPS to generate 3D point clouds.

3D HPE with UWB MIMO Radar via SGA-Net
561
Temporal samples of 20 frames are then constructed via ﬁxed-interval frame sampling 
(interval s = 3) and sliding window mechanism (window length W = 10).
The proposed SGA-Net is implemented based on PyTorch and runs on the Ubuntu 
22.04 system with a single NVIDIA RTX 4090 GPU for training and inference. The 
batch size is set to 8, and the sequence length T = 20. The Adam optimizer is used with 
an initial learning rate of 5\times 1{0}^{-5}, combined with the ReduceLROnPlateau learning rate 
scheduler, and the minimum learning rate is set to 1\times 1{0}^{-7}. The total number of training 
epochs is set to 100. The loss function takes MPJPE as the base loss and introduces the 
BioLoss constraint with the following weight parameters: {\lamb da }_{\text {bone}}=0.2, {\lambd a }_{\text {angle}}=0.1, 
{\lam bda }_{\text {sym}}=0.1, and {\lam bda }_{\text {acc}}=0.2. 
4.2 
Ablation Study 
Ablation Study on Core Modules. Ablation studies are conducted under uniﬁed exper-
imental conditions to validate the effectiveness of each core module in the proposed 
SGA-Net. The baseline model adopts FPS, global max pooling and standard MPJPE 
loss. We incrementally integrate the DAS module, CAB module and BioLoss into the 
baseline for comparative analysis. 
Quantitative results are summarized in Table 1. The baseline model achieves an 
MPJPE of 42.15 mm. With the integration of the DAS module, the MPJPE decreases to 
36.20 mm, corresponding to an error reduction of 5.95 mm. This demonstrates that the 
proposed sampling strategy adapts to the non-uniform distribution of radar point clouds 
and preserves critical features of distal limbs. 
On the basis of the DAS-equipped model, the addition of the CAB module further 
reduces the MPJPE to 32.45 mm with an additional error reduction of 3.75 mm. This 
module achieves semantic alignment from unordered point cloud features to structured 
human skeletons and mitigates the feature misalignment problem caused by conventional 
global pooling. 
After incorporating the BioLoss constraint, the model achieves the optimal MPJPE of 
29.95 mm with a further error reduction of 2.5 mm. The combination of biomechanical 
priors and data-driven loss further improves reconstruction accuracy. The synergistic 
effect of all modules enables the steady improvement of the overall model performance. 
Table 1. Ablation Experiment Results of Core Innova-
tive Modules. 
DAS
CAB
BioLoss
MPJPE (mm)⬇ 
42.15
√
36.20
√
√
32.45
√
√
√
29.95 
Ablation Study on Temporal Feature Extraction Backbones. This paper conducts 
comparative experiments using the controlled variable method to validate the temporal

562
Y. Song et al.
information extraction performance of the proposed SGA-Net. All baseline models adopt 
the identical PointNet++ spatial feature extraction front-end and training conﬁgurations, 
with only the temporal feature extraction backbone replaced. The comparative methods 
include GRU, LSTM and their bidirectional variants, Transformer, and x-LSTM. 
Table 2. Comparison of Different Temporal Feature Extraction Backbones on Overall Network 
Performance and Computational Efﬁciency. 
Temporal Backbones
MPJPE (mm)
Params (M)
FLOPs (G) 
PointNet++, LSTM
33.41
5.87
8.72 
PointNet++, bi-LSTM
31.96
11.24
8.87 
PointNet++, GRU
37.49
4.51
8.68 
PointNet++, bi-GRU
32.43
10.09
8.80 
PointNet++, Transformer
31.74
17.25
8.61 
SGA-Net
29.95
2.93
7.18 
The performance comparison results of different temporal backbones are presented 
in Table 2. Conventional recurrent neural networks exhibit inherent limitations in mod-
eling long-range dependencies. Their bidirectional variants achieve moderate accuracy 
improvements but incur a twofold increase in model parameters. The standard Trans-
former reduces the MPJPE to 31.74 mm, yet at the cost of a sharp parameter surge 
to 17.25 M. The proposed SGA-Net integrates DAS, feature alignment, and temporal 
decoding architecture, achieving a ﬁnal MPJPE of 29.95 mm. It achieves an approx-
imately 5.6% error reduction over the standard Transformer, while maintaining only 
2.93 M parameters and 7.18 G FLOPs, which basically meets the practical deployment 
requirements of edge-side radar perception systems. 
Five typical human actions covering both static and dynamic motions, namely punch-
ing, waving, standing, arm stretching, and clapping, are selected for visual compara-
tive analysis, and the results are shown in Fig. 5. Conventional recurrent neural net-
works exhibit obvious temporal jitter and distal joint deviation in fast dynamic actions. 
Transformer-based methods are prone to global pose deformation. x-LSTM still suffers 
from defects such as abnormal limb length. The proposed SGA-Net achieves a high 
degree of ﬁt with the ground truth across all actions. It suppresses inter-frame jitter 
via the spatio-temporal graph Transformer module, avoids static joint drift through the 
DAS encoder, and ensures pose plausibility with the biomechanical loss. Meanwhile, its 
localization accuracy for distal limbs is signiﬁcantly higher than that of all comparative 
baselines. 
Performance Analysis of the DAS Module. To further analyze the individual contri-
butions of the density calculation and quota allocation components in the DAS module, 
we design four comparative schemes: the baseline group using conventional FPS sam-
pling, Ablation Group 1 without quota allocation, Ablation Group 2 without density 
calculation, and the complete DAS module.

3D HPE with UWB MIMO Radar via SGA-Net
563
The baseline group adopts traditional uniform sampling, which ignores the non-
uniform distribution of radar point clouds and fails to retain the critical features of distal 
limbs, leading to the worst model performance. Ablation Group 2 only retains the quota 
allocation component, which lacks adaptability to the dynamic variation of point cloud 
distribution and thus achieves unsatisfactory feature extraction. Ablation Group 1 only 
retains the density calculation component; without quota constraints, it cannot precisely 
focus on key limb regions, resulting in limited performance improvement. 
The complete DAS module leverages the collaborative design of density perception 
and quota allocation to dynamically adapt to point cloud distribution and allocate differ-
entiated sampling resources for different limb regions. It effectively preserves the critical 
features of distal limbs, effectively improves the problem of key feature loss caused by 
the sparsity and non-uniformity of radar point clouds, and yields the optimal overall 
model performance. 
Visualization and Performance Analysis of the CAB Module. To intuitively verify 
the feature alignment performance and interpretability of the proposed CAB module, 
we perform visualization analysis, and the results are illustrated in Fig. 6. Four compar-
ative schemes are designed for the experiment, including the human pose ground truth 
benchmark, the model integrated with the CAB module, the control model where the 
CAB module is replaced by a linear layer, and the baseline model without the CAB mod-
ule. A color scale ranging from blue to red is used to characterize the attention weight 
distribution. Cyan points, green lines, and red regions denote human joints, topologi-
cal constraints of the human skeleton, and high-attention areas corresponding to joint 
queries, respectively. 
Fig. 5. Visualization Analysis of Different Temporal Modules.

564
Y. Song et al.
The model embedded with the CAB module achieves precise alignment between 
joint queries and limb point clouds. In the entire sequence of the punching action, the 
attention regions are accurately matched with the positions of real human joints and 
migrate smoothly along the limb motion trajectory, ensuring the temporal coherence of 
input features. In contrast, the control model and the baseline model suffer from severe 
semantic-geometric misalignment. Their high-attention areas are dispersed and deviated 
with numerous invalid points, and the attention distribution exhibits abrupt mutations and 
disorder in the temporal dimension, making it difﬁcult to establish stable spatiotemporal 
semantic correspondences. 
Experimental results demonstrate that the CAB module can realize accurate spatial 
semantic alignment from unordered point clouds to structured human joints within a 
single frame and effectively mitigate the semantic-geometric misalignment problem. 
Meanwhile, it provides temporally coherent and semantically explicit feature inputs 
for the subsequent spatiotemporal decoding module, laying a solid foundation for the 
modeling of long-range temporal dependencies. 
Parameter and Performance Analysis of the Biomechanical Loss Function. Bi-oLoss 
can correct pose prediction biases and suppress motion jitters by imposing anatomi-
cal prior constraints of the human body. As shown by the ablation results in Table 1, 
after incorporating BioLoss into the model integrated with the DAS and CAB mod-
ules, the MPJPE of the model is reduced from 32.45 mm to 29.95 mm. This improve-
ment enhances the reconstruction accuracy while effectively ensuring the anatomical 
rationality of human poses and the temporal smoothness of motion. 
Fig. 6. Visualization Analysis of the Cross-Attention Bridge Module. 
4.3 
Comparative Experiment and Analysis 
This paper conducts comprehensive comparative experiments between the proposed 
SGA-Net and two state-of-the-art mainstream methods for UWB radar-based human

3D HPE with UWB MIMO Radar via SGA-Net
565
pose estimation, namely mm-Pose [11] and UWB-Pose [12]. All experiments are per-
formed under identical training strategies, datasets and hardware environments to ensure 
fair comparison. 
Quantitative comparison results are summarized in Table 3. mm-Pose performs pose 
estimation based on projection data. It achieves an extremely low computational cost 
of only 0.01 GFLOPs with its minimalist design, but loses critical 3D spatial structure 
information and results in the worst reconstruction accuracy with an MPJPE as high as 
60.44 mm. UWB-Pose takes single-frame point clouds as input and fully preserves 3D 
spatial features. It reduces the MPJPE to 49.66 mm with a lightweight design, but fails to 
exploit the temporal information required for dynamic pose reconstruction and thus can-
not adapt to complex dynamic motions. SGA-Net takes multi-frame point clouds as input 
and achieves competitive accuracy among all compared methods. It obtains an MPJPE 
as low as 29.95 mm, which represents a reduction of 19.71 mm and 30.49 mm compared 
with UWB-Pose and mm-Pose respectively. SGA-Net achieves superior pose reconstruc-
tion accuracy with only 2.93 M parameters and reasonably controlled computational cost, 
and effectively balances model complexity and computational efﬁciency. 
To intuitively demonstrate the reconstruction performance, three typical motions 
including punching, waving and arm spreading are selected. The 3D pose reconstruction 
visualization results are presented in Fig. 7. The reconstruction results of SGA-Net are 
highly consistent with the ground truth, with accurate joint positions and limb topology 
conforming to human anatomical principles. In contrast, mm-Pose suffers from severe 
joint misalignment and limb structure distortion in all three motions. UWB-Pose is 
limited by single-frame input and exhibits signiﬁcant joint offset in dynamic motions, 
and cannot guarantee the continuity of motion sequences. Joint-level error distribution 
further indicates that SGA-Net achieves uniform and stable error distribution on both 
dynamic and static joints, while the errors of compared methods increase sharply at 
high-dynamic distal limb joints. 
Fig. 7. Visualization Analysis of Results from Different Methods.

566
Y. Song et al.
Table 3. Error and Parameter Comparison of Different Methods. 
Network Model
Input
MPJPE
Params
FLOPs 
UWB-Pose
Single-frame Point Cloud
49.66
1.47
2.35 
mm-Pose
Projection
60.44
2.39
0.01 
SGA-Net
Multi frame point cloud
29.95
2.93
7.18 
In summary, SGA-Net efﬁciently fuses spatiotemporal features of multi-frame point 
clouds and addresses the problems of semantic-geometric misalignment and insufﬁcient 
temporal feature utilization in existing methods. It outperforms mainstream methods in 
terms of accuracy, computational efﬁciency and practical deployment feasibility. 
5 
Conclusion 
We This paper proposes the SGA-Net to address the accuracy limitation of 3D human 
pose estimation caused by the inherent sparsity and inhomogeneity of UWB radar point 
clouds. The network incorporates three core innovations: First, a DAS is designed to 
preserve critical features of distal limbs from the input stage. Second, a CAB mod-
ule is proposed to achieve precise alignment between geometric features and semantic 
joints via learnable joint queries. Third, a data-driven BioLoss is introduced to suppress 
physically implausible poses by integrating statistical distribution boundaries and kine-
matic constraints. Experimental results demonstrate that SGA-Net reduces the MPJPE 
to 29.95 mm, outperforming existing state-of-the-art methods in terms of accuracy, 
computational efﬁciency, and pose plausibility. 
Acknowledgments. This work was supported in part by the National Natural Science Foundation 
of China under Grant 62401086 and in part by the Youth Fund Project of the Natural Science 
Foundation of Hunan Province under Grant 2024JJ6065. 
Disclosure of Interests. The authors have no competing interests to declare that are relevant to 
the content of this article. 
References 
1. Zhao, M., et al.: Through-wall human pose estimation using radio signals. In: IEEE/CVF 
Conference on Computer Vision and Pattern Recognition (CVPR), pp. 7356–7365 (2018) 
2. An, S., Ogras, U.Y.: MARS: millimeter-wave pattern radar for skeleton-based human pose 
estimation. IEEE Trans. Biomed. Circuits Syst. 16(2), 267–279 (2022) 
3. Zheng, J., et al.: MmMesh: Towards 3D real-time dynamic human mesh construction using 
millimeter-wave radar. IEEE Trans. Mob. Comput. 21(11), 3875–3888 (2022) 
4. Gurbuz, S.A., Amin, M.G.: Radar-based human-computer interaction, homeland security, and 
assisted living: a tutorial. IEEE Geosci. Remote Sens. Mag. 7(3), 87–123 (2019) 
5. Zhao, H., Jiang, L., Jia, J., Torr, P.H., Koltun, V.: Point transformer. In: IEEE/CVF International 
Conference on Computer Vision (ICCV), pp. 16259–16268 (2021)

3D HPE with UWB MIMO Radar via SGA-Net
567
6. Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., Zagoruyko, S.: End-to-end 
object detection with transformers. In: Vedaldi, A., Bischof, H., Brox, T., Frahm, J.M. (eds.) 
Computer Vision – ECCV 2020, ECCV 2020. LNCS, vol. 12346, pp. 213–229. Springer, 
Cham (2020). https://doi.org/10.1007/978-3-030-58452-8_13 
7. Pavllo, D., Feichtenhofer, C., Grangier, D., Auli, M.: 3D human pose estimation in video with 
temporal convolutions and semi-supervised training. In: IEEE/CVF Conference on Computer 
Vision and Pattern Recognition (CVPR), pp. 7753–7762 (2019) 
8. Zenkova, M., Kholodov, I., Shcherbatyi, A., Tishchenko, N.: MotionAGFormer: enhanc-
ing 3D human pose estimation with a graph-transformer architecture. In: IEEE/CVF Winter 
Conference on Applications of Computer Vision (WACV), pp. 4267–4276 (2024) 
9. Qi, C.R., Yi, L., Su, H., Guibas, L.J.: PointNet++: deep hierarchical feature learning on point 
sets in a metric space. In: Advances in Neural Information Processing Systems (NeurIPS), 
vol. 30, pp. 5099–5108 (2017) 
10. Zheng, Y., Zhang, D., Gu, Y., Qin, T., Chen, Y., Li, Z.: Vi-Fi: cross-modal human pose 
estimation using WiFi. IEEE Trans. Pattern Anal. Mach. Intell. 46(3), 1667–1682 (2023) 
11. Sengupta, A., Jin, F., Zhang, R., Cao, Y.: Mm-Pose: Real-time human skeletal posture 
estimation using mmWave radars and CNNs. IEEE Sens. J. 20(17), 10032–10044 (2020) 
12. Song, Y., Jin, T., Dai, Y., Wang, X.: Efﬁcient through-wall human pose reconstruction using 
UWB MIMO radar. IEEE Antennas Wirel. Propag. Lett. 21(3), 571–575 (2021) 
Open Access This chapter is licensed under the terms of the Creative Commons Attribution-
NonCommercial-NoDerivatives 4.0 International License (http://creativecommons.org/licenses/ 
by-nc-nd/4.0/), which permits any noncommercial use, sharing, distribution and reproduction in 
any medium or format, as long as you give appropriate credit to the original author(s) and the 
source, provide a link to the Creative Commons license and indicate if you modiﬁed the licensed 
material. You do not have permission under this license to share adapted material derived from 
this chapter or parts of it. 
The images or other third party material in this chapter are included in the chapter’s Creative 
Commons license, unless indicated otherwise in a credit line to the material. If material is not 
included in the chapter’s Creative Commons license and your intended use is not permitted by 
statutory regulation or exceeds the permitted use, you will need to obtain permission directly from 
the copyright holder.