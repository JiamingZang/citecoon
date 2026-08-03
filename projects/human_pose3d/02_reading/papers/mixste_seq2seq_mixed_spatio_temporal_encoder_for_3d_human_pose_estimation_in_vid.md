# MixSTE: Seq2seq Mixed Spatio-Temporal Encoder for 3D Human Pose Estimation in Video

> 2022 · id: W4312417903 · arXiv: 2203.00859 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Recent transformer-based solutions have been intro-
duced to estimate 3D human pose from 2D keypoint se-
quence by considering body joints among all frames glob-
ally to learn spatio-temporal correlation. We observe that
the motions of different joints differ significantly.
How-
ever, the previous methods cannot efficiently model the solid
inter-frame correspondence of each joint, leading to insuf-
ficient learning of spatial-temporal correlation.
We pro-
pose MixSTE (Mixed Spatio-Temporal Encoder), which has
a temporal transformer block to separately model the tem-
poral motion of each joint and a spatial transformer block
to learn inter-joint spatial correlation. These two blocks
are utilized alternately to obtain better spatio-temporal fea-
ture encoding. In addition, the network output is extended
from the central frame to entire frames of the input video,
thereby improving the coherence between the input and out-
put sequences.
Extensive experiments are conducted on
three benchmarks (i.e. Human3.6M, MPI-INF-3DHP, and
HumanEva). The results show that our model outperforms
the state-of-the-art approach by 10.9% P-MPJPE and 7.6%
MPJPE. The code is available at https://github.
com/JinluZhang1126/MixSTE.

## introduction
3D human pose estimation from monocular observations
is a fundamental vision task that reconstructs 3D body joint
locations from the input images or video. Since this task
can obtain meaningful expressions of body geometry and
motion, it has a wide range of applications, such as action
recognition [54, 55], virtual human [5–7, 52], and human-
robot interaction [11, 43, 50]. Most recent works are based
on the 2D-to-3D lifting pipeline [1, 4, 28, 31, 37, 46, 57],
which detects 2D keypoints firstly and then lift them to 3D.
Due to the depth ambiguity of monocular data, multiple po-
tential 3D poses may be mapped from the same 2D pose, so
*Corresponding author: tuzhigang@whu.edu.cn
†Work done at Wuhan University
Ours(T=243)
Ours(T=81)
0
200
400
600
800
1000
1200
40
41
42
43
44
45
46
47
48
FPS(frame/s)
MPJPE(mm)
PoseFormer (T=81) [57]
Anatomy-aware (T=81) [4]
Anatomy-aware (T=243) [4]
STE (T=243) [23]
AM (T=243) [28]
VideoPose3D (T=243) [37]
Spatial Correlation
of each frame
Alternated 
Learning of
S-T Correlation 
time
Each joint has a different motion
Separate Temporal Correlation
of each joint
Figure 1. Top: Overview of spatio-temporal correlation modeling.
Each 2D keypoint is separated in the temporal domain to learn dif-
ferent motion trajectories of body joints, and the spatial and tem-
poral correlation are alternately stacked to improve the sequence
coherence modeling ability. Bottom: Accuracy (MPJPE) and effi-
ciency (FPS) comparison with different methods on Human3.6M
dataset, the blue and orange colors indicate that the input sequence
length T is equal to 81 and 243, respectively.
it is difficult to recover an accurate 3D pose merely based
on the information of a single frame 2D keypoints.
Notable progress has been made by exploiting tempo-
ral information contained in the input video to address the
above issues in a single frame [1, 4, 16, 28, 37, 46]. Re-
cently, driven by the success of transformer [45] for its abil-
ity to model sequence data, Zheng et al. [57] introduces a
transformer-based 3D human pose estimation network. It
takes advantage of spatio-temporal information for estimat-
ing the more accurate central-frame pose in video. By mod-
eling spatial correlations between all joints and temporal

correlations among consecutive frames, PoseFormer [57]
achieves performance improvement. However, it ignores
the motion differences among body joints, which causes the
insufficient learning of spatio-temporal correlation. More-
over, it increases the dimension of the temporal transformer
module, which limits the usage of longer input sequence.
Poseformer [57] takes a video as input and only esti-
mates the human pose of the central frame, which we sum-
marize this pipeline as the seq2frame approach. Many re-
cent methods [1,4,28,37,57] follow it and they utilize adja-
cent frames to improve the accuracy of estimating the pose
of a certain moment, but the sequence coherence is ignored
due to the single frame output. Additionally, during the in-
ference, these seq2frame solutions need to input a 2D key-
point sequence repeatedly with large overlap to obtain 3D
poses of all frames, which brings redundant calculation. In
contrast to the seq2frame approach, there is also the seq2seq
approach, which regresses the 3D pose sequence from the
input 2D keypoints. These methods [16,46] mainly depend
on long short-term memory (LSTM) [15] cell or graph con-
volution network (GCN) [21], and perform well in learning
temporal information among continuous estimation results.
However, current seq2seq networks lack the global model-
ing ability between input and output sequences, which tend
to be excessively smooth [37] in the output poses of a long
sequence. The low efficiency of LSTM [15] is also a severe
issue for estimating human pose from video.
While previous work has focused on associating all
joints in the spatial and temporal domains, we observe that
the motion trajectories of the different body joints vary from
frame to frame and should be learned separately. Addition-
ally, the input 2D keypoint sequence and the output 3D pose
sequence have solid global coherence, and they should be
tightly coupled to promote accurate and smooth 3D poses.
Motivated by the above observations, in this work, we
propose MixSTE to learn the separate temporal motion of
each body joint and imbue sequential coherent human pose
sequence in a seq2seq approach. In contrast to the prior
method [57] which reconstructs the central frame and ig-
nores the single joint motion, the MixSTE lifts 2D key-
point sequence to 3D pose sequence via a novel seq2seq
architecture and a set of motion-aware constraints. Specif-
ically, as shown at the top of Figure 1, we propose the
joint separation to consider temporal motion information
of each joint. It takes each 2D joint as an individual fea-
ture (which is referred to as a token in transformer) to suf-
ficiently learn spatio-temporal correlation and helps to re-
duce the dimension of the joint features in temporal do-
main.
Moreover, we propose an alternating design with
seq2seq to flexibly obtain better sequence coherence within
a long sequence, which decreases redundant calculation and
excessive smoothness. In this way, temporal motion tra-
jectories of different body joints could be adequately con-
sidered to predict accurate 3D pose sequence. To the best
of our knowledge, the proposed method is the first to uti-
lize the transformer encoder in the seq2seq pipeline, which
enhances learning spatio-temperal correlation for accurate
pose estimation and significantly improves the inference
speed from seq2frame methods (see the bottom of Fig.1)
Besides, our approach can easily adapt to any length of the
input sequence.
Our contributions to 3D human pose estimation can be
summarized in three folds:
• The MixSTE is proposed to effectively capture the
temporal motion of different body joints over the
long sequence, which helps to model sufficient spatio-
temporal correlation.
• We
propose
a
novel
alternating
design
with
transformer-based
seq2seq

## method
PCK↑
AUC↑
MPJPE↓
Mehta et al. [33]
ACM TOG 2017
79.4
41.6
-
Lin et al. [24](T=25)
BMVC2019
83.6
51.4
79.8
Li et al. [22]
CVPR2020
81.2
46.1
99.7
Wang et al. [46](T=96)ECCV2020
86.9
62.1
68.1
Chen et al. [4](T=243) TCSVT2021
87.8
53.8
79.1
Gong et al. [12]
CVPR2021
88.6
57.3
73.0
Zheng et al. [57]
ICCV2021
88.6
56.4
77.1
Ours(T=1)
94.2
63.8
57.9
Ours(T=27)
94.4
66.5
54.9
Table 3. Detailed quantitative comparison results on MPI-INF-
3DHP with three metrics. The ↑indicates the higher, the better, the
↓indicates the lower, the better. The best and second-best results
are highlighted in bold and underlined formats, respectively.
ity of the proposed method and the impact of finetuning
from large datasets. The MPJPE results on HumanEva fine-
tuning from Human3.6M are reported in the Table 4. Due
to seq2seq setting and limitation of transformer in small
dataset, our method without fine-tuning is slightly worse
than our baseline. But the performance can be improved
by using smaller data sample strides (interval=1). The ex-
periment shows that our model has a better generalization
ability than previous methods.
#Protocol1
Walk
Jog
Avg.
Pavllo et al. [37](T=81)
13.1
10.1
39.8
20.7
13.9
15.6
18.9
Pavllo et al. [37](T=81, FT)
14.0
12.5
27.1
20.3
17.9
17.5
18.2
Zheng et al. [57](T=43)
16.3
11
47.1
25
15.2
15.1
21.6
Zheng et al. [57](T=43, FT)
14.4
10.2
46.6
22.7
13.4
13.4
20.1
Ours(T=43)
20.3
22.4
34.8
27.3
32.1
34.3
28.5
Ours(T=43, interval=1)
16.2
14.2
21.6
24.6
23.2
25.8
20.9
Ours(T=43, FT)
12.7
10.9
17.6
22.6
15.8
17.0
16.1
Table 4. The MPJPE on HumanEva testset under Protocol 1. FT
indicates using the pretrained model on Human3.6M for finetun-
ing. The best result is highlighted in bold.
4.4. Ablation Study
To evaluate the impact and performance of each com-
ponent in our model, we evaluate their effectiveness in this
section. The Human3.6M dataset and the CPN [8] detector
are employed to provide 2D keypoints.
Effect of Each Component.
As shown in Table 5,
we first modify the central frame 3D pose output to the
sequence output without any other optimization to get the
seq2seq baseline model.
For a fair comparison, the pa-
rameter setting of the seq2seq baseline is directly applied
to the proposed method, and the MPJPE loss is utilized in
the baseline model. After applying the alternating design,
the result shows that our method decreases 6.2mm MPJPE
(from 51.7mm to 45.5mm). Then joint separation is utilized
to demonstrate its advantage in both improving the perfor-
mance (from 45.5 to 41.7) and reducing computing cost
(FLOPs for each frame decreases to 645 from 186405). By
applying our loss function to replace MPJPE loss, our result
achieves the best (40.9mm MPJPE with 645 FLOPs). The
MixSTE with our loss function improves 20.9% (from 51.7
to 40.9) compared to the seq2seq baseline, and it proves the
rationality of our network design.
Effect of Loss Function.
We have explored the con-
tribution of our loss function in detail. As shown in Ta-
ble 6, the MPJPE metric decreases from 41.7 to 41.3 after
applying the WMPJPE loss. The result demonstrates that
the WMPJPE is an essential loss to improve accuracy. Then
the temporal consistency loss (TCLoss) following [16] is
employed to improve the temporal smoothness performance
(MPJVE) by 1.0 (decreases from 4.6 to 3.6), and the coher-
ence gets better after using the MPJVE loss (decreases from
4.6 to 2.6). The motion loss [46] has less contribution to
the coherence than TCLoss and MPJVE loss. Finally, after
applying the T-Loss and WMPJPE loss to our method, the
result achieves the best on the MPJPE and MPJVE metrics
Seq2seq
Alternating
Design
Joint
Separation
Our
Loss
MPJPE
FLOPs (M)
Baseline
!
51.7
186405
!
!
45.5
186405
!
!
!
41.7
645
Ours
!
!
!
!
40.9
645
Table 5. Ablation study for each component used in our method.
The evaluation is performed on Human3.6M with MPJPE (mm)
and FLOPs.
MPJPE
MPJVE
MPJPE Loss
41.7
5.0
WMPJPE Loss
41.3
4.6
WMPJPE Loss + Motion Loss [46]
41.3
4.3
WMPJPE Loss + TCLoss [16]
41.2
3.6
WMPJPE Loss + MPJVE Loss
41.2
2.6
Ours (WMPJPE Loss + T-Loss)
40.9
2.3
Table 6.
Ablation study for loss function in our method with
MPJPE and MPJVE.

(40.9mm MPJPE, 2.3 MPJVE). The ablation study demon-
strates that our loss function is comprehensive for the pro-
posed model regarding accuracy and smoothness.
Parameter Setting Analysis. Table 7 shows how the
setting of different hyper-parameters in our method impacts
the performance under Protocol 1 with MPJPE. There are
three main hyper-parameters for the network: the depth of
MixSTE (dl), the dimension of model (dm), and the in-
put sequence length (T). We divide the configurations into
3 groups row-wise, and different values are assigned for
one hyper-parameters while keeping the other two hyper-
parameters fixed to evaluate the impact and choice of
each configuration. Based on the results in the table, we
choose the combination of Depth=8, Channel=512, and
Input Length=243. Note that we choose the Depth = 8
rather than Depth = 10 because the latter setting intro-
duces a more significant number of parameters (33.7M vs.
42.2M).
Depth (dl)
Dimension (dm)
Input Length (T)
MPJPE
4
64
27
54.3
6
64
27
53.2
8
64
27
51.8
10
64
27
51.1
8
128
27
47.9
8
256
27
46.1
8
512
27
45.1
8
640
27
46.0
8
512
81
42.7
8
512
128
42.0
8
512
243
40.9
8
512
300
41.8
Table 7. Ablation study for hyper-parameter setting in depth (dl),
dimension (dm) and input length (T). The evaluation is performed
on Human3.6M with MPJPE (mm).
4.5. Qualitative Results
As shown in Figure 5, we further conduct visualization
on spatial and temporal attention. The selected action (Sit-
tingDown of testset S11) is applied for visualization. More-
over, attention outputs of different heads are averaged to
observe the overall correlations of joints and frames, and
the attention outputs are normalized to [0, 1]. It can be eas-
ily observed from spatial attention map (left of Figure 5)
that our model learns different dependencies between joints.
Furthermore, we also visualize the temporal attention map
(right of Figure 5) from the last temporal attention layer.
The two parts with light color have similar poses with adja-
cent frames, while the dark color corresponded frame (the
middle image in the frame sequence) has a more different
pose with adjacent frames. We also evaluate the visual re-
sult of estimated poses and 3D ground truth of Human3.6M
in Figure 6 to show that we can estimate more accurate
poses compared to PoseFormer [57].
x
y
[0] Hip
[1] R Hip
[2] R Knee
[3] R Foot
[4] L Hip
[5] L Knee
[6] L Foot
[7] Spine
[8] Thorax
[9] Neck
[10] Head
[11] L Shoulder
[12] L Elbow
[13] L Wrist
[14] R Shoulder
[15] R Elbow
[16] R Wrist
Lower Body Joints
Upper Body Joints
Figure 5. Visualization of self-attentions among body joints and
frames. The x-axis and y-axis correspond to the queries and the
predicted outputs, respectively.
Each row shows the attention
weight wi,j of the j-th query for the i-th output.
PoseFormer
Ours
Ground Truth
Figure 6. Qualitative comparison between our method (MixSTE)
and [57] with the Photo and SittingDown actions on on Hu-
man3.6M. The green circle highlights locations where our method
has better results.

## experiments
4.1. Datasets and Evaluation Protocols
We evaluate our model on three 3D human pose estima-
tion datasets: Human3.6M [3,19], MPI-INF-3DHP [32] and
HumanEva [40] individually.
Human3.6M is the most commonly used indoor dataset
for the 3D human pose estimation tasks.
Following the
same policy of previous methods [4, 28, 31, 35–37, 57], the
3D human pose in Human3.6M is adopted as a 17-joint
skeleton, and the subjects S1, S5, S6, S7, S8 from the dataset
are applied during training, the subjects S9 and S11 are used
for testing.
The two commonly used evaluation metrics
(MPJPE and P-MPJPE) are involved in this dataset. In addi-
tion, mean per-joint velocity error (MPJVE) [37] is applied
to measure the smoothness of the prediction sequence. We
also compute the variance (VAR.) of MPJPE between action
categories to evaluate the stability.
MPI-INF-3DHP is also a recently popular large-scale
3D human pose dataset.
Our setting follows previous
works [46, 57]. The area under the curve (AUC), percent-
age of correct keypoints (PCK), and MPJPE are reported as
evaluation metrics.
HumanEva is a smaller dataset than above datasets. As
the same setting of [28,57], actions (Walk, Jog) in subjects
S1, S2, S3 are evaluation data. The metrics MPJPE and P-
MPJPE are applied.
4.2. Implementation Details
The proposed model is implemented with Pytorch. We
use 2D keypoints from 2D pose detector [8, 41] or 2D
ground truth to analyze the performance of our frame-
work. Although the proposed model can easily adapt to any
length of input sequence, to be fair, we select some spe-
cific sequence lengths T for three datasets to compare our
method with other methods which must have a certain 2D
input length [4,28,37]: Human3.6M (T=81,243), MPI-INF-
3DHP (T=1,27), HumanEva (T=43). Analysis about the
frame length setting is discussed in the ablation study Sec-
tion 4.4. The W in WMPJPE is set based on different joint
groups (torso, head, middle limb, and terminal limb) with
different values (1.0, 1.5, 2.5, and 4.0, respectively). The
Adam optimizer [20] is employed for the training model.
The batch size, dropout rate, and activation function for
datasets are set to 1024, 0.1, and GELU. We utilize the
stride data sample strategy with interval is as same as the
input length to make there no overlapping frames between
sequences(more details in the supplementary material).
4.3. Comparison with State-of-the-art Methods
Results on Human3.6M. Two types of 2D joint detec-
tion data are applied in the experiment: CPN [8], which is
the most typical 2D estimator used in previous approaches,
and HRNet [41] which is used to further investigate the up-
per bound of our method. The results compared with other
methods, including the error of all 15 actions and the av-
erage error, are reported in Table 1. For CPN [8] detec-
tor, our model obtains the best result of average MPJPE
of 40.9mm under Protocol 1 and 32.6mm P-MPJPE under
Protocol 2, which outperforms PoseFormer [57] by 3.4mm
MPJPE (7.6%). Furthermore, our method achieves the best
under T = 243 setting and second-best under T = 81 set-
ting in all actions.
Utilizing more powerful 2D detector HRNet [41], our
model further improves roughly 4.5mm (10.2%) under Pro-
tocol 1. We also compare our method with [4,28,37,46,57]
using 2D ground truth, and the results are illustrated in the
Table 2.
Our method significantly outperforms all other
methods and achieves approximately 31.0% improvement
of average MPJPE compared with PoseFormer [57].
Furthermore, we compare the MPJPE distribution in the
testset S9 and S11 with other methods [37, 57] to evaluate
the ability of estimating difficult poses. It can be observed
in Figure 3 that there are much fewer poses with high errors
in our method. Moreover, the proportion of poses with over
40mm MPJPE, which causes loss of accuracy, is consis-

Protocol #1
Dir.
Disc.
Eat
Greet
Phone
Photo
Pose
Pur.
Sit
SitD.
Smoke
Wait
WalkD.
Walk
WalkT.
Avg.
Pavlakos et al. [35]
CVPR2018
48.5
54.4
54.4
52.0
59.4
65.3
49.9
52.9
65.8
71.1
56.6
52.9
60.9
44.7
47.8
56.2
Pavllo et al. [37](CPN, T=243)(†)
CVPR2019
45.2
46.7
43.3
45.6
48.1
55.1
44.6
44.3
57.3
65.8
47.1
44.0
49.0
32.8
33.9
46.8
Cai et al. [1](CPN, T=7)(†)
ICCV2019
44.6
47.4
45.6
48.8
50.8
59.0
47.2
43.9
57.9
61.9
49.7
46.6
51.3
37.1
39.4
48.8
Yeh et al. [51](†)
NIPS2019
44.8
46.1
43.3
46.4
49.0
55.2
44.6
44.0
58.3
62.7
47.1
43.9
48.6
32.7
33.3
46.7
Liu et al. [28](CPN, T=243)(†)
CVPR2020
41.8
44.8
41.1
44.9
47.4
54.1
43.4
42.2
56.2
63.6
45.3
43.5
45.3
31.3
32.2
45.1
Wang et al. [46](CPN, T=96)(†)
ECCV2020
40.2
42.5
42.6
41.1
46.7
56.7
41.4
42.3
56.2
60.4
46.3
42.2
46.2
31.7
31.0
44.5
Chen et al. [4](CPN, T=243)(†)
TCSVT2021
41.4
43.5
40.1
42.9
46.6
51.9
41.7
42.3
53.9
60.2
45.4
41.7
46.0
31.5
32.7
44.1
Xu et al. [48](T=1)
CVPR2021
45.2
49.9
47.5
50.9
54.9
66.1
48.5
46.3
59.7
71.5
51.4
48.6
53.9
39.9
44.1
51.9
Lin et al. [25](T=1)(*)
CVPR2021
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
54.0
Zeng et al. [53](†)
ICCV2021
43.1
50.4
43.9
45.3
46.1
57.0
46.3
47.6
56.3
61.5
47.7
47.4
53.5
35.4
37.3
47.9
Zheng et al. [57](CPN, T=81)(†)(*) ICCV2021
41.5
44.8
39.8
42.5
46.5
51.6
42.1
42.0
53.3
60.7
45.5
43.3
46.1
31.8
32.2
44.3
Ours(CPN, T=81)(†)(*)
39.8
43.0
38.6
40.1
43.4
50.6
40.6
41.4
52.2
56.7
43.8
40.8
43.9
29.4
30.3
42.4
Ours(CPN, T=243)(†)(*)
37.6
40.9
37.3
39.7
42.3
49.9
40.1
39.8
51.7
55.0
42.1
39.8
41.0
27.9
27.9
40.9
Wang et al. [46](HRNet, T=96)(†)
ECCV2020
38.2
41.0
45.9
39.7
41.4
51.4
41.6
41.4
52.0
57.4
41.8
44.4
41.6
33.1
30.0
42.6
Wehrbein et al. [47](HRNet, T=200) ICCV2021
38.5
42.5
39.9
41.7
46.5
51.6
39.9
40.8
49.5
56.8
45.3
46.4
46.8
37.8
40.4
44.3
Ours(HRNet, T=243)
36.7
39.0
36.5
39.4
40.2
44.9
39.8
36.9
47.9
54.8
39.6
37.8
39.3
29.7
30.6
39.8
Protocol #2
Dir.
Disc.
Eat
Greet
Phone
Photo
Pose
Pur.
Sit
SitD.
Smoke
Wait
WalkD.
Walk
WalkT.
Avg.
Wang et al. [46](CPN, T=96)(†)
ECCV2020
31.8
34.3
35.4
33.5
35.4
41.7
31.1
31.6
44.4
49.0
36.4
32.2
35.0
24.9
23.0
34.5
Liu et al. [28](CPN, T=243)(†)
CVPR2020
32.3
35.2
33.3
35.8
35.9
41.5
33.2
32.7
44.6
50.9
37.0
32.4
37.0
25.2
27.2
35.6
Zheng et al. [57](CPN, T=81)(†)(*) ICCV2021
34.1
36.1
34.4
37.2
36.4
42.2
34.4
33.6
45.0
52.5
37.4
33.8
37.8
25.6
27.3
36.5
Ours(CPN, T=81)(†)(*)
32.0
34.2
31.7
33.7
34.4
39.2
32.0
31.8
42.9
46.9
35.5
32.0
34.4
23.6
25.2
33.9
Ours(CPN, T=243)(†)(*)
30.8
33.1
30.3
31.8
33.1
39.1
31.1
30.5
42.5
44.5
34.0
30.8
32.7
22.1
22.9
32.6
Wang et al. [46](HRNet)(†)
ECCV2020
28.4
32.5
34.4
32.3
32.5
40.9
30.4
29.3
42.6
45.2
33.0
32.0
33.2
24.2
22.9
32.7
Wehrbein et al. [47](HRNet, T=200) ICCV2021
27.9
31.4
29.7
30.2
34.9
37.1
27.3
28.2
39.0
46.1
34.2
32.3
33.6
26.1
27.5
32.4
Ours(HRNet, T=243)
28.0
30.9
28.6
30.7
30.4
34.6
28.6
28.1
37.1
47.3
30.5
29.7
30.5
21.6
20.0
30.6
MPJVE
Dir.
Disc.
Eat
Greet
Phone
Photo
Pose
Pur.
Sit
SitD.
Smoke
Wait
WalkD.
Walk
WalkT.
Avg.
Pavllo et al. [37](†)
CVPR2019
3.0
3.1
2.2
3.4
2.3
2.7
2.7
3.1
2.1
2.9
2.3
2.4
3.7
3.1
2.8
2.8
Chen et al. [4](†)
TCSVT2021
2.7
2.8
2.0
3.1
2.0
2.4
2.4
2.8
1.8
2.4
2.0
2.1
3.4
2.7
2.4
2.5
Zheng et al. [57](†)(*)
ICCV2021
3.2
3.4
2.6
3.6
2.6
3.0
2.9
3.2
2.6
3.3
2.7
2.7
3.8
3.2
2.9
3.1
Ours(CPN, T=243)(†)(*)
2.5
2.7
1.9
2.8
1.9
2.2
2.3
2.6
1.6
2.2
1.9
2.0
3.1
2.6
2.2
2.3
Table 1. Detailed quantitative comparison results of MPJPE in millimeters (mm) on Human3.6M under Protocol 1 (no rigid alignment
applied) and Protocol 2 (rigid alignment). Top table: results under Protocol 1 (MPJPE); Middle table: results under Protocol 2 (P-
MPJPE); Bottom table: results of MPJVE. T denotes the number of input frames estimated by the respective approaches, (†) indicates
using temporal information, and (*) indicates the transformer-based methods. The best and second-best results are highlighted in bold and
underlined formats, respectively.
Protocol #1
Dir.
Disc.
Eat
Greet
Phone
Photo
Pose
Pur.
Sit
SitD.
Smoke
Wait
WalkD.
Walk
WalkT.
Avg.
Liu et al. [28](T=243)(†)
CVPR2020
34.5
37.1
33.6
34.2
32.9
37.1
39.6
35.8
40.7
41.4
33.0
33.8
33.0
26.6
26.9
34.7
Wang et al. [46](GT, T=96)
ECCV2020
23.0
25.7
22.8
22.6
24.1
30.6
24.9
24.5
31.1
35.0
25.6
24.3
25.1
19.8
18.4
25.6
Zheng et al. [57](T = 81)(†)(*)ICCV2021
30.0
33.6
29.9
31.0
30.2
33.3
34.8
31.4
37.8
38.6
31.7
31.5
29.0
23.3
23.1
31.3
Ours(T=81)
25.6
27.8
24.5
25.7
24.9
29.9
28.6
27.4
29.9
29.0
26.1
25.0
25.2
18.7
19.9
25.9
Ours(T=243)
21.6
22.0
20.4
21.0
20.8
24.3
24.7
21.9
26.9
24.9
21.2
21.5
20.8
14.7
15.7
21.6
Table 2. Detailed quantitative comparison results of MPJPE in millimeters (mm) on Human3.6M under Protocol 1 using 2D ground truth
keypoints as input. The best results are highlighted in bold.
tently lower, and the proportion of less than 30mm MPJPE
is much higher than other methods. The results demonstrate
our method performs better on difficult actions.
12.07%
17.74%
18.26%
14.83%
10.83%
7.82%
5.68%
3.89%
2.75%
1.91%
4.22%
6.12%
7.90%
4.12%
9.16%
1.00%
3.00%
5.00%
7.00%
9.00%
11.00%
13.00%
15.00%
17.00%
19.00%
<25
25-30
30-35
35-40
40-45
45-50
50-55
55-60
60-65
65-70
>70
Propotion
The  Dis

## related_work
3D Human Pose Estimation.
Estimating 3D human
pose from monocular data was started by relying on the
kinematics feature or the skeleton structure prior [17, 18,
38,39]. With the development of deep learning, more data-
driven methods have been proposed, and these methods
can be divided into end-to-end manner and 2D-to-3D lift-
ing manner. The end-to-end manner directly estimates the
3D coordinates from the input without the intermediate 2D
pose representation. Some methods [36, 42, 44] followed
this manner but required a high computation cost due to re-
gressing directly from the image space. Different from the
end-to-end manner, 2D-to-3D lifting pipeline first estimates
2D keypoints in the RGB data and then leverages the corre-
spondences between 2D and 3D human structures to lift the
2D keypoints to 3D pose. Benefiting from the reliable effort
of 2D keypoint detection works [8,13,29,34,41], recent 2D-
to-3D lifting methods [9,27,30,31,48,56,58] outperformed
end-to-end approaches. Therefore, we follow the 2D-to-3D
lifting manner to obtain robust 2D intermediate supervision.
Seq2frame and Seq2seq under 2D-to-3D Lifting. Re-
cently, temporal information from video has been exploited
to produce more robust predictions by many methods. With
the video input, many influential works (seq2frame) pay at-
tention to predicting the central frame of the input video
to produce a more robust prediction and less sensitivity
to noise.
Pavllo et al. [37] proposed the dilated tempo-
ral convolutions based on the temporal convolution network
(TCN) to extract temporal features. Some following works
improved the performance of TCN by utilizing the attention

mechanism [28], or decomposing the pose estimation task
into bone length and bone direction prediction [4], but they
have to fix the receptive field of the input sequence. In con-
trast to them, our approach is no need to preset the length of
each input with respect to the convolution kernel or the slid-
ing window size. Besides, GCN [21] was also applied to the
task by [1] to learn multi-scale features of human and hand
poses. These works achieved good performance; however,
calculation redundancy is a common flaw of these methods.
On the other hand, some works (seq2seq) improve the
coherence and efficiency of 3D pose estimation and recon-
struct all frames of input sequence at once. LSTM [15] was
introduced to estimate 3D poses in video from a set of 2D
keypoints [26]. Hossain et al. [16] presented a temporal
derivative loss function to ensure the temporal consistency
over a sequence, but it faces the low computing efficiency
issue. Wang et al. [46] exploited a GCN-based approach
and designed a corresponding loss to model motion in both
short temporal intervals and long temporal ranges, but it
lacks global modeling ability of input sequence. In contrast
to [16, 46], our method has the advantage of global model-
ing ability of each joint in the spatial and temporal domains.
Besides, it enables parallel processes for frames and joints
to address the low-efficiency issue of LSTM [15].
Self-attention and Transformer The transformer archi-
tecture with self-attention was firstly proposed by [45], and
then was applied to various visual tasks, e.g. classifica-
tion with visual transformer (ViT) [10], and detection with
DETR [2]. For the human pose estimation task, [49] pro-
posed the Transpose to estimate 2D pose from images. [25]
presented a transformer framework for both human mesh
recovery and pose estimation from a single image but ig-
nored the temporal information in the video.
Some re-
searchers also explored the multi-view 3D human pose es-
timation scheme [14]. The stride transformer encoder [23]
was introduced to incorporate local contexts. Furthermore,
PoseFormer [57] constructed a model based on ViT [10] to
capture the spatial and temporal dependency sequentially.
Both [23] and [57] have to fix the order of spatial and tem-
poral encoders, and only the central frame of video is re-
constructed. Our approach is similar to them in applying
transformer architecture. But we consider motion trajecto-
ries of different body joints and apply the seq2seq to better
model sequence coherence.
From the above analysis and comparison of related
works, further exploration for transformer-based methods
in 3D human pose estimation is necessary and feasible, but
there is no method combining the transformer with seq2seq
framework in the 3D human pose task.
3. Our Approach
As shown in Figure 2, our network takes a concatenated
2D coordinates CN,T ∈RN×T ×2 with N joints and T
Mixed Spatial-Temporal Encoder (MixSTE)
Linear Embedding
T-Loss
WMPJPE Loss
Regression Head
Spatial 
Position Embedding
Temporal 
Position Embedding
Layer Norm
Spatial 
Self-Attention
Layer Norm
MLP
Spatial Transformer Block
𝒅𝒅𝒍𝒍𝐿𝐿𝐿𝐿𝐿𝐿𝐿𝐿𝐿𝐿
2D Keypoints
Frame Sequence
3D Pose Sequence
Layer Norm
Temporal         
Self-Attention
Layer Norm
MLP
Temporal Transformer Block
Joint Separation
Figure 2. Overview of the proposed framework. The MixSTE
is stacked for dl loops, and each MixSTE models spatio-temporal
dependencies independently.
The WMPJPE Loss denotes the
weighted per-joint position error loss. The T-Loss indicates the
loss function of temporal coherence in Section 3.3.
frames as input, where the channel size of the input is 2.
Firstly, we project the input keypoint sequence CN,T to
high-dimensional feature PN,T ∈RN×T ×dm with feature
dimension dm for each joint representation. Then we uti-
lize the position embedding matrix for retaining the posi-
tion information of the spatial and temporal domains. The
proposed MixSTE takes the PN,T as input and aims to al-
ternately learn the spatial correlation and separate tempo-
ral motion. Finally, we use a regression head to concate-
nate the outputs X ∈RN×T ×dm of encoder, and take the
dimension dm to 3 to get the 3D human pose sequence
Out ∈RN×T ×3.
3.1. Mixed Spatio-Temporal Encoder
We utilize the MixSTE to model spatial dependency and
temporal motion for a given 2D input keypoint sequence, re-
spectively. MixSTE consists of a Spatial Transformer Block
(STB) and a Temporal Transformer Block (TTB). Here, the
STB computes the self-attention between joints and aims to
learn the body joint relations of each frame, while the TTB
computes the self-attention between frames and focuses on
learning the global temporal correlation of each joint.
3.1.1
Separate Temporal Correlation Learning
To imbue effective motion trajectories into the learned rep-
resentations, we consider the temporal correspondence of
each joint in order to explicitly model correlations on the
same joint over the dynamic sequence. Different from the

previous method [57], we do not treat all body joints as a
token in the temporal transformer block. We separate dif-
ferent joints in time dimension, so that the trajectory of each
joint is an individual token p ∈R1×T ×dm, and different
joints of body are modeled paralleled. From the perspective
of the time dimension, different motion trajectories of body
joints are modeled separately to represent temporal corre-
lations better. The joint separation is operated as follows:
  
\ b egin {aligned} X_l^ {t} = C onc a t (\mathcal {F}(p_{i,1},p_{i,2},...p_{i,T})), \ i\in N, \end {aligned} 
(1)
where pi,j ∈PN,T denotes the i-th joint in the j-th frame,
F indicates the temporal encoder function and the output
of the l-th TTB encoder is Xl ∈RN×T ×dm.
Further-
more, treating each body joint as an individual token can
decrease dimension of the model to dm from N × dm of
PoseFormer [57], and it also enables the longer sequence
processed in the model.
3.1.2
Spatial Correlation Learning
We employ the spatial transformer block (STB) to learn spa-
tial correlations among joints in each frame. Given 2D key-
points with N joints, we consider each joint as a token in
spatial attention. Firstly, we take 2D keypoints as input and
project each keypoint to a high-dimensional feature with the
linear embedding layer. The feature is referred to as a spa-
tial token in STB. We then embed the spatial position infor-
mation with a positional matrix Es−pos ∈RN×dm. After
that, spatial tokens Pi ∈RN×dm of the i-th frame is fed into
spatial self-attention mechanism of STB to model depen-
dencies across all joints and output the high-dimensional
tokens Xs
l ∈RN×T ×dm in l-th STB.
3.1.3
Alternating design with Seq2seq
Alternating design in spatio-temporal correlation. The
STB and TTB are designed in an alternating way to encode
different high-dimensional tokens. The process of alternat-
ing design is like recurrent neural network (RNN), but we
can parallel over joint and time dimensions. We stack STB
and TTB for dl loops, and the dimension of the feature is
preserved as a fixed size dm to promise that spatial-temporal
correlation learning focuses on the same joint. Specifically,
the spatial and temporal position embedding is applied only
in the first encoder to retain two kinds of position infor-
mation. Moreover, th

## conclusion
We have presented MixSTE, a novel transformer-based
seq2seq approach for 3D pose estimation from monocular
video. The model can better capture global sequence co-
herence and temporal motion trajectories of different body
joints. Moreover, the efficiency of 3D human pose estima-
tion is much improved. Comprehensive evaluation results
show that our model obtains the best performance. As a
new universal baseline, the proposed method also opens up
many possible directions for future works. Nonethless, our
method is still limited by inaccurate 2D detection results
e.g. missing and noisy keypoints. It may be alleviated by
applying better 2D detector, but modeling distribution of
input noise is also a feasible and valuable exploration.
Acknowledgements. This work was supported by the Na-
tional Natural Science Foundation of China under Grant
62106177 and 61773272.