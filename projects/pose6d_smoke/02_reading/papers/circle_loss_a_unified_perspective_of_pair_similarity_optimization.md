# Circle Loss: A Unified Perspective of Pair Similarity Optimization

> 2020 · id: W3034303554 · arXiv: 2002.10857 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
This paper provides a pair similarity optimization view-
point on deep feature learning, aiming to maximize the
within-class similarity sp and minimize the between-class
similarity sn.
We ﬁnd a majority of loss functions, in-
cluding the triplet loss and the softmax cross-entropy loss,
embed sn and sp into similarity pairs and seek to reduce
(sn −sp). Such an optimization manner is inﬂexible, be-
cause the penalty strength on every single similarity score
is restricted to be equal. Our intuition is that if a similarity
score deviates far from the optimum, it should be empha-
sized. To this end, we simply re-weight each similarity to
highlight the less-optimized similarity scores. It results in
a Circle loss, which is named due to its circular decision
boundary. The Circle loss has a uniﬁed formula for two
elemental deep feature learning paradigms, i.e., learning
with class-level labels and pair-wise labels. Analytically,
we show that the Circle loss offers a more ﬂexible optimiza-
tion approach towards a more deﬁnite convergence target,
compared with the loss functions optimizing (sn −sp). Ex-
perimentally, we demonstrate the superiority of the Circle
loss on a variety of deep feature learning tasks. On face
recognition, person re-identiﬁcation, as well as several ﬁne-
grained image retrieval datasets, the achieved performance
is on par with the state of the art.

## introduction
This paper holds a similarity optimization view towards
two elemental deep feature learning paradigms, i.e., learn-
ing from data with class-level labels and from data with
pair-wise labels. The former employs a classiﬁcation loss
function (e.g., softmax cross-entropy loss [25, 16, 36]) to
optimize the similarity between samples and weight vec-
tors. The latter leverages a metric loss function (e.g., triplet
loss [9, 22]) to optimize the similarity between samples. In
our interpretation, there is no intrinsic difference between
these two learning approaches. They both seek to minimize
∗Equal contribution.
†Corresponding author.
𝑠!
A
C
𝑠"
B
T
T’
𝑠!
A
C
𝑠"
B
T
0
1
1
0
1
1
T’
(a)
(b)
Figure 1: Comparison between the popular optimization
manner of reducing (sn−sp) and the proposed optimization
manner of reducing (αnsn −αpsp). (a) Reducing (sn −sp)
is prone to inﬂexible optimization (A, B and C all have
equal gradients with respect to sn and sp), as well as am-
biguous convergence status (both T and T ′ on the decision
boundary are acceptable). (b) With (αnsn −αpsp), the Cir-
cle loss dynamically adjusts its gradients on sp and sn, and
thus beneﬁts from a ﬂexible optimization process. For A, it
emphasizes on increasing sp; for B, it emphasizes on reduc-
ing sn. Moreover, it favors a speciﬁed point T on the circu-
lar decision boundary for convergence, setting up a deﬁnite
convergence target.
between-class similarity sn, as well as to maximize within-
class similarity sp.
From this viewpoint, we ﬁnd that many popular loss
functions (e.g., triplet loss [9, 22], softmax cross-entropy
loss and its variants [25, 16, 36, 29, 32, 2]) share a similar
optimization pattern. They all embed sn and sp into sim-
ilarity pairs and seek to reduce (sn −sp). In (sn −sp),
increasing sp is equivalent to reducing sn. We argue that
this symmetric optimization manner is prone to the follow-
ing two problems.
• Lack of ﬂexibility for optimization.
The penalty
strength on sn and sp is restricted to be equal. Given the
speciﬁed loss functions, the gradients with respect to sn
and sp are of same amplitudes (as detailed in Section 2).
In some corner cases, e.g., sp is small and sn already ap-
proaches 0 (“A” in Fig. 1 (a)), it keeps on penalizing sn
with a large gradient. It is inefﬁcient and irrational.
arXiv:2002.10857v2  [cs.CV]  15 Jun 2020

• Ambiguous convergence status. Optimizing (sn−sp)
usually leads to a decision boundary of sp −sn = m (m
is the margin). This decision boundary allows ambiguity
(e.g., “T” and “T ′” in Fig. 1 (a)) for convergence. For ex-
ample, T has {sn, sp} = {0.2, 0.5} and T ′ has {s′
n, s′
p} =
{0.4, 0.7}. They both obtain the margin m = 0.3. However,
comparing them against each other, we ﬁnd the gap between
s′
n and sp is only 0.1. Consequently, the ambiguous conver-
gence compromises the separability of the feature space.
With these insights, we reach an intuition that different
similarity scores should have different penalty strength. If
a similarity score deviates far from the optimum, it should
receive a strong penalty. Otherwise, if a similarity score
already approaches the optimum, it should be optimized
mildly.
To this end, we ﬁrst generalize (sn −sp) into
(αnsn −αpsp), where αn and αp are independent weight-
ing factors, allowing sn and sp to learn at different paces.
We then implement αn and αp as linear functions w.r.t. sn
and sp respectively, to make the learning pace adaptive to
the optimization status: The farther a similarity score de-
viates from the optimum, the larger the weighting factor
will be. Such optimization results in the decision boundary
αnsn −αpsp = m, yielding a circle shape in the (sn, sp)
space, so we name the proposed loss function Circle loss.
Being simple, Circle loss intrinsically reshapes the char-
acteristics of the deep feature learning from the following
three aspects:
First, a uniﬁed loss function. From the uniﬁed simi-
larity pair optimization perspective, we propose a uniﬁed
loss function for two elemental learning paradigms, learn-
ing with class-level labels and with pair-wise labels.
Second, ﬂexible optimization.
During training, the
gradient back-propagated to sn (sp) will be ampliﬁed by
αn (αp). Those less-optimized similarity scores will have
larger weighting factors and consequentially get larger gra-
dients. As shown in Fig. 1 (b), the optimization on A, B
and C are different to each other.
Third, deﬁnite convergence status. On the circular de-
cision boundary, Circle loss favors a speciﬁed convergence
status (“T” in Fig. 1 (b)), as to be demonstrated in Sec-
tion 3.3. Correspondingly, it sets up a deﬁnite optimization
target and beneﬁts the separability.
The main contributions of this paper are summarized as
follows:
• We propose Circle loss, a simple loss function for deep
feature learning. By re-weighting each similarity score
under supervision, Circle loss beneﬁts the deep feature
learning with ﬂexible optimization and deﬁnite conver-
gence target.
• We present Circle loss with compatibility to both class-
level labels and pair-wise labels. Circle loss degener-
ates to triplet loss or softmax cross-entropy loss with
slight modiﬁcations.
• We conduct extensive experiments on a variety of deep
feature learning tasks, e.g. face recognition, person re-
identiﬁcation, car image retrieval and so on. On all
these tasks, we demonstrate the superiority of Circle
loss with performance on par with the state of the art.
2. A Uniﬁed Perspective
Deep feature learning aims to maximize the within-class
similarity sp, as well as to minimize the between-class sim-
ilarity sn. Under the cosine similarity metric, for example,
we expect sp →1 and sn →0.
To this end, learning with class-level labels and learn-
ing with pair-wise labels are two elemental paradigms.
They are conventionally considered separately and signif-
icantly differ from each other w.r.t to the loss functions.
Given class-level labels, the ﬁrst one basically learns to
classify each training sample to its target class with a clas-
siﬁcation loss, e.g. L2-Softmax [21], Large-margin Soft-
max [15], Angular Softmax [16], NormFace [30], AM-
Softmax [29], CosFace [32], ArcFace [2]. These methods
are also known as proxy-based learning, as they optimize
the similarity between samples and a set of proxies rep-
resenting each class. In contrast, given pair-wise labels,
the second one directly learns pair-wise similarity (i.e., the
similarity between samples) in the feature space and thus
requires no proxies, e.g., constrastive loss [5, 1], triplet
loss [9, 22], Lifted-Structure loss [19], N-pair loss [24], His-
togram loss [27], Angular loss [33], Margin based loss [38],
Multi-Similarity loss [34] and so on.
This paper views both learning approaches from a uni-
ﬁed perspective, with no preference for either proxy-based
or pair-wise similarity. Given a single sample x in the fea-
ture space, let us assume that there are K within-class sim-
ilarity scores and L between-class similarity scores associ-
ated with x. We denote these similarity scores as {si
p} (i =
1, 2, · · · , K) and {sj
n} (j = 1, 2, · · · , L), respectively.
To minimize each sj
n as well as to maximize si
p, (∀i ∈
{1, 2, · · · , K}, ∀j ∈{1, 2, · · · , L}), we propose a uniﬁed
loss function by:
Luni = log
h
1 +
K
X
i=1
L
X
j=1
exp(γ(sj
n −si
p + m))
i
= log
h
1 +
L
X
j=1
exp(γ(sj
n + m))
K
X
i=1
exp(γ(−si
p))
i
,
(1)
in which γ is a scale factor and m is a margin for better
similarity separation.
Eq. 1 is intuitive. It iterates through every similarity pair
to reduce (sj
n −si
p). We note that it degenerates to triplet
loss or classiﬁcation loss, through slight modiﬁcations.
Given class-level labels, we calculate the similarity
scores between x and weight vectors wi (i = 1, 2, · · · , N)

𝑑𝐿
𝑑𝑠$
𝑑𝐿
𝑑𝑠%
𝑠$
𝑠%
𝑠$
𝑠%
𝑠$
𝑠%
𝑠$
𝑠%
𝑠$
𝑠%
𝑠$
𝑠%
𝑑𝐿
𝑑𝑠$
𝑑𝐿
𝑑𝑠%
𝑑𝐿
𝑑𝑠$
𝑑𝐿
𝑑𝑠%
(a) Triplet loss
(b) AMSoftmax loss
(c) Circle loss
A
A
B
B
B
A
B
A
B
A
B
A
Figure 2: The gradients of the loss functions. (a) Triplet loss. (b) AM-Softmax loss. (c) The proposed Circle loss. Both
triplet loss and AM-Softmax loss present the lack of ﬂexibility for optimization. The gradients with respect to sp (left) and sn
(right) are restricted to equal and undergo a sudden decrease upon convergence (the similarity pair B). For example, at A, the
within-class similarity score sp already approaches 1, and still incurs a large gradient. Moreover, the decision boundaries are
parallel to sp = sn, which allows ambiguous convergence. In contrast, the proposed Circle loss assigns different gradients
to the similarity scores, depending on their distances to the optimum. For A (both sn and sp are large), Circle loss lays
emphasis on optimizing sn. For B, since sn signiﬁcantly decreases, Circle loss reduces its gradient and thus enforces a
moderated penalty. Circle loss has a circular decision boundary, and promotes accurate convergence status.
(N is the number of training classes) in the classiﬁcation
layer. Speciﬁcally, 

## method
Market-1501
MSMT17
R-1
mAP
R-1
mAP
PCB [26] (Softmax)
93.8
81.6
68.2
40.4
MGN [31] (Softmax+Triplet)
95.7
86.9
-
-
JDGL [42]
94.8
86.0
77.2
52.3
ResNet50 + AM-Softmax
92.4
83.8
75.6
49.3
ResNet50 + CircleLoss(ours)
94.2
84.9
76.3
50.2
MGN + AM-Softmax
95.3
86.6
76.5
51.8
MGN + CircleLoss(ours)
96.1
87.4
76.9
52.1
ent backbones. For example, with ResNet34 as the back-
bone, Circle loss surpasses the most competitive one (Ar-
cFace) by +0.13% at rank-1 accuracy. With ResNet100 as
the backbone, while ArcFace achieves a high rank-1 accu-
racy of 98.36%, Circle loss still outperforms it by +0.14%.
The same observations also hold for the veriﬁcation metric.
Table
2
summarizes
face
veriﬁcation

## experiments
on
LFW [10], YTF [37] and CFP-FP [23]. We note that perfor-
mance on these datasets is already near saturation. Specif-
ically, ArcFace is higher than AM-Softmax by +0.05%,
+0.03%, +0.07% on three datasets, respectively.
Circle
loss remains the best one, surpassing ArcFace by +0.05%,
+0.06% and +0.18%, respectively.
We further compare Circle loss with AM-Softmax
and ArcFace on IJB-C 1:1 veriﬁcation task in Table 3.
Under both ResNet34 and ResNet100 backbones, Cir-
cle loss presents considerable superiority.
For example,
with ResNet34, Circle loss signiﬁcantly surpasses Arc-
Face by +1.16% and +2.55% on “TAR@FAR=1e-4” and
“TAR@FAR=1e-5”, respectively.
4.3. Person Re-identiﬁcation
We evaluate Circle loss on re-ID task in Table 4.
MGN [31] is one of the state-of-the-art methods and is
featured for learning multi-granularity part-level features.
Originally, it uses both Softmax loss and triplet loss to fa-
cilitate joint optimization. Our implementation of “MGN
(ResNet50) + AM-Softmax” and “MGN (ResNet50)+ Cir-
cle loss” only use a single loss function for simplicity.
We make three observations from Table 4.
First, we
ﬁnd that Circle loss can achieve competitive re-ID accu-
racy against state of the art.
We note that “JDGL” is
slightly higher than “MGN + Circle loss” on MSMT17 [35].
JDGL [42] uses a generative model to augment the training
data, and signiﬁcantly improves re-ID over the long-tailed
dataset. Second, comparing Circle loss with AM-Softmax,
we observe the superiority of Circle loss, which is consis-
tent with the experimental results on the face recognition
task. Third, comparing “ResNet50 + Circle loss” against

Table 5: Comparison of R@K(%) on three ﬁne-grained image retrieval datasets. Superscript denotes embedding size.
Loss function
CUB-200-2011 [28]
Cars196 [14]
Stanford Online Products [19]
R@1
R@2
R@4
R@8
R@1
R@2
R@4
R@8
R@1
R@10
R@102
R@103
LiftedStruct64 [19]
43.6
56.6
68.6
79.6
53.0
65.7
76.0
84.3
62.5
80.8
91.9
97.4
HDC384 [18]
53.6
65.7
77.0
85.6
73.7
83.2
89.5
93.8
69.5
84.4
92.8
97.7
HTL512 [3]
57.1
68.8
78.7
86.5
81.4
88.0
92.7
95.7
74.8
88.3
94.8
98.4
ABIER512 [20]
57.5
71.5
79.8
87.4
82.0
89.0
93.2
96.1
74.2
86.9
94.0
97.8
ABE512 [13]
60.6
71.5
79.8
87.4
85.2
90.5
94.0
96.1
76.3
88.4
94.8
98.2
Multi-Simi512 [34]
65.7
77.0
86.3
91.2
84.1
90.4
94.0
96.5
78.2
90.5
96.0
98.7
CircleLoss512
66.7
77.4
86.2
91.2
83.4
89.8
94.1
96.5
78.3
90.5
96.1
98.6
(a) scale factor 𝛾
(b) relaxation factor m
Rank-1 accuracy (%) on MFC1
Rank-1 accuracy (%) on MFC1
Figure 3: Impact of two hyper-parameters. In (a), Circle
loss presents high robustness on various settings of scale
factor γ. In (b), Circle loss surpasses the best performance
of both AM-Softmax and ArcFace within a large range of
relaxation factor m.
“MGN + Circle loss”, we ﬁnd that part-level features bring
incremental improvement to Circle loss.
It implies that
Circle loss is compatible with the part-model specially de-
signed for re-ID.
4.4. Fine-grained Image Retrieval
We evaluate the compatibility of Circle loss to pair-wise
labeled data on three ﬁne-grained image retrieval datasets,
i.e., CUB-200-2011, Cars196, and Standford Online Prod-
ucts. On these datasets, majority methods [19, 18, 3, 20,
13, 34] adopt the encouraged setting of learning with pair-
wise labels. We compare Circle loss against these state-
of-the-art methods in Table 5.
We observe that Circle
loss achieves competitive performance, on all of the three
datasets. Among the competing methods, LiftedStruct [19]
and Multi-Simi [34] are specially designed with elaborate
hard mining strategies for learning with pair-wise labels.
HDC [18], ABIER [20] and ABE [13] beneﬁt from model
ensemble. In contrast, the proposed Circle loss achieves
performance on par with the state of the art, without any
bells and whistles.
Figure 4: The change of sp and sn values during training.
We linearly lengthen the curves within the ﬁrst 2k iterations
to highlight the initial training process (in the green zone).
During the early training stage, Circle loss rapidly increases
sp, because sp deviates far from the optimum at the initial-
ization and thus attracts higher optimization priority.
4.5. Impact of the Hyper-parameters
We analyze the impact of two hyper-parameters, i.e., the
scale factor γ in Eq. 6 and the relaxation factor m in Eq. 8
on face recognition tasks.
The scale factor γ determines the largest scale of each
similarity score. The concept of the scale factor is critical in
a lot of variants of Softmax loss. We experimentally eval-
uate its impact on Circle loss and make a comparison with
several other loss functions involving scale factors. We vary
γ from 32 to 1024 for both AM-Softmax and Circle loss.
For ArcFace, we only set γ to 32, 64 and 128, as it becomes
unstable with larger γ in our implementation. The results
are visualized in Fig. 3. Compared with AM-Softmax and
ArcFace, Circle loss exhibits high robustness on γ. The
main reason for the robustness of Circle loss on γ is the au-
tomatic attenuation of gradients. As the similarity scores
approach the optimum during training, the weighting fac-
tors gradually decrease. Consequentially, the gradients au-
tomatically decay, leading to a moderated optimization.
The relaxation factor m determines the radius of the
circular decision boundary. We vary m from −0.2 to 0.3

𝑠𝑠𝑛𝑛
𝑠𝑠𝑝𝑝
𝑠𝑠𝑝𝑝
𝑠𝑠𝑝𝑝
𝑠𝑠𝑛𝑛
𝑠𝑠𝑛𝑛
(a) AMSoftmax (m=0.35)
(b) Circle loss  (m=0.325)
(c) Circle loss  (m=0.25)
Figure 5: Visualization of the similarity distribution after convergence. The blue dots mark the similarity pairs crossing
the decision boundary during the whole training process. The green dots mark the similarity pairs after convergence. (a)
AM-Softmax seeks to minimize (sn −sp). During training, the similarity pairs cross the decision boundary through a wide
passage. After convergence, the similarity pairs scatter in a relatively large region in the (sn, sp) space. In (b) and (c), Circle
loss has a circular decision boundary. The similarity pairs cross the decision boundary through a narrow passage and gather
into a relatively concentrated region.
(with 0.05 as the interval) and visualize the results in Fig. 3
(b). It is observed that under all the settings from −0.05 to
0.25, Circle loss surpasses the best performance of Arcface,
as well as AM-Softmax, presenting a considerable degree
of robustness.
4.6. Investigation of the Characteristics
Analysis of the optimization process.
To intuitively
understand the learning process, we show the change of sn
and sp during the whole training process in Fig. 4, from
which we draw two observations:
First, at the initialization, all the sn and sp scores are
small. It is because randomized features are prone to be
far away from each other in the high dimensional feature
space [40, 7]. Correspondingly, sp get signiﬁcantly larger
weights (compared with sn), and the optimization on sp
dominates the training, incurring a fast increase in similar-
ity values in Fig. 4. This phenomenon evidences that Circle
loss maintains a ﬂexible and balanced optimization.
Second, at the end of the training, Circle loss achieves
both better within-class compactness and between-class dis-
crepancy (on the training set), compared with AM-Softmax.
Because Circle loss achieves higher performance on the
testing set, we believe that it indicates better optimization.
Analysis of the convergence.
We analyze the conver-
gence status of Circle loss in Fig. 5. We investigate two
issues: how the similarity pairs consisted of sn and sp cross
the decision boundary during training and how they are dis-
tributed in the (sn, sp) space after convergence. The results
are shown in Fig. 5. In Fig. 5 (a), AM-Softmax loss adopts
the optimal setting of m = 0.35. In Fig. 5 (b), Circle loss
adopts a compromised setting of m = 0.325. The decision
boundaries of (a) and (b) are tangent to each other, allowing
an intuitive comparison. In Fig. 5 (c), Circle loss adopts its
optimal setting of m = 0.25. Comparing Fig. 5 (b) and (c)
against Fig. 5 (a), we ﬁnd that Circle loss presents a rela-
tively narrower passage on the decision boundary, as well
as a more concentrated distribution for convergence (espe-
cially when m = 0.25). It indicates that Circle loss fa-
cilitates more consistent convergence for all the similarity
pairs, compared with AM-Softmax loss. This phenomenon
conﬁrms that Circle loss has a more deﬁnite convergence
target, which promotes the separability in the feature space.

## conclusion
This paper provides two insights into the optimization
process for deep feature learning.
First, a majority of
loss functions, including the triplet loss and popular clas-
siﬁcation losses, conduct optimization by embedding the
between-class and within-class similarity into similarity
pairs. Second, within a similarity pair under supervision,
each similarity score favors different penalty strength, de-
pending on its distance to the optimum.
These insights
result in Circle loss, which allows the similarity scores to
learn at different paces. The Circle loss beneﬁts deep fea-
ture learning with high ﬂexibility in optimization and a
more deﬁnite convergence target. It has a uniﬁed formula
for two elemental learning approaches, i.e., learning with
class-level labels and learning with pair-wise labels. On
a variety of deep feature learning tasks, e.g., face recog-
nition, person re-identiﬁcation, and ﬁne-grained image re-
trieval, the Circle loss achieves performance on par with the
state of the art.