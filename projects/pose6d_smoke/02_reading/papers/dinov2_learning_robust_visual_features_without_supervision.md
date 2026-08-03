# DINOv2: Learning Robust Visual Features without Supervision

> 2023 · id: arxiv:2304.07193 · arXiv: 2304.07193 · pdf: https://arxiv.org/pdf/2304.07193 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

Published in Transactions on Machine Learning Research (01/2024)
DINOv2: Learning Robust Visual Features
without Supervision
Maxime Oquab∗∗, Timothée Darcet∗∗, Théo Moutakanni∗∗,
Huy V. Vo∗, Marc Szafraniec∗, Vasil Khalidov∗, Pierre Fernandez, Daniel Haziza,
Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba,
Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat,
Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal1,
Patrick Labatut∗, Armand Joulin∗, Piotr Bojanowski∗
Meta AI Research
1Inria
∗core team
∗∗equal contribution
Reviewed on OpenReview: https://openreview.net/forum?id=a68SUt6zFt
Abstract
The recent breakthroughs in natural language processing for model pretraining on large
quantities of data have opened the way for similar foundation models in computer vision.
These models could greatly simplify the use of images in any system by producing general-
purpose visual features, i.e., features that work across image distributions and tasks without
finetuning. This work shows that existing pretraining methods, especially self-supervised
methods, can produce such features if trained on enough curated data from diverse sources.
We revisit existing approaches and combine different techniques to scale our pretraining in
terms of data and model size. Most of the technical contributions aim at accelerating and
stabilizing the training at scale. In terms of data, we propose an automatic pipeline to build
a dedicated, diverse, and curated image dataset instead of uncurated data, as typically done
in the self-supervised literature. In terms of models, we train a ViT model (Dosovitskiy
et al., 2021) with 1B parameters and distill it into a series of smaller models that surpass
the best available general-purpose features, OpenCLIP (Ilharco et al., 2021) on most of the
benchmarks at image and pixel levels.
1
Introduction
Learning task-agnostic pretrained representations have become the standard in Natural Language Process-
ing (NLP) (Radford et al., 2019; Raffel et al., 2020; Chowdhery et al., 2022; Hoffmann et al., 2022; Touvron
et al., 2023). One can use these features “as they are”, i.e., without fine-tuning, and achieve performances
on downstream tasks that are significantly better than those produced by task-specific models (Brown et al.,
2020). This success has been fueled by pretraining on large quantities of raw text using pretext objectives,
such as language modeling (Radford et al., 2017) or word vectors (Devlin et al., 2019), that require no
supervision.
Following this paradigm shift in NLP, we expect similar “foundation” models to appear in computer vi-
sion (Bommasani et al., 2021). These models should generate visual features that work out of the box on
any task, both at the image level, e.g., image classification, and pixel level, e.g., segmentation. Most promis-
ing efforts towards these foundation models focus on text-guided pretraining, i.e., using a form of textual
supervision to guide the training of the features (Joulin et al., 2016; Mahajan et al., 2018; Radford et al.,
All the authors are affiliated to Meta, except Julien Mairal who is affiliated to Inria. Timothée Darcet and Pierre Fernandez
have a co-affiliation with Inria. Théo Moutakanni has a co-affiliation with Université Paris Saclay. Alaaeldin El-Nouby has a
co-affiliation with Inria and ENS-PSL. Correspondence: {qas, timdarcet, theomoutakanni, ajoulin, bojanowski}@meta.com
1
arXiv:2304.07193v2  [cs.CV]  2 Feb 2024

Published in Transactions on Machine Learning Research (01/2024)
Figure 1: Visualization of the first PCA components. We compute a PCA between the patches of the
images from the same column (a, b, c and d) and show their first 3 components. Each component is matched
to a different color channel. Same parts are matched between related images despite changes of pose, style
or even objects. Background is removed by thresholding the first PCA component.
2021). This form of text-guided pretraining limits the information that can be retained about the image
since captions only approximate the rich information in images, and complex pixel-level information may
not surface with this supervision. Furthermore, these image encoders require aligned text-image corpora and
hence, do not offer the flexibility of their text counterparts, that is, to learn from raw data alone.
An alternative to text-guided pretraining is self-supervised learning (Caron et al., 2018; Chen et al., 2020;
He et al., 2022) where features are learned from images alone. These approaches are conceptually closer to
pretext tasks such as language modeling and can capture information at the image and pixel level (Caron
et al., 2021). Additionally, the features output by self-supervised models have been shown to exhibit various
useful properties, and have enabled enabled a wide variety of applications (Amir et al., 2022; Tumanyan
et al., 2022; Ofri-Amar et al., 2023; Hamilton et al., 2022). However, despite their potential to learn general-
purpose features, most of the advances in self-supervised learning were made in the context of pretraining on
a small curated dataset, ImageNet-1k (Russakovsky et al., 2015). Some efforts on scaling these approaches
beyond ImageNet-1k have been attempted (Caron et al., 2019; Goyal et al., 2021; 2022a), but they focused on
uncurated datasets, which typically lead to a significant drop in the quality of the features. This is explained
by the lack of control over the data quality and diversity, which are essential to produce good features.
In this work, we explore if self-supervised learning has the potential to learn general-purpose visual features if
pretrained on a large quantity of curated data. We revisit existing discriminative self-supervised approaches
that learn features at both the image and patch level, such as iBOT (Zhou et al., 2022a), and we reconsider
some of their design choices under the lens of a larger dataset. Most of our technical contributions are tailored
toward stabilizing and accelerating discriminative self-supervised learning when scaling in model and data
sizes. These improvements make our approach around 2× faster and require 3× less memory than similar
discriminative self-supervised methods, allowing us to leverage longer training with larger batch sizes.
Regarding pretraining data, we have built an automatic pipeline to filter and rebalance datasets from an
extensive collection of uncurated images. This pipeline is inspired by pipelines used in NLP (Wenzek et al.,
2020), where data similarities are used instead of external metadata and do not require manual annotation.
A major difficulty when dealing with images in the wild is to rebalance concepts and avoid overfitting on a
few dominant modes. In this work, a naive clustering approach works reasonably well to resolve this issue.
We gathered a small but diverse corpus of 142M images to validate our approach.
2

Published in Transactions on Machine Learning Research (01/2024)
1010
1011
1012
flops
75
78
81
84
87
Accuracy
Inet-1k
1010
1011
1012
flops
40
48
56
64
mIoU
Segmentation
1010
1011
1012
flops
0.9
1.2
1.5
1.8
R-MSE
Monocular Depth 
1010
1011
1012
flops
80
84
88
92
Accuracy
Classification
1010
1011
1012
flops
48
56
64
72
80
Accuracy
Finegrained Classification
1010
1011
1012
flops
30
45
60
75
mAP
Instance Retrieval
1010
1011
1012
flops
40
50
60
70
80
Accuracy
ImageNet-{A,R,Sketch}
1010
1011
1012
flops
50
55
60
65
70
Accuracy
Video Understanding
SSL
WSL
DINOv2
Figure 2: Evolution of performance when scaling in parameters. We show performance on eight
types of vision tasks, as presented in Sec. 7, and average metrics with each type. Features are extracted
from our self-supervised encoders, DINOv2 (dark blue), and we compare them with self-supervised methods
(pale orange), as well as weakly-supervised methods (dark pink). We report the best-performing weakly-
supervised model’s performance as a dashed horizontal line. Our family of models drastically improves over
the previous state of the art in self-supervised learning and reaches performance comparable with weakly-
supervised features. See Sec. 7 for a detailed analysis.
Finally, we provide a variety of pretrained visual models, called DINOv2, trained with different Vision
Transformers (ViT) (Dosovitskiy et al., 2016) architectures on our data. We release all the models and
the code to retrain DINOv2 on any data. We validate the quality of DINOv2 on various computer vision
benchmarks at both image and pixel levels as we scale them, as summarized in Fig. 2. We conclude that self-
supervised pretraining alone is a good candidate for learning transferable frozen features that are competitive
with the best openly available weakly-supervised models.
2
Related Work
Intra-image self-supervised training.
A first family of self-supervised methods focuses on pretext tasks
built from the image, i.e., extracting a signal from the image to be predicted from the rest of the image.
This idea has become prevalent with the work of Doersch et al. (2015), where they train by predicting the
context of a given patch. Many other pretext tasks were introduced based on, for example, re-colorizing
images (Zhang et al., 2016), predicting transformations (Gidaris et al., 2018), inpainting (Pathak et al.,
2016) or patch re-ordering (Noroozi & Favaro, 2016; Misra & Maaten, 2020). Recently, the emergence of
patch-based architectures, like ViTs, has led to a revisit of inpainting for pre-training (He et al., 2022; Bao
et al., 2021; El-Nouby et al., 2021), potentially in feature space (Assran et al., 2023; Baevski et al., 2022).
Of particular interest, He et al. (2022) show that a masked auto-encoder (MAE) learns features that provide
substantial improvements when finetuned on downstream tasks. This property of MAEs has been further
validated on video (Tong et al., 2022), audio (Xu et al., 2022), and across other modalities (Girdhar et al.,
2023). However, their features require supervised finetuning, while our features perform well out of the box.
Discriminative self-supervised learning.
The second line of work, closer to ours, is using discriminative
signals between images or groups of images to learn features. This family of methods has roots in early
deep learning work (Hadsell et al., 2006) but became popular with the emergence of instance classification
methods (Dosovitskiy et al., 2016; Bojanowski & Joulin, 2017; Wu et al., 2018).
Several improvements
3

Published in Transactions on Machine Learning Research (01/2024)
Uncurated Data 
Augmented Curated Data
Curated Data
Embedding
Deduplication
Retrieval
Figure 3: Overview of our data processing pipeline. Images from curated and uncurated data sources
are first mapped to embeddings. Uncurated images are then deduplicated before being matched to curated
images. The resulting combination augments the initial dataset through a self-supervised retrieval system.
were made based either on instance-level objectives (Hénaff et al., 2019; He et al., 2020; Chen & He, 2021;
Chen et al., 2020; Grill et al., 2020; Caron et al., 2021) or clustering (Caron et al., 2018; Asano et al.,
2020; Caron et al., 2020). These methods provide performant frozen features on standard benchmarks like
ImageNet (Russakovsky et al., 2015), but they are hard to scale to larger model sizes (Chen et al., 2021). In
this work, we revisit the training of these approaches in the context of large pretraining datasets and models.
In particular, we build on top of Zhou et al. (2022a) that we find particularly suited for scaling.
Scaling self-supervised pretraining.
A growing body of work has focused on the scaling abilities of
self-supervised learning in terms of data and model size (Caron et al., 2019; Goyal et al., 2019; Tian et al.,
2021; Goyal et al., 2022a).
Most of these works use large quantities of uncurated data to train models
without supervision. They show evidence that discriminative methods scale with data, but because of the
poor quality of the pretraining data, most of the results are obtained by finetuning the features. Of particular
interest, Goyal et al. (2021) have also shown that these methods benefit from scaling in model size given
enough pretrained data. This line of work questions the ability of self-supervised methods to work on any
data while we focus on producing the best pretrained encoders.
Automatic data curation.
Our dataset construction borrows from the image retrieval community (Wein-
zaepfel et al., 2021; Radenović et al., 2018b; Berman et al., 2019; Douze et al., 2009; Tolias et al., 2016; Revaud
et al., 2019). In particular, the use of retrieval to augment the training set has been studied in the context of
semi-supervised learning (Yalniz et al., 2019). Similarly, others have used hashtags or other metadata (Ma-
hajan et al., 2018; Radford et al., 2021) or pretrained vision encoders (Schuhmann et al., 2021; 2022) to
filter uncurated datasets. Unlike these works, we use no pretrained encoders, metadata nor supervision
to filter images and leverage visual similarity between images. Our approach is inspired by text curation
pipelines (Wenzek et al., 2020), where a language model is trained on Wikipedia to score texts extracted
from an uncurated source.
3
Data Processing
We assemble our curated LVD-142M dataset by retrieving, from a large pool of uncurated data, images that
are close to those in several curated datasets. We describe below the main components in our data pipeline
including the curated/uncurated data sources, the image deduplication step and the retrieval system. Our
pipeline does not require any metadata or text and directly works with images, as shown in Fig. 3. We refer
the reader to appendix A for more details on our approach.
Data sources.
Our selection of curated datasets is detailed in the appendix (Table 15) and contains
ImageNet-22k, the train split of ImageNet-1k, Google Landmarks and several fine-grained datasets. For the
4

Published in Transactions on Machine Learning Research (01/2024)
uncurated data source, we collect a raw unfiltered dataset of images from a publicly available repository of
crawled web data. From each web page in the repository, we extract URL links of images from <img> tags.
We discard URLs that are unsafe or restricted by domains, and post-process the downloaded images (PCA
hash deduplication, NSFW filtering, and blurring identifiable faces). This results in 1.2B unique images.
Deduplication. We apply the copy detection pipeline of Pizzi et al. (2022) to the uncurated data and
remove near-duplicate images. This reduces redundancy and increases diversity among images. We also
remove near-duplicates of images contained in the test or validation set of any benchmark used in this work.
Self-supervised image retrieval. We build our curated pretraining dataset by retrieving images from
our uncurated data source that are close to images in our curated sources. In order to do this, we first
compute an image embedding using a self-supervised ViT-H/16 network pretrained on ImageNet-22k, and
use cosine-similarity as a distance measure between images. Then, we perform k-means clustering of the
uncurated data. Given a query dataset for retrieval, if it is large enough we retrieve N (typically 4) nearest
neighbors for each query image. If it is small, we sample M images from the cluster corresponding to each
query image. Although visual inspection seemed to indicate good retrieval quality for N much larger than
4, this leads to more collisions (images that are nearest-neighbor retrievals of multiple queries). We choose
N = 4 as it provides a good tradeoff in that sense.
Implementation Details.
The deduplication and retrieval stages of our pipeline rely on the Faiss li-
brary (Johnson et al., 2019) to efficiently index and compute batch searches of nearest embeddings.
In
particular, we heavily leverage its support for GPU-accelerated indices, using inverted file indices with prod-
uct quantization codes (Jegou et al., 2010). The whole processing is distributed on a compute cluster of 20
nodes equipped with 8 V100-32GB GPUs and takes less than two days to produce the LVD-142M dataset.
4
Discriminative Self-supervised Pre-training
We learn our features with a discriminative self-supervised method that can be seen as a combination of
DINO and iBOT losses with the centering of SwAV (Caron et al., 2020). We also add a regularizer to spread
features and a short high-resolution training phase. We rapidly introduce each of these approaches, but more
details can be found in the related papers, or in our open-sourced code.
• Image-level objective (Caron et al., 2021). We consider the cross-entropy loss between the
features extracted from a student and a teacher network. Both features are coming from the class
token of a ViT, obtained from different crops of the same image.
We pass the student class token
through the student DINO head. This head is an MLP model outputting a vector of scores, that
we call "prototype scores". We then apply a softmax to obtain ps. Similarly, we apply the teacher
DINO head to the teacher class token to obtain teacher prototype scores. We then apply a softmax
followed by a centering with moving average (or a Sinkhorn-Knopp centering as detailed thereafter)
to obtain pt. The DINO loss term corresponds to:
LDINO = −
X
pt log ps
We learn the parameters of the student and build the teacher head with an exponential moving
average of past iterates (He et al., 2020).
• Patch-level objective (Zhou et al., 2022a). We randomly mask some of the input patches
given to the student, but not to the teacher.
We then apply the student iBOT head to the student
mask tokens.
Similarly, we apply the teacher iBOT head to the (visible) teacher patch tokens
corresponding to the ones masked in the student. We then apply the softmax and centering steps
as above, and obtain the iBOT loss term:
LiBOT = −
X
i
pti log psi
5

Published in Transactions on Machine Learning Research (01/2024)
, where i are patch indices for masked tokens. Similarly to above, we learn the parameters of the
student, and build the teacher head through exponential moving average.
• Untying head weights between both objectives.
Both the DINO and the iBOT loss use a
learnable MLP projection head. It is applied to the output tokens and the loss is compute atop. In
Zhou et al. (2022a), an ablation study shows that sharing parameters between the DINO and iBOT
heads leads to better performance. At scale, we observed that the opposite is true, and we therefore
use two separate heads in all our experiments.
• Sinkhorn-Knopp centering (Caron et al., 2020). Ruan et al. (2023) recommend to replace the
teacher softmax-centering step of DINO and iBot by the Sinkhorn-Knopp (SK) batch normalization
of SwAV (Caron et al., 2020). We run the Sinkhorn-Knopp algorithm steps for 3 iterations. For the
student, we apply the softmax normalization.
• KoLeo regularizer
(Sablayrolles et al., 2019).
The KoLeo regularizer derives from the
Kozachenko-Leonenko differential entropy estimator (see Beirlant et al. (1997); Delattre & Fournier
(2017)) and encourages a uniform span of the features within a batch. Given a set of n vectors
(x1, . . . , xn), it is defined as
Lkoleo = −1
n
n
X
i=1
log(dn,i),
where dn,i = minj̸=i ∥xi −xj∥is the minimum distance between xi and any other point within the
batch. We also ℓ2-normalize the features before computing this regularizer.
• Adapting the resolution (Touvron et al., 2019). Increasing image resolution is key to pixel-
level downstream tasks such as segmentation or detection, where small objects disappear at low
resolutions. However, training at high resolution is time and memory demanding, and instead, we
increase the resolution of images to 518×518 during a short period at the end of pretraining. This is
also similar to UniViT training from Likhomanenko et al. (2021) and FlexiViT training from Beyer
et al. (2023).
5
Efficient implementation
We consider several improvements to train models at a larger scale. We train models on A100 GPUs using
PyTorch 2.0. The code and pretrained models are made available under Apache 2.0 license 1. The details of
our models are in the appendix, Table 17. With the same hardware, compared to the iBOT implementation,
the DINOv2 code runs around 2× faster using only 1/3 of the memory.
Fast and memory-efficient attention.
We implemented our own version of FlashAttention (Dao et al.,
2022) to improve memory usage and speed on the self-attention layers.
Our version is on par with or
better than the original on all cases considered, while covering more use-cases and hardware. Due to the
GPU hardware specifics, the efficiency is best when the embedding dimension per head is a multiple of
64, and the matrix operations are even better when the full embedding dimension is a multiple of 256.
As a consequence, our ViT-g architecture slightly differs from the architecture proposed by Zhai et al.
(2022) in order to maximize compute efficiency, and we use an embedding dimension of 1536 with 24 heads
(64 dim/head), rather than 1408 with 16 heads (88 dim/head). Our experiments did not show significant
differences in final accuracy, and our ViT-g backbone counts 1.1B parameters.
Sequence packing.
The DINO algorithm requires forwarding both large crops (at resolution 224) and
small crops (resolution 98). When split into patches, these two groups are represented by token sequences
of different lengths and cannot be forwarded together. In order to accelerate training, we use a trick called
"sequence packing," which originates from NLP (Krell et al., 2022). The idea is simple: we concatenate the
1https://github.com/facebookresearch/dinov2
6

Published in Transactions on Machine Learning Research (01/2024)
sequences we must forward through the transformers into a single long sequence. We pass this sequence
through the transformer blocks as usual. However, a block-diagonal mask is applied to the self-attention
matrix in attention layers, preventing attention between different sequences. This way, the forward is strictly
equivalent to forwarding each sequence separately. This trick gives us significant compute efficiency gains
compared to using separate forward and backward passes, as in prior implementations.
The lower-level
components of our setup are available in the xFormers library2 (Lefaudeux et al. (2022)).
Efficient stochastic depth.
We implement an improved version of stochastic depth (Huang et al., 2016)
that skips the computation of the dropped residuals rather than masking the result. This saves memory and
compute in proportion approximately equal to the drop rate, thanks to specific fused kernels. With high
drop rates (d = 40% in this work), this allows a drastic improvement in compute efficiency and memory
usage. The implementation consists of randomly shuffling the B samples over the batch dimension, and
slicing the first (1 −d) × B samples for the computations in the block.
Fully-Sharded Data Parallel (FSDP).
Minimizing our objective with the AdamW optimizer requires
4 model replicas in float32 precision – student, teacher, optimizer first moments, optimizer second moments.
This sums to 16 GB of memory for a billion-parameter model such as our ViT-g. In order to reduce this
memory footprint per GPU, we split the model replicas across GPUs, i.e., sharding 16 GB across GPUs
using the PyTorch implementation of FSDP. Consequently, the model size is not bounded by the memory of
a single GPU but by the total sum of GPU memory across compute nodes. The Pytorch implementation of
FSDP brings a second advantage, which is to save on the cross-GPU communication costs: the weight shards
are stored in float32 precision as required by the optimizer, but broadcasting weights and reducing gradients
is done in float16 precision for the backbone (MLP heads gradients are reduced in float32 to avoid training
instabilities). This leads to approximately 50% reduction in communication costs compared to the float32
gradient all-reduce operation used in DistributedDataParallel (DDP), which is used in other self-supervised
pretraining methods (Caron et al., 2021; Zhou et al., 2022a). As a consequence, the training procedure
scales more efficiently than DDP with float16 autocast when scaling the number of GPU nodes. Overall,
Pytorch-FSDP mixed-precision is superior to DDP with autocast in virtually all cases we encountered.
Model distillation.
Most of our technical improvements to the training loop aim at improving the training
of large models over large quantities of data. For smaller models, we distill them from our largest model,
the ViT-g, instead of training them from scratch.
Knowledge distillation (Hinton et al., 2014) aims at
reproducing the output of a large model with a smaller model by minimizing some distance between both
outputs for a set of given inputs. Since our objective function is a form of distillation from the teacher
network to the student network, we leverage the same training loop with a few exceptions: we use a larger
model as a frozen teacher, keep a spare EMA of the student that we use as our final model, remove the
masking and stochastic depth, and, apply the iBOT loss on the two global crops. In our ablations, we
observe that this approach achieves better performance than training from scratch, even for a ViT-L. Our
distillation method ends up close to the one described by Duval et al. (2023), except we do not modify the
loss terms for distillation and evaluate the EMA of the student.
6
Ablation Studies
We present a set of ablations to empirically validate different components of our pipeline: the technical
modifications described in Sec. 4, the pretraining data and the impact of model distillation. We consider
various downstream tasks that are described in Sec. 7.
6.1
Improved Training Recipe
Our approach improves over the iBOT method by combining it with several existing components described
in Sec. 4. To evaluate their importance, we train multiple models where we successively add components to
a baseline iBOT model. We report the Top-1 accuracy on the validation set of ImageNet-1k with a k-NN
2https://github.com/facebookresearch/xformers
7

Published in Transactions on Machine Learning Research (01/2024)
INet-1k k-NN
INet-1k linear
iBOT
72.9
82.3
+(our reproduction)
74.5 ↑1.6
83.2 ↑0.9
+LayerScale, Stochastic Depth
75.4 ↑0.9
82.0 ↓1.2
+128k prototypes
76.6 ↑1.2
81.9 ↓0.1
+KoLeo
78.9 ↑2.3
82.5 ↑0.6
+SwiGLU FFN
78.7 ↓0.2
83.1 ↑0.6
+Patch size 14
78.9 ↑0.2
83.5 ↑0.4
+Teacher momentum 0.994
79.4 ↑0.5
83.6 ↑0.1
+Tweak warmup schedules
80.5 ↑1.1
83.8 ↑0.2
+Batch size 3k
81.7 ↑1.2
84.7 ↑0.9
+Sinkhorn-Knopp
81.7 =
84.7 =
+Untying heads = DINOv2
82.0 ↑0.3
84.5 ↓0.2
Table 1: Ablation study of the training differences between iBOT and DINOv2. We optimize
for k-NN performance, as in our experience, the linear probe performance is lower-bounded by the k-NN
performance. Some modifications, like LayerScale and a high Stochastic Depth (rate=0.4), incur a decrease
in linear probe performance, but have the benefits of increasing the stability of training by avoiding NaN
loss values during training (Touvron et al., 2022). Overall, these modifications allowed for the next set of
improvements to be added. Experiments are run using the ViT-Large architecture on ImageNet-22k.
Training Data
INet-1k
Im-A
ADE-20k
Oxford-M
iNat2018
iNat2021
Places205
INet-22k
85.9
73.5
46.6
62.5
81.1
85.6
67.0
INet-22k \ INet-1k
85.3
70.3
46.2
58.7
80.1
85.1
66.5
Uncurated data
83.3
59.4
48.5
54.3
68.0
76.4
67.2
LVD-142M
85.8
73.9
47.7
64.6
82.3
86.4
67.6
Table 2: Ablation of the source of pretraining data.
We compare the INet-22k dataset that was
used in iBOT to our dataset, LVD-142M. Each model is trained for the same number of iterations, that is
smaller than in our final run, without high-resolution adaptation. Pretraining on LVD-142M maintains the
performance over INet-1k while leading to models that perform better in other domains.
and a linear probe in Table 1. Generally, we observe that each component improves the performance on
either k-NN or linear probing and even both in most cases. Only LayerScale and Stochastic Depth incur a
performance drop in linear probing but significantly improve the training stability in our experience.
6.2
Pretraining Data Source
The quality of features is directly related to the quality of the pretraining data. In this experiment, we
probe the impact of LVD-142M compared to ImageNet-22k, a commonly used pretraining dataset, or using
directly raw and uncurated data. For the uncurated dataset, we randomly sample 142 million images from
the same data source as LVD-142M. We train a ViT-g/14 on each dataset for the same number of iterations.
We also include a variant of ImageNet-22k obtained by removing the synsets of ImageNet-1k (INet-22k \
INet-1k) for completeness. We report the comparisons in Table 2.
The most salient observation is that training on a curated set of images works better on most benchmarks
than training on uncurated data.
This confirms the benefit of curating data, even in the case of self-
supervised pretraining. When compared with models trained on ImageNet-22k, training on LVD-142M is
also superior on all the benchmarks but ImageNet-1k. This confirms that training on a more diverse set of
images improves the quality of the features in domains that are not covered by ImageNet-22k. We also see
that training on our curated data increases the performances on domains that are not used for the curation
process (INaturalist 2018, 2021 and Places205), proving that scale and diversity can benefit unseen domains.
8

Published in Transactions on Machine Learning Research (01/2024)
L
H
g
84
85
86
ImageNet-1k
INet-22k
LVD142M
L
H
g
74
76
78
ImageNet-V2
L
H
g
50
60
ImageNet-Sketch
L
H
g
92
94
Food101
L
H
g
80
90
Cars
L
H
g
35
40
AmsterTime
L
H
g
20
40
Oxford-H
Figure 4: Model scale versus data scale.
Evolution of performance as a function of model size for
two different pretraining datasets: ImageNet-22k (14M images) and LVD-142M (142M images). The ViT-g
trained on LVD-142M surpasses the ViT-g trained on ImageNet-22k on most benchmarks.
KoLeo
INet-1k
Im-A
ADE-20k
Oxford-M
✕
85.3
70.6
47.2
55.6
✓
85.8
72.8
47.1
63.9
(a) Koleo loss
MIM
INet-1k
Im-A
ADE-20k
Oxford-M
✕
85.3
72.0
44.2
64.3
✓
85.8
72.8
47.1
63.9
(b) MIM objective in iBOT
Table 3: (a) Effect of the KoLeo loss term. (b) Effect of the iBOT Masked Image Modeling (MIM) loss
term. Evaluation performed on ImageNet-{1k,A} (classification with linear probe, accuracy %), ADE-20k
(segmentation with linear layer, mIoU) and Oxford-M (image retrieval, mAP). Each model is trained on the
same number of iterations, that is smaller than our final run. The KoLeo loss term improves nearest-neighbor
search tasks (e.g. retrieval), and the MIM loss improves patch-level tasks (e.g. segmentation).
Overall, the conclusion of this ablation is that our dataset provides a good balance of different types of
images that leads to the best performance overall.
6.3
Model Size and Data
We quantify the importance of scaling data with the model size in Fig. 4. As the size of models grow, training
on LVD-142M becomes more beneficial than training on ImageNet-22k. For instance, a ViT-g trained on
LVD-142M matches the performance on ImageNet-1k of a model trained on ImageNet-22k while significantly
outperforming it on the other benchmarks.
6.4
Loss Components
We validated the proposed technical improvements in Sec. 6.1 by adding them incrementally. This section
analyzes the performance hit observed if we ablate specific loss terms, starting from our best-performing
model. We ablate the importance of the KoLeo loss and the impact of the masked image modeling term.
For both, we report performance on ImageNet-1k using a linear classifier, ADE-20k segmentation using a
linear classifier, and nearest-neighbor image retrieval on Oxford-M. Table 3a shows the impact of using the
KoLeo loss. We see that the instance retrieval performance improves by more than 8%, confirming that this
term helps spread features in the output space. At the same time, the other metrics do not suffer from this
regularization. In Table 3b, we show the impact of using the masked image modeling term from iBOT. This
term is critical for dense prediction tasks, leading to almost 3% performance improvement.
6.5
Impact of Knowledge Distillation
For small architectures, we distill larger models instead of training them from scratch. We use the distillation
procedure described in Sec. 5. We evaluate the effectiveness of this approach by comparing a ViT-L/14
trained from scratch with one distilled from a ViT-g/14 over 12 benchmarks in Fig. 5. We also report the
performance of the ViT-g/14 used for distillation as a topline. The distilled model outperforms the one
trained from scratch on all 12 benchmarks, validating our pretraining approach for small models.
9

Published in Transactions on Machine Learning Research (01/2024)
INet-1k
Food
Cars
iNat18
iNat21
Places 205
Oxford-H
Paris-H
INet-A
INet-R
Kitti
NYUd
ViT-L/14 Scratch
ViT-L/14 Distill
ViT-g/14 Scratch
84.5
86.3 86.5
92.8
94.3
94.7
81.8
90.1
91.4
77.8
80.4
81.6
83.1
85.1
85.7
66.0
67.3
67.5
47.7
52.6 52.1
77.6
84.4
82.7
61.7
71.3
75.9
68.1
74.1
78.8
2.57
2.5
2.35
0.345
0.333
0.298
(a) Comparison on individual metrics
Arch
Method
INet-1k
Segm.
Depth↓
Classif.
ViT-g/14
Scratch
86.5
73.4
1.00
92.1
ViT-L/14
Scratch
84.5
72.2
1.10
90.2
ViT-L/14
Distill
86.3
73.3
1.08
91.2
Arch
Method
Finegr.
Retriev.
ARSketch
Video
ViT-g/14
Scratch
78.3
75.2
77.0
69.3
ViT-L/14
Scratch
75.8
71.3
69.5
67.3
ViT-L/14
Distill
77.6
76.3
74.5
67.5
(b) Averaged metrics on 8 vision tasks
Figure 5: Effectiveness of knowledge distillation. Comparison between a ViT-L trained from scratch
or distilled from DINOv2 using ViT-g/14. For reference, we also report the performance of the ViT-g/14
teacher. We show that a ViT-L model distilled from a frozen ViT-g outperforms a the same model trained
from scratch on all benchmarks, sometimes even outperforming the distillation target.
224
336
512
640
768
resolution
78
79
80
81
82
83
84
Accuracy
ImageNet-1k
224
336
512
640
768
resolution
39
41
43
45
47
mIoU
ADE-20K
224
416
224
416
Figure 6: Role of resolution. Performance of ViT-L/16 trained on ImageNet-1k at fixed resolution (“224”
and “416”) or trained at 224 then 416 for a short duration (“224→416”). We train linear classifiers on top of
frozen features at different resolutions and report Top-1 accuracy on ImageNet and mIoU on ADE-20k. We
observe that performing SSL training at high resolution for a short duration achieve behavior and results
close to training at the same high resolution for the full training, at a fraction of the cost.
6.6
Impact of Resolution
We measure the impact of changing the resolution during the pretraining on the performance of image and
patch-level features. We consider models trained from scratch using a fixed resolution of either 224 × 224 or
416×416, and a model trained from scratch at 224×224, then resumed for 10k more iterations at 416×416.
High-resolution training is compute-intensive, so we conduct this ablation on a small setup: a ViT-L/16
trained on ImageNet1k. In Fig. 6, we report the performance of a linear probe on ImageNet-1k and ADE-
20k, evaluated at various resolutions. The model trained on high-resolution images performs best across
resolutions, but this comes at a high cost: training at 416 is approximately 3 × more compute-intensive
than training at 224. On the other hand, training at high resolution for only 10k iterations at the end of the
training is almost as good and only requiring a fraction of the compute. As a consequence, we include this
step at the end of the training rather than training at a high resolution from scratch.
10

Published in Transactions on Machine Learning Research (01/2024)
7
Results
In this section, we present the empirical evaluation of our models on many image understanding tasks. We
evaluate both global and local image representations, on category and instance-level recognition, semantic
segmentation, monocular depth prediction, and action recognition.
We detail the list of benchmarks in
Appendix C. The goal of this evaluation is twofold. First, we show that our self-supervised features outper-
form the current state of the art by a very large margin. Second, we show that they match, or surpass the
performance of weakly-supervised ones on a substantial number of tasks.
Baselines. In our comparisons, we use two kinds of models as baselines. We compare to the best performing
self-supervised models that are openly available. First, we run our evaluations for MAE (He et al., 2022),
DINO (Caron et al., 2021), SEERv2 (Goyal et al., 2022a), MSN (Assran et al., 2022), EsViT (Li et al.,
2022a), Mugs (Zhou et al., 2022b) and iBOT (Zhou et al., 2022a). When several architectural variants were
proposed for a given method, we report results for the one that leads to best top-1 accuracy on ImageNet-
1k. Second, we report performance of open-source weakly-supervised models such as CLIP (Radford et al.,
2021), OpenCLIP (Ilharco et al., 2021; Cherti et al., 2023), and SWAG (Singh et al., 2022). When evaluating
models on ImageNet-1k, we report the performance for each of the aforementioned methods. For all other
evaluations, we report the four best-performing models amongst SSL ones. Also, for reference, we report the
best performing OpenCLIP-G for weakly-supervised ones.
7.1
ImageNet Classification
As a first evaluation, we probe the quality of the holistic image representation produced by the model on the
ImageNet-1k classification dataset. We evaluate the quality of features by training a simple classifier over a
frozen backbone, and do not perform finetuning of the backbone weights. Following previous work, we use
a linear model for simplicity, ensuring a reproducible evaluation, despite the fact that classes may not be
linearly separable. Because most SSL methods were developped using ImageNet-1k validation performance
as a debugging signal, we also report the top-1 accuracy on ImageNet-ReaL and ImageNet-V2. In order
to report this additional validation performance, for all models, we run the evaluation with our code. We
compare our frozen features to the best publicly available SSL features in Table 4, regardless of architecture
or pretraining data. We see the components proposed in this work lead to a very significant improvement
(+4.2%) over the previous state of the art (iBOT ViT-L/16 trained on ImageNet-22k) on linear evaluation.
At the same time, we also see that the performance increase on the alternative test sets is larger for our
method, indicating stronger generalization. We describe details of our linear evaluation in Appendix B.3.
How far are we from weakly-supervised models?
We also want to validate that our features are com-
petitive with state-of-the-art open-source weakly supervised models. To this end, we compare on ImageNet-
1k, using the linear evaluation, to three off-the-shelf methods with several architectural variants. For all
models, we run the linear evaluation using our code, after making sure that our numbers match those re-
ported in technical reports and papers. We show the result of this evaluation in Table 4. We see that our
backbone, surpases the performance of OpenCLIP with a ViT-G/14 architecture (+0.3%) and EVA-CLIP
with a ViT-g/14 (+0.1%). At the same time, we also observe that our performance on the ImageNet-V2 test
set is significantly better (+1.1% versus EVA-CLIP), indicating better generalization. For the remainder of
this section, we report OpenCLIP-G as a reference for weakly-supervised models.
Can we finetune the encoders?
We question if the ability of our models to produce high quality frozen
features impact their performance when finetuned with supervision on a specific dataset. While this is not
core t