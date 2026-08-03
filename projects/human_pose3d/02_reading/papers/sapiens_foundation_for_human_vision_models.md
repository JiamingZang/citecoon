# Sapiens: Foundation for Human Vision Models

> 2024 · id: arxiv:2408.12569 · arXiv: 2408.12569 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
We present Sapiens, a family of models for four fun-
damental human-centric vision tasks – 2D pose estima-
tion, body-part segmentation, depth estimation, and surface
normal prediction. Our models natively support 1K high-
resolution inference and are extremely easy to adapt for in-
dividual tasks by simply fine-tuning models pretrained on
over 300 million in-the-wild human images. We observe
that, given the same computational budget, self-supervised
pretraining on a curated dataset of human images signifi-
cantly boosts the performance for a diverse set of human-
centric tasks. The resulting models exhibit remarkable gen-
eralization to in-the-wild data, even when labeled data is
scarce or entirely synthetic. Our simple model design also
brings scalability – model performance across tasks im-
proves as we scale the number of parameters from 0.3 to
2 billion. Sapiens consistently surpasses existing baselines
across various human-centric benchmarks.
We achieve
significant improvements over the prior state-of-the-art on
Humans-5K (pose) by 7.6 mAP, Humans-2K (part-seg) by
17.1 mIoU, Hi4D (depth) by 22.4% relative RMSE, and
THuman2 (normal) by 53.5% relative angular error.
arXiv:2408.12569v3  [cs.CV]  27 Aug 2024

“Sapiens—pertaining to, or resembling modern humans.”

## introduction
Recent years have witnessed remarkable strides towards
generating photorealistic humans in 2D [17, 28, 50, 118]
and 3D [69, 89, 102, 109].
The success of these meth-
ods is greatly attributed to the robust estimation of var-
ious assets such as 2D keypoints [14, 67], fine-grained
body-part segmentation [119], depth [113], and surface nor-
mals [89, 108]. However, robust and accurate estimation
of these assets is still an active research area, and compli-
cated systems to boost performance for individual tasks of-
ten hinder wider adoption. Moreover, obtaining accurate
ground-truth annotation in-the-wild is notoriously difficult
to scale. Our goal is to provide a unified framework and
models to infer these assets in-the-wild to unlock a wide
range of human-centric applications for everybody.
We argue that such human-centric models should satisfy
three criteria: generalization, broad applicability, and high
fidelity. Generalization ensures robustness to unseen con-
ditions, enabling the model to perform consistently across
varied environments. Broad applicability indicates the ver-
satility of the model, making it suitable for a wide range
of tasks with minimal modifications. High fidelity denotes
the ability of the model to produce precise, high-resolution
outputs, essential for faithful human generation tasks. This
paper details the development of models that embody these
attributes, collectively referred to as Sapiens.
Following the insights from [34, 79, 91], leveraging large
datasets and scalable model architectures is key for gener-
alization. For broader applicability, we adopt the pretrain-
then-finetune approach, enabling post-pretraining adapta-
tion to specific tasks with minimal adjustments. This ap-
proach raises a critical question:
What type of data is
most effective for pretraining? Given computational lim-
its, should the emphasis be on collecting as many human
images as possible, or is it preferable to pretrain on a less
curated set to better reflect real-world variability? Exist-
ing methods often overlook the pretraining data distribution
in the context of downstream tasks. To study the influence
of pretraining data distribution on human-specific tasks, we
collect the Humans-300M dataset, featuring 300 million di-
verse human images. These unlabelled images are used to
pretrain a family of vision transformers [27] from scratch,
with parameter counts ranging from 300M to 2B.
Among various self-supervision methods for learning
general-purpose visual features from large datasets [5, 19,
34, 47, 48, 121], we choose the masked-autoencoder (MAE)
approach [48] for its simplicity and efficiency in pretrain-
ing. MAE, having a single-pass inference model compared
to contrastive or multi-inference strategies, allows process-
ing a larger volume of images with the same computational
resources. For higher-fidelity, in contrast to prior methods,
we increase the native input resolution of our pretraining to
1024 pixels, resulting in a ∼4× increase in FLOPs com-
pared to the largest existing vision backbone [91]. Each
model is pretrained on 1.2 trillion tokens. Table 1 outlines
a comparison with earlier approaches. For finetuning on
human-centric tasks [15, 101, 113, 119], we use a consis-
tent encoder-decoder architecture. The encoder is initial-
ized with weights from pretraining, while the decoder, a
lightweight and task-specific head, is initialized randomly.
Both components are then finetuned end-to-end. We focus
on four key tasks - 2D pose estimation, body-part segmen-
tation, depth, and normal estimation, as shown in Fig. 1.
Consistently with prior studies [56, 122], we affirm the
critical impact of label quality on the model’s in-the-wild
performance. Public benchmarks [23, 40, 55] often con-
tain noisy labels, providing inconsistent supervisory signals
during model fine-tuning. At the same time, it is impor-
tant to utilize fine-grained and precise annotations to align
closely with our primary goal of 3D human digitization. To
this end, we propose a substantially denser set of 2D whole
body keypoints for pose estimation and a detailed class vo-
cabulary for body part segmentation, surpassing the scope
of previous datasets (please refer to Fig. 1). Specifically, we
introduce a comprehensive collection of 308 keypoints en-
compassing the body, hands, feet, surface, and face. Addi-
tionally, we expand the segmentation class vocabulary to 28
classes, covering body parts such as the hair, tongue, teeth,
upper/lower lip, and torso. To guarantee the quality and
consistency of annotations and a high degree of automation,
we utilize a multi-view capture setup to collect pose and
segmentation annotations. We also utilize human-centric
synthetic data for depth and normal estimation, leverag-
ing 600 detailed scans from RenderPeople [84] to generate
high-resolution depth maps and surface normals.
We show that the combination of domain-specific large-
scale pretraining with limited, yet high-quality annotations
leads to robust in-the-wild generalization.
Overall, our
method demonstrates an effective strategy for developing
highly precise discriminative models capable of perform-
ing in real-world scenarios without the need for collecting a
costly and diverse set of annotations.

## method
3.1. Humans-300M Dataset
We utilize a large proprietary dataset for pretraining of ap-
proximately 1 billion in-the-wild images, focusing exclu-
sively on human images. The preprocessing involves dis-
carding images with watermarks, text, artistic depictions,
or unnatural elements. Subsequently, we use an off-the-
shelf person bounding-box detector [103] to filter images,
retaining those with a detection score above 0.9 and bound-
ing box dimensions exceeding 300 pixels. Fig. 2 provides
an overview of the distribution of the number of people per
image in our dataset, noting that over 248 million images
contain multiple subjects.

Ground Truth
Mask Ratio 75%
Mask Ratio 80%
Mask Ratio 85%
Mask Ratio 90%
Mask Ratio 95%
Figure 3. Sapiens reconstruction on unseen images. Top: Each triplet contains the ground truth (left), the masked image (center), and the
MAE reconstruction (right), with a masking ratio of 75%, a patch size of 16, and an image size of 1024. Bottom: Varying the mask ratio
between [0.75, 0.95] during inference reveals a minimal reduction in quality, underscoring the model’s understanding of human images.
3.2. Pretraining
We follow the masked-autoencoder [48] (MAE) approach
for pretraining. Our model is trained to reconstruct the orig-
inal human image given its partial observation. Like all au-
toencoders, our model has an encoder that maps the visible
image to a latent representation and a decoder that recon-
structs the original image from this latent representation.
Our pretraining dataset consists of both single and multi-
human images; each image is resized to a fixed size with a
square aspect ratio. Similar to ViT [27], we divide an image
into regular non-overlapping patches with a fixed patch size.
A subset of these patches is randomly selected and masked,
leaving the rest visible. The proportion of masked patches
to visible ones is defined as the masking ratio, which re-
mains fixed throughout training. We refer to MAE [48] for
more details. Fig. 3 (Top) shows the reconstruction of our
pretrained model on unseen human images.
Our models exhibit generalization across a variety of im-
age characteristics including scales, crops, the age and eth-
nicity of subjects, and number of subjects. Each patch token
in our model accounts for 0.02% of the image area com-
pared to 0.4% in standard ViTs, a 16× reduction - this pro-
vides a fine-grained inter-token reasoning for our models.
Fig.3 (Bottom) shows that even with an increased mask ra-
tio of 95%, our model achieves a plausible reconstruction
of human anatomy on held-out samples.
3.3. 2D Pose Estimation
We follow the top-down paradigm, which aims to detect
the locations of K keypoints from an input image I ∈
RH×W ×3. Most methods pose this problem as heatmap
prediction, where each of K heatmaps represents the prob-
ability of the corresponding keypoint being at any spatial
location.
Similar to [111], we define a pose estimation
transformer, P, for keypoint detection. The bounding box
at training and inference is scaled to H × W and is pro-
vided as an input to P. Let y ∈RH×W ×K denote the
K heatmaps corresponding to the ground truth keypoints
for a given input I. The pose estimator transforms input I
to a set of predicted heatmaps, ˆy ∈RH×W ×K, such that
ˆy = P(I). P is trained to minimize the mean squared loss
Lpose = MSE(y, ˆy). During finetuning, the encoder of P
is initialized with the weights from pretaining, and the de-
coder is initialized randomly. The aspect ratio H : W is set
to be 4 : 3, with the pretrained positional embedding being
interpolated accordingly[58]. We use lightweight decoders
with deconvolution and convolution operations.
We finetune the encoder and decoder in P across multi-
ple skeletons, including K = 17 [67], K = 133 [55] and
a new highly-detailed skeleton, with K = 308, as shown
in Fig. 4 (Left). Compared to existing formats with at most
68 facial keypoints, our annotations consist of 243 facial
keypoints, including representative points around the eyes,
lips, nose, and ears. This design is tailored to meticulously
capture the nuanced details of facial expressions in the real
world. With these keypoints, we manually annotated 1 mil-
lion images at 4K resolution from an indoor capture setup.
3.4. Body-Part Segmentation
Commonly referred to as human parsing, body-part seg-
mentation aims to classify pixels in the input image I into
C classes. Most methods [40] transform this problem to
estimating per-pixel class probabilities to create a proba-
bility map ˆp ∈RH×W ×C such that ˆp = S(I), where

a) Full-Body – 308 kps
b) Hands – 40 kps
c) Face – 243 kps
d) Body-Part Segmentation: 28 Classes
Figure 4. Ground-truth annotations for 2D pose estimation and body-part segmentation.
S is the segmentation model. As outlined previously, we
adopt the same encoder-decoder architecture and initializa-
tion scheme for S. S is finetuned to minimize the weighted
cross-entropy loss between the actual p and predicted ˆp
probability maps, Lseg = WeightedCE(p, ˆp).
We finetune S across two part-segmentation vocabular-
ies: a standard set with C = 20 [40] and a new larger vo-
cabulary with C = 28, as illustrated in Fig.4 (Right). Our
proposed vocabulary goes beyond previous datasets in im-
portant ways. It distinguishes between the upper and lower
halves of limbs and incorporates more detailed classifica-
tions such as upper/lower lips, teeth, and tongue. To this
end, we manually annotate 100K images at 4K resolution
with this vocabulary.
3.5. Depth Estimation
For depth estimation, we adopt the architecture used for
segmentation, with the modification that the decoder output
channel is set to 1 for regression. We denote the ground-
truth depth map of image I by d ∈RH×W , the depth es-
timator by D, where ˆd = D(I), and M as the number of
human pixels in the image. For the relative depth estima-
tion, we normalize d to the range [0, 1] using max and min
depths in the image. The Ldepth loss [32] for D is defined as
follows:
∆d = log(d) −log(ˆd),
(1)
∆d = 1
M
M
X
i=1
∆di,
(∆d)2 = 1
M
M
X
i=1
(∆di)2,
(2)
Ldepth =
r
(∆d)2 −1
2(∆d)2.
(3)
We render 500, 000 synthetic images using 600 high-
resolution photogrammetry human scans as shown in Fig. 5
to obtain a robust monocular depth estimation model with
high-fidelity. A random background is selected from a 100
HDRI environment map collection. We place a virtual cam-
era within the scene, randomly adjusting its focal length,
rotation, and translation to capture images and their associ-
ated ground-truth depth maps at 4K resolution.
Figure 5. Ground-truth synthetic annotations for depth and surface normal estimation.

## experiments
In this section, we initially provide an overview of the im-
plementation details. Subsequently, we conduct compre-
hensive benchmarking across four tasks: pose estimation,
part segmentation, depth estimation, and normal estimation.
4.1. Implementation Details
Our largest model, Sapiens-2B, is pretrained using 1024
A100 GPUs for 18 days using PyTorch.
We use the
AdamW [73] optimizer for all our experiments. The learn-
ing schedule includes a brief linear warm-up, followed by
cosine annealing [72] for pretraining and linear decay [65]
for finetuning. All models are pretrained from scratch at
a resolution of 1024 × 1024 with a patch size of 16. For
finetuning, the input image is resized to a 4:3 ratio, i.e.
1024 × 768. We use standard augmentations like cropping,
scaling, flipping, and photometric distortions. A random
background from non-human COCO [67] images is added
for segmentation, depth, and normal prediction tasks. Im-
portantly, we use differential learning rates [114] to pre-
serve generalization i.e. lower learning rates for initial lay-
ers and progressively higher rates for subsequent layers.
The layer-wise learning rate decay is set to 0.85 with a
weight decay of 0.1 for the encoder. We detail the design
specifications of Sapiens in Table. 2. Following [34, 100],
we prioritize scaling models by width rather than depth.
Note that the Sapiens-0.3B model, while architecturally
similar to the traditional ViT-Large, consists of twentyfold
more FLOPs due to its higher resolution.
4.2. 2D Pose Estimation
We finetune Sapiens for face, body, feet, and hand (K =
308) pose estimation on our high-fidelity annotations. For
training, we use the train set with 1M images and
for evaluation, we use the test set, named Humans-
5K, with 5K images. Our evaluation is top-down [111]
i.e. we use an off-the-shelf detector [37] for bounding-
box and conduct single human pose inference.
Table 3
shows a comparison of our models with existing methods
for whole-body pose estimation.
We evaluate all meth-
ods on 114 common keypoints between our 308 keypoint
vocabulary and the 133 keypoint vocabulary from COCO-
WholeBody [55]. Sapiens-0.6B surpasses the current state-
of-the-art, DWPose-l [115] by +2.8 AP. Contrary to DW-
Pose [115], which utilizes a complex student-teacher frame-
work with feature distillation tailored for the task, Sapiens
adopts a general encoder-decoder architecture with large
human-centric pretraining.
Interestingly, even with the same parameter count, our
models demonstrate superior performance compared to
their counterparts.
For instance, Sapiens-0.3B exceeds
VitPose+-L by +5.6 AP, and Sapiens-0.6B outperforms
VitPose+-H by +7.9 AP. Within the Sapiens family, our re-
sults indicate a direct correlation between model size and
performance. Sapiens-2B sets a state-of-the-art with 61.1
AP, a significant improvement of +7.6 AP to the prior art.
Despite fine-tuning with annotations from a indoor capture
studio, Sapiens demonstrate robust generalization to real-
world, as shown in Fig. 6.
Figure 6. Pose estimation with Sapiens-1B for 308 keypoints on in-the-wild images.

## related_work
Our work explores the limits of training large architectures
on a large number of in-the-wild human images. We build
on prior work from different areas: pretraining at scale, hu-
man vision tasks, and large vision transformers.
Pretraining at Scale.
The remarkable success of large-
scale pretraining [26, 95] followed by task-specific finetun-
ing for language modeling [2, 13, 53, 96, 99, 100] has estab-
lished this approach as a standard practice. Similarly, com-
puter vision methods [1, 4, 33, 34, 42, 79, 82, 85, 87, 120]
are progressively embracing extensive data scales for pre-
training. The emergence of large datasets, such as LAION-
5B [90], Instagram-3.5B [77], JFT-300M [92], LVD-
142M [79], Visual Genome [60], and YFCC100M [97], has
enabled the exploration of a data corpus well beyond the
scope of traditional benchmarks [61, 67, 86]. Salient work
in this domain includes DINOv2 [79], MAWS [91], and
AIM [34]. DINOv2 achieves state-of-the-art performance
in generating self-supervised features by scaling the con-
trastive iBot [121] method on the LDV-142M dataset [79].
MAWS [91] studies the scaling of masked-autoencoders
(MAE) [48] on billion images.
AIM [34] explores the
scalability of autoregressive visual pretraining similar to
BERT [26] for vision transformers [27]. In contrast to these
methods which mainly focus on general image pretrain-
ing or zero-shot image classification, we take a distinctly
human-centric approach: our models leverage a vast col-
lection of human images for pretraining, subsequently fine-
tuning for a range of human-related tasks.
Human Vision Tasks. The pursuit of large-scale 3D hu-
man digitization [8, 44, 64, 74] remains a pivotal goal
in computer vision [12].
Significant progress has been
made within controlled or studio environments [3, 59,
63, 69, 70, 76, 89], yet challenges persist in extending
these methods to unconstrained environments [29]. To ad-
dress these challenges, developing versatile models capa-
ble of multiple fundamental tasks such as keypoint estima-
tion [21, 35, 46, 51, 57, 78, 80, 93, 106], body-part seg-
mentation [36, 40, 41, 41, 75, 104, 105], depth estima-
N=1 
17%
N=2
14%
N=3
13%
N>=4
56%
Number of Humans (N)
Figure 2.
Overview of number of humans per image in the
Humans-300M dataset.
tion [9, 10, 32, 43, 52, 66, 83, 113], and surface normal
prediction [6, 7, 31, 39, 62, 88, 101, 108] from images in
natural settings is crucial. In this work, we aim to develop
models for these essential human vision tasks which gener-
alize to in-the-wild settings.
Scaling Architectures.
Currently, the largest publicly-
accessible language models contain upwards of 100B pa-
rameters [49], while the more commonly used language
models [94, 100] contain around 7B parameters. In contrast,
Vision Transformers (ViT) [27], despite sharing a similar
architecture, have not been scaled to this extent success-
fully. While there are notable endeavors in this direction,
including the development of a dense ViT-4B [20] trained
on both text and images, and the formulation of techniques
for the stable training of a ViT-22B [25], commonly uti-
lized vision backbones still range between 300M to 600M
parameters [24, 38, 45, 68] and are primarily pretrained
at an image resolution of about 224 pixels. Similarly, ex-
isting transformer-based image generation models, such as
DiT [81] use less than 700M parameters, and operate on a
highly compressed latent space. To address this gap, we in-
troduce Sapiens - a collection of large, high-resolution ViT
models that are pretrained natively at a 1024 pixel image
resolution on millions of human images.

## conclusion
Importance of Pretraining Data Source.
The feature
quality is closely linked to the pretraining data quality. We
assess the importance of pretraining on various data sources
for human-centric tasks by pretraining Sapiens-0.3B on
each dataset under identical training schedules and number
of iterations. We fine-tune the model on each task and select
early checkpoints for evaluation, reasoning that early-stage
Image
PIFuHD
Sapiens-1B
ECON
Figure 9. Qualitative comparison of Sapiens-1B with PIFuHD [89] and ECON [108] for surface normal estimation on in-the-wild images.

40.5
44.9
50.8
57.9
62.5
66.1
1M
10M
20M
30M
40M
50M
35
40
45
50
55
60
65
70
Normal Estimation (% within 30 deg)
Number of Unique Human Images in Pretraining
Figure 10. Sapiens-0.3B’s normal estimation performance with
unique human images seen during pretraining.
fine-tuning better reflects the model’s generalization capa-
bility. We investigate the impact of pretraining at scale on
general images (which may include humans) versus exclu-
sively human images using Sapiens. We randomly select
100 million and 300 million general images from our 1 bil-
lion image corpus to create the General-100M and General-
300M datasets, respectively. Table 7 showcases the com-
parison of pretraining outcomes. We report mAP for pose
on Humans-5K, mIoU for segmentation on Humans-2K,
RMSE for depth on THuman2.0, and mean angular error
in degrees for normal estimation on Hi4D. Aligned with
findings from [112], our results show that pretraining with
Human300M leads to superior performance across all met-
rics, highlighting the benefits of human-centric pretraining
within a fixed computational budget.
We also study the effect of number of unique human im-
ages seen during pretraining with normal estimation per-
formance. We report % within 30◦. Again, we maintain
identical conditions for Sapiens-0.3B pretraining and fine-
Pretraining Source
#Images Pose (↑)
Seg(↑) Depth(↓) Normal(↓)
Random Initialization
-
30.2
40.3
0.720
35.4
General-100M
100M
35.7
50.1
0.351
27.5
General-300M
300M
37.3
52.8
0.347
26.8
Humans-100M
100M
43.6
61.2
0.316
24.0
Humans-300M (Full)
300M
47.0
66.5
0.288
21.8
Table 7. Comparison of Sapiens-0.3B pretrained on various data
sources.
A domain-specific pretraining yields superior results
compared to general data sources.
tuning. Fig.10 shows a steady improvement in performance
as the pretraining data size increases without saturation. In
summary, the diversity of human images observed during
pretraining directly correlates with improved generalization
to down-stream tasks.
Zero-Shot Generalization. Our models exhibit broad gen-
eralization to a variety of settings. For instance, in seg-
mentation, Sapiens are finetuned on single-human images
with limited subject diversity, minimal background varia-
tion, and solely third-person views (see Fig. 4).
Never-
theless, our large-scale pretraining enables generalization
across number of subjects, varying ages, and egocentric
views, as shown in Fig. 11. These observations similarly
hold for other tasks.
Limitations.
While our models generally perform well,
they are not perfect.
Human images with complex/rare
poses, crowding, and severe occlusion are challenging (see
supplemental for details). Although aggressive data aug-
mentation and a detect-and-crop strategy could mitigate
these issues, we envision our models as a tool for acquiring
large-scale, real-world supervision with human-in-the-loop
to develop the next generations of human vision models.