# FAST: Efficient Action Tokenization for Vision-Language-Action Models

> 2025 · id: arxiv:2501.09747 · arXiv: 2501.09747 · pdf: https://arxiv.org/pdf/2501.09747 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

FAST: Efficient Action Tokenization for
Vision-Language-Action Models
Karl Pertsch∗,1,2,3, Kyle Stachowicz∗,2,
Brian Ichter1, Danny Driess1, Suraj Nair1, Quan Vuong1, Oier Mees2, Chelsea Finn1,3, Sergey Levine1,2
1Physical Intelligence, 2UC Berkeley, 3Stanford
https://pi.website/research/fast
Abstract—Autoregressive
sequence
models,
such
as
Transformer-based
vision-language
action
(VLA)
policies,
can be tremendously effective for capturing complex and
generalizable robotic behaviors. However, such models require
us to choose a tokenization of our continuous action signals,
which determines how the discrete symbols predicted by the
model map to continuous robot actions. We find that current
approaches for robot action tokenization, based on simple
per-dimension, per-timestep binning schemes, typically perform
poorly when learning dexterous skills from high-frequency
robot data. To address this challenge, we propose a new
compression-based tokenization scheme for robot actions, based
on the discrete cosine transform. Our tokenization approach,
Frequency-space Action Sequence Tokenization (FAST), enables
us to train autoregressive VLAs for highly dexterous and
high-frequency tasks where standard discretization methods fail
completely. Based on FAST, we release FAST+, a universal robot
action tokenizer, trained on 1M real robot action trajectories.
It can be used as a black-box tokenizer for a wide range of
robot action sequences, with diverse action spaces and control
frequencies. Finally, we show that, when combined with the
π0 VLA, our method can scale to training on 10k hours of
robot data and match the performance of diffusion VLAs, while
reducing training time by up to 5x.
I. INTRODUCTION
Large, high-capacity Transformer models can be tremen-
dously effective for capturing complex and generalizable
robotic behaviors both from scratch [8, 69, 51, 6, 20, 62]
and using models pre-trained for next-token prediction on
Internet-scale image-text corpora [10, 39, 63, 7, 65]. How-
ever, these models require choosing a tokenization of the
continuous action signal, which determines how the discrete
symbols predicted by the model map to continuous robot
actions [64, 34, 41, 12]. It is widely known that a good choice
of tokenization can be critical to the performance of sequence
models [55, 57]. Prior robotic policies of this sort typically
use na¨ıve tokenization strategies based on a per-dimension,
per-timestep binning scheme [9, 10, 39]. We find that such
methods perform poorly when learning dexterous skills with
high-frequency control (see Figure 2, right). We observe that
correlations between time steps are a major challenge for
na¨ıve tokenization strategies when predicting sequences of
∗: Core contributors
Correspondence to: research@physicalintelligence.company
Fig. 1: We propose FAST, a simple yet effective approach
for tokenization of robot action trajectories via time-series
compression. FAST enables training of autoregressive VLAs
that solve complex dexterous manipulation tasks and gener-
alize broadly to new scenes. We use it to train π0-FAST,
a generalist robot policy that matches the performance of
the state-of-the-art π0 diffusion VLA on dexterous and long-
horizon manipulation tasks, while training 5x faster (top).
arXiv:2501.09747v1  [cs.RO]  16 Jan 2025

future actions, i.e., action “chunks”, as is common for high-
frequency control. Highly correlated action tokens diminish
the effectiveness of the next token prediction objective used
in autoregressive VLAs. Intuitively, in such cases low token
prediction loss can often be achieved with mappings as trivial
as simply copying the most recent action token, leaving models
in poor local optima.
In this work, we propose a new tokenization strategy
from first principles. Our key insight is that robot action
signals need to be compressed before training, to reduce
correlation between consecutive tokens. We take inspiration
from compression-based tokenization strategies, such as the
byte-pair encoding method commonly used by language mod-
els [27, 57]. However, since robotic actions are continuous,
the corresponding compression strategy should be chosen
accordingly. We therefore base our method off of the discrete
cosine transform (DCT) encoding, which is widely used for
compressing continuous signals such as images (e.g., JPEG
compression). We find that the resulting tokenization approach,
Frequency-space Action Sequence Tokenization (FAST), en-
ables us to train autoregressive VLA policies via simple
next token prediction (see Figure 2, left) for highly dexter-
ous and high-frequency tasks where standard discretization
methods fail entirely. Additionally, FAST for the first time
enables efficient VLA training on the recently introduced
DROID dataset [38], a large-scale multitask “in-the-wild”
robot manipulation dataset. The resulting policy is the first
language-conditioned generalist manipulation policy that can
be successfully evaluated zero-shot in unseen environments,
simply by prompting it in natural language.
Based on FAST, we develop FAST+, a universal robot ac-
tion tokenizer, trained on 1M real robot action trajectories that
cover a large diversity of robot embodiments, action spaces
and control frequencies. We demonstrate that the FAST+ to-
kenizer effectively tokenizes a wide range of robot action
sequences, from single-arm to bi-manual and mobile robots,
and is a good off-the-shelf tokenizer for training autoregressive
VLA models. When integrated with the π0 VLA, FAST-based
autoregressive VLAs scale to training on 10k hours of robot
data and achieve performance comparable to diffusion-based
VLAs across a variety of tasks, while reducing training time
by up to 5x (see Figure 1).
II. RELATED WORK
Tokenization for language, text, and audio. Tokenization is
a key component of training pipelines for modern transformer-
based autoregressive sequence models, and the choice of
tokenization approach can have significant impact on model
training and downstream performance [55]. While there are
multiple works exploring the training of “tokenization-free”
language models [28, 53] that directly operate on bit streams,
most language models today rely on a text tokenization
stage prior to training. A common approach is byte pair
encoding [27, 55], which compresses input text by merging
frequently occurring token sequences into new tokens. For
images, learned compression schemes present an effective
DCT
Vision-Language-Action Model
Vision-Language-Action Model
Vision-Language-Action Model
“fold the shirt”
“fold the shirt”
“fold the shirt”
FAST
OpenVLA-style
Score
Data Control Frequency (Hz)
2
10
20
5
10
8
6
4
2
0
Comparing Action Tokenizers across 
Control Frequencies
FAST Action 
tokenization
High-
Frequency 
Robot Data
Fig. 2: Left: FAST tokenization enables training of autoregres-
sive Transformers for dexterous robot control via simple next
token prediction. Right: FAST outperforms popular binning
tokenization schemes, e.g., used in OpenVLA [39], particu-
larly for high-frequency robot data.
approach: input images can be represented as “soft tokens”
produced by a pre-trained vision encoder [44], and full au-
toregressive image input-output can be achieved with a vector-
quantizing autoencoder [22, 59]. Similar approaches can be
extended to the video domain [66]. In audio generation and
speech synthesis, which share the time-series structure of ac-
tion prediction, state-of-the-art models typically encode time-
series audio data using either frequency-domain spectrogram
images [29] or using learned vector quantizers [68].
Vision-language-action models. Recently, multiple works
have developed generalist robot policies [9, 51, 6, 10, 20, 39,
62, 11] that are trained on increasingly large robot learning
datasets [52, 38, 60, 24, 47, 35]. One promising approach for
training generalist policies are vision-language-action models
(VLAs; [10, 17, 39, 67, 7, 63, 73, 71, 13, 11]). VLAs fine-
tune vision-language models, that are pre-trained on internet-
scale image and text data, for robot control. This has multiple
benefits: using large vision-language model backbones, with
billions of parameters, provides policies with the necessary
expressivity for fitting large robot datasets. Reusing weights
pre-trained on internet-scale datasets also improves the ability
of VLAs to follow diverse language commands and generalize,
e.g., to new objects and scene backgrounds [10, 39, 67, 63, 36].
Most VLA models today are confined to rather simple, low-
frequency control tasks, particularly models that use the most
common autoregressive VLA design [10, 39]. We show that
this is a direct consequence of the action tokenization schemes
employed by these models, which make training on dexterous
tasks challenging. We introduce a new action tokenization

approach that allows us to train the first autoregressive VLAs
on dexterous and high-frequency robot data.
Action representations for VLA training. Prior works have
explored various action parameterizations for training robot
policies, including VLAs. One line of work uses “semantic”
action representations like language sub-tasks [21, 2, 4], or
keypoints [50, 32, 25, 19]. Such approaches can often learn
from few examples or even perform tasks zero-shot without
any robot examples [50, 32, 25], but require hand-designed
low-level controllers for task execution, limiting their gener-
ality. An alternative approach directly trains VLAs to output
low-level robot control commands given image and language
instruction inputs. The most common design directly embeds
actions into discrete tokens, that can be generated with stan-
dard autoregressive sequence models, like any popular vision-
language model. Existing approaches map from continuous
robot actions to discrete action tokens using a simple per-
dimension, per-timestep binning scheme [9, 10, 39]. We find
that this scheme struggles to scale to high-frequency robot
control tasks. We propose a new tokenization scheme for
robot actions, based on time-series compression techniques,
that allows us to train autoregressive VLAs on high-frequency
data. A number of works have also proposed alternatives
to tokenization, for example by using regression heads or
introducing new weights for diffusion decoding [20, 7, 41, 63].
In comparison, our approach does not require modifications of
the underlying pre-trained transformer model, can easily be ap-
plied to any pre-trained autoregressive transformer model, and
achieves competitive performance to state-of-the-art diffusion-
based VLAs [7] across many tasks, while being significantly
more compute efficient to train.
Another set of related work explores vector-quantized action
representations [41, 3, 49]. Such approaches train a vector-
quantized encoder-decoder network, for which reconstruction
quality can be sensitive to hyperparameter choices and struc-
ture [66]. We find that these methods perform well at coarse,
low-fidelity reconstruction tasks, but fail on high-frequency
tasks when fine-grained control is required. In comparison, our
FAST tokenization scheme has few hyperparameters and can
reconstruct actions with high precision while offering strong
compression properties.
III. PRELIMINARIES
Problem formulation. Our goal is to train policies π(a1:H|o)
that map an observation o to a sequence of future robot
actions a1:H. We assume that policies output an “action
chunk” [69, 40], a sequence of H actions [15, 7, 69], which
makes it easier to produce temporally-consistent actions and
reduces compounding error. The goal of action tokenization is
to define a mapping Ta : a1:H →[T1, . . . , Tn] from a sequence
of continuous actions a1:H, with dimensionality |A|, to a
sequence of n discrete tokens T ∈|V| from a vocabulary of
size |V|. Note that the number of tokens n may differ between
action sequences, just like sentences of the same length may
be tokenized into a variable number of text tokens.
Fig. 3: Effect of sampling rate on prediction performance.
We train a small autoregressive transformer model on a
didactic interpolation task, in which the network must predict
the black dashed curve given the four circles. We find that
models trained with the binning tokenization approach used in
prior VLAs [10, 39] produce increasingly poor predictions as
we increase the sampling frequency of the underlying signal,
due to strong correlation between consecutive tokens at high
frequencies. Our FAST tokenization approach, based on the
discrete cosine transform (DCT), addresses the problem and
leads to high-quality predictions across all sampling rates.
Binning-based action tokenization. The most commonly
used approach for action tokenization is a simple binning dis-
cretization scheme [8, 10, 39, 72, 56]. For a given action a, this
approach discretizes each dimension independently, dividing
the range of values in the training dataset into N uniform
bins, most commonly using N = 256. For a sequence of D-
dimensional actions a1:H, this tokenization scheme would be
applied to each time step, resulting in a final token sequence
Ta
 a1:H

= [T1,1, . . . , T1,D, . . . , TH,1, . . . , TH,D]. For high-
frequency robot data, this tokenization scheme is sub-optimal:
it can easily produce hundreds of tokens per action chunk,
which make training challenging and lead to slow inference.
IV. CASE STUDY: HOW DOES TOKENIZATION AFFECT
VLA TRAINING?
To illustrate the challenge of training autoregressive poli-
cies with current action tokenization approaches, we start

with a simple didactic example. We create a synthetic time-
series dataset where the goal is to predict a cubic spline
that interpolates four randomly-generated points (see Fig-
ure 3, bottom). This toy problem reflects the challenge faced
by policies trained on high-frequency action chunks, which
must predict a sequence of continuous actions given some
conditioning information. We tokenize the target sequences
using the na¨ıve tokenization scheme employed in previous
VLA policies, which discretizes each element in the sequence
separately into one of 256 bins (see Section III). We then
train a small, autoregressive transformer policy to predict the
tokenized signal given the conditioning points. We repeat this
experiment for different sampling rates of the target signal,
from 25 to 800 timesteps per sequence, without changing
the underlying dataset. This emulates training autoregressive
policies on action data collected at different frequencies.
The average prediction MSE of autoregressive models
trained at different frequencies is shown in Figure 3, top
(“naive”). We observe that the model with binning tokenization
achieves good prediction performance (i.e., low MSE) for
low sampling rates. But as the sampling rate increases, the
prediction error steeply increases, until eventually the model
simply copies the first action, as seen in the qualitative
visualization in Figure 3, bottom left. Note that this issue
cannot be attributed to the data itself: the complexity of the
underlying data distribution does not change, and we would
expect a model with the same capacity trained for the same
number of steps to achieve comparable performance across all
sampling rates. So what happened?
To understand how the tokenization scheme impacts learn-
ing performance, we need to look at the learning objective
itself. Fundamentally, autoregressive models are trained to
predict the next token, given all previous tokens. As such, their
learning signal is proportional to the marginal information
content of Ti given T1:i−1. Crucially, when using the na¨ıve
per-timestep tokenization scheme, this marginal information
approaches zero as the control frequency of the training signal
increases: for smooth signals, as timesteps get shorter the
change per timestep decreases proportionally. This greatly
slows down the rate of convergence during training and can
make it challenging to fit complex, high-frequency datasets.
Indeed, such challenges have been observed in prior work.
For instance, OpenVLA worked well on the low-frequency
BridgeV2 and RT-1 datasets, but has struggled to fit the higher-
frequency DROID dataset [39]. The result of our case study
underlines the importance of designing better tokenization
schemes for robot actions.
V. EFFICIENT ACTION TOKENIZATION VIA TIME-SERIES
COMPRESSION
We saw in the previous section how redundancy in high-
frequency action trajectories can lead to low marginal in-
formation for each action token, and thereby poor training
performance. To address this, we need a tokenization approach
that compresses the highly redundant action signal into a
smaller number of high-information tokens. In this section,
we will first describe a simple approach for compressing
continuous time series (V-A), then use it to design an action
tokenization algorithm (Section V-B), and finally explain how
we train a universal tokenizer for robot actions (Section V-C).
A. Time-Series Compression via Discrete Cosine Transform
There is a rich body of work on effectively compressing
continuous time series, from approaches that compress signals
after transforming them into the frequency domain [18, 1, 61]
to learned compression approaches, e.g., based on vector
quantization [59, 48]. One key takeaway of our work is that
any sufficiently effective compression approach, when applied
to the action targets, is suited to improve the training speed of
VLA models. In practice, there are a few considerations that
may still lead us to favor some compression algorithms over
others, e.g., the complexity of training the tokenizer, and how
efficient is it at tokenizing and detokenizing actions.
In this work, we use a compression algorithm based on
the discrete cosine transform (DCT) [1]. DCT is a frequency-
space transform that represents a continuous signal as a sum
of cosine elements of various frequencies. Low frequencies
capture the overall shape of the signal, while high-frequency
components reflect sharp jumps. DCT is a commonly used
transformation for compression algorithms, e.g., for JPEG im-
age compression [61], due to its simplicity and computational
efficiency, and its strong compression property on practical
images: since pixels often vary smoothly, DCT can often
represent most of the information of an input signal in only
a few coefficients. Signals can be compressed by omitting
frequency components with low weights. Compared to learned
compression approaches based on vector quantization, DCT-
based compression is an analytical approach, thus extremely
simple and fast.
B. The FAST Tokenization Algorithm
We use the discrete cosine transform to design FAST, a
quick and effective tokenization approach for robot actions.
We detail the steps from raw robot actions to action tokens
in Figure 4. We first normalize the input actions, such that
the 1st and 99th quantile of values in the training dataset for
each action dimension maps to the range [−1, . . . , 1]. This
initial normalization step is useful to bring the data into a
specified range and also makes tokenization of cross-embodied
datasets with different action scales easier. We use quantiles
to be robust to outlier actions which occasionally occur in
large robot datasets. After the data is normalized, we apply the
discrete cosine transform to each action dimension separately.
To compress the DCT-converted signal we can simply omit
insignificant coefficients, which we implement through a scale-
and-round operation, where the scaling coefficient is a hyper-
parameter that trades off between lossiness and compression
rate of the tokenization operation.
After the rounding operation, the DCT coefficient matrix
is typically sparse, with most entries being zero and only a
few significant coefficients remaining per action dimension. To
actually realize the compression, we must convert this sparse

Normalized action chunk
(first 2 dimensions displayed)
1
Frequency components
(first 2 dimensions displayed)
2
Sparse frequency matrix (each dim = 1 row)
3
124
-86
344
-45
178
12
0
3
15
-3
0
1
-1
0
0
0
1
12
0
5
-2
Low-frequency components first
4
124
-86
344
-45
178
12
0
3
0
15
• • •
Compressed action tokens
5
978
233
19
1022
1
124
-86
344
-45
178
12
0
3
0
15
•••
Flatten
Byte
Pair
Encoding
(BPE)
Discrete
Cosine
Transform
(DCT)
Quantize
Fig. 4: Overview of the FAST action tokenization pipeline. Given a normalized chunk of actions, we apply discrete cosine
transform (DCT) to convert the signal to the frequency domain. We then quantize the DCT coefficients and use byte-pair
encoding (BPE) to compress the flattened sequence of per-dimension DCT coefficients into the final action token sequence.
See Section V-B for a detailed description.
matrix into a sequence of dense tokens. We flatten the matrix
into a 1-dimensional vector of integers, interleaving action di-
mensions by including all low-frequency components first, and
train a byte pair encoding (BPE) tokenizer [27] to losslessly
compress it into dense action tokens. The BPE step “squashes”
the zero-valued components and merges frequently-occurring
coefficient combinations across action dimensions. We choose
BPE to compress the DCT matrix, since many efficient im-
plementations exist and it can produce a fixed-size output
vocabulary that can be easily integrated into the existing
vocabulary of vision-language models for VLA training. Other
lossless compression algorithms like Huffman coding [33] or
Lempel-Ziv methods [75] (the algorithms underlying the gzip
compression approach) could be used instead, but we leave
this investigation for future work.
Note that the order of flattening the |A|×H DCT coefficient
matrix prior to BPE encoding can have significant impact on
policy training. There are two options: column-first flattening,
i.e., concatenate the lowest-frequency components for each
dimension first, or row-first flattening, i.e., concatenating all
frequency components for a single action dimension first. We
choose the former, since we find that predicting the low-
frequency components, that characterize the overall shape of
the output sequence, first during autoregressive prediction
leads to more stable policy rollouts.
All operations in our tokenization pipeline are easily invert-
ible, allowing fast decoding of predicted actions. The tokenizer
has only two hyperparameters: the scale applied to the DCT
coefficients before rounding, and the vocabulary size of the
BPE compression step. We find that both parameters are not
very sensitive, and we use the same values across all our
single-dataset tokenization experiments (rounding scale 10,
BPE vocabulary size 1024). This is in contrast to end-to-end
learned compression modules that rely on vector quantiza-
Algorithm 1 FAST Tokenizer
Require: scale γ, (for inference) BPE dictionary Φ
procedure FASTTOKENIZER(a1:H)
Ci
j ←DCT
 ai
1:H

▷Compute DCT coefficients
¯Ci
j ←round
 γ · Ci
j

▷Quantize coefficients
[Tk] ←
 ¯C1
1, ¯C2
1, . . . , C1
2, . . . , Cn
H

▷Flatten tokens
BPE Training:
ϕ ←TrainBPE(D := {[Tk]})
Tokenization:
 ¯T1, . . . , ¯T¯k

←BPE ([T1, . . . , Tk], ϕ)
return action tokens
tion [59]. Such networks are often tedious to train, and require
careful dataset-specific hyperparameter selection to achieve
good reconstruction [66, 48]. Our experiments show that
our DCT-based tokenization approach trains higher-performing
policies than VQ-based approaches, while being significantly
simpler and easier to tune.
We empirically demonstrate the benefits of our DCT-
based tokenization in the toy example from Section IV.
Figure 3 shows that training the autoregressive model on DCT-
compressed target tokens achieves constantly low prediction
error across a wide range of sampling frequencies. We provide
a concise summary of our tokenization approach in Algo-
rithm 1 and test the effectiveness of FAST tokenization on
robot control problems in Section VI.
C. A Universal Robot Action Tokenizer
The only learned component of our tokenizer is the vo-
cabulary of the BPE encoder, which needs to be trained for
each new dataset that the tokenizer is being applied to. While
this learning process is fast (typically only a few minutes),
it adds additional friction to using FAST tokenization. Thus,

we aim to train a universal action tokenizer, that can encode
chunks of robot actions from any robot. To this end, we train a
tokenizer using the pipeline described above on a large, cross-
embodied robot action dataset, consisting of approximately
one million 1-second action chunks from single-arm, bi-
manual and mobile manipulation robots, with joint and end-
effector control action spaces and various control frequencies.
We provide a detailed breakdown of the data mixture used
for training the universal tokenizer in Appendix A. Once
trained, our universal action tokenizer, FAST+, can be applied
as a black-box tokenizer on 1-second action sequences from
any robot setup. Our experimental evaluation shows that it is
competitive to tokenizers tuned for individual datasets.
Code
release. We release our pre-trained universal ac-
tion
tokenizer,
FAST+,
in
a
convenient
HuggingFace
AutoProcessor class, that makes it easy to apply the
tokenizer to any new robot action chunk in three lines of code:
from transformers import AutoProcessor
tokenizer = AutoProcessor.from_pretrained(
"physical-intelligence/fast",
trust_remote_code=True
)
tokens = tokenizer(action_chunk)
For best compression results, we recommend normalizing
input actions to range [−1, . . . , 1] via quantile normalization
as described in Section V-B, and tokenizing 1-second action
chunks at a time. Our module also makes it easy to train a
new FAST tokenizer on a given dataset of action chunks:
from transformers import AutoProcessor
tokenizer = AutoProcessor.from_pretrained(
"physical-intelligence/fast",
trust_remote_code=True
)
new_tokenizer = tokenizer.fit(action_dataset)
VI. EXPERIMENTS
In our experiments, we test FAST with two VLA backbones:
π0 [7] and OpenVLA [39]. We compare FAST to alternative
action tokenization schemes and ablate key design decisions.
We then compare π0 models trained with FAST tokenization
to the state-of-the-art π0 flow-matching (diffusion) VLA, and
test the scaling of autoregressive VLA training with FAST to
large, cross-embodied datasets with 10k hours of dexterous
robot manipulation data.
A. Experimental Setup
Policy
implementation.
We
test
different
tokenization
schemes for autoregressive VLA training with popular VLA
backbones. For most of our experiments, we use π0 [7],
a VLA based on PaliGemma-3B [5]. We also test with
OpenVLA [39], which is built on Prismatic 7B [37]. During
training, we tokenize 1-second action chunks and overwrite the
least used tokens in the VLM vocabulary with the resulting
action tokens, following prior VLAs [10, 39]. We fine-tune
Fig. 5: Evaluation environments. We test FAST across
7 evaluation environments: 6 real-robot tasks and 1 simulation
environment. The tasks are designed to test VLA performance
on highly dexterous tasks, like folding cloths from a laundry
basket (“Laundry Folding”), and generalization, e.g., zero-shot
table-top manipulation in unseen environments (“DROID”).
the VLA models for robot action prediction, without weight
freezing. We provide more details on the policy training setup
in Appendix C.
Evaluation tasks. We develop a suite of 7 evaluation tasks
(6 real robot, 1 simulated; see Figure 5), designed to test
VLA performance on both, highly dexterous tasks like laundry
folding, and generalization tasks, like performing table-top
manipulations 0-shot in unseen environments.
• Libero: We test on the Libero [43] simulated benchmark
suites. We measure average performance across Libero-
Spatial, Libero-Object, Libero-Goal, and Libero-10.
• Table bussing [7] (20 Hz): a UR5 single-arm robot needs
to clean a table, sorting 12 objects into a trash bin (for

trash) and a plastic container (for plates, bowls, cups and
cutlery). The task requires precise grasping of various
objects.
• T-Shirt folding [7] (50 Hz): a bi-manual ARX robot
setup needs to fold various shirts on a stationary table
top. At the beginning of the task, the shirts are placed
flat on the table. Succeeding at the task requires precise
grasps and movements to fold the shirt.
• Grocery bagging [7] (20 Hz): a UR5 single-arm robot
needs to pack seven objects from a table into a grocery
bag, taking care to not topple or rip the bag in the process.
This task requires picking a diverse set of objects and
carefully inserting them into the bag.
• Toast out of toaster [7] (50 Hz): a bimanual Trossen
Viper-X robot needs to remove two slices of bread from
a toaster and place them on a plate. This task requires
precise grasping and placement of the bread slices.
• Laundry folding [7] (50 Hz): a bi-manual ARX robot
needs to take shirts and shorts from a basket, flatten them
on a table, fold and stack them. This is the most dexterous
task we test. It requires precise grasps, dynamic motions
to flatten the cloths, retrying and corrections when cloths
got tangled up, and precise placements of the folded
cloths on the existing stack of cloths. We report success
rate on individual clothing items.
• Zero-shot DROID tabletop manipulation [38] (15 Hz):
we test a policy trained on the full DROID dataset across
various table-top manipulation tasks like picking and
placing objects, wiping, opening and closing drawers etc.
Importantly, we test the policy in a completely unseen
environment, with a new table setup, background, novel
objects, viewpoint and table height. To our knowledge,
this is the first “zero-shot” evaluation of DROID policies
in a completely unseen environment, without co-training
or fine-tuning, simply by prompting a pre-trained model
with natural language.
Following Black et al. [7], we use grocery bagging, the toaster
task, and laundry folding only to evaluate our most powerful,
generalist VLA in Section VI-F. We provide additional details
on training datasets and evaluation tasks in Appendix E.
Comparisons. We test FAST, our DCT-based action tokeniza-
tion approach, trained on each evaluation dataset individually,
and FAST+, our universal DCT-based action tokenizer, trained
on a large dataset of 1M action sequences. Note that we
trained the universal tokenizer on the most diverse real robot
dataset we could assemble, which includes data from our real-
robot evaluation tasks. We compare both tokenizers to the
per-dimension binning scheme used by prior autoregressive
VLAs like RT-2 [10], RT-2-X [52] and OpenVLA [39], dubbed
na¨ıve tokenization. We apply the binning tokenization to each
time step in the action chunk separately and then concatenate.
Finally, while our approach provides a compressed tokeniza-
tion without the need to train any separate model, we can
consider an alternative compression scheme that instead trains
a model to produce a quantized representation of the action
chunk via FSQ [48], a simpler alternative to VQ-VAE [59].
This tokenization strategy has been previously used to tokenize
high-dimensional image data [48, 66], and can be viewed
as an ablation of our compression-based approach, utilizing
compressed representations but with a more complex learning-
based alternative to our relatively simple DCT-based method.
B. Comparing Action Tokenizers for VLA Training
Dataset
Action
Dimension
Control
Frequency
Avg. Token
Compression
Naive
FAST
BridgeV2
7
5 Hz
35
20
1.75
DROID
7
15 Hz
105
29
3.6
Bussing
7
20 Hz
140
28
5.0
Shirt Fold
14
50 Hz
700
53
13.2
TABLE I: Comparison of the average token count per
action chunk for na¨ıve tokenization and FAST. We use 1-
second chunks in all datasets. With our method, each chunk
requires many fewer tokens, particularly for high-frequency
domains such as the T-shirt folding task, indicating that it is
more effective at removing redundancy.
We first provide a comparison of compression rates between
our proposed FAST tokenizer and the na¨ıve binning scheme
used in prior works in Table I. We use 1-second action chunks
from datasets with various action dimensionalities and control
frequencies. For both approaches we use the default hyper-
parameters, which have comparable tokenization errors. We
see that FAST achieves a significant compression of the input
action sequences across all datasets. The compression benefits
are especially pronounced for datasets with high-frequency
action data. Interestingly, FAST consistently generates roughly
30 action tokens per chunk per robot arm (i.e., 60 tokens for
the bi-manual setup) in each of the domains. This suggests that
FAST finds a representation that approximates the complexity
of the underlying action signal, and is largely independent of
the frequency of the action data.
We note that this compression is not entirely lossless,
with a trade-off between compression ratio and reconstruction
accuracy determined by the scale parameter γ from Algo-
rithm 1. Figures in Table I are at comparable reconstruction
accuracy. Please see Appendix B for plots showing the trade-
off between compression and fidelity for each of the tokenizers
we compare.
Next, we train policies using the policy architecture and
tokenization approaches described in Section VI-A. We report
results in Figure 6.
Overall, we find that the na¨ıve tokenization applied in
prior works struggles to learn effective policies on high-
frequency robot data. This is particularly apparent for the
highest frequency tasks in our evaluations: Table Bussing
(20Hz) and T-Shirt Folding (50Hz). On both tasks, policies
trained with na¨ıve tokenization are unable to make progress
on the task.
In contrast, we find that compression-based tokenization
leads to effective training. Comparing FAST to our FSQ

T-SHIRT FOLDING 
(50Hz)
TABLE BUSSING 
(20Hz)
DROID 
(15Hz)
LIBERO
% Success
% Task Progress
% Success
% Task Progress
% Success
100
80
60
40
20
0
AVERAGE
Naive
FSQ
FAST
FAST+
Fig. 6: Comparison of policy performance using different tokenization approaches. We find that tokenization approaches
that compress action targets (FAST, FSQ) lead to substantially more efficient training than the na¨ıve binning tokenization used
in prior VLAs. Overall, we find that FAST leads to more effective policy training than FSQ, particularly on dexterous real-robot
tasks. Our universal tokenizer, FAST+, matches the performance of dataset-specific tokenizers. We report mean and 95% CI.
Berkeley
Stanford
U. Washington
Fig. 7: Evaluation environments of FAST policy trained
on DROID [38]. We find that the same policy checkpoint
generalizes robustly, and performs various simple table-top
tasks zero-shot across three university campuses.
baseline, we find that FAST is as good or at times better,
particularly on the dexterous, high-frequency tasks, despite
being much simpler and requiring no separate neural network
training.
Notably, FAST tokenization enables the first success-
ful training of a strong generalist policy on the DROID
dataset [38], which can be evaluated zero-shot in unseen
environments, without fine-tuning, by simply prompting it
in natural language. All prior works, including the original
DROID paper [38] and OpenVLA [39], did not show zero-shot
results and focused entirely on co-training or fine-tuning eval-
uations instead. We demonstrate the generality of our DROID
policy by testing it on various table-top manipulation tasks
15
10
5
1
Single Arm
SERL
Bussing
T-DROID Joint
T-DROID EEF
SOAR
UMI
UMI
UMI on Legs
Dexterous
NYU Hand
HATO
Berkeley Dex Arm
Berkeley Dex Hand
Humanoid
Humanplus
Open Television
Compression Ratio  
(Naive / FAST)
Nav
Waymo
Fig. 8: Universal tokenizer. We test the compression rate
achieved by our FAST+ tokenizer vs. na¨ıve tokenization across
diverse robot datasets, unseen during tokenizer training. We
find that FAST is effective across a wide range of robot
morphologies, action spaces and control frequencies.
in environments across three university campuses (Figure 7).
Out of the box, the policy can competently perform simple
manipulation tasks, like picking and placing objects, opening
and closing cupboards and turning on faucets, across a wide
range of scenes and camera viewpoints. Even unsuccessful
trials show sensible behavior, like approaching the handles of
microwave and dish washer doors, even if ultimately failing
to open them. We show success and failure videos on our
website. While far from perfect, the level of generality and
robustness of this policy substantially exceeds that of prior
DROID policies.
C. Universal Action Tokenizer
In this section, we evaluate the performance of our universal
action tokenizer, FAST+, which we trained on 1M real robot
action sequences (see Section V-C). To test the generality
of the tokenizer, we assemble a diverse set of small testing
datasets. This set spans a wide range of robot morphologies,
action spaces, and control frequencies (see Figure 8, with a full
list of datasets in Table III). Note that none of these datasets is
part of the tokenizer training set. They thus test a scenario in

which the tokenizer is applied to a completely new robot setup
without recomputing the tokenization. We find that the FAST+
tokenizer achieves good compression performance across a
wide range of robot datasets, reducing the number of action
tokens by 2x across all datasets, and significantly more on
some.
We also test performance of the universal tokenizer for
policy training, and report results alongside the per-dataset
tokenizers in Figure 6. Across all tasks, the universal tok-
enizer closely matches the performance of the dataset-specific
FAST tokenizers, suggesting that the universal tokenizer can
be used as a strong default for robot action tokenization.
D. Ablation Studies
We analyze two key aspects of our method: (1) Is our
FAST tokenization approach independent of the underlying
VLA backbone? (2) How important is the BPE compression
step, the only learned component of our tokenization pipeline.
T-Shirt Folding
% Success
OpenVLA
OpenVLA + FAST
100
80
60
40
20
To answer the first question, we
train an OpenVLA policy [39] on
the challenging high-frequency T-
shirt folding dataset, comparing
the na¨ıve tokenization approach
originally used in OpenVLA to
our FAST+ tokenizer. To comply
with the task setup, we modify the
OpenVLA model code to accept
multiple input images and predict
1-second action chunks. The re-
sults on the right demonstrate that FAST is able to significantly
boost performance of OpenVLA, enabling it to train effectively
on high-frequency robot manipulation data. This suggests, that
our tokenization approach is independent of the underlying
model backbone, and may be easily applied to a wide range
of pre-trained autoregressive transformer models.
FAST
FAST without BPE
100
80
60
40
20
TABLE BUSSING
% Task Progress
T-SHIRT FOLD
% Success
Secondly, we ablate the BPE
encoding step on the table buss-
ing and T-shirt folding tasks. The
figure on the right shows that
the resulting policies without BPE
encoding achieve worse rollout
performance (but still outperform
na¨ıve tokenization). Intuitively, the
DCT transform still concentrates
most of the signal’s information
in a few tokens, improving the
learning signal. However, without
BPE, there is a large number of repeated 0-tokens which dilute
the learning signal and also significantly slow down inference,
since models need to autoregressively predict hundreds of
action tokens, ultimately leading to worse policy performance.
E. Comparing FAST to Diffusion
In this section, we compare π0, a state-of-the-art diffusion
VLA, to our model that combines π0 with FAST and uses
π0 FAST
π0
π0 (3x compute)
LIBERO
TABLE
BUSSING
T-SHIRT
FOLDING
DROID
AVERAGE
100
80
60
40
20
0
% Task Progress
Single Task Performance
Fig. 9: Comparison of diffusion π0 [7] to our π0 model with
FAST decoding on single-task training. On small datasets
(Libero, T-Shirt Folding), both perform comparably. On large
datasets (Table Bussin