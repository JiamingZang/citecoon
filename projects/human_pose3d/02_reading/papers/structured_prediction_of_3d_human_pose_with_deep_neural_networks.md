# Structured Prediction of 3D Human Pose with Deep Neural Networks

> 2016 · id: W2404595106 · arXiv: 1605.05180 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Most recent approaches to monocular 3D pose estimation rely on Deep Learning.
They either train a Convolutional Neural Network to directly regress from image to 3D
pose, which ignores the dependencies between human joints, or model these dependen-
cies via a max-margin structured learning framework, which involves a high computa-
tional cost at inference time.
In this paper, we introduce a Deep Learning regression architecture for structured
prediction of 3D human pose from monocular images that relies on an overcomplete
auto-encoder to learn a high-dimensional latent pose representation and account for joint
dependencies. We demonstrate that our approach outperforms state-of-the-art ones both
in terms of structure preservation and prediction accuracy.
1

## introduction
3D human pose can now be estimated reliably by training algorithms to exploit depth data [7,
27] or video sequences [3, 11, 30]. However, estimating such a 3D pose from single ordi-
nary images remains challenging because of the many ambiguities inherent to monocular 3D
reconstruction, including occlusions, complex backgrounds, and, more generally, the loss of
depth information resulting from the projection from 3D to 2D.
These ambiguities can be mitigated by exploiting the structure of the human pose, that
is, the dependencies between the different body joint locations. This has been done by ex-
plicitly enforcing physical constraints at test time [25, 29] and by data-driven priors over the
pose space [6, 28, 33]. Recently, dependencies have been modeled within a Deep Learning
framework using a max-margin formalism [20], which resulted in state-of-the-art prediction
accuracy. While effective, these methods suffer from the fact that they require solving a
computationally expensive optimization problem to estimate the 3D pose.
c⃝2016. The copyright of this document resides with its authors.
It may be distributed unchanged freely in print or electronic forms.
∗indicates equal contribution.
arXiv:1605.05180v1  [cs.CV]  17 May 2016

2
TEKIN ET AL.: STRUCTURED PREDICTION OF 3D HUMAN POSE
(a)
(b)
(c)
Figure 1: Our architecture for the structured prediction of the 3D human pose. (a) An
auto-encoder whose hidden layers have a larger dimension than both its input and output
layers is pretrained. In practice we use either this one or more sophisticated versions that are
described in more detail in Section 3.1 (b) A CNN is mapped into the latent representation
learned by the auto-encoder. (c) the latent representation is mapped back to the original pose
space using the decoder.
By contrast, regression-based methods, such as [19], directly and efﬁciently predict the
3D pose given the input image. While this often comes at the cost of ignoring the underlying
structure, several methods have been proposed to account for it [14, 26]. In [14], this was
achieved by making use of Kernel Dependency Estimation (KDE) [5, 36], which maps both
input and output to high-dimensional Hilbert spaces and learns a mapping between these
spaces. Because this approach relies on handcrafted features and does not exploit the power
of Deep Learning, it somewhat under-performs more recent CNN-based techniques [19, 20].
In this paper, we demonstrate that we can account for the human pose structure within a
deep learning framework by ﬁrst training an overcomplete auto-encoder that projects body
joint positions to a high dimensional space represented by its middle layer, as depicted
by Fig. 1(a).
We then learn a CNN-based mapping from the input image to this high-
dimensional pose representation as shown in Fig. 1(b). This is inspired by KDE in that it
can be understood as replacing kernels by the auto-encoder layers to predict the pose param-
eters in a high dimensional space that encodes complex dependencies between different body
parts. As a result, it enforces implicit constraints on the human pose, preserves the human
body statistics, and improves prediction accuracy, as will be demonstrated by our experi-
ments. Finally, as illustrated in Fig. 1(c), we connect the decoding layers of the auto-encoder
to this network, and ﬁne-tune the whole model for pose estimation.
In short, our contribution is to show that combining traditional CNNs for supervised
learning with auto-encoders for structured learning preserves the power of CNNs while also
accounting for dependencies, resulting in increased performance. In the remainder of the
paper, we ﬁrst brieﬂy discuss earlier approaches. We then present our structured prediction
approach in more detail and ﬁnally demonstrate that it outperforms state-of-the-art methods
on the Human3.6m dataset.
2

## method
In this work, we aim at directly regressing from an input image x to a 3D human pose. As
in [4, 13, 19], we represent the human pose in terms of the 3D locations y ∈R3J of J body
joints relative to a root joint. An alternative would have been to predict the joint angles and
limb lengths, however this is a less homogeneous representation and is therefore rarely used
for regression.
As discussed above, a straightforward approach to creating a regressor is to train a con-
ventional CNN such as the one used in [19]. However, this fails to encode dependencies

4
TEKIN ET AL.: STRUCTURED PREDICTION OF 3D HUMAN POSE
(a) Auto-encoder training
(b) Regression in latent space
(c) Fine-tuning
Figure 2: Our approach. (a) We train a stacked denoising auto-encoder that learns and
enforces implicit constraints about human body in its latent middle layer hL. (b) Our CNN
architecture maps the image to the latent representation hL learned by the auto-encoder. (c)
We stack the decoding layers of the auto-encoder on top of the CNN for reprojection from
the latent space to the original pose space and ﬁne-tune the entire network by updating the
parameters of all layers.
between joint locations. In [20], this limitation was overcome by introducing a substantially
more complex, deep architecture for maximum-margin structured learning. Here, we encode
dependencies in a simpler, more efﬁcient, and ultimately more accurate way by learning a
mapping between the output of a conventional CNN and a latent representation obtained us-
ing an overcomplete auto-encoder, as illustrated in Fig. 2. The auto-encoder is pre-trained
on human poses and comprises a hidden layer of higher dimension than its input and output.
In effect, this hidden layer and the CNN-based representation of the image play the same
role as the kernel embeddings in KDE-based approaches [5, 12, 14], thus allowing us to ac-
count for structure within a direct regression framework. Once the mapping between these
two high-dimensional embeddings is learned, we further ﬁne-tune the whole network for the
ﬁnal pose estimation task, as depicted at the bottom of Fig. 2.
In the remainder of this section, we describe the different stages of our approach.
3.1
Using Auto-Encoders to Learn Structured Latent Representations
We encode the dependencies between human joints by learning a mapping of 3D human
pose to a high-dimensional latent space. To this end, we use a denoising auto-encoder that
can have one or more hidden layers.
Following standard practice [35], given a training set of pose vectors {yi}, we add
isotropic Gaussian noise to create noisy versions {˜yi} of these vectors. We then train our
auto-encoder to take as input a noisy ˜yi and return a denoised yi as output. The correspond-
ing reconstruction function fae(·) must satisfy
ˆy = fae(y,θae) ,
(1)

TEKIN ET AL.: STRUCTURED PREDICTION OF 3D HUMAN POSE
5
where ˆy is the reconstruction and θae = (Wenc,j,benc,j,Wdec,j,bdec,j)L
j=1 contains the model
parameters, that is, the weights and biases for L encoding and decoding layers. We take
the middle layer to be our latent pose representation and denote it by hL. We use ReLU as
the activation function of the encoding layer. This favors a sparse hidden representation [8],
which has been shown to be effective at modeling a wide range of human poses [2, 23]. A
linear activation function is used at the decoding layer of the auto-encoder to reproject to both
negative and positive joint coordinates. To keep the number of parameters small and reduce
overﬁtting, we use tied weights for the encoder and the decoder, that is, Wdec,j = W T
enc,j.
To learn the parameters θae, we rely on the square loss between the reconstruction, ˆy,
and the original input, y, over the N training examples. To increase robustness to small
pose changes, we regularize the cost function by adding the squared Frobenius norm of
the Jacobian of the hidden mapping g(·), that is, J(y) = ∂g
∂y(y) where g(·) is the encoding
function that maps the input ˜y to the middle hidden layer, hL. Training can thus be expressed
as ﬁnding
θ ∗
ae = argmin
θae
N
∑
i
||yi −f(yi,θae)||2
2 +λ∥J(yi)∥2
F ,
(2)
where λ is the regularization weight. Unlike when using KDE, we do not need to solve a
complex pre-image problem to go from the latent pose representation to the pose itself. This
mapping, which corresponds to the decoding part of our auto-encoder, is learned directly
from data.
3.2
Regression in Latent Space
Once the auto-encoder is trained, we aim to learn a mapping between the input image and the
latent representation of the human pose. To this end, and as shown in Fig. 2(b), we make use
of a CNN to regress the image to a high-dimensional representation, which is itself mapped
to the latent pose representation.
More speciﬁcally, let θcnn be the parameters of the CNN, including the mapping to the
latent pose representation. Given an input image x, we consider the square loss function be-
tween the representation predicted by the CNN, fcnn(x,θcnn), and the one that was previously
learned by the auto-encoder, hL. Given our N training samples, learning amounts to ﬁnding
θ ∗
cnn = argmin
θcnn
N
∑
i
||fcnn(xi,θcnn)−hL,i||2
2 .
(3)
In practice, as shown in Fig. 2(b), we rely on a standard CNN architecture similar to the
one of [19, 32]. It comprises three convolutional layers—C1, C2 and C3—each followed by
a pooling layer—P1, P2, and P3. In our implementation, the input volume is a three channel
image of size 128 × 128. P3 is directly connected to a cascade of fully-connected layers—
FC1, FC2 and FC3—that produces a 4096-dimensional image representation, which is then
mapped linearly to the latent pose embedding. Except for this last linear layer, each layer
uses a ReLU activation function.
As in [19], prior to training our CNN, we ﬁrst initialize the convolutional layers using a
network trained for the detection of body joints in 2D. We then replace the fully-connected
layers of the detection network with those of the regressor to further train for the pose esti-
mation task.

6
TEKIN ET AL.: STRUCTURED PREDICTION OF 3D HUMAN POSE
3.3
Fine-Tuning the Whole Network
Finally, as shown in Fig. 2(c), we append the decoding layers of the auto-encoder to the
CNN discussed above, which reprojects the latent pose estimates to the original pose space.
We then ﬁne-tune the resulting complete network for the task of human pose estimation. We
take the cost function to be the squared difference between the predicted and ground-truth
3D poses, which yields the optimization problem
θ ∗
ft = argmin
θ ft
N
∑
i
||f ft(xi,θ ft)−yi||2
2 ,
(4)
where θft are the complete set of model parameters and f ft is the mapping function.
4

## experiments
In this section, we ﬁrst describe the large-scale dataset we tested our approach on. We then
give implementation details and describe the evaluation protocol. Finally, we compare our
results against those of the state-of-the-art methods.
4.1
Dataset
We evaluate our method on the Human3.6m dataset [14], which comprises 3.6 million image
frames with their corresponding 2D and 3D poses. The subjects perform complex motion
scenarios based on typical human activities such as discussion, eating, greeting and walking.
The videos were captured from 4 different camera viewpoints. Following the standard pro-
cedure of [19], we collect the input images by extracting a square region around the subject
using the bounding box present in the dataset and resize it to 128×128. The output pose is
a vector of 17 3D joint coordinates.
4.2
Implementation Details
We trained our auto-encoder using a greedy layer-wise training scheme followed by ﬁne-
tuning as in [9, 35]. We set the regularization weight of Eq. 2 to λ = 0.1. We experimented
with single-layer auto-encoders, as well as with 2-layer ones. The size of the layers were set
to 2000 and 300-300 for the 1-layer and 2-layer cases, respectively. We corrupted the input
pose with zero-mean Gaussian noise with standard deviation of 40 for 1-layer and 40-20 for
2-layer auto-encoders. In all cases, we used the ADAM optimization procedure [17] with a
learning rate of 0.001 and a batch size of 128.
The number and individual sizes of the layers of our CNNs are given in Fig. 2. The ﬁlter
sizes for the convolutional layers are consecutively 9×9, 5×5 and 5×5. Each convolutional
layer is followed by a 2×2 max-pooling layer. The activation function is the ReLU in all the
layers except for the last one that uses linear activation. As for the auto-encoders, we used
ADAM [17] with a learning rate of 0.001 and a batch size of 128. To prevent overﬁtting, we
applied dropout with a probability of 0.5 after each fully-connected layer and augment the
data by randomly cropping 112×112 patches from the 128×128 input images.
4.3
Evaluation Protocol
For a fair comparison, we used the same data partition protocol as in earlier work [19, 20] for
training and test splits. The data from 5 subjects (S1,S5,S6,S7,S8) was used for training and

TEKIN ET AL.: STRUCTURED PREDICTION OF 3D HUMAN POSE
7
Figure 3: 3D poses for the Walking, Eating, Taking Photo, Greeting, Discussion and Walking
Dog actions of the Human3.6m database. In each case, the ﬁrst skeleton depicts the ground-
truth pose and the second one the pose we recover. Best viewed in color.

## related_work
Following recent trends in Computer Vision, human pose estimation is now usually for-
mulated within a Deep Learning framework. The switch away from earlier representations

TEKIN ET AL.: STRUCTURED PREDICTION OF 3D HUMAN POSE
3
started with 2D pose estimation by learning a regressor from an input image to either directly
the pose vectors [32] or the heatmaps encoding 2D joint locations [15, 22, 31]. Recently, this
trend has extended to 3D pose estimation [19], where the problem is typically formulated in
terms of continuous 3D joint locations, since discretizing the 3D space is more challenging
than in the case of 2D.
Another important difference between 2D and 3D pose estimation comes from the ad-
ditional ambiguities in the latter one due to the fact that the input only shows a projection
of the output. To overcome these ambiguities, recent algorithms have attempted to encode
the dependencies between the different joints within Deep Learning approaches, thus ef-
fectively achieving structured prediction. In particular, [10] uses auto-encoders to learn a
shared representation for 2D silhouettes and 3D poses. This approach, however, relies on
accurate foreground masks and exploits handcrafted features, which mitigate the beneﬁts of
Deep Learning. In the context of hand pose estimation, [21] introduces a bottleneck, low-
dimensional layer that aims at accounting for joint dependencies. This layer, however, is
obtained directly via PCA, which limits the kind of dependencies it can model.
To the best of our knowledge, the work of [20] constitutes the most effective approach to
encoding dependencies within a Deep Learning framework for 3D human pose estimation.
This approach extends the structured SVM model to the Deep Learning setting by learning a
similarity score between feature embeddings of the input image and the 3D pose. This pro-
cess, however, comes at a high computational cost at test time, since, given an input image,
the algorithm needs to search for the highest-scoring pose. Furthermore, the ﬁnal results are
obtained by averaging over multiple high-scoring ground-truth training poses, which might
not generalize well to unseen data since the prediction can thus only be in the convex hull
of the ground-truth training poses. By contrast, we draw inspiration from the KDE-based
approaches [12, 14], that map both image and 3D pose to high-dimensional Hilbert spaces
and learn a mapping between these spaces. Here, however, we show how to do this in a Deep
Learning context with CNNs and auto-encoders. The beneﬁts are twofold: We can leverage
the power of learned features that have proven more effective than handcrafted ones such
as HOG [1], and our framework relies on a direct and efﬁcient regression between the two
spaces, thus avoiding the computational burden of the state-of-the-art approach of [20].
Using auto-encoders for unsupervised feature learning has proven effective in several
recognition tasks [16, 18, 35]. In particular, denoising auto-encoders [34] that aim at recon-
structing the perfect data from a corrupted version of it have demonstrated good generaliza-
tion ability. Similarly, contractive auto-encoders have been shown to produce intermediate
representations that are robust to small variations of the input data [24]. All these methods,
however, rely on auto-encoders to learn features for recognition tasks. By contrast, here, we
exploit auto-encoders to model the output structure for regression purposes.
3

## conclusion
Eating
Greeting
Taking Photo
Walking
Walking Dog
LinKDE( [14]
183.09
132.50
162.27
206.45
97.07
177.84
DconvMP-HML [19]
148.79
104.01
127.17
189.08
77.60
146.59
StructNet-Max [20]
149.09
109.93
136.90
179.92
83.64
147.24
StructNet-Avg [20]
134.13
97.37
122.33
166.15
68.51
132.51
OURS
129.06
91.43
121.68
162.17
65.75
130.53
Table 1: Average Euclidean distance in mm between the ground-truth 3D joint locations and
those predicted by competing methods [14, 19, 20] and ours.
the data from 2 different subjects (S9,S11) was used for testing. We evaluate the accuracy
of 3D human pose estimation in terms of average Euclidean distance between the predicted
and ground-truth 3D joint positions as in [19, 20]. The accuracy numbers are reported in
milimeters for all actions on which the authors of [19, 20] provided results. Training and
testing were carried out monocularly in all camera views for each separate action.
4.4