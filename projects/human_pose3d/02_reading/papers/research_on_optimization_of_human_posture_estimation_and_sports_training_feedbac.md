# Research on optimization of human posture estimation and sports training feedback system based on deep learning

> 2025 · id: W7117136731 · pdf: https://link.springer.com/content/pdf/10.1007/s11760-025-05019-1.pdf · 来源: pdf_url
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

Signal, Image and Video Processing (2025) 19:1454
https://doi.org/10.1007/s11760-025-05019-1
ORIGINALPAPER
Research on optimization of human posture estimation and sports
training feedback system based on deep learning
NianYun Tao1 · HaiXiang Jia2 · WenQian Li2
Received: 14 July 2025 / Revised: 10 November 2025 / Accepted: 27 November 2025 / Published online: 24 December 2025
© The Author(s) 2025
Abstract
As the requirements for accuracy and efﬁciency in sports training continue to increase, the application of traditional posture
estimation algorithms in high-dynamic and complex training environments faces problems such as image redundancy, noise
interference, and stability in long-term training. To solve these problems, this study proposes a sports training system based
on an optimized human posture estimation algorithm, aiming to improve the accuracy and stability of athletes in training. By
introducing optimization strategies such as deep image processing, enhanced feature extraction, focusing mechanism, and
feature fusion, the joint estimation accuracy and system stability are improved. Based on tennis training data, the experiment
analyzes the performance of different algorithms in joint angle similarity, long-term similarity, and system stability. The
results show that the optimization algorithm improves joint angle similarity by about 10%, long-term stability by 12%, and
system stability by 8%. These experimental results verify the advantages of the optimization algorithm in sports training,
especially in complex movements and multi-person training scenarios, and can provide athletes with high-precision and
real-time feedback.
Keywords Human pose estimation · Deep image processing · Optimization algorithm · Motion training · System stability
1 Introduction
With the rapid development of computer technology, elec-
tronic information technology and artiﬁcial intelligence,
traditional sports training methods have gradually exposed
theirlimitationsinaccuracy,real-timeandpersonalization.In
order to better improve the effect of sports training, more and
more new sports training systems have emerged. These sys-
tems combine cutting-edge technologies such as computer
vision, virtual reality and deep learning, and can provide
B WenQian Li
wqli8997@163.com
NianYun Tao
2072@mail.usts.edu.cn
HaiXiang Jia
withwind5000@sina.com
1
Department of Physical Education, Suzhou University of
Science and Technology, Sports training, Suzhou 215011,
China
2
Department of Physical Education, China University of
Political Science and Law, Sports training, Beijing 102249,
China
more accurate motion analysis and feedback [1–3], greatly
improving the training effect and sports skills of athletes.
Especially in the ﬁeld of sports training, human posture
estimation algorithm, as one of the core technologies, has
been widely used in sports training accuracy improvement
and motion optimization [4]. By using image or video anal-
ysis technology, human posture estimation algorithm can
extract the spatial position of human joints in real time,
thereby providing athletes with scientiﬁc and systematic
motion data support. With the development of deep learning
and computer vision technology, human posture estimation
technology has made signiﬁcant progress in image under-
standing, motion analysis, motion optimization and other
ﬁelds. It can not only accurately track each movement of
athletes, but also provide real-time feedback in sports train-
ing, helping athletes to ﬁnd deﬁciencies in their movements
in time and make improvements.
In recent years, the combination of virtual reality and aug-
mented reality has provided a new direction for the accuracy
and immersion of sports training systems. These systems
can create a highly realistic virtual training environment and
work together with human posture estimation technology to
achieve more intelligent and personalized sports training [5,
123

1454
Page 2 of 14
Signal, Image and Video Processing (2025) 19 :1454
6]. Human posture estimation can not only help athletes to
train skills, but also play a huge role in sports rehabilita-
tion and post-injury recovery. By accurately identifying the
changes in athletes’ postures, sports training systems can
adjust training plans in real time to avoid injuries caused
by incorrect movements and improve the training efﬁciency
and safety of athletes. Although human posture estimation
technology has made signiﬁcant progress in sports train-
ing, traditional algorithms still face many challenges [7, 8].
In practical applications, traditional human posture estima-
tion algorithms mostly rely on depth image data, which
often contains a lot of redundant information and noise,
which is difﬁcult to effectively remove, which has a direct
impact on the accuracy of estimation. Especially in com-
plex environments and dynamic training scenarios, how to
remove redundant information and improve feature extrac-
tion efﬁciency through optimization algorithms has become a
problem that needs to be solved by current posture estimation
algorithms.
In addition, the accuracy of traditional posture estimation
algorithms in multi-person posture estimation also needs to
be improved [9, 10]. In sports training, especially in scenarios
ofmulti-personparalleltrainingorteamtraining,thepostures
of multiple athletes are often intertwined, causing the error
of the algorithm in the estimation process to increase. Tradi-
tional algorithms are usually unable to effectively handle the
interaction between athletes, especially when dealing with
complex postures and large-scale movements, showing great
instability and inefﬁciency. This makes the accuracy and
real-time performance of posture estimation in multi-person
training scenarios still a difﬁcult problem to be overcome.
In addition, in highly dynamic training scenarios, the real-
time and robustness of traditional algorithms are also greatly
limited.Forexample,athleteshavefastmovementspeedsand
complex posture changes, and traditional algorithms often
cannot meet the needs of training in terms of real-time perfor-
mance. Therefore, how to improve the real-time performance
of the algorithm, improve the accuracy of posture estimation,
and maintain high efﬁciency in multi-person and complex
training scenarios is still the core challenge in the develop-
ment of human posture estimation technology.
This study aims at the practical application of human
posture estimation in sports training, proposes an optimized
human posture trajectory algorithm, combines deep image
theory with optimization algorithms, to solve the noise prob-
lem faced by traditional posture estimation algorithms, and
makes innovative contributions in many aspects.
First, we proposed a new data optimization method to
address the interference problem of redundant information
and noise when traditional algorithms process depth image
data [11–13]. By introducing the theory of depth image cor-
relation, this method can effectively ﬁlter out unnecessary
noise information and enhance the key information in the
image. Speciﬁcally, we optimize the extraction process of
joint position data in the image by constructing a new fea-
ture extraction network to ensure the accurate and reliable
position of each key point, thereby improving the accuracy
of pose estimation. In addition, this optimization method
effectively reduces redundant information during training
and improves computational efﬁciency, providing a reliable
foundation for subsequent dynamic motion analysis.
Second, in terms of multi-person pose estimation, we
designed a high-resolution multi-person pose high-precision
network model and introduced a focusing mechanism to deal
with the complex interaction problem of multiple athletes
in the same picture. Traditional multi-person pose estima-
tion algorithms often cannot accurately handle occlusion and
interweaving between people. The focusing mechanism we
proposed can more accurately track the pose of each ath-
lete and effectively separate the interaction between people
by ﬁnely modeling the spatial relationship between differ-
ent athletes. The network model adopts a multi-scale feature
fusion strategy [14, 15], which can capture the changes of the
athlete’s joints at different scales and ensure efﬁcient estima-
tion under complex motion postures.
In addition, we also proposed an optimized dynamic
feedback mechanism that can provide personalized train-
ing suggestions based on the real-time performance of the
athlete during training. This mechanism relies on an opti-
mized posture trajectory algorithm to continuously track the
athlete’s motion trajectory and posture changes, detect the
accuracy of the movement in real time, and give feedback.
Unlike the traditional training model that simply relies on
post-feedback, our system can dynamically adjust the move-
ment plan during training and make real-time adjustments
based on the athlete’s actual performance, thereby helping
athletes avoid incorrect movement patterns and improving
the efﬁciency and accuracy of sports training. In terms of
algorithm implementation, we used an efﬁcient deep learning
framework and a large-scale dataset to train a high-precision
and efﬁcient posture estimation network. Compared with
traditional methods, the proposed system shows signiﬁcant
advantages in accuracy, efﬁciency, and robustness, especially
in the scenarios of multi-human posture estimation and com-
plex movements, showing better adaptability and stability.
In order to verify the accuracy and effect of the proposed
system in sports training, this paper designs an experimen-
tal case based on tennis training to verify the application
effect and performance of the proposed system in the actual
sports training environment. As a typical high-dynamic and
high-precision sport, tennis training can well test the effect
of human posture estimation algorithm in dynamic and com-
plex training scenarios. The experimental scenario setting
includes the simultaneous training of multiple athletes, espe-
cially the switching between high-intensity fast movement
and complex movements, to test the accuracy and real-time
123

Signal, Image and Video Processing (2025) 19 :1454
Page 3 of 14
1454
performance of the system for multiple athletes’ posture
estimation. We selected multiple tennis training data sets,
including different training scenarios and training intensi-
ties, to ensure the comprehensiveness of the veriﬁcation
experiment and the adaptability of the system. In the exper-
iment, we ﬁrst calibrated the athlete’s posture data, and
veriﬁed the improvement of the proposed system in the
accuracy of single-body posture estimation by comparing
it with the traditional posture estimation algorithm. In this
part of the experiment, we focused on testing the perfor-
mance of the algorithm in the athlete’s fast movements and
complex movements, such as serving, receiving and running.
The experimental results show that the proposed system can
accurately track the changes in the athlete’s joints in these
high-dynamic and high-precision movements, and the esti-
mation accuracy is signiﬁcantly higher than that of traditional
methods.
In summary, the results of the experimental design fully
demonstrate the accuracy, efﬁciency and robustness of the
proposed system in sports training, indicating that the appli-
cation of the system in athlete posture estimation and training
feedback has signiﬁcant potential, can provide athletes with
scientiﬁc and accurate training support, and promote sports
training to develop in a more intelligent and personalized
direction.
2 Related Works
2.1 Traditional methods
The capture and analysis of human motion in sports training
systems usually rely on a variety of methods such as physi-
cal observation, image processing, image segmentation and
detection technology, and sensor algorithms. These meth-
ods can be divided into multiple technical routes, including
image-based processing methods and sensor-based analysis
methods. Among these traditional methods, image segmen-
tation and detection technology is widely used to identify and
track athletes’ movements, while sensor algorithms are used
to accurately capture athletes’ dynamic data.
Chen et al. [16] proposed a method based on image seg-
mentation and sensor fusion for the recognition and tracking
of human motion. The authors successfully improved the
accuracy of athlete posture recognition by combining sensor
data with image processing, especially in dynamic training
environments. The image segmentation technology used in
the study can separate human features from complex back-
grounds, and the sensor data provides more accurate motion
trajectory information. Batchuluun et al. [17] proposed a new
method to analyze and identify human motion using con-
volutional neural networks (CNNs). Through deep learning
models, researchers can effectively capture the details of ath-
letes’ movements, thereby achieving high-precision motion
recognition. This method demonstrates the great potential
of deep learning in sports training, especially when dealing
with complex sports scenes. The algorithm can better sup-
press noise and enhance the ability to extract athlete’s motion
features. Nweke et al. [18] used multi-sensor fusion technol-
ogy to combine image and sensor data for human posture
estimation. By fusing visual information with sensor data
such as accelerometers and gyroscopes, the system can more
accurately capture the athlete’s posture changes and improve
the real-time and accuracy of the estimation. This method
has a good application prospect for complex motion analysis
in sports training.
2.2 Research Progress of Human Pose Estimation
Algorithms
2.2.1 Single Person Pose Estimation Algorithm
Single-person pose estimation algorithms mainly focus on
the action recognition and behavior analysis of a single
athlete. The methods are diverse, ranging from random deci-
sion trees to sparse regression and image structure models.
Through these methods, researchers can accurately capture
the spatial position of each joint of the human body and thus
identify speciﬁc actions.
Lan et al. [21] reviewed the research progress of single-
person pose estimation and detailed the transition from
traditional algorithms to deep learning methods. The authors
focused on methods based on random decision trees and
sparseregression,whichperformedwellinearlyposeestima-
tion tasks and were relatively simple and easy to implement.
However, with the rise of deep learning, these traditional
methods have gradually been replaced by more efﬁcient
deep neural networks. The paper systematically summa-
rizes the advantages and disadvantages of these technologies
and looks forward to future development directions. Deep-
Pose [22] proposed the DeepPose algorithm, a human pose
estimation method based on deep neural networks. The algo-
rithm successfully transformed the pose estimation task into
a regression problem through an end-to-end training model,
signiﬁcantly improving the accuracy of pose estimation.
The introduction of DeepPose marks a new era for pose
estimation technology, especially in complex backgrounds.
Yang et al.[23] proposed a posture estimation method based
on sparse regression. By introducing a ﬂexible component
hybrid model, the model can be adaptively adjusted accord-
ing to the movement characteristics of various parts of the
human body. This method can accurately capture the changes
of the human body under different actions, and is particularly
suitable for the estimation of complex actions and different
postures. Andriluka et al.[24] proposed a new 2D human
posture estimation benchmark and designed a new estima-
123

1454
Page 4 of 14
Signal, Image and Video Processing (2025) 19 :1454
tion method in combination with the image structure model.
This method can efﬁciently estimate the 2D posture of a sin-
gle human body in a changing environment, and veriﬁed the
performance of the model on different data sets through new
benchmark tests. Luvizon et al.[25] performed human pos-
ture estimation based on deep learning methods and applied
it to behavior recognition tasks. By training deep neural
networks, the system can achieve high-precision posture esti-
mation in complex action scenes, providing strong support
for action recognition.
2.2.2 Multi-Human Pose Estimation Algorithm
MobiPose[26] proposed a multi-person posture estimation
method based on convolutional neural networks, using a
cascade structure to solve the problem of multi-person inter-
action. Experiments show that this method performs well in
handling multi-person motion scenes and can achieve accu-
rate posture estimation in real-time environments. Newell
et al.[27] proposed a stacked hourglass network structure,
which can gradually capture the spatial position of each joint
of the human body in a high-resolution space by stacking
multiple layers of networks. This method performs particu-
larly well in multi-person scenes and can effectively handle
complex postures and occlusion problems. Wei et al.[28]
proposed the Convolutional Pose Machine (CPM) method,
which gradually extracts human posture features through a
convolutional network and is successfully applied to multi-
personpostureestimationtasks.Thismethodcanimprovethe
accuracy of posture estimation through iterative optimiza-
tion, especially in complex backgrounds and multi-person
interactions. Dai et al.[29] proposed a hybrid deep learning
framework that combines multiple neural network models
to achieve efﬁcient processing of multi-person posture esti-
mation. Experimental results show that the proposed method
performs well in processing complex multi-person interac-
tion scenarios. Cao et al.[30] proposed a multi-person posture
estimation method based on part afﬁnity ﬁeld (PAF), which
can maintain high estimation accuracy in a multi-person
interaction training environment and has good real-time per-
formance.
2.2.3 3D Human Pose Estimation
With the widespread application of 2D human pose esti-
mation, 3D human pose estimation has gradually become
a research focus. 3D human pose estimation provides a more
comprehensive 3D human pose analysis by extending the
traditional 2D pose estimation method, and can accurately
estimate the 3D position and motion trajectory of the human
body. This method can not only describe the spatial pose of
the human body, but also reveal the spatial dynamic changes
behind the action. It has important application value in vir-
tual reality, motion analysis, robot control and other ﬁelds.
The key challenges of 3D pose estimation include the lack of
depth information, pose inference under different perspec-
tives, and the impact of complex background on estimation
accuracy.
Vnect[31]proposedareal-time3Dhumanposeestimation
method based on monocular video, which solves the prob-
lem of extracting 3D pose from monocular video without
a depth sensor. The method uses a deep learning regres-
sion model to convert 2D pose data into 3D information and
can run efﬁciently in real-time video streams. A new regres-
sion method is used in the study to gradually restore the
3D joint position through a deep convolutional neural net-
work, thereby achieving excellent performance on standard
datasets. This method is particularly suitable for application
scenarios where 3D human pose estimation relies on ordinary
video data under resource constraints. Nie et al.[32] proposed
a new 3D pose estimation method based on a monocular cam-
era, which achieves 3D pose reconstruction by predicting
depth information on a 2D heat map. Different from tradi-
tional 3D pose estimation methods, this method infers joint
positions in 3D space by enhancing the depth information of
the 2D heat map, avoiding the use of expensive depth sen-
sors. Experimental results show that this method can provide
stable and accurate 3D pose estimation in complex training
environments, especially in pose estimation under occlusion
and different viewpoints, showing strong robustness. Zhao
et al.[33] proposed a 3D human pose estimation method
based on regression network. This method gradually maps
the 2D joint positions extracted from the image to 3D space
coordinates by constructing a multi-layer regression net-
work. During the learning process, the regression network is
trained with a large amount of labeled data and can accurately
estimate the 3D human pose. This method is superior to tra-
ditional regression methods in accuracy and can handle more
complex movements. The study also shows its potential for
application in motion analysis and virtual reality. Martinez et
al.[34] proposed a 3D human pose estimation method based
on depth information. By combining depth images with tradi-
tional 2D pose estimation technology, this method can handle
3D human pose estimation problems in real time in practical
applications. This method improves real-time performance
while maintaining high estimation accuracy, and can run sta-
bly in complex motion scenes. By comparing with existing
methods,thedepthinformationcombinedwith2Destimation
method proposed in this paper shows excellent performance
on multiple standard datasets, especially in dynamic environ-
ments. Yang et al.[35] proposed a 3D human pose estimation
method based on adversarial learning. This method uses the
generative adversarial network (GANs) framework to pre-
dict 3D human poses and optimizes the network training
process through adversarial learning. By introducing adver-
sarial learning, the network can better capture human pose
123

Signal, Image and Video Processing (2025) 19 :1454
Page 5 of 14
1454
Fig. 1 Overall model structure
informationincomplexenvironmentsandavoidthedeviation
of traditional methods in noisy environments. Experimental
results show that the proposed method performs better than
existing technologies in "wild" environments and can handle
pose estimation tasks in various backgrounds and scenarios.
3 Method
Our model structure is shown in Figure 1.
3.1 Deep Image Processing and Enhancement
Optimization module
Thecoregoalofthismoduleistosolvetheproblemsofredun-
dant information, noise interference and poor image quality
in traditional human posture estimation algorithms by opti-
mizing the depth image data processing process, and improve
the robustness and accuracy of the system. During motion
training, depth images are often affected by factors such as
environmental noise, image blur, and incomplete depth infor-
mation, which directly affects subsequent posture estimation
and motion analysis. Therefore, the preprocessing and opti-
mization of depth images become key steps to improve the
accuracy of posture estimation.
First, in order to remove noise from depth images, we
use a combination of median ﬁltering and Gaussian ﬁltering
to remove noise. As a nonlinear ﬁltering method, median
ﬁltering can effectively remove salt and pepper noise and
maintain edge details by replacing the median of each pixel
in the image. Speciﬁcally, the mathematical representation
of median ﬁltering is:
Imed(x, y) = med

I(x′, y′)

,
(x′, y′) ∈N(x, y)
(1)
Among them, I(x, y) represents the original image, N(x, y)
is the neighborhood centered on the pixel (x, y), and med(·)
representsthemedianofthepixelvaluesintheneighborhood.
Through this method, the noise in the image is effectively
removed while retaining important edge information.
Furthermore, in order to enhance the image contrast and
improve the detail performance, Gaussian ﬁltering is used.
Gaussian ﬁltering can smooth the noise in the image and
enhance the low-frequency information of the image by per-
forming weighted averaging on the image. The mathematical
formula of Gaussian ﬁltering is:
G(x, y) =
∞

i=−∞
∞

j=−∞
I(x + i, y + j) · e−(i2+ j2)/2σ 2
(2)
Among them, G(x, y) represents the image after Gaus-
sian ﬁltering, σ is the standard deviation of the Gaussian
kernel(σ=0.01), which controls the smoothness of the ﬁlter.
This method effectively reduces the high-frequency noise in
the image, making the details in the image more prominent.
Image contrast enhancement is another important part
of improving the accuracy of pose estimation. To this end,
we combine histogram equalization and contrast-limited
adaptive histogram equalization (CLAHE) technology. His-
togramequalizationenhancesimagecontrastbyadjustingthe
grayscale distribution of the image. Its mathematical expres-
sion is:
Hr(x) = 1
n
x

i=0
p(i)
(3)
Among them, Hr(x) is the cumulative distribution function,
and p(i) is the probability density of the pixel of gray level i
in the image. The histogram equalization method can effec-
tively improve the global contrast of the image, especially
for the enhancement of low-contrast images. For the contrast
123

1454
Page 6 of 14
Signal, Image and Video Processing (2025) 19 :1454
enhancement of local areas, the CLAHE method can avoid
the noise ampliﬁcation caused by over-enhancement, and its
formula is:
CL AH E(x, y) = H(x, y) −μ
σ + ϵ
· C
(4)
Among them, H(x, y) is the histogram of the local image, μ
and σ are the mean and standard deviation of the local area
respectively, C is the enhancement factor, and ϵ is a small
constant to prevent division by zero. In this way, the local
contrast of the image is effectively improved, especially in
the detail area.
In the process of processing deep image sequences, the
continuity and dynamic changes between images have an
important impact on the accuracy of pose estimation. In
order to improve the smoothness of the image sequence
and reduce the estimation error caused by drastic changes
between images, we use the weighted sliding average method
to smooth the image sequence. The weighted sliding average
formula is:
St(x, y) = α · It(x, y) + (1 −α) · St−1(x, y)
(5)
Among them, St(x, y) is the smoothed image of the tth frame,
It(x, y) is the original depth image of the tth frame, and α is
the smoothing factor(α=0.001), which controls the weighted
ratio between the current frame and the previous frame. This
method can eliminate the jumping or mutation phenomenon
in the image sequence and improve the stability of the image.
Feature extraction of depth images is the basis for subse-
quent posture estimation. In this process, each joint position
of the human body is usually manifested as a high-contrast
area in the image. Therefore, by extracting features from the
depth image, the spatial information of the joint can be accu-
rately obtained. Assuming that each pixel in the depth image
D(x, y) represents the depth value of a certain position in
space, we extract the depth feature of each joint from the
image:
fD = [D(x1, y1), D(x2, y2), . . . , D(xn, yn)]
(6)
Among them, fD is a feature vector containing the depth val-
ues of all joints. These depth features will be used as input for
subsequent posture estimation to help the system accurately
infer the actual position of each joint in three-dimensional
space.
In posture estimation, gradient features play an impor-
tant role in edge detection and precise positioning of joint
positions. By calculating the gradient of the image, the edge
information in the image can be extracted, further enhancing
the positioning accuracy of the joints. The gradient feature
can be obtained by the following formula:
fG = [∇x D(x, y), ∇y D(x, y)]
(7)
Among them, ∇x and ∇y represent the gradient operations
on the horizontal and vertical directions of the depth image,
respectively. Through this operation, the system can capture
subtle changes in the image, especially the edge areas of the
joints.
The depth image processing and enhancement optimiza-
tion module can effectively improve image quality, enhance
image features and eliminate noise interference through these
technical means. The preprocessed depth image provides a
stable and reliable foundation for subsequent feature extrac-
tion and posture estimation, ensuring efﬁcient operation in
complex dynamic environments.
3.2 Feature Extraction and Optimization module
3.2.1 Deep feature extraction and optimization
In a depth image, the joint position is usually represented by
the depth change in the image, so the depth feature is one of
the most basic features in human posture estimation. In order
to extract the spatial features of the joints from the depth
image, we ﬁrst use the convolutional neural network (CNN)
to extract the local features in the depth image, and gradu-
ally build a high-dimensional feature representation through
multiple convolution layers. Assume that the depth image
D(x, y) is the input image, where each pixel (x, y) corre-
sponds to a depth value of a spatial position. Through the
convolution operation, we get the output of each layer of
convolution features:
f(l)
D = Conv(l)(D(x, y)),
l ∈[1, L]
(8)
Among them, Conv(l) represents the l-th convolution opera-
tion, L is the number of layers of the convolutional network,
and f(l)
D
is the depth feature extracted by the l-th layer.
Through the combination of multiple layers of convolution,
the model can gradually extract detailed features in the image
and provide support for subsequent joint position estimation.
In order to further optimize the deep feature extraction
process, we introduced the deep comparison feature, which
can enhance the spatial resolution of the image by comparing
the depth differences between different regions. The calcu-
lation formula of the deep comparison feature is as follows:
fDdiff = |D(x1, y1) −D(x2, y2)|
(9)
Among them, (x1, y1) and (x2, y2) are pixels at two adjacent
positions in the image, and fDdiff is the depth difference fea-
ture. By extracting the depth difference feature, the system
123

Signal, Image and Video Processing (2025) 19 :1454
Page 7 of 14
1454
can more keenly capture the spatial changes between joints
and improve the estimation accuracy.
3.2.2 Gradient Feature Extraction and Optimization
Based on the depth feature, the gradient feature can effec-
tively capture the edge information in the image and help
the system accurately locate the human joints. The gradient
feature is usually extracted by calculating the rate of change
of the image in the horizontal and vertical directions. In the
depth image, the human joints usually show strong gradient
changes, so the gradient feature can effectively assist in the
positioning of the joints.
Assume that the gradient calculation formula in the depth
image D(x, y) is as follows:
gG =
∂D(x, y)
∂x
, ∂D(x, y)
∂y

(10)
In this case, ∂D(x,y)
∂x
and ∂D(x,y)
∂y
represent the gradient oper-
ations on the depth image in the horizontal and vertical
directions, respectively, and gG is the gradient feature vector.
This modiﬁcation preserves the meaning while updating the
symbols as requested.
In order to further optimize the gradient feature extraction
process, we introduced a binary phase ﬁlter (BPF). BPF can
more accurately capture the detailed changes in the image,
especially in the joints, by extracting the phase information
of the image. Through the calculation of BPF, we can obtain
more detailed edge information and further improve the accu-
racy of posture estimation. The calculation formula of BPF
is:
fBPF(x, y) = Re
⎛
⎝
∞

i=−∞
∞

j=−∞
ei·θi j · D(x + i, y + j)
⎞
⎠
(11)
Among them, θi j is the phase angle of the image region
(x + i, y + j), and fBPF(x, y) is the feature extracted by the
phase ﬁlter. By enhancing the phase information, BPF can
provide clearer boundary features for accurate estimation of
joint positions.
3.2.3 Contrast Feature Extraction and Optimization
Contrast features are of great signiﬁcance for improving
image quality and enhancing the extraction of joint informa-
tion. Improving image contrast can highlight joint positions
and make posture estimation more accurate. In depth images,
joints are usually shown as high-contrast areas in the image,
so extracting contrast features is a key step in the posture
estimation process.
To improve image contrast, we introduced a local contrast
enhancement method to enhance the performance of details
by adjusting the brightness and contrast of local areas of the
image. The mathematical formula for local contrast enhance-
ment is:
fLC(x, y) = D(x, y) −μlocal
σlocal + ϵ
(12)
Among them, μlocal and σlocal represent the mean and stan-
dard deviation of the local area of the image, respectively,
and ϵ is a constant to prevent division by zero. Through this
method, we can signiﬁcantly improve the recognizability of
joint areas in the image, especially in low-contrast areas,
where joint features are more prominent.
In the process of optimizing contrast features, we also
combine gradient weighted contrast features to further
enhance the effect of image contrast through gradient infor-
mation. The formula of this method is as follows:
fGWC(x, y) = fG(x, y) · fLC(x, y)
(13)
By combining gradient and contrast features, the system can
more effectively locate joint positions in the image, further
improving the accuracy of pose estimation.
3.2.4 Feature Fusion and Optimization
In the process of extracting depth features, gradient features,
and contrast features, various features can describe the spa-
tial information of the image from different dimensions. In
order to maximize the relevance of information and reduce
redundancy, we integrate multiple features through a feature
fusion algorithm to improve the accuracy and computational
efﬁciency of pose estimation. The goal of feature fusion is to
reduce the error that may be caused by processing each fea-
ture separately by weighted combination of the advantages
of different features.
Assuming that the extracted depth features, gradient fea-
tures, and contrast features are fD, fG, and fC, respectively,
the feature fusion process is performed by the following for-
mula:
ffusion = λ1 · fD + λ2 · fG + λ3 · fC
(14)
Among them, λ1(0.3), λ2(0.3), λ3(0.4) are the weight coef-
ﬁcients of the features, which are obtained through learning
optimization. The feature fusion algorithm can dynamically
adjust the weights according to the importance of each
feature, thereby ensuring the accuracy of posture estima-
tion in complex environments. Through this feature fusion
strategy, the system can comprehensively consider various
feature information and improve the stability and robust-
ness of posture estimation. Especially in multi-person scenes
123

1454
Page 8 of 14
Signal, Image and Video Processing (2025) 19 :1454
and complex action environments, the fusion algorithm can
signiﬁcantly reduce redundant information and enhance the
accuracy of joint positions.
3.3 Focusing Mechanism and Feature Response
Optimization module
In order to further improve the accuracy and robustness of
human posture estimation, this module introduces a focus-
ing mechanism to optimize the response of each feature in
the depth image sequence. Traditional posture estimation
algorithms usually rely on simple global feature processing,
which often ignores the importance of features in differ-
ent regions of the image, resulting in poor performance
in complex motion environments. The focusing mechanism
simulates the visual observation mechanism of the human
body and can assign different weights to each input feature,
thereby emphasizing the key information in the image and
optimizing the posture estimation process.
3.3.1 Focusing Mechanism Principle
The core idea of the focusing mechanism is to assign a weight
value to each feature in the image, focusing on the areas that
have a greater impact on pose estimation. Suppose we extract
features from the input depth image sequence It(x, y), where
t represents the time frame, (x, y) is the pixel position in the
image, and the feature response weight αt(x, y) at a certain
position in the image can be calculated by the following for-
mula:
αt(x, y) =
e−(x−x0)2+(y−y0)2
2σ2

i, j e−(i−x0)2+( j−y0)2
2σ2
(15)
Among them, (x0, y0) is the initial position of the key points
of the human body, and σ is the standard deviation of the
Gaussian function, which controls the range of focus. In
this way, we can assign weights according to the positions
of joints and limbs, ensuring that the system can prioritize
important areas when processing depth images and sup-
press the inﬂuence of background and irrelevant areas. This
weighting mechanism achieves the focusing effect by assign-
ing different weights to each pixel position in each frame of
the image. Areas in the image that are far away from the joint
position will be assigned lower weights, thereby reducing the
interference of these areas on the posture estimation process.
Areas close to the joints will be given higher weights, ensur-
ing that the system can focus on the key parts of the athlete
and provide more accurate estimation results.
3.3.2 Mathematical Modeling of Focusing Mechanism
The introduction of the focusing mechanism makes the opti-
mization process of the feature response more reﬁned. We
use the weighted feature response formula to optimize the
accuracy of the pose estimation by adjusting the weights of
each position in the image sequence. For each frame of the
depth image It(x, y), the product of the feature response
ft(x, y) of each pixel in the image and the weight αt(x, y)
is the optimized feature response:
ˆft(x, y) = αt(x, y) · ft(x, y)
(16)
In this way, the feature response of key areas in the image is
enhanced, while the inﬂuence of background noise is effec-
tively suppressed. This weighted feature response can help
the system extract joint positions more accurately and reduce
the impact of complex background, multi-person interaction
or motion blur on estimation accuracy.
In order to further improve the system’s responsiveness
to features, we embed the optimization process of weighted
feature response into the neural network model. Through
end-to-end training, the network can automatically learn the
optimal weight distribution based on the input depth image,
thereby continuously improving the accuracy of posture esti-
mation. Assuming that the feature response of each frame
in the depth image sequence is ft, the ﬁnal focused feature
response Ffocus can be expressed as:
Ffocus =
T

t=1
ˆft
(17)
Among them, T represents the total number of frames in
the image sequence, and
ˆft is the feature response after
optimization by the focusing mechanism. Through this
accumulation method, the system can integrate the feature
response information at multiple time points to provide more
comprehensive and accurate data support for the ﬁnal posture
estimation.
3.3.3 Combination of Focusing Mechanism and Pose
Estimation
In human pose estimation, the focusing mechanism can
signiﬁcantly improve the performance of the algorithm in
multi-person training and complex dynamic scenes. Tra-
ditional pose estimation algorithms are usually unable to
effectively handle pose estimation problems in multi-person
interactions, especially when there is occlusion or interweav-
ing of poses between athletes, traditional methods often make
estimation errors. The focusing mechanism can effectively
distinguish the poses of different athletes by strengthening
123

Signal, Image and Video Processing (2025) 19 :1454
Page 9 of 14
1454
theresponseofkeyareafeatures,therebyimprovingtheaccu-
racy of multi-person pose estimation.
Suppose in a complex training scenario, multiple athletes
are training at the same time, and the joint position of each
athlete is affected by other athletes. In this case, the system
assigns different weights through the focusing mechanism,
which can prioritize the pose changes of key athletes and
ignore the interference of other athletes. For each athlete’s
pose estimation, the weighted process of the focusing mecha-
nism will cause the algorithm to give a higher response to the
joint position of the athlete, thereby ensuring that the system
can accurately estimate the athlete’s pose.
Speciﬁcally, assuming that in a multi-person pose estima-
tion scenario, the system processes the poses of N athletes
si