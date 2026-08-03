# 3D Human pose estimation: A review of the literature and analysis of covariates

> 2016 · id: W2515603221 · 来源: web-agent
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

Computer Vision and Image Understanding 152 (2016) 1–20 
Contents lists available at ScienceDirect 
Computer Vision and Image Understanding 
journal homepage: www.elsevier.com/locate/cviu 
3D Human pose estimation: A review of the literature and analysis of 
covariates 
Nikolaos Saraﬁanos a , Bogdan Boteanu b , Bogdan Ionescu b , Ioannis A. Kakadiaris a , ∗
a Computational Biomedicine Lab, Department of Computer Science, University of Houston, 4800 Calhoun Rd. Houston, TX 77004, United States 
b Image Processing and Analysis Lab, University Politehnica of Bucharest, 61071 Romania 
a r t i c l e 
i n f o 
Article history: 
Received 8 December 2015 
Revised 2 September 2016 
Accepted 3 September 2016 
Available online 8 September 2016 
Keywords: 
3D Human pose estimation 
Articulated tracking 
Anthropometry 
Human motion analysis 
a b s t r a c t 
Estimating the pose of a human in 3D given an image or a video has recently received signiﬁcant atten- 
tion from the scientiﬁc community. The main reasons for this trend are the ever increasing new range 
of applications (e.g., human-robot interaction, gaming, sports performance analysis) which are driven by 
current technological advances. Although recent approaches have dealt with several challenges and have 
reported remarkable results, 3D pose estimation remains a largely unsolved problem because real-life 
applications impose several challenges which are not fully addressed by existing methods. For exam- 
ple, estimating the 3D pose of multiple people in an outdoor environment remains a largely unsolved 
problem. In this paper, we review the recent advances in 3D human pose estimation from RGB images 
or image sequences. We propose a taxonomy of the approaches based on the input (e.g., single image 
or video, monocular or multi-view) and in each case we categorize the methods according to their key 
characteristics. To provide an overview of the current capabilities, we conducted an extensive experi- 
mental evaluation of state-of-the-art approaches in a synthetic dataset created speciﬁcally for this task, 
which along with its ground truth is made publicly available for research purposes. Finally, we provide 
an in-depth discussion of the insights obtained from reviewing the literature and the results of our ex- 
periments. Future directions and challenges are identiﬁed. 
© 2016 Elsevier Inc. All rights reserved. 
1. Introduction 
Articulated pose and motion estimation is the task that em- 
ploys computer vision techniques to estimate the conﬁguration of 
the human body in a given image or a sequence of images. This 
is an important task in computer vision, being used in a broad 
range of scientiﬁc and consumer domains, a sample of which are: 
(i) Human-Computer Interaction (HCI): Human motion can pro- 
vide natural computer interfaces whereby computers can be con- 
trolled by human gestures or can recognize sign languages ( Erol 
et al., 2007; Song et al., 2012 ); (ii) Human-Robot Interaction: To- 
day’s robots must operate closely with humans. In household en- 
vironments, and especially in assisted living situations, a domes- 
tic service robot should be able to perceive the human body pose 
to interact more effectively ( Droeschel and Behnke, 2011; McColl 
et al., 2011 ); (iii) Video Surveillance: In video-based smart surveil- 
lance systems, human motion can convey the action of a human 
∗Corresponding author. 
E-mail addresses: nsaraﬁanos@uh.edu (N. Saraﬁanos), ikakadia@central.uh.edu 
(I.A. Kakadiaris). 
subject in a scene. Since manual monitoring of all the data ac- 
quired is impossible, a system can assist security personnel to fo- 
cus their attention on the events of interest ( Chen et al., 2011a; 
Sedai et al., 2009 ); (iv) Gaming: The release of the Microsoft Kinect 
sensor ( Shotton et al., 2013a; 2013b ) along with toolkit extensions 
that facilitate the integration of full-body control with games and 
Virtual Reality applications ( Suma et al., 2011 ) are the most illus- 
trative examples of how human motion capture can be used in the 
gaming industry; (v) Sport Performance Analysis: In most sports, 
the movements of the athletes are studied in great depth from 
multiple views and, as a result, accurate pose estimation systems 
can help in analyzing these actions ( Fastovets et al., 2013; John- 
son and Everingham, 2010; Unzueta et al., 2014 ); (vi) Scene Under- 
standing: Estimating the 3D human pose can be used in a human- 
centric scene understanding setup to help in the prediction of the 
“workspace” of a human in an indoor scene ( Gupta et al., 2011; 
Zheng et al., 2015 ); (vii) Proxemics Recognition: Proxemics recog- 
nition refers to the task of understanding how people interact. It 
can be combined with robust pose estimation techniques to di- 
rectly decide whether and to what extent there is an interaction 
between people in an image ( Yang et al., 2012 ) and at the same 
time improves the pose estimation accuracy since it addresses oc- 
http://dx.doi.org/10.1016/j.cviu.2016.09.002 
1077-3142/© 2016 Elsevier Inc. All rights reserved. 

2 
N. Saraﬁanos et al. / Computer Vision and Image Understanding 152 (2016) 1–20 
Fig. 1. A summary of real-life applications of human motion analysis and pose estimation (images from left to right and top to bottom): Human-Computer Interaction, Video 
Surveillance, Gaming, Physiotherapy, Movies, Dancing, Proxemics, Sports, Human-Robot Interaction. Flickr image credits: The Conmunity – Pop Culture Geek, Intel Free Press, 
Patrick Oscar Boykin, Rae Allen, Christopher Prentiss Michel, Yuiseki Aoba, DIUS Corporate, Dalbra J.P., and Grook Da Oger. 
clusions between body parts; (viii) Estimating the anthropometry 
of a human from a single image ( Barron and Kakadiaris, 20 0 0; 
2003; Kakadiaris et al., 2016 ); (ix) 3D Avatar creation ( Barmpoutis, 
2013; Zhang et al., 2013 ) or controlling a 3D Avatar in games 
( Pugliese et al., 2015 ); (x) Understanding the camera wearer’s ac- 
tivity in an egocentric vision scenario ( Jiang and Grauman, 2016 ); 
and (xi) Describing clothes in images ( Chen et al., 2012; Yamaguchi 
et al., 2012 ) which can then be used to improve the pose identiﬁ- 
cation accuracy. 
In Fig. 1 some of the aforementioned applications are depicted, 
which along with recent technological advances, and the release of 
new datasets have resulted in an increasing attention of the scien- 
tiﬁc community on the ﬁeld. However, human pose estimation still 
remains an open problem with several challenges, especially in the 
3D space. 
Fig. 2 shows the number of publications with the keywords: (i) 
“3D human pose estimation”, (ii) “3D motion tracking”, (iii) “3D 
pose recovery”, and (iv) “3D pose tracking” in their title after du- 
plicate and not relevant results are discarded. Note that, there are 
other keywords that return relevant publications such as “3D hu- 
man pose recovery” ( Chen et al., 2011a ) or “3D human motion 
tracking” ( Kakadiaris and Metaxas, 20 0 0 ). Thus, Fig. 2 does not 
cover all the methods we discuss but, even restricted to this par- 
Fig. 2. Depiction of the number of papers published during the last decade that 
include the keywords “3D human pose estimation”, “3D motion tracking”, “3D pose 
recovery”, and “3D pose tracking” in their title after duplicate and irrelevant results 
are discarded. 
ticular search, still shows the increase of interest by the scientiﬁc 
community. 1 
To cover the recent advances in the ﬁeld and at the same time 
to be effective in our approach, we narrowed this survey to a class 
1 Results of the search on May 1 st , 2016. We excluded searches related to patents 
or articles which other scholarly articles have referred to, but which cannot be 
found online. 

N. Saraﬁanos et al. / Computer Vision and Image Understanding 152 (2016) 1–20 
3 
of techniques which are currently the most popular, namely the 3D 
human body pose estimation from RGB images. Apart from using 
RGB data, another major class of methods, which have received a 
lot of attention lately, are the ones using depth information such 
as RGB-D. Although an increasing number of papers has been pub- 
lished on this topic during the last few years with remarkable re- 
sults ( Shotton et al., 2013b; Pons-Moll et al., 2013, 2015 ), 3D Pose 
Estimation from RGB-D images will not be covered in this work 
because Helten et al. (2013) and Ye et al. (2013) published surveys 
on this topic recently which cover in detail the recent advances 
and trends in the ﬁeld. 
1.1. Previous surveys and other resources 
The reader is encouraged to refer to the early works of 
Aggarwal and Cai (1997) and Gavrila (1999) to obtain an overview 
of the initial methods in the ﬁeld. The most recent surveys on hu- 
man pose estimation by Moeslund et al. (2006) and Poppe (2007) , 
date back to 2006 and 2007, respectively, and since they cover 
in great breadth and depth the whole vision-based human mo- 
tion capture domain, they are highly recommended. However, they 
do not focus speciﬁcally on the 3D human pose estimation and 
are now outdated. Other existing reviews, focus on more speciﬁc 
tasks. For instance, a review on view-invariant pose representa- 
tion and estimation is offered by Ji and Liu (2010) . In the work of 
Sminchisescu (2008) , an overview of the problem of reconstructing 
3D human motion from monocular image sequences is provided, 
whereas Holte et al. (2012) present a 3D human pose estimation 
review, which covers only model-based methods in multi-view set- 
tings. 
The primary goal of our review is to summarize the recent ad- 
vances of the 3D pose estimation task. We conducted a systematic 
research of single-view approaches published in the 2008–2015 
time frame. For multi-view scenarios, we focused on methods ei- 
ther published after the work of Holte et al. (2012) or published 
before, but not discussed in their work. The selected time frames 
ensure that all approaches discussed in this survey are not refer- 
enced in previous reviews. However, for an incipient overview of 
this ﬁeld, the reader is encouraged to refer to the publications of 
Sigal et al. (2010) ; Sigal and Black (2010) where, inspired by the 
introduction of the HumanEva dataset, they present some aspects 
of the image- and video-based human pose and motion estima- 
tion tasks. In the recent work of Sigal (2014) , the interested reader 
can ﬁnd a well-structured overview of the articulated pose esti- 
mation problem. Finally, Moeslund et al. (2011) offer an illustrative 
introduction to the problem and provide a detailed analysis and 
overview of different human pose estimation approaches. 
1.2. Taxonomy and scope of this survey 
Fig. 3 presents the pool of steps which apply to most 3D hu- 
man pose estimation systems and illustrates all the stages covered 
in this review. Three-dimensional pose estimation methods include 
some of the action steps shown which are: (i) the use of a priori 
body model which determines if the approach will be model-based 
or model-free, (ii) the utilization of 2D pose information which can 
be used not only as an additional source of information but also 
as a way to measure the accuracy by projecting the estimated 3D 
pose to the 2D image and comparing the error, (iii) the use of pre- 
processing techniques, such as background subtraction, (iv) feature 
extraction/selection approaches that obtain key features from the 
human subject which are fed to the estimation algorithms, (v) the 
process of obtaining an initial 3D pose which is used thereafter by 
optimization techniques that are employed to estimate the 3D pose 
and (vi) the pose estimation approach proposed each time that of- 
ten is discussed along with constraints that are enforced to dis- 
Fig. 3. Pool of the stages of a common 3D human pose estimation system. Given 
an input signal the 3D pose is estimated by employing some or even all of the 
depicted steps. 
card anthropometrically unrealistic poses, and ﬁnally how the ﬁnal 
pose is inferred. A more speciﬁc categorization of the approaches 
wouldn’t be practical since different approaches follow different 
paths according to the problem they are trying to address. 
Despite the increasing interest from the scientiﬁc community, a 
well-structured taxonomy for the 3D human pose estimation task 
has not been proposed. To group approaches with similar key char- 
acteristics, we categorized the problem based on the input signal. 
We investigate articulated 3D pose and motion estimation when 
the input is a single image or a sequence of RGB frames. In the 
latter case approaches focus on capturing how the 3D human pose 
changes over time from an image sequence. A noteworthy amount 
of publications address the articulated 3D human pose estimation 
problem in multi-view scenarios. Since these approaches overcome 
some diﬃculties, while at the same time introducing new chal- 
lenges to the pose estimation task, they are discussed separately 
in each case. 
Similar to the aforementioned surveys and resources, we ap- 
proach the pose estimation methods focusing on how they inter- 
pret the structure of the body: generative (model-based), discrim- 
inative (model-free), part-based which is a subcategory of gener- 
ative models, and ﬁnally hybrid approaches. The taxonomy of 3D 
Pose Estimation methods is depicted in Fig. 4 . 
Generative model approaches (also referred to as model-based 
or top-down approaches) employ a known model based on 
a priori information such as speciﬁc motion ( Daubney et al., 
2012 ) and context ( Ning et al., 2008 ). The pose recovery process 
comprises two distinct parts, the modeling and the estimation 
( Sminchisescu, 2002 ). In the ﬁrst stage, a likelihood function is 
constructed by considering all the aspects of the problem such as 
the image descriptors, the structure of the human body model, the 
camera model and also the constraints being introduced. For the 
estimation part, the most likely hidden poses are predicted based 
on image observations and the likelihood function. 
Another category of generative approaches found in the lit- 
erature is part-based (also referred to as bottom-up approaches), 
which follows a different path by representing the human skele- 
ton as a collection of body parts connected by constraints im- 
posed by the joints within the skeleton structure. The Pictorial 
Structure Model (PSM) is the most illustrative example of part- 
based models. It has been mainly used for 2D human pose es- 
timation ( Eichner et al., 2012; Felzenszwalb and Huttenlocher, 
2005; Pishchulin et al., 2013 ) and has lately been extended for 
3D pose estimation ( Belagiannis et al., 2014a; Burenius et al., 
2013 ). It represents the human body as a collection of parts ar- 
ranged in a deformable conﬁguration. It is a powerful body model 
which results in an eﬃcient inference of the respective parts. An 
extension of the PSM is the Deformable Structures model pro- 
posed by Zuﬃet al. (2012) , which replaces the rigid part tem- 

4 
N. Saraﬁanos et al. / Computer Vision and Image Understanding 152 (2016) 1–20 
Image
Video
Mul-view
Part-based
Discriminave
Learning-
based
Hybrid
Generave
Example-
based
Generave
Discriminave
Fig. 4. Taxonomy of 3D Pose Estimation methods. Given an image or a video in a monocular or multi-view setup, methods can be classiﬁed as generative (a subcategory 
of which are part-based approaches), discriminative (which can be classiﬁed into learning-based and example-based) and ﬁnally hybrid which are a combination of the 
previous two. 
plates with deformable parts to capture body shape deforma- 
tions and to model the boundaries of the parts more accurately. 
A graphical model which captures and ﬁts a wide range of hu- 
man body shapes in different poses is proposed by Zuﬃand 
Black (2015) . It is called Stitched Puppet (SP) and is a realistic 
part-based model in which each body part is represented by a 
mean shape. Two subspaces of shape deformations are learned us- 
ing principal component analysis (PCA), independently accounting 
for variations in intrinsic body shape and pose-dependent shape 
deformations. 
Discriminative approaches (also referred to as model-free) do not 
assume a particular model since they learn a mapping between 
image or depth observations and 3D human body poses. They 
can be further classiﬁed into learning-based and example-based 
approaches. Learning-based approaches learn a mapping function 
from image observations to the pose space, which must general- 
ize well for a new image from the testing set ( Huang and Yang, 
2009a; Sedai et al., 2010 ). In example-based approaches, a set of 
exemplars with their corresponding pose descriptors is stored and 
the ﬁnal pose is estimated by interpolating the candidates obtained 
from a similarity search ( Grauman et al., 2003; Huang and Yang, 
2009a ). Such methods beneﬁt in robustness and speed from the 
fact that the set of feasible human body poses is smaller than the 
set of anatomically possible ones ( Van den Bergh et al., 2009 ). The 
main advantage of generative methods is their ability to infer poses 
with better precision since they generalize well and can handle 
complex human body conﬁgurations with clothing and accessories. 
Discriminative approaches have the advantage in execution time 
because the employed models have fewer dimensions. According 
to Sigal and Black (2010) , the performance of discriminative meth- 
ods depends less on the feature set or the inference method than 
it does for generative approaches. 
Additionally, there are hybrid approaches , in which discrimina- 
tive and generative approaches are combined to predict the pose 
more accurately. To combine these two methods, the observation 
likelihood obtained from a generative model is used to verify the 
pose hypotheses obtained from the discriminative mapping func- 
tions for pose estimation ( Rosales and Sclaroff, 2006; Sedai et al., 
2013b ). For example, Salzmann and Urtasun (2010) introduced a 
uniﬁed framework that combines model-free and model-based ap- 
proaches by introducing distance constraints into the discrimina- 
tive methods and employing generative methods to enforce con- 
straints between the output dimensions. An interesting discussion 
on generative and discriminative approaches can be found in the 
work of Bishop and Lasserre (2007) . 
In the following, we present a detailed analysis of 3D pose es- 
timation techniques in different setups. The rest of the paper is 
organized as follows. In Section 2 , we discuss the main aspects of 
the body model employed by model-based methods and the most 
common features and descriptors used. In Section 3 , we present 
the proposed taxonomy by discussing the key aspects of pose es- 
timation approaches from a single image. Section 4 presents the 
recent advances and trends in 3D human pose estimation from a 
sequence of images. In both sections, we discuss separately single- 
and multi-view input approaches. In Section 5 , we discuss some of 
the available datasets, summarize the evaluation measures found 
in the literature, and offer a summary of performance of several 
methods on the HumanEva dataset. Section 6 introduces a new 
synthetic dataset in which humans with different anthropometric 
measurements perform actions. An evaluation of the performance 
of state-of-the-art 3D pose estimation approaches is also provided. 
We conclude this survey in Section 7 with a discussion of promis- 
ing directions for future research. 
2. Human body model and feature representation 
The human body is a very complex system composed of 
many limbs and joints and a realistic estimation of the posi- 
tion of the joints in 3D is a challenging task even for humans. 
Marinoiu et al. (2013) investigated how humans perceive the pic- 
torial 3D pose space, and how this perception can be connected 
with the regular 3D space we move in. Towards this direction, they 
created a dataset which, in addition to 2D and 3D poses, contains 
synchronized eye movement recordings of human subjects shown 
a variety of human body conﬁgurations and measured how accu- 
rately humans re-create 3D poses. They found that people are not 
signiﬁcantly better at re-enacting 3D poses in laboratory environ- 
ments given visual stimuli, on average, than existing computer vi- 
sion algorithms. 
Despite these challenges, automated techniques provide valu- 
able alternatives for solving this task. Model-based approaches em- 
ploy a human body model which introduces prior information to 
overcome this diﬃculty. The most common 3D human body mod- 
els in the literature are the skeleton (or stick ﬁgure), a common 
representation of which is shown in Fig. 5 along with its struc- 
ture, and shape models. They both deﬁne kinematic properties, 

N. Saraﬁanos et al. / Computer Vision and Image Understanding 152 (2016) 1–20 
5 
Fig. 5. Left: Human skeleton body model with 15 joints. Right: Tree-structured representation with the pelvis as the root node (Sh. - Shoulder, Elb. - Elbow and Ank. - 
Ankle). 
whereas the shape models also deﬁne appearance characteristics. 
The cylindrical and the truncated cone body models are illustra- 
tive examples of shape models. After constructing the body model, 
constraints are usually enforced to constrain the pose parameters. 
Kinematic constraints, for example, ensure that limb lengths, limb- 
length proportions, and joint angles follow certain rules. Other 
popular constraints found in the literature are occlusion constraints 
that allow more realistic poses in which some body parts (legs or 
arms) are occluded by others and prevent double-counting phe- 
nomena, appearance constraints introduced by the symmetry of 
left and right body part appearances ( Gupta et al., 2008 ), and 
smoothness constraints in the angle of the joints which are used 
to avoid abrupt changes between sequential video frames. 
Whether a body model is employed or not (model-based or 
model-free approaches), the next action step in the study of 3D 
human motion, is the accurate feature extraction from the input 
signal. Early approaches in the ﬁeld used low-level features such 
as edges, color, optical ﬂow or silhouettes which are obtained af- 
ter performing background subtraction. Silhouettes are invariant to 
texture and lighting but require good segmentation of the subject, 
and can easily lose speciﬁc details of human parts. Image descrip- 
tors are then employed to describe these features and to reduce 
the size of the feature space. Common feature representations em- 
ployed in the literature include the use of Scale Invariant Feature 
Transforms (SIFT) ( Müller and Arens, 2010 ), Shape Context (SC) 
( Amin et al., 2013 ) and Appearance and Position Context (APC) de- 
scriptors ( Ning et al., 2008 ). APC is a sparse and local image de- 
scriptor, which captures the spatial co-occurrence and context in- 
formation of the local structure as well as their relative spatial po- 
sitions. Histograms of Oriented Gradients (HoG) have been used 
a lot lately ( Gkioxari et al., 2013; Yang and Ramanan, 2011 ), be- 
cause they perform well when dealing with clutter and can capture 
the most discriminative information from the image. Instead of ex- 
tracting features from the image, some approaches ( Chen et al., 
2011a; Huang and Yang, 2009a ) select the most discriminative fea- 
tures. Pons-Moll et al. (2014) proposed posebits which are seman- 
tic pose descriptors which represent geometrical relationships be- 
tween body parts and can take binary values depending on the an- 
swer to simple questions such as “Left foot in front of the torso”. 
Posebits can provide suﬃcient 3D pose information without requir- 
ing 3D annotation, which is a diﬃcult task, and can resolve depth 
ambiguities. 
3. Recovering 3D human pose from a single image 
The reconstruction of an arbitrary conﬁguration of 3D points 
from a single monocular RGB image has three characteristics that 
affect its performance: (i) it is a severely ill-posed problem because 
similar image projections can be derived from different 3D poses; 
(ii) it is an ill-conditioned problem since minor errors in the loca- 
tions of the 2D body joints can have large consequences in the 3D 
space; and (iii) it suffers from high dimensionality ( Agarwal and 
Triggs, 2006 ). Existing approaches propose different solutions to 
compensate for these constraints and are discussed in Section 3.1 . 
3.1. Three-dimensional human pose estimation from a single 
monocular image 
The recovery of 3D human poses in monocular images is a dif- 
ﬁcult task in computer vision since highly nonlinear human mo- 
tions, pose and appearance variance, cluttered backgrounds, occlu- 
sions (both from other people or objects and self-occlusions), and 
the ambiguity between 2D and 3D poses are common phenomena. 
The papers described in this category estimate the human pose 
explicitly from a single monocular image and are summarized in 
Table 1 . Publications that ﬁt into both the single image and the 
video categories are discussed in Section 4 . 
Deep-Learning 
Methods: 
Deep-learning 
methods 
are 
representation-learning approaches ( Bengio et al., 2013 ) com- 
posed of multiple non-linear transformations. Feature hierarchies 
are learned with features from higher and more abstract levels of 
the hierarchy formed by the composition of lower level features 
( Bengio, 2009; LeCun et al., 2015 ). Depending on the method 
used and how the architecture is set-up, it ﬁnds applications 
in both unsupervised and supervised learning as well as hybrid 
approaches ( Deng and Yu, 2014 ). After its early introduction by 
Hinton et al. (2006) ; Hinton and Salakhutdinov (2006) , employing 
deep architectures, is found to yield signiﬁcantly better results 
in many computer vision tasks such as object recognition, image 
classiﬁcation and face veriﬁcation ( Krizhevsky et al., 2012; Szegedy 
et al., 2013; Taigman et al., 2014 ). Following that, approaches 
which employ deep-learning techniques to address the 2D pose 
estimation task with great success, have been proposed ( Charles 
et al., 2016; Chen and Yuille, 2014; Tompson et al., 2014; Toshev 
and Szegedy, 2014 ) and only recently the 3D pose estimation 
task was approached using deep learning. In the work of Li and 
Chan (2014) , deep convolutional networks (ConvNets) are trained 
for two distinct approaches: (i) they jointly train the pose re- 
gression task with a set of detection tasks in a heterogeneous 
multi-task learning framework and (ii) pre-train the network using 
the detection tasks, and then reﬁne the network using the pose 
regression task alone. They show that the network in its last 
layers has an internal representation for the positions of the left 
(or right) side of the person, and thus, has learned the structure 
of the skeleton and the correlation between output variables. 
Li et al. (2015) proposed a framework which takes as an input an 

6 
N. Saraﬁanos et al. / Computer Vision and Image Understanding 152 (2016) 1–20 
Table 1 
3D human pose estimation from a single monocular RGB image. Wherever a second reference is provided, it denotes the availability of source code for the method. The 
Body Model column indicates whether a body model is employed. The Method Highlights column reﬂects the most important steps in each approach. 
Year 
First author 
Body 
model 
Method highlights 
Evaluation 
datasets 
Evaluation metrics 
2016 
Yasin et al. (2016a) , 
Yasin et al. (2016b) 
Yes 
Training: 3D poses are projected to 2D and a regression model is learned from 
the 2D annotations; Testing: 2D pose is estimated, the nearest 3D poses are 
predicted; ﬁnal 3D pose is obtained by minimizing the projection error 
HumanEva-I, 
Human3.6M 
3D pose 
2015 
Li et al. (2015) 
No 
The input is an image and a potential 3D pose and the output a score 
matching value; ConvNet for image feature extraction; Two sub-networks for 
transforming features and pose into a joint embedding 
Human3.6M 
MPJPE 
2014 
Kostrikov and Gall (2014) 
Yes 
Predict the relative 3D joint position using depth sweep regression forests 
trained with three groups of features; 3DPS model for inference 
Human3.6M, 
HumanEva-I 
3D, 3D pose 
2014 
Li and Chan (2014) 
No 
Train a deep ConvNet; and joint point regression to estimate the positions of 
joint points relative to the root position and joint point detection to classify 
whether one local window contains the speciﬁc joint 
Human3.6M 
MPJPE 
2014 
Wang et al. (2014a) , 
Wang et al. (2014b) 
Yes 
2D part detector and a sparse basis representation in an overcomplete 
dictionary; Anthropometric constraints are enforced and an L 1 -norm 
projection error metric is used; Optimization with ADMM 
HumanEva-I, 
CMU MoCap, 
UVA 3D 
3D pose 
2014 
Zhou et al. (2015a) , 
Zhou et al. (2015b) 
Yes 
Convex formulation by using the convex relaxation of the orthogonality 
constraint; ADMM for optimization 
CMU MoCap 
3D 
2013 
Radwan et al. (2013) 
Yes 
Employ a 2D part detector with an occlusion detection step; Create multiple 
views synthetically with a twin-GPR in a cascaded manner; Kinematic and 
orientation constraints to resolve remaining ambiguities 
HumanEva-I, 
CMU MoCap 
3D pose 
2013 
Simo-Serra et al. (2013) 
Yes 
Bayesian approach using a model with discriminative 2D part detectors and a 
probabilistic generative model based on latent variables; Inference using the 
CMA-ES 
HumanEva-I, 
TUD Stadmitte 
3D, 3D pose 
2012 
Brauer et al. (2012) 
Yes 
ISM to obtain vote distributions for the 2D joints; Example-based 3D prior 
modeling and comparison of their projections with the respective joint votes 
UMPM 
MJAE, Orientation 
Angle 
2012 
Ramakrishna et al. (2012a) , 
Ramakrishna et al. (2012b) 
Yes 
Enforce anthropometric constraints and estimate the parameters of sparse 
linear representation in an overcomplete dictionary with a matching pursuit 
algorithm 
CMU MoCap 
3D 
2012 
Simo-Serra et al. (2012) 
Yes 
2D part detector and a stochastic sampling to explore each part region; Set of 
hypotheses enforces reprojection and length constraints; OCSVM to ﬁnd the 
best sample 
HumanEva-I, 
TUD Stadmitte 
3D, 3D pose 
2011 
Greif et al. (2011) 
No 
Train an action-speciﬁc classiﬁer on improved HoG features; use a people 
detector algorithm and treat 3D pose estimation as a classiﬁcation problem 
HumanEva-I 
3D 
2009 
Guo and Patras (2009) 
No 
Pose tree is learned by hierarchical clustering; Multi-class classiﬁers are 
learned and the relevance vector machine regressors at each leaf node 
estimate the ﬁnal 3D pose 
HumanEva-I 
3D 
2009 
Huang and Yang (2009a) , 
Huang and Yang (2009b) 
No 
Occluded test images as a sparse linear combination of training images; 
Pose-dependent (HoG) feature selection and L 1 -norm minimization to ﬁnd 
the sparest solution 
HumanEva-I, 
Synthetic 
3D, MJAE 
2008 
Ning et al. (2008) 
No 
Employ an APC descriptor and learn in a jointly supervised manner the visual 
words and the pose estimators 
HumanEva-I, 
Quasi-synthetic 
3D, MJAE 
image and a 3D pose and produces a score value that represents 
a multi-view similarity between the two inputs (i.e., whether 
they depict the same pose). A ConvNet for feature extraction is 
employed and two sub-networks are used to perform a non-linear 
transformation of the image and pose into a joint embedding. 
A maximum-margin cost function is used during training which 
enforces a re-scaling margin between the score values of the 
ground truth image-pose pair and the rest image-pose pairs. The 
score function is the dot-product between the two embeddings. 
However, the lack of training data for ConvNet-based techniques 
remains a signiﬁcant challenge. Towards this direction, the meth- 
ods of Chen et al. (2016) and Rogez and Schmid (2016) propose 
techniques to synthesize training images with ground truth pose 
annotations. Finally, the task of estimating the 3D human pose 
from image sequences has also been explored using deep learning, 
Elhayek et al. (2015 , 2016) ; Hong et al. (2015) ; Tekin et al. (2016a , 
2016b) ; Zhou et al. (2016b) and the respective methods are going 
to be discussed individually in Sections 4.1 and 4.2 . 
Two-dimensional detectors for 3D pose estimation: To overcome 
the diﬃculty and the cost of acquiring images of humans along 
with their respective 3D poses, Yasin et al. (2016a) proposed a 
dual-source approach which employs images with their annotated 
2D poses and 3D motion capture data to estimate the pose of 
a new test image in 3D. During training, 3D poses are projected 
to a 2D space and the projection is estimated from the anno- 
tated 2D pose of the image data through a regression model. At 
testing time, the 2D pose of the new image is ﬁrst estimated 
from which the most likely 3D poses are retrieved. By minimiz- 
ing the projection error the ﬁnal 3D pose is obtained. Aiming 
to perform 3D human pose estimation from noisy observations, 
Simo-Serra et al. (2012) proposed a stochastic sampling method. 
As a ﬁrst step, they employ a state-of-the-art 2D body part de- 
tector ( Yang and Ramanan, 2011 ) and then convert the bounding 
boxes of the parts to a Gaussian distribution by computing the co- 
variance matrix of the classiﬁcation scores within each bounding 
box. To obtain a set of ambiguous candidate poses from the sam- 
ples generated in the 3D space by the Gaussian distribution, they 
use the Covariance Matrix Adaptation Evolution Strategy (CMA- 
ES) to simultaneously minimize re-projection and length errors. 
The most anthropometric pose between the candidates is deter- 
mined by using a One-Class Support Vector Machine (OCSVM). To 
exploit the advantages of both generative and discriminative ap- 
proaches, Simo-Serra et al. (2013) proposed a hybrid Bayesian ap- 
proach. Their method comprises 2D HoG-based discriminative part 
detectors which constrain the 2D location of the body parts and 
a probabilistic generative latent variable model which (i) maps 
points from the high dimensional 3D space to the lower dimen- 
sional latent space, (ii) speciﬁes the dependencies between the la- 
tent states, (iii) enforces anthropometric constraints, and (iv) pre- 
vents double counting. To infer the ﬁnal 3D pose they use a vari- 
ation of CMA-ES. Brauer et al. (2012) employ a slightly modiﬁed 
Implicit Shape Model (ISM) to generate vote distributions for po- 

N. Saraﬁanos et al. / Computer Vision and Image Understanding 152 (2016) 1–20 
7 
tential 2D joint locations. Using a Bayesian formulation, 3D and 
2D poses are estimated by modeling (i) the pose prior following 
an example-based approach and (ii) the likelihood by comparing 
the projected joint locations of the exemplar poses with the corre- 
sponding nearby votes. 
Discussion of Norms and Camera Parameter Estimation: To re- 
solve the ambiguities that arise when performing pose estimation 
from a single image, some methods also estimate the relative pose 
of the camera. The approaches of Ramakrishna et al. (2012a) and 
Wang et al. (2014a) belong to this category. Both methods require 
the locations of the joints in the 2D space as an input, use a sparse 
basis model representation, and employ an optimization scheme 
which alternatively estimates the 3D pose estimation and the cam- 
era parameters. In the ﬁrst case, the authors constrain the sum of 
the limb lengths and use a matching pursuit algorithm to perform 
reconstruction. Their method can also recover the 3D pose of mul- 
tiple people in the same view. In the latter case, L 1 -norm is used 
as a reprojection error metric that is more robust when the joint 
locations in 2D are inaccurate. This approach also enforces not 
only limb length constraints, which eliminate implausible poses, 
but also L 1 -norm constraints on the basis coeﬃcients. A discussion 
on why L 2 -norms are insuﬃcient for estimating 3D pose similarity 
is provided by Chen et al. (2011b) . However, Zhou et al. (2015a, 
2016c) argue that the solution to such alternating minimization 
approaches is sensitive to initialization. Using the 2D image land- 
marks as an input, they used an augmented shape-space model 
to give a linear representation of both intrinsic shape deforma- 
tion and extrinsic viewpoint changes. They proposed a convex for- 
mulation that guarantees global optimality and solved the opti- 
mization problem with a novel algorithm based on the Alternat- 
ing Direction Method of Multipliers (ADMM) and the proximal op- 
erator of the spectral norm. Their method is applicable not only 
to human pose but also to car and face reconstruction. An ap- 
proach which also uses a sparse image representation and solves 
a convex optimization problem with the L 1 -norm is proposed by 
Huang and Yang (2009a) . Aiming to estimate 3D human pose when 
humans are occluded, they proposed a method which exploits the 
advantages of both example-based and learning-based approaches 
and represents each test sample as a sparse linear combination of 
training samples. The background clutter in the test sample is re- 
placed with backgrounds from the training images which results 
in pose-dependent feature selection. They use a Gaussian process 
regressor to learn the mapping between the image features (HoG 
from original or corrupted images and recovered features) and the 
corresponding 3D parameters. They observed that when a sparse 
linear representation of the training images is used for the probes, 
the set of coeﬃcients from the corrupted (i.e., occluded) test im- 
age is recovered with minimum error via solving an L 1 -norm min- 
imization problem. 
Discriminative Approaches: Ning et al. (2008) proposed a dis- 
criminative bag of words approach. As a ﬁrst step, they utilize an 
APC descriptor, and learn in a supervised manner a separate metric 
for each visual word from the labeled image-to-pose pairs. They 
use a Bayesian Mixture of Experts (BME) model to represent the 
multi-modal distribution of the 3D human pose conditioned on the 
feature space and also a gradient ascent algorithm which jointly 
optimizes the metric learning and the BME model. Kostrikov and 
Gall (2014) approached the pose estimation task from a different 
perspective, and proposed a discriminative depth sweep forest re- 
gression approach. After extracting features from 2D patches sam- 
pled from different depths, the proposed method sweeps with a 
plane through the 3D volume of potential joint locations and uses 
a regression forest that learns 2D-2D or 3D-3D mappings from the 
relative feature locations. Thus, they predict the relative 3D posi- 
tion of a joint, given the hypothesized depth of the feature. Finally, 
the pose space is constrained by employing a 3D pictorial structure 
model used to infer the ﬁnal pose. Okada and Soatto (2008) intro- 
duced a method comprising three main parts that estimates the 
3D pose in clutter backgrounds. Given a test image with a win- 
dow circumscribing a speciﬁc subject, (i) they extract a HoG-based 
feature vector of the window; (ii) they use a Support Vector Ma- 
chine (SVM) classiﬁer that selects the pose cluster which the cur- 
rent pose belongs to; and (iii) having taken into consideration that 
the relevance of features selected depends on the pose, they re- 
cover the 3D pose using a piecewise linear regressor of the se- 
lected cluster. 
Guo and Patras (2009) and Jiang (2010) proposed exemplar- 
based approaches. In the ﬁrst approach a tree is learned by hi- 
erarchical clustering on pose manifold via aﬃnity propagation 
and the ﬁnal 3D pose is estimated by applying the learned rele- 
vance vector machine regressor that is attached to the leaf node 
to which the example is classiﬁed. In the second method, the 
3D pose is reconstructed by using a k-dimensional tree (kd-tree) 
to search in a database containing mi