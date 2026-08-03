# DAÏO: A High-Specificity Real-TimeVision-Based Fall Detection SystemUsing Advanced Kinematic Tracking andComputer Vision

> 2026 · id: W7169498896 · pdf: https://www.researchsquare.com/article/rs-10374436/latest.pdf · 来源: pdf_url
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

DAÏO: A High-Specificity Real-TimeVision-Based
Fall Detection SystemUsing Advanced Kinematic
Tracking andComputer Vision
Alamin Abubakar Nataala 
SEGi University College Kota Damansara: SEGi University Kota Damansara https://orcid.org/0009-
0001-9833-0846
Research Article
Keywords: AI systems, Machine Learning, Artificial Intelligence, Kalman
Posted Date: July 17th, 2026
DOI: https://doi.org/10.21203/rs.3.rs-10374436/v1
License:   This work is licensed under a Creative Commons Attribution 4.0 International License.  
Read Full License
Additional Declarations: The authors declare no competing interests.

DAÏO: A High-Speciůcity Real-Time 
Vision-Based Fall Detection System 
Using Advanced Kinematic Tracking and 
Computer Vision 
Author: Al-Amin Abubakar Nataala 
Date: November 11, 2025 
ORCID: hŵps://orcid.org/0009-0001-9833-0846​
 
Abstract 
Rapid and highly speciůc fall detection is critical to preserving the health and autonomy of 
vulnerable elderly populations. While existing vision-based fall detection architectures promise 
non-intrusive monitoring, they suŦer from high false positive rates triggered by ordinary 
Activities of Daily Living (ADLs) and severe performance degradation under joint occlusions. 
This paper presents DAÏO (Intelligent Vision-Based Fall Detection System), a novel, 
edge-capable, high-speciůcity computer vision framework designed to address these 
limitations. DAÏO operates on a hybrid architecture combining a lightweight human pose 
estimator, an Extended Kalman Filter (EKF) with adaptive measurement noise, and a stateful 
Triple-Check Logic Engine. By tracking joint dynamics and mapping them through an adaptive 
mathematical state-space model, the system adjusts to occlusion events dynamically using 
visibility conůdence scores. Falling actions are validated across spatial, rotational, and kinematic 
dimensions in parallel, ůltering out high-velocity non-fall events (such as prostrations, deep 
squats, and rapid siŵing) via a velocity latching mechanism. 
Evaluated against a composite dataset comprising 340 video sequences (~153,000 frames) 
from the UR Fall Dataset, the Multiple Cameras Fall Dataset (MCFD), and customized 
stress-test scenarios, the system achieved an overall accuracy of 95.3%, a sensitivity of 93.3%, 
and a speciůcity of 96.4%. Crucially, the velocity latch cut false positive alarms by 79.7% 
compared to baseline posture-only models, demonstrating 100% speciůcity during active, 
repetitive religious prostrations. Running locally at 35.7 FPS on consumer workstations and 
12.3% on a Raspberry Pi 4, the proposed system demonstrates high speciůcity, operational 
reliability, and privacy preservation at the network edge. 
 
Keywords: Fall detection, Computer vision, Extended Kalman Filter, MediaPipe Pose, Kinematic 

tracking, Edge computing, High speciůcity. 
1. Introduction 
The global aging demographic presents unprecedented challenges to healthcare systems and 
independent living structures. According to the World Health Organization, falls represent the 
second leading cause of accidental or unintentional injury deaths worldwide, with adults older 
than 65 years suŦering the greatest number of fatal falls (World Health Organization, 2021). The 
clinical prognosis of a fallen individual is strictly bound to the "lie time"—the duration an 
individual remains on the ground before receiving medical aŵention. Prolonged lie times are 
associated with severe complications, including rhabdomyolysis, pressure ulcers, hypothermia, 
and long-term psychological dread of subsequent falls. Consequently, autonomous, real-time 
fall detection systems have emerged as a core focus in assistive technology. 
To date, research in autonomous fall detection has been bifurcated into wearable sensor 
systems and environmental vision-based frameworks (Wang et al., 2017). Wearable systems 
(e.g., waist-mounted tri-axial accelerometers and gyroscopes) are highly sensitive to impact 
dynamics but suŦer from poor user compliance. Elderly individuals frequently forget to wear, 
charge, or properly calibrate these devices, and active physical movements can generate 
high-velocity motion artifacts that lead to high false-alarm rates. Conversely, vision-based fall 
detection oŦers a non-intrusive alternative that eliminates user compliance constraints. 
However, state-of-the-art vision systems exhibit three major limitations: 
1.​ Kinematic Instability: Sudden, erratic camera movements, fast background illumination 
changes, and fast skeletal transitions cause high-frequency jiŵer in estimated joint 
coordinates. 
2.​ High False Positive Rates (FPR): Routine Activities of Daily Living (ADLs) that mimic fall 
physics—such as tying a shoe, deep squats, collapsing onto a sofa, or executing 
prostrations during prayer—frequently trigger false alarms, causing alert fatigue for 
caregivers. 
3.​ Privacy and Latency Overheads: Traditional vision approaches stream raw video feeds 
to centralized cloud servers for deep learning inference. This compromises personal 
privacy in private spaces (such as bedrooms or bathrooms) and introduces latency that 
can delay emergency responses. 
To resolve these challenges, this study presents DAÏO (Intelligent Vision-Based Fall Detection 
System). DAÏO is a localized, edge-computing pipeline that tracks human skeletons, ůlters 
tracking noise dynamically, and validates falls across three physical criteria. The system is 
designed to run entirely locally without external cloud dependencies, safeguarding personal 
privacy while achieving low system latency. 
 
 

 
2. Literature Review 
The academic landscape of automated fall detection is divided into wearable technologies, 
ambient sensors, and computer vision systems. Each paradigm represents a unique balance of 
sensitivity, speciůcity, computational cost, and user comfort. 
2.1. Wearable and Ambient Technologies 
Historically, wearable architectures have relied on tri-axial accelerometers, magnetometers, 
and barometric altimeters to track the kinematic state of a user (Wang et al., 2017). These 
systems utilize predeůned threshold-based algorithms or shallow machine learning classiůers 
to identify the impact paŵerns characteristic of a fall. While these systems demonstrate 
excellent accuracy when aŵached ůrmly to the center of mass, user non-compliance remains 
their fatal Ųaw. Ambient systems, such as Ųoor vibration sensors, acoustic arrays, and 
micro-radar modules, oŦer a passive, non-wearable alternative. However, their deployment is 
hindered by high installation costs, blind spots, and interference from environmental noise (e.g., 
dropped objects or pets). 
2.2. Video-Based Posture and Skeletal Models 
Computer vision systems represent the most robust non-wearable alternative. Early vision 
models relied on background subtraction to analyze the bounding box aspect ratios, silhoueŵe 
orientations, and optical Ųow ůelds of moving human subjects. These models are highly 
sensitive to changes in illumination, shadows, and the presence of moving non-human objects. 
To overcome these environmental dependencies, researchers transitioned to human pose 
estimation (HPE) frameworks. HPE models extract structured 2D or 3D skeletal topologies from 
standard RGB video feeds. Advanced deep learning HPE networks, such as those relying on 
Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs), provide highly accurate 
joint coordinates (Kim et al., 2023). However, these architectures require high-performance 
GPUs, rendering them unsuitable for low-power edge nodes. 
To address this, Google developed MediaPipe Pose, an open-source framework optimized for 
real-time mobile and edge CPU execution (Kim et al., 2023). MediaPipe Pose extracts 33 
skeletal landmarks along with their respective visibility metrics (Hammadi et al., 2022). While 
MediaPipe Pose provides a lightweight, real-time tracking solution, raw landmark coordinates 
still suŦer from high-frequency jiŵer, self-occlusions, and tracking lag during sudden 
movements. 
2.3. Fall Detection Classiůers and the Speciůcity Deůcit 
Once skeletal coordinates are extracted, deep-learning classiůers (e.g., Spatio-Temporal Graph 
Convolutional Networks, or ST-GCNs) are commonly deployed to classify falls from ADLs (Li et 
al., 2018). While deep-learning approaches demonstrate high classiůcation rates on controlled, 
closed datasets, they lack physical interpretability and generalize poorly to out-of-distribution 

movements (Yuan et al., 2022). Consequently, everyday actions, such as a user bending over 
quickly to pick up an object or kneeling to pray, are frequently classiůed as falls, producing 
excessive false alarms (Zi et al., 2023). 
Moreover, deep learning classiůers do not explicitly account for joint occlusion, which is a major 
source of tracking failure when furniture or structural columns block the camera's line of sight. 
These systemic challenges require a hybrid approach: one that combines the lightweight 
processing of edge pose estimators with robust physical models that ůlter out measurement 
noise and validate the mechanical properties of a fall. 
 
3. Methodology 
This 
study 
adopts 
an 
engineering-based case-study and algorithmic optimization 
methodology. Rather than treating fall detection as an unconstrained black-box classiůcation 
problem, DAÏO enforces physical laws, speciůcally, gravity and kinematic constraints, on the 
tracked skeleton. The system architecture is composed of ůve distinct, sequential processing 
stages designed for local execution. 
​

 
 
 
 
 
3.1. Edge-Based Human Pose Estimation 
The primary input pipeline receives a 2D RGB video frame at index 
 with dimensions 
. The frame is converted to the RGB color space and passed to the MediaPipe Pose 
API (Kim et al., 2023). For each frame, the engine returns a set of 33 landmarks, where each 
landmark  is represented as a tuple: 
where 
 and 
 are the normalized coordinates in 
 mapped to the interval 
, 

and 
 represents the visibility probability that the joint is not occluded by 
another body part or environmental obstacle. 
​
3.2. State-Space Modeling and EKF Tracking 
To smooth out high-frequency tracking jiŵer and reconstruct missing joint coordinates during 
occlusion events, DAÏO implements independent Extended Kalman Filter (EKF) instances to 
track critical skeletal junctions. Speciůcally, the system tracks the mid-point of the hips (
) 
and the mid-point of the shoulders (
), as these form the central coordinate vector of 
the human torso. 
For a targeted joint, the state of its motion is modeled as a 4-dimensional continuous vector: 
where 
 and 
 represent the ůltered coordinate positions in the image plane, and 
 and 
 represent the corresponding horizontal and vertical pixel velocities. 
3.2.1. Process Model 
Assuming a constant velocity model with state transition interval 
, the state 
transition equation is formulated as: 
where the transition matrix 
 is deůned linearly as: 

The process noise vector 
 models acceleration perturbations, with its 
covariance matrix 
 parameterized by spatial noise 
 and velocity noise 
: 
Through empirical validation, these parameters were tuned to 
 and 
. 
This conůguration forces the ůlter to maintain strict geometric consistency during slow, 
steady-state walking (low 
), while allowing the state estimation to adapt quickly to rapid 
acceleration transients during a fall (high 
). 
​
3.2.2. Measurement Model and Adaptive Noise 
The measurement vector 
 represents the raw observations returned by 
MediaPipe Pose for the target joint: 
where the measurement mapping matrix 
 is deůned as: 
 

and the measurement noise covariance vector is modeled as white noise 
. 
A core contribution of the DAÏO system is the formulation of Adaptive Measurement Noise (
). During occlusion, the raw landmark estimate 
 becomes unreliable, which can 
introduce massive tracking spikes. To solve this, the measurement noise covariance matrix 
 is updated dynamically in every frame based on the joint visibility score 
 returned 
by MediaPipe: 
where 
 is the base measurement noise variance (set to 
) and 
 is a tiny 
regularizer that prevents division-by-zero errors. 
When a joint is fully visible (
), 
 remains small, causing the EKF update stage 
to trust the raw computer vision observations. However, when the joint is occluded (e.g., 
 as a person falls behind a table), 
. This causes the Kalman Gain 
 to scale down to zero: 
Consequently, the correction step of the ůlter is bypassed, and the state vector 
 relies 
entirely on the internal physical process model (
). This prevents tracking failure 
during momentary self-occlusions or partial environment blocks. 
 
 
 
 
 

3.2.3. Jacobian Derivations for Torso Angle Analysis 
To track the dynamic tilt of the torso, the system evaluates the angle 
 of the vector spanning 
from the hip midpoint (
) to the shoulder midpoint (
): 
Because 
 is a non-linear function, we construct the ůrst-order Jacobian matrices 
 to 
linearize 
and 
propagate 
the 
covariance boundaries. Leŵing 
 and 
, the partial derivatives with respect to the state coordinates are derived as: 
These linearized transformations prevent mathematical instability and coordinate lag during 
rapid rotational maneuvers. 
3.3. The Triple-Check Logic Engine 
To address the high false positive rates of traditional threshold-based systems, DAÏO uses a 
parallel multi-metric evaluation architecture. A fall alert is triggered if and only if three 
independent structural, spatial, and kinematic conditions are met simultaneously: 
                             
 
 
 
 

 
​
Check 1: Dynamic Position Check 
This check measures the absolute vertical displacement of the user's center of mass relative to 
their baseline height, ensuring scale-invariance across diŦerent camera distances. During the 
initial calibration sequence (ůrst 100 frames of steady standing), the system records the 
average vertical coordinate of the hip midpoint, denoted as 
. In each frame 
, the 
position metric evaluates: 
The ůrst condition is satisůed if: 
This means a fall is suspected only if the user's hips drop below 
 of their calibrated 
standing height. 
 

Check 2: Torso Angle Analysis 
A true fall terminates with the human body in a horizontal orientation on the ground. Check 2 
measures the angle of the torso relative to the ground plane: 
This prevents vertical movements that maintain an upright torso—such as deep squats or rapid 
siŵing, from triggering a false fall detection. 
​
Check 3: Kinematic Velocity Latch 
To diŦerentiate between controlled downward transitions (such as bowing, prostrations, or 
kneeling) and uncontrolled falls, the third module checks for gravitational acceleration proůles. 
The vertical velocity of the hip midpoint, 
, is extracted directly from the EKF state 
vector. The kinematic check is formulated as: 
Once high-velocity impact is registered, a velocity latching mechanism maintains a True value 
for a shiųing conůrmation window, ensuring the system can process the post-fall static lying 
phase even as velocity drops back to zero. 
Weighted Conůdence Scoring 
If a fall is conůrmed, DAÏO generates a continuous conůdence score 
 to 
represent the severity of the event. Because velocity is the primary physical diŦerentiator of 
impact injury, it is assigned a heavier weighting coeũcient: 
where 
, 
, and 
. The individual score components 
 
represent the normalized deviation of each metric beyond its minimum activation threshold. 

3.4. State-Machine Architecture 
To prevent single-frame tracking glitches from triggering false alarms, system state transitions 
are managed by a Finite State Machine (FSM): 
 
 
●​ CALIBRATING: Captures the upright calibration baseline (
). Transition to 
MONITORING occurs aųer 100 stable frames. 
●​ MONITORING: Continually executes the EKF and evaluations. If the Triple-Check Logic 
returns True, the system transitions to FALL SUSPECTED. 
●​ FALL SUSPECTED: A temporal buŦer state. If the fall conditions are maintained for 30 
consecutive frames (1.0 second at 30 FPS), the state transitions to FALL CONFIRMED. 
If any check fails before the buŦer is ůlled, the system resets to MONITORING. 
●​ FALL CONFIRMED: Evaluates a valid fall. The system locks this state and broadcasts 
secure alerts via MQTT over TLS and Telegram. 
●​ RECOVERY: Monitors the post-fall state. The system transitions to RECOVERY only if 
the user returns to an upright posture (
) and height displacement 
normalizes for at least 150 frames (5.0 seconds), preventing Ųickering alerts. 
 

4. Findings and Analysis 
The DAÏO pipeline was validated against a large composite video database containing 340 
sequences (~153,000 frames). This evaluation set combined the UR Fall Dataset, the Multiple 
Cameras Fall Dataset (MCFD), and custom stress-test clips designed to induce false positives 
(e.g., fast prostrations, kneeling, rapid siŵing, and dropping objects). 
4.1. Overall System Classiůcation Performance 
The performance of the classiůcation framework was measured using standard sensitivity 
(recall), speciůcity (true negative rate), and accuracy metrics: 
The system achieved excellent performance across all three metrics: 
●​ Sensitivity (Recall): 93.3% 
●​ Speciůcity (True Negative Rate): 96.4% 
●​ Overall Accuracy: 95.3% 
●​ Mean Alert Latency: 1.23 seconds (from initial vertical descent to alert transmission over 
local MQTT). 
 
4.2. Ablation Study: Impact of the Velocity Latch Module 
To quantify the beneůt of the physics-informed velocity latch, an ablation study was 
conducted. The baseline posture model (using only the Position and Angle checks) was 
compared against the full DAÏO Triple-Check pipeline: 
Metric 
Conůguration 
Posture-Only 
Model (Position + 
Angle) 
Full DAÏO Pipeline 
(Triple-Check) 
Relative Delta 
Sensitivity 
 
 
 
Speciůcity 
 
 
 
False Positive Rate 
 
 
 
 

Speciůcity Improvements on ADLs 
The baseline conůguration frequently misclassiůed controlled downward transitions as falls. 
Introducing the velocity latch caused a 79.7% relative reduction in the False Positive Rate 
(FPR). 
The 0.9% decrease in sensitivity was isolated to slow, sliding falls (e.g., syncope where an 
individual slides slowly down a wall). Because these movements do not exceed the 1.5 m/s 
velocity threshold, they bypassed the velocity latch. However, such slow-sinking actions carry a 
lower risk of immediate traumatic brain injury compared to high-velocity impacts, which the 
velocity latch detected with 100% accuracy. 
Prayer Prostration Isolation 
During evaluations involving active, repetitive religious prostrations, the posture-only baseline 
misclassiůed 100% of prostration sequences as falls due to the horizontal torso angle and low 
hip height. In contrast, the full DAÏO pipeline achieved a 100% True Negative Rate 
(speciůcity) on all prayer sequences. Although prostrations meet the height and angle 
thresholds, they are performed under controlled muscle deceleration, staying well below the 
1.5 m/s  velocity threshold. 
4.3. Real-Time Execution Benchmark 
The processing time for each stage of the pipeline was benchmarked across two diŦerent local 
hardware environments to verify its edge capability: 
Hardware 
Environment 
Pose 
Extracti
on 
EKF 
Tracking 
Logic 
Check 
Overhead / 
Rendering 
Total 
Frame 
Time 
Maximum 
Throughput 
Workstation 
(i7-11700K, 
RTX 3070) 
 
 
 
 
 
 
Raspberry Pi 
4 
(Quad-Core 
1.5GHz) 
 
 
 
 
 
 
These execution times demonstrate that the mathematical simplicity of the EKF and logic 
checks adds negligible computational overhead (5ms on workstation, 8ms on Raspberry Pi). 
This allows the system to achieve real-time tracking performance even on low-cost edge 
computers. 

5. Discussion 
The empirical results conůrm that integrating a physics-informed EKF tracking loop with 
structured triple-check logic provides a robust solution for real-world fall detection. 
​
5.1. Inductive Physical Bias vs. Deep Learning 
Most contemporary computer vision systems rely on deep temporal classiůers, such as 
CNN-LSTMs or Spatio-Temporal Graph Convolutional Networks (Li et al., 2018). While these 
models can learn highly complex motion features, they lack an inductive physical bias. They 
operate as black-box estimators, mapping raw pixel coordinates to target fall classes without 
modeling the laws of motion. 
DAÏO addresses this limitation by embedding physical constraints directly into the tracking 
loop. Modeling state-space variables with an Extended Kalman Filter enforces gravitational and 
physical limits on estimated joint behaviors. The dynamic position, angle, and velocity checks 
provide a transparent, rule-based decision boundary that is easy to debug and verify, unlike 
deep neural networks. 
5.2. Edge Processing and Privacy Preservation 
A major obstacle to deploying vision-based fall detection in home seŵings is the invasion of 
privacy. Streaming continuous video of private rooms to external servers is unacceptable to 
most users. 
Because DAÏO uses highly optimized, CPU-friendly pose extraction (Kim et al., 2023) and 
eũcient mathematical tracking equations, the entire pipeline runs locally on low-cost hardware 
like a Raspberry Pi 4. The raw video stream never leaves the local memory space of the edge 
device; only structured state telemetry and system alerts are transmiŵed externally. This oŬine, 
local execution model eliminates cloud service subscription fees and safeguards user privacy, 
facilitating wider deployment in residential and clinical seŵings. 
5.3. Limitations 
Despite its high speciůcity, the system has two main limitations: 
1.​ Slow-Sinking Falls: As shown in the ablation study, slow-descending falls (such as a 
person slowly sliding down a wall due to fainting) do not generate high velocities, which 
can cause the velocity latch to fail. 
2.​ Extreme Occlusion: If key tracking joints (the hips and shoulders) are fully occluded for 
more than 2.0 Seconds (60 frames), the EKF's state uncertainty boundary covariance 
 grows excessively. When the target joints emerge from behind the obstacle, this 
large uncertainty can cause coordinate tracking snaps and momentary instability. 
 

6. Conclusion 
This paper presented DAÏO, a high-speciůcity, real-time vision-based fall detection system 
designed for local edge deployment. By integrating MediaPipe Pose extraction with an 
Extended Kalman Filter and an adaptive measurement noise model, DAÏO maintains stable 
skeletal tracking even during partial joint occlusions. The system's Triple-Check Logic Engine 
combines height displacement, torso tilt, and dynamic vertical velocity checks to evaluate 
potential falls, successfully reducing false-alarm rates during routine Activities of Daily Living. 
Evaluated on a large, multi-dataset testing framework, DAÏO achieved an overall accuracy of 
95.3%, with a speciůcity of 96.4% and a sensitivity of 93.3%. The velocity latching module 
eliminated 79.7% of false positive alerts compared to baseline posture-only models, showing 
excellent resilience during active religious prayer movements. 
Future research will focus on: 
●​ Expanding the single-camera architecture to a multi-camera distributed edge network to 
handle room crossings and eliminate blind spots. 
●​ Integrating infrared and depth-sensing feeds to maintain tracking stability in zero-light 
environments. 
●​ Utilizing active learning algorithms to dynamically adjust the user's calibrated standing 
baseline (
) as they age or their gait paŵerns change over time. 
 
 

References 
Bazarevsky, V., Grishchenko, I., Raveendran, K., Zhu, T., Zhang, F., & Grundmann, M. (2020). 
BlazePose: On-device real-time body pose tracking. arXiv preprint arXiv:2006.12472. 
hŵps://arxiv.org/abs/2006.12472 
Hammadi, Y., Grondin, F., Ferland, F., & Lebel, K. (2022). Evaluation of various state of the art 
head 
pose 
estimation 
algorithms 
for 
clinical 
scenarios. 
Sensors, 
22(18), 
6850. 
hŵps://doi.org/10.3390/s22186850 
Kim, J.-W., Choi, J.-Y., Ha, E.-J., & Choi, J.-H. (2023). Human pose estimation using MediaPipe 
Pose and optimization method based on a humanoid model. Applied Sciences, 13(4), 2700. 
hŵps://doi.org/10.3390/app13042700 
Li, C., Cui, Z., Zheng, W., Xu, C., & Yang, J. (2018). Spatio-temporal graph convolution for 
skeleton based action recognition. Proceedings of the AAAI Conference on Artiůcial 
Intelligence, 32(1), 7444–7452. hŵps://doi.org/10.1609/aaai.v32i1.11776 
Wang, Z., Yang, Z., & Dong, T. (2017). A review of wearable technologies for elderly care that can 
accurately track indoor position, recognize physical activities and monitor vital signs in real 
time. Sensors, 17(2), 341. hŵps://doi.org/10.3390/s17020341 
World 
Health 
Organization. 
(2021). 
Falls 
[Fact 
sheet]. 
hŵps://www.who.int/news-room/fact-sheets/detail/falls 
Yuan, C., Zhang, P., Yang, Q., & Wang, J. (2022). Fall detection and direction judgment based on 
posture 
estimation. 
Discrete 
Dynamics 
in 
Nature 
and 
Society, 
2022, 
1–12. 
hŵps://doi.org/10.1155/2022/8372291 
Zi, X., Chaturvedi, K., Braytee, A., Li, J., & Prasad, M. (2023). Detecting human falls in poor 
lighting: Object detection and tracking approach for indoor safety. Electronics, 12(5), 1259. 
hŵps://doi.org/10.3390/electronics12051259​