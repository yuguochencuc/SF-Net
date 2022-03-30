### SF-Net for fullband SE
This is the repo of the manuscript "Optimizing Shoulder to Shoulder: A Coordinated Sub-Band Fusion Model for Real-Time Full-Band Speech Enhancement", which is submitted to Interspecch 2022. Some audio samples are provided here and the code for network is released soon.


Abstract：Due to the high computational complexity to model more frequency bands, it is still intractable to conduct real-time full-band speech enhancement based on deep neural networks. Recent studies typically utilize the compressed perceptually motivated features with relatively low frequency resolution to filter the full-band spectrum by one-stage networks, leading to limited speech quality improvements. In this paper, we propose a coordinated sub-band fusion network for full-band speech enhancement, which aims to recover the low- (0-8 kHz), middle- (8-16 kHz), and high-band (16-24 kHz) in a step-wise manner. Specifically, a dual-stream network is first pretrained to recover the low-band complex spectrum, and another two sub-networks are designed as the middle- and high-band noise suppressors in the magnitude-only domain. To fully capitalize on the information intercommunication, we employ a sub-band interaction module to provide external knowledge guidance across different frequency bands. Extensive experiments show that the proposed method yields consistent performance advantages over state-of-the-art full-band baselines.

### [Demo page of audio samples](https://yuguochencuc.github.io/sfnet_demo/) 
(https://yuguochencuc.github.io/sfnet_demo/)  

### System flowchart of SF-Net
![image](https://user-images.githubusercontent.com/51236251/160518968-0a074c93-7400-467c-9c16-2887f8466737.png)

### Results:
## Abaltion study
![lQLPDhtH56hfHqXNAjHNA7OwrtrfF1S_QEECRntyY8DWAA_947_561](https://user-images.githubusercontent.com/51236251/160519109-49617410-5611-4abb-85fc-d30e13225280.png)

## Comparison with SOTA
![lQLPDhtH6ITooSDNAz7NA7GwySl8YbYqe-8CRnzbbEA6AA_945_830](https://user-images.githubusercontent.com/51236251/160519878-e0aa93ea-9573-40f9-9665-2d8de9d90efe.png)

## Visualization of spectrograms
## VB dataset
![lQLPDhtH6NmMG6rNAwTNA6uw8qUBkZFtUxgCRn1lwQBsAA_939_772](https://user-images.githubusercontent.com/51236251/160520076-b580c1bb-253c-44ee-a781-1243caa7924d.png)
## DNS blind set
![lQLPDhtH8iT2-N_NAubNA42wBREh0WKHv5wCRoygD0BOAA_909_742](https://user-images.githubusercontent.com/51236251/160527589-97dff384-cb72-4354-8ab0-1f953ab5564c.png)



