# [ICCV 2025] Beyond Spatial Frequency: Pixel-wise Temporal Frequency-based Deepfake Video Detection
This repository contains the official implementation of our ICCV 2025 paper,
"Beyond Spatial Frequency: Pixel-wise Temporal Frequency-based Deepfake Video Detection."
 [arxiv](https://arxiv.org/abs/2507.02398)  [page](https://rama0126.github.io/PwTF-DVD/)





# Abstract
![그림1](https://github.com/user-attachments/assets/47093264-f235-4197-ac85-76f9c14653e3)

We introduce a deepfake video detection approach that exploits pixel-wise temporal inconsistencies, which traditional spatial frequency-based detectors often overlook. Traditional detectors represent temporal information merely by stacking spatial frequency spectra across frames, resulting in the failure to detect temporal artifacts in the pixel plane. Our approach performs a 1D Fourier transform on the time axis for each pixel, extracting features highly sensitive to temporal inconsistencies, especially in areas prone to unnatural movements. To precisely locate regions containing the temporal artifacts, we introduce an attention proposal module trained in an end-to-end manner. Additionally, our joint transformer module effectively integrates pixel-wise temporal frequency features with spatio-temporal context features, expanding the range of detectable forgery artifacts. Our framework represents a significant advancement in deepfake video detection, providing robust performance across diverse and challenging detection scenarios.



## Updates
- **[NEW]  New Code Uploaded**  
  Code under `preprocess/`, `inference/` 


- Additional modules will be uploaded continuously.

---
