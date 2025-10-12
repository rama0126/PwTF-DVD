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

- inference: `./inference/test_on_raw_video.py --video [video_path] --out_dir [output_path] --model_path [model_path]`
- Additional codes will be uploaded continuously.

## Introduction of Previous Research for Video Deepfake Detection
[FTCN: Exploring Temporal Coherence for More General Video Face Forgery Detection (ICCV 2021)](https://arxiv.org/abs/2108.06693)
- GitHub:[https://github.com/yinglinzheng/FTCN](https://github.com/yinglinzheng/FTCN)
- Paper: [arXiv:2108.06693](https://arxiv.org/abs/2108.06693)


[AltFreezing: Alternating Freezing for More General Video Face Forgery Detection (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_AltFreezing_for_More_General_Video_Face_Forgery_Detection_CVPR_2023_paper.pdf)
- GitHub: [https://github.com/ZhendongWang6/AltFreezing](https://github.com/ZhendongWang6/AltFreezing)
- Paper: [CVPR 2023 Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_AltFreezing_for_More_General_Video_Face_Forgery_Detection_CVPR_2023_paper.pdf)


[StyleFlow: Exploiting Style Latent Flows for Generalizing Deepfake Video Detection (CVPR 2024)](https://arxiv.org/abs/2403.06592)
- GitHub: [https://github.com/jongwook-Choi/StyleFlow](https://github.com/jongwook-Choi/StyleFlow)
- Paper: [arXiv:2403.06592](https://arxiv.org/abs/2403.06592)

