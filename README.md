# MSCA-CSSC-UNet: A multi-scale cross-attention network with cross-scale skip connections for point-wise blood pressure estimation from single-lead ECG 

This is an official repository related to the paper **"MSCA-CSSC-UNet"**.

- **Paper link:** []  
- **Author:** [Shunbin Chen]，[Dongxiao Zhang]，[Jialiang Xie]  
- **Dataset link:** [https://www.kaggle.com/datasets/mkachuee/BloodPressureDataset]

## 1. Project Introductio
MSCA-CSSC-UNet is an enhanced framework based on UNet++, designed specifically for point-wise blood pressure estimation.Evaluations on the MIMIC-III dataset show that the proposed model outperforms baseline methods in mean squared error (MSE), coefficient of determination (R2), and mean absolute error (MAE).It meets the British Hypertension Society (BHS) Grade C standard as well as the requirements of the Association for the Advancement of Medical Instrumentation (AAMI). Ablation studies validate the effectiveness of the proposed CSSC and MSCA mechanisms.
- Core Contributions:
  1. By utilizing a single-modality input, our method avoids the inherent issue of SMO associated with multimodal systems, thereby reducing both device complexity, alleviating the operational burden on clinicians, and minimizing potential discomfort for patients.
  2. Moving beyond the limitations of discrete BP prediction, our model provides a real-time, instantaneous BP values. This advancement is critically important for clinical practice, as it facilitates detailed hemodynamic assessment and supports more informed, rapid therapeutic interventions.
  3. With the support of the PIP, the proposed MSCACSSC-UNet model enables accurate and real-time continuousBPmonitoring.Experimentalresultsdemonstrate that the model’s performance meets the AAMI standard and achieves a Grade C rating under the stringent BHS protocol. Furthermore, in comparative assessments, our model consistently outperforms established classical machine learning and contemporary deep learning approaches on all evaluated metrics.

## 2. Core Challenges
1. **Signal Modality Overhead, SMO**: Most existing high-precision methods rely on multimodal input (such as using ECG and PPG signals simultaneously), which brings problems such as complex device integration, difficult data synchronization, and heavy computational burden, which is not conducive to clinical deployment and wearable device applications.
2. **Lack of point-wise continuous blood pressure estimation capability**: Existing methods generally adopt a "collect-then-generate" (CTG) paradigm, requiring the completion of one or more complete cardiac cycle signals before outputting blood pressure values ​​(such as SBP/DBP or entire waveforms). This makes it difficult to achieve real-time, instantaneous blood pressure prediction, making it difficult to support dynamic hemodynamic monitoring and timely clinical intervention.
## 3. Parameter Configuration

## 4. Usage Guide

