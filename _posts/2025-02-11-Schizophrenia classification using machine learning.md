# Schizophrenia Classification Using Machine Learning

## Introduction
Schizophrenia is a chronic brain disorder characterized by hallucinations, delusions, and cognitive impairment. Traditional diagnostic methods rely heavily on subjective clinical assessments. Objective biomarkers derived from neuroimaging data, particularly EEG, can enhance the accuracy and reliability of schizophrenia diagnosis.  

Machine learning models are very useful in classification problems. In this project, the five most popular machine learning algorithms are used on an EEG dataset from patients with schizophrenia and healthy controls at rest.  

## Dataset and Processing  
Dataset can be downloaded from [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0188629). Data consists of fifteen minutes of EEG data of 28 subjects during an eyes-closed resting state condition. Data were acquired with a sampling frequency of 250 Hz using the standard 10–20 EEG montage with 19 EEG channels:  

**Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2**.  

The reference electrode was placed at **FCz**.  

Data processing was performed using **MNE-Python**. The steps involved:  
1. **Filtering**: Each EEG signal was filtered with a bandpass filter (0.5Hz – 45Hz) to remove unwanted noise like low-frequency drift or high-frequency noise.  
2. **Re-referencing**: The data was re-referenced using the default average reference method.  
3. **Artifact Removal**: Blinks and other artifacts were extracted using **Independent Component Analysis (ICA)**.  
4. **Epoching**: After artifact removal, the continuous EEG data was segmented into epochs to be used for machine learning models.  
5. **Autoreject**: Used to remove any remaining bad epochs.  

To understand the visualization and data processing pipeline in detail, follow this notebook:  
[Preprocessing Pipeline](https://github.com/Deepika-kumar-chd/EEG_Schizophrenia_classification/blob/main/Preprocessing_Pipeline.ipynb)  

## Extracting Features from EEG  
Five different measures were extracted from the EEG data:  

- **Gamma Power**  
- **Relative Gamma Power**  
- **Connectivity**  
- **Phase-Locking Value (PLV)**  
- **Coherence between Channel Pairs**  

These features are used to capture abnormal brain dynamics characteristic of schizophrenia and aid in the classification process by revealing disruptions in brain activity, connectivity, and synchronization.  

### Gamma Power  
Gamma power refers to the **power spectral density (PSD)** in the gamma frequency band (30-100 Hz), which reflects high-frequency brain activity associated with cognition, attention, and perception. In schizophrenia, gamma power is often dysregulated, indicating abnormal neural processing and network dysfunction.  

Gamma power is computed using the **Power Spectral Density (PSD)** obtained via **Welch's method**:  

<img src="./images/eeg schiz/psd.png">

Where:
- X(f) is the Fourier Transform of the EEG signal,  
- N is the number of segments used in Welch's method, 
- PSD(f) represents the distribution of signal power across frequencies.

Gamma Power is computed as:

<img src="./images/eeg schiz/gamma.png">

Where:
- Pγ is the mean gamma power,  
- Fγ is the set of frequencies in the gamma range (30-100 Hz).

### Relative Gamma Power
Relative gamma power measures the ratio of gamma band power (30-100 Hz) to the total power across all frequencies in an EEG signal. It normalizes gamma power, providing a frequency-independent measure of neural activity. Absolute gamma power can vary due to signal amplitude differences across subjects. The relative measure corrects for these variations. The relative gamma power is the ratio of gamma power to the total power across all frequencies:

<img src="./images/eeg schiz/relgamma.png">

Where:
- P<sub>y</sub><sup>rel</sup> is the relative gamma power, 
- P<sub>total</sub> is the sum of power over all frequencies.

### Connectivity
Connectivity in EEG refers to how different brain regions communicate with each other. It measures the statistical or functional dependence between EEG signals recorded from different electrodes. In schizophrenia, abnormal connectivity patterns have been widely reported, making it a crucial feature for classification. Studies show weakened connections between distant brain areas, affecting cognition and perception. Some regions may exhibit excessive local connectivity, which disrupts global integration. Connectivity is computed using Pearson correlation, which measures the linear relationship between two EEG channels.

<img src="./images/eeg schiz/connectivity.png">

Where:
- Xi, Xj are EEG signals from two different channels, 
- X&#772;i X&#772;j are their mean values,
- r<sub>ij</sub> is the correlation coefficient, ranging from -1 to 1:
  - 1 → Perfect positive correlation (signals move together)
  - 0 → No correlation (independent signals)
  - -1 → Perfect negative correlation (inverse signals)

### Phase-Locking Value (PLV)
Phase-Locking Value (PLV) is a measure of neural synchrony, quantifying how consistently the phase difference between two EEG signals remains over time. Unlike correlation, which captures amplitude relationships, PLV focuses purely on phase synchronization, which is crucial in neural communication and cognitive processes. PLV is based on the Hilbert Transform, which extracts the instantaneous phase of an EEG signal.

**Steps to Compute PLV**
1.	Compute the analytic signal Xa(t) using the Hilbert Transform:
<img src="./images/eeg schiz/hilbert.png">
where H(X(t)) is the Hilbert Transform of X(t).
2.	Extract the instantaneous phase:
<img src="./images/eeg schiz/angle.png">
3.	Compute the phase difference between two EEG channels i and j:
<img src="./images/eeg schiz/phase.png">
4.	Compute the PLV using the complex exponential function:
<img src="./images/eeg schiz/plv.png">

Where:
- N is the number of time points, 
- e<sup>jΔθ</sup> represents the unit vector of phase difference.

Extract upper triangular matrix to reduce redundancy.

**PLV Ranges**
- PLV=1 → Perfect synchronization (phases are locked)
- PLV=0 → No synchronization (random phase differences)

### Coherence
Coherence measures the synchronization between two EEG signals in the frequency domain. It quantifies how well two signals maintain a constant phase relationship at specific frequencies, indicating functional connectivity between brain regions. Coherence is based on the cross-spectral density (CSD) and power spectral density (PSD) of two signals. Given two EEG signals xi(t) and xj(t) coherence is defined as:  

<img src="./images/eeg schiz/coherence.png">

Where:
- Sij(f) is the cross-spectral density (CSD) between channels i and j.
- Sii(f) and Sjj(f) are the power spectral densities (PSD) of channels i and j.
- ∣Sij(f)∣<sup>2</sup> represents the magnitude squared of the cross-spectral density.

**Coherence Values**
-	Cij(f)=1 → Perfect synchronization at frequency f.
-	Cij(f)=0 → No synchronization at frequency f.


## Machine Learning Algorithms

### Logistic Regression (LR)
A linear classification algorithm that predicts the probability of a class. Uses a logistic function (sigmoid) to map outputs between 0 and 1. Ideal for linearly separable data and provides interpretable feature importance.
Hyperparameter Tuning:
-	C: Inverse regularization strength (lower values apply stronger regularization).
-	solver: Optimization algorithm (liblinear for small datasets, lbfgs for larger ones).
-	scaler: StandardScaler (zero mean, unit variance) and MinMaxScaler (scales between 0 and 1).

Best hyperparameters found:
- **C** = 0.0215
- **Solver** = liblinear
- **Scaler** = MinMaxScaler
- **Best Score**: 0.612

### k-Nearest Neighbors (k-NN)
A non-parametric classification algorithm. Finds the k nearest neighbors and assigns the most common class label. Sensitive to feature scaling, hence the use of StandardScaler.
Hyperparameter Tuning:
-	n_neighbors: Tested values [3, 5, 7, 9, 11].
-	weights: Uniform vs. Distance.
-	p: Manhattan vs. Euclidean distance.

Best hyperparameters found:
- **n_neighbors** = 11
- **p** = 2 (Euclidean distance)
- **Weights** = uniform
- **Best Score**: 0.602

### Decision Tree (DT)
A tree-based algorithm that makes sequential decisions based on feature splits. Uses the entropy criterion to minimize uncertainty. Can overfit, so depth control is necessary.
Hyperparameter Tuning:
-	max_depth: [5, 10, 15]
-	min_samples_split: [5, 10]
-	min_samples_leaf: [2, 5, 10]

Best hyperparameters found:
- **criterion** = entropy
- **Max Depth** = 10
- **Min Samples Leaf** = 5
- **Min Samples Split** = 5
- **Best Score**: 0.553

### Random Forest Classifier (RF)
An ensemble of decision trees. Uses bagging to improve robustness. Handles high-dimensional features well.
Hyperparameter Tuning:
-	n_estimators: [80, 100, 120]
-	max_depth: [15, 20, 25]
-	min_samples_split: [2, 3]
-	min_samples_leaf: [1, 2]
-	max_features: ['sqrt', 'log2']

Best hyperparameters found:
- **n_estimators** = 120
- **Max Depth** = 15
- **Min Samples Leaf** = 1
- **Min Samples Split** = 2
- **Max Features** = 'log2'
- **Best Score**: 0.576

### Support Vector Machine (SVM)
A kernel-based classifier that maximizes the margin between classes. Works well in high-dimensional feature spaces. Uses the Radial Basis Function (RBF) kernel to handle non-linearity.
Hyperparameter Tuning:
-	C: Regularization parameter (controls margin softness).
-	gamma: Kernel coefficient (affects decision boundary complexity).

Best hyperparameters found:
- **C** = 0.0336
- **Gamma** = scale
- **Best Score**: 0.646

## Comparison of Algorithms
<table>
  <tr>
    <th>Algorithm</th>
    <th>Best Score</th>
  </tr>
  <tr>
    <td>SVM</td>
    <td>0.646</td>
  </tr>
  <tr>
    <td>Logistic Regression</td>
    <td>0.612</td>
  </tr>
  <tr>
    <td>k-NN</td>
    <td>0.602</td>
  </tr>
  <tr>
    <td>Random Forest</td>
    <td>0.576</td>
  </tr>
  <tr>
    <td>Decision Tree</td>
    <td>0.553</td>
  </tr>
</table>


### Results
- **SVM performed the best**, likely due to its ability to capture non-linear relationships in EEG features.
- **Random Forest and Decision Tree had lower scores**, likely due to overfitting.
- **k-NN and Logistic Regression had moderate performance**, showing that both local neighborhood patterns and linear separation contribute to classification.

## Code Implementation
For a detailed walkthrough of the code, refer to the project's [GitHub repository](https://github.com/Deepika-kumar-chd/EEG_Schizophrenia_classification/tree/main).


