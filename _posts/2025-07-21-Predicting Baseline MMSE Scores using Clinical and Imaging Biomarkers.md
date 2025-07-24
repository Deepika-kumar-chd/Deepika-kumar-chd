# Predicting Baseline MMSE Scores using Clinical and Imaging Biomarkers from the ADNI Dataset


## Introduction
This project aims to develop machine learning regression models to predict baseline Mini-Mental State Examination (MMSE) scores using a combination of demographic, clinical, and neuroimaging biomarkers available in the ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset. MMSE is a widely used screening tool to measure cognitive impairment. Early and accurate estimation of MMSE can aid in timely clinical intervention and support diagnostic decision-making in the context of Alzheimer’s Disease (AD).


## Dataset and Processing  
The project uses the ADNI baseline dataset, focusing on the following categories:

1. **Demographic Feature**:
- AGE

2. **Genetic Risk Factor**:

- APOE4 (0, 1, 2 copies of the E4 allele): APOE ε4 is strongly associated with increased risk of Alzheimer’s disease.APOE4 status is often used in cognitive decline and Alzheimer's research as a genetic biomarker.Having one ε4 allele increases Alzheimer's risk by ~2–3x.Having two ε4 alleles increases risk by ~10–15x.

3. **Clinical Cognitive Assessments**: 

- ADAS11_bl (Alzheimer's Disease Assessment Scale): It is a standardized test used to assess cognitive dysfunction in individuals with Alzheimer's disease (AD) or Mild Cognitive Impairment (MCI). The ADAS-Cog11 includes 11 tasks that evaluate: Word recall, Naming objects and fingers, Following commands, Constructional praxis (copying shapes), Ideational praxis (doing multi-step tasks), Orientation, Word recognition, Remembering test instructions, Spoken language ability, Word-finding difficulty, Comprehension. The maximum score is 70, but in practice, scores above 40 are rare outside of severe AD cases.

- ADAS13_bl (Expanded version): ADAS13 builds on ADAS11 by including two additional tasks, which improve sensitivity to early cognitive changes, especially in MCI.It includes all 11 tasks from ADAS11, plus: Delayed Word Recall and Digit Cancellation (a task of attention and executive function). Maximum score is 85, but typical ranges are much lower in early stages.

- CDRSB_bl (Clinical Dementia Rating – Sum of Boxes at baseline): It is a clinical score used to quantify the severity of dementia symptoms in individuals, particularly in Alzheimer's disease studies. CDR (Clinical Dementia Rating) is composed of 6 domains: Memory, Orientation, Judgment & Problem Solving, Community Affairs, Home & Hobbies, Personal Care. 0.0 means normal cognition and 18.0 means severe dementia. 

- RAVLT_immediate_bl: It refers to the Rey Auditory Verbal Learning Test (RAVLT) — Immediate Recall score at baseline. It is a key neuropsychological test used to assess episodic verbal memory. A low score may indicate early memory impairment, especially when compared to delayed recall or recognition.

- RAVLT_learning_bl: It measures how much verbal information a person learns across repeated exposures to the same list of words — a key indicator of learning efficiency and cognitive function. Higher values mean better learning ability.

- RAVLT_forgetting_bl: This feature represents how many words the person forgot after a 30-minute delay, compared to their best performance during the earlier trials. Higher scores = more forgetting, indicating poorer memory consolidation.

- RAVLT_perc_forgetting_bl(Rey Auditory Verbal Learning Test): This is the percentage of previously recalled words that were forgotten after the delay. It provides a normalized measure of memory decay.

- FAQ_bl (Functional Activities Questionnaire): It refers to the Functional Activities Questionnaire score assessed at the baseline visit of a subject. It's a clinical tool used to measure a person's ability to perform instrumental activities of daily living (IADLs) — which are more complex than basic self-care tasks. Managing finances, Preparing meals, Shopping alone, Using a telephone, Traveling outside the neighborhood, Handling paperwork, Keeping track of current events, Playing games/hobbies, Remembering appointments, Watching television or reading. Higher FAQ scores = greater functional impairment.

4. **Imaging Biomarkers**:  

- Ventricles_bl: It is the volume of the brain ventricles (fluid-filled spaces).As brain tissue shrinks (atrophy), the ventricles enlarge. Larger ventricular volume at baseline often correlates with greater brain atrophy and cognitive decline.

- Hippocampus_bl: The hippocampus is central to memory formation. It is one of the first regions to atrophy in Alzheimer's. Lower values means worse memory-related decline.

- WholeBrain_bl: Total brain volume (excluding ventricles and cerebrospinal fluid) at baseline. It is a global marker of neurodegeneration. Decreases as disease progresses.

- Entorhinal_bl: Volume of the entorhinal cortex, part of the medial temporal lobe. It is involved in memory, navigation, and perception. One of the earliest regions to show volume loss in AD.

- Fusiform_bl: Volume of the fusiform gyrus, which supports visual processing and facial recognition. Atrophy here has been linked to cognitive impairment and late-stage AD. Lower values means greater impairment.

- MidTemp_bl: Volume of the middle temporal gyrus at baseline. It is important for language and semantic memory. Shrinkage here is associated with language dysfunction in AD.Lower values means poorer language and memory performance

5. **Target Variable**:

- MMSE_bl (Mini-Mental State Examination score at baseline): Higher scores indicate better cognitive function. It tests include several cognitive domains like orientation (time and place), registration (repeat named prompts),attention and calculation, recall (short-term memory), language (naming, repetition, comprehension, visuospatial (copying a design)



Data was preprocessed for missing values, encoded appropriately, and scaled. MongoDB was used as the document-oriented database to store structured data records extracted from CSV files.
### Preprocessing Pipelines:
- Categorical Pipeline (APOE4): Imputed using most frequent values and encoded using OneHotEncoder.
- Clinical Pipeline: Median imputation and StandardScaler.
- Imaging Pipeline: KNNImputer (k=5) followed by StandardScaler.
- Age Pipeline: Only scaled.


## Model Experiments and Performance Summary:  
Seven regression models were evaluated:
- Linear Regression (Default)
- Ridge Regression (Alpha + Max Iter tuned)
- Lasso Regression (Alpha + Max Iter tuned)
- Support Vector Regressor (Kernel + C + Gamma tuned)
- Random Forest (Depth + Estimators tuned)
- Gradient Boosting (Learning Rate + Depth + Estimators tuned)
- XGBoost (Learning Rate + Depth + Estimators tuned)

GridSearchCV was applied to all models to find the best parameters. Performance metrics (R2, RMSE, MAE) were used to evaluate for test sets.
<table>
  <tr>
    <th>Model</th>
    <th>R2</th>
    <th>RMSE</th>
    <th>RMAE</th>
  </tr>
  <tr>
    <td>Ridge Regression</td>
    <td>0.670</td>
    <td>1.565</td>
    <td>1.224</td>
  </tr>
  <tr>
    <td>Lasso Regression </td>
    <td>0.669</td>
    <td>1.567</td>
    <td>1.224</td>
  </tr>
  <tr>
    <td>Linear Regression</td>
    <td>0.668</td>
    <td>1.568</td>
    <td>1.225</td>
  </tr>
  <tr>
    <td>Support Vector Regressor</td>
    <td>0.665</td>
    <td>1.577</td>
    <td>1.226</td>
  </tr>
  <tr>
    <td>Random Forest</td>
    <td>0.659</td>
    <td>1.590</td>
    <td>1.252</td>
  </tr>
  <tr>
    <td>Gradient Boosting</td>
    <td>0.655</td>
    <td>1.599</td>
    <td>1.262</td>
  </tr>
  <tr>
    <td>XGBoost</td>
    <td>0.655</td>
    <td>1.599</td>
    <td>1.264</td>
  </tr>
</table>


## Machine Learning Lifecycle:

<img src="https://deepika-kumar-chd.github.io/Deepika-kumar-chd/images/Flowchart.png" width=80% height=100%>

- Data Ingestion 
- Data validation
- Data transformation
- Model trainer
- Model pusher

## Code Implementation
For a detailed walkthrough of the code, refer to the project's [GitHub repository](https://github.com/Deepika-kumar-chd/MMSE_Prediction_ML).

Code deployed on https://mmse-prediction.onrender.com

