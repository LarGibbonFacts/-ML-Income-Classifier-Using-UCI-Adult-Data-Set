# -ML-Income-Classifier-Using-UCI-Adult-Data-Set
ECON 418 Project
### **Project Summary**

This project focuses on building and evaluating predictive models to classify individuals' income levels based on demographic and employment-related features. The dataset contains two income classes: "Low" (≤ $50K) and "High" (> $50K), with a significant class imbalance favoring the "Low" income group.

The project involved the following steps:
1. **Data Preprocessing**:
   - Removed redundant and less informative features.
   - Scaled numerical variables and encoded categorical variables into binary or numeric formats.

2. **Modeling**:
   - Trained **Lasso** and **Ridge regression models** to predict income while incorporating regularization to handle feature importance and multicollinearity.
   - Developed **Random Forest models** with 100, 200, and 300 trees, using cross-validation to tune the number of features (`mtry`) considered at each split.

3. **Evaluation**:
   - Evaluated models on both training and testing datasets using metrics such as **accuracy**, **precision**, **recall**, and **F1-score**.
   - The **Random Forest model with 100 trees** achieved the highest testing accuracy (82.09%) and F1-score (88.96%), outperforming Lasso and Ridge models.

4. **Class Imbalance**:
   - The imbalance in the dataset led to challenges in identifying the minority class ("High" income). Metrics like precision, recall, and F1-score were used to better assess model performance for both classes.

5. **Insights**:
   - Random Forest models demonstrated robustness and outperformed regression-based models by capturing complex relationships between features.
   - While recall was high across all models, precision slightly lagged, suggesting the need for further tuning or resampling strategies to address class imbalance.
