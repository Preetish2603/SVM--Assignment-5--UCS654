# SVM Optimization for Multi-Class Classification

## Project Overview
This project implements an optimized Support Vector Machine (SVM) classifier for multi-class letter recognition using the UCI Letter Recognition dataset. The implementation includes:

- Comprehensive data analysis and visualization  
- 10 different training-test splits (70-30 ratio)  
- Hyperparameter optimization with 100 iterations per sample  
- Performance comparison across all samples  
- Convergence graph visualization for the best-performing model  

---

## Dataset
- **Source**: [UCI Machine Learning Repository - Letter Recognition Dataset](https://archive.ics.uci.edu/ml/datasets/letter+recognition)  
- **Size**: 20,000 instances  
- **Features**: 16 numerical attributes describing letter characteristics  
- **Classes**: 26 (A-Z)  

---

## Key Findings
- **Best cross-validation accuracy**: 89.50% (Sample S3)  
- **Best test accuracy**: 97.02%  
- **Optimal parameters across samples consistently used RBF kernel**  

### Most Common Optimal Parameters:
- **Kernel**: `rbf`  
- **Nu (C parameter)**: ~75.904  
- **Epsilon (gamma parameter)**: ~0.08337  

---

## Implementation Details

### Data Preprocessing
- Standard scaling of features  
- Label encoding of target classes (A-Z → 0-25)  
- 10 different random splits (70% training, 30% testing)  

### Model Optimization
- `RandomizedSearchCV` with 100 iterations  
- **Parameter space**:
  - Kernels: `linear`, `rbf`, `poly`  
  - C (Nu): loguniform between `1e0` and `1e2`  
  - gamma (Epsilon): loguniform between `1e-4` and `1e-1`  
- 3-fold cross-validation during optimization  

---

## Performance Metrics
- Accuracy scores for all samples  
- Confusion matrix for best model  
- Classification report with precision, recall, and F1-score  

---

## Results

### Comparative Performance Table

| Sample # | Best Accuracy | Best SVM Parameters (Kernel, Nu, Epsilon)     |
|----------|----------------|----------------------------------------------|
| S1       | 88.57%         | rbf, 3.416, 0.08508                          |
| S2       | 89.46%         | rbf, 75.904, 0.08337                         |
| S3       | 89.50%         | rbf, 75.904, 0.08337                         |
| S4       | 88.32%         | rbf, 3.416, 0.08508                          |
| S5       | 89.25%         | rbf, 75.904, 0.08337                         |
| S6       | 89.07%         | rbf, 75.904, 0.08337                         |
| S7       | 89.07%         | rbf, 48.801, 0.06167                         |
| S8       | 88.25%         | rbf, 75.904, 0.08337                         |
| S9       | 88.68%         | rbf, 48.801, 0.06167                         |
| S10      | 89.39%         | rbf, 75.904, 0.08337                         |

---

## Visualizations
- `class_distribution.png`: Shows frequency of each letter class  
- `feature_correlation.png`: Displays relationships between features  
- `convergence_plot.png`: Tracks accuracy improvement during optimization  
- `confusion_matrix.png`: Visualizes prediction patterns for best model  

---

## Dependencies
- Python 3.7+  
- scikit-learn  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- joblib  
- scipy  

---

## Files Included
- `svm_final_check.py`: Main implementation script  
- `class_distribution.png`: Class frequency visualization  
- `feature_correlation.png`: Feature relationships heatmap  
- `convergence_plot.png`: Optimization progress graph  
- `confusion_matrix.png`: Prediction patterns for best model  
- `README.md`: This documentation file  

---

## Future Work
- Experiment with different optimization techniques  
- Try alternative multi-class classification approaches  
- Implement feature selection to improve performance  
- Explore deep learning alternatives for comparison  
