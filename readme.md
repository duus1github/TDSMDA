Preface:
===========================
This is the code of my own small paper, should be to use this code, at present, I want to use the architecture of this article is the graph expression learning +transformer+mlp.

Data set:
===========================
miRNA similarity data, disease similarity data, protein similarity data, m-d association pairs, m-p association pairs, p-d association pairs
label: [1,0]: indicates that miRNA and disease are correlated; [0,1] indicates no correlation
case study:
    50:Breast Neoplasms
    236:Lung Neoplasms    
    240:Lymphoma
    
Evaluation index:
==========================
- Accuracy (acc) : the proportion of correctly classified samples in the total number of samples.
- precision: The proportion of samples predicted to be positive by the model to those predicted to be positive.
- Recall: samples that are actually positive, and the proportion of samples predicted to be positive that are actually      positive. The best is 1, the worst is 0.
- F1 Score: is the harmonic average of accuracy rate (pre) and recall. The higher the F1 score, the more robust the model.
- ROC curve: The closer to the top left corner, the better the performance of the classifier.
- AUC: =1, is a perfect classifier; 1>auc>0.5: better than random guess; Properly set the threshold value, it has predictive value; =0.5: No predictive value, just like a random guess.