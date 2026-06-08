# Task-2-Vennelakanchanapalli
# DecodeLabs - Data Classification (Project 2)

An end-to-end Machine Learning pipeline utilizing the **K-Nearest Neighbors (KNN)** algorithm to classify the classic Iris dataset. Moving beyond rule-based architectures, this project implements complete data preprocessing, automated hyperparameter tuning (Elbow Method), comprehensive performance evaluation, and decision boundary visualization.

---

## 🚀 Key Pipeline Features

* **Complete ML Lifecycle:** Implements feature scaling, training-testing splits, model training, evaluation, hyperparameter optimization, and boundary mapping.
* **Feature Scaling Pipeline:** Employs `StandardScaler` to ensure distance metrics in the KNN engine are completely unbiased across diverse feature magnitudes.
* **Auto-Generated Analytics:** Automatically renders and exports **6 analytical graphics** detailing feature physics, validation metrics, and performance landscapes.
* **Automated Hyperparameter Tuning:** Integrates an Elbow Curve analyzer to dynamically calculate and display the optimal neighborhood value ($K$).

---

## 🛠️ Project Visualizations Gallery

The script automatically processes data matrices and writes 6 high-definition visualization plots directly into the workspace directory:

| Filename | Type | Analytical Purpose |
| --- | --- | --- |
| `viz1_dataset_overview.png` | Scatter Matrix | Plots feature pairs against one another to identify cluster linear/non-linear separability. |
| `viz2_feature_distributions.png` | Box Plot Matrix | Reveals statistical distributions, medians, quartiles, and range spreads per class. |
| `viz3_confusion_matrix.png` | Heatmap Matrix | Quantifies true positives, false positives, false negatives, and true negatives clearly. |
| `viz4_classification_report.png` | Grouped Bar | Graphically breaks down Precision, Recall, and F1-Score metrics across every plant class. |
| `viz5_optimal_k.png` | Line Curve | Implements the **Elbow Curve** by tracking validation error across values of $K$ from 1 to 20. |
| `viz6_decision_boundary.png` | 2D Contour Map | Computes and plots the actual classification decision margins across the feature space. |

---

## 💻 Technical Stack

* **Language:** Python 3.x
* **Core Machine Learning:** `scikit-learn`
* **Data Processing & Engineering:** `numpy`
* **Visualization Engine:** `matplotlib`, `seaborn`

---

## 📁 Repository Structure

```text

├── data_classification.py          # Project 2: Main KNN ML Pipeline script

└── README.md                       # Complete project portfolio documentation

```

---

## ⚙️ Installation & Running the Engine

### 1. Configure the Environment

Ensure you have the required engineering libraries installed on your device:

```bash
pip install numpy matplotlib seaborn scikit-learn

```

### 2. Run the Classification Pipeline

Execute the processing core through your terminal terminal prompt:

```bash
python data_classification.py

```

---

## 📊 Pipeline Terminal Output Snapshot

```text
==================================================
     DecodeLabs - Data Classification
==================================================

 Samples: 150 | Features: 4 | Classes: 3

[VIZ 1 SAVED] viz1_dataset_overview.png
[VIZ 2 SAVED] viz2_feature_distributions.png

[MODEL TRAINED] Accuracy: 100.00%

[VIZ 3 SAVED] viz3_confusion_matrix.png
[VIZ 4 SAVED] viz4_classification_report.png
[VIZ 5 SAVED] viz5_optimal_k.png
[VIZ 6 SAVED] viz6_decision_boundary.png

==================================================
     FINAL SUMMARY
==================================================

 Algorithm : KNN (K=5)
 Dataset   : Iris (150 samples, 4 features, 3 classes)
 Split     : 80% Train / 20% Test
 Accuracy  : 100.00%
 Optimal K : 11

 6 Visualizations saved to current directory.

[PIPELINE COMPLETE] Data Classification with KNN - All steps executed successfully!

```

---

## 🤝 Contributing

Contributions, optimizations, and additions (such as comparing this against Support Vector Machines or Random Forests) are always welcome.

1. Fork the Repository
2. Scale the codebase (`git checkout -b feature/AmazingClassifier`)
3. Commit optimizations (`git commit -m 'Added comparative classifier analysis'`)
4. Push updates (`git push origin feature/AmazingClassifier`)
5. Open a formal Pull Request

---

## 👤 Contact

**Vennela Kanchanapalli** * **Organization:** DecodeLabs

* **LinkedIn:** https://www.linkedin.com/in/vennelakanchanapalli
* **GitHub Repository:** https://github.com/vennela0811/Task-2-Vennelakanchanapalli

---

## 📄 License

Distributed under the MIT License. See the snippet below for terms:

```text
MIT License

Copyright (c) 2026 Vennela Kanchanapalli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

```
