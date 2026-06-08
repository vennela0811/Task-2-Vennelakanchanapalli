import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names
colors = ['#E74C3C', '#2ECC71', '#3498DB']

print("=" * 50)
print("     DecodeLabs - Data Classification")
print("=" * 50)
print(f"\n Samples: {X.shape[0]} | Features: {X.shape[1]} | Classes: 3")

# Dataset Overview (pair plot style)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('VISUALIZATION 1: Dataset Overview - Feature Distributions',
             fontsize=14, fontweight='bold', y=1.01)

feature_pairs = [(0, 1), (0, 2), (0, 3), (2, 3)]
pair_titles   = ['Sepal L vs Sepal W', 'Sepal L vs Petal L',
                'Sepal L vs Petal W', 'Petal L vs Petal W']

for ax, (fx, fy), title in zip(axes.flatten(), feature_pairs, pair_titles):
    for cls in range(3):
        mask = y == cls
        ax.scatter(X[mask, fx], X[mask, fy],
                   color=colors[cls], label=class_names[cls],
                   alpha=0.7, edgecolor='white', linewidths=0.5, s=60)
    ax.set_xlabel(feature_names[fx], fontsize=9)
    ax.set_ylabel(feature_names[fy], fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('viz1_dataset_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[VIZ 1 SAVED] viz1_dataset_overview.png")

# Feature Distributions (Box plot style)
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle('VISUALIZATION 2: Feature Distributions per Class (Box plot)',
             fontsize=14, fontweight='bold')
for i, ax in enumerate(axes):
    data_per_class = [X[y == cls, i] for cls in range(3)]
    bp = ax.boxplot(data_per_class, patch_artist=True, notch=False)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(feature_names[i].replace(' (cm)', ''), fontsize=10, fontweight='bold')
    ax.set_xticklabels(class_names, rotation=15, fontsize=8)
    ax.set_ylabel('cm', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('viz2_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[VIZ 2 SAVED] viz2_feature_distributions.png")

# Scale -> Split -> Train -> Predict
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=True)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy    = model.score(X_test, y_test)

print(f"\n[MODEL TRAINED] Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix (Heatmap)
cm = confusion_matrix(y_test, predictions)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            linewidths=1, linecolor='white',
            annot_kws={"size": 16, "weight": 'bold'})

ax.set_title('VISUALIZATION 3: Confusion Matrix\n(TP / FP / FN / TN)',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Predicted Label', fontsize=10, fontweight='bold')
ax.set_ylabel('True Label', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('viz3_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[VIZ 3 SAVED] viz3_confusion_matrix.png")

# F1 / Precision / Recall Report
report = classification_report(y_test, predictions,
                                target_names=class_names, output_dict=True)

metrics = ['precision', 'recall', 'f1-score']
# FIX: Renamed 'X' to 'x_indices' to prevent overwriting the dataset features matrix
x_indices = np.arange(len(class_names)) 
bar_width = 0.25 # FIX: Unified variable name from 'width' to 'bar_width'
metric_colors = ['#E74C3C', '#2ECC71', '#3498DB']

fig, ax = plt.subplots(figsize=(10, 6))

for i, (metric, color) in enumerate(zip(metrics, metric_colors)):
    values = [report[cls][metric] for cls in class_names]
    # FIX: Used 'x_indices' and corrected the bar_width references
    bars = ax.bar(x_indices + i * bar_width, values, bar_width,
                    label=metric.capitalize(), color=color, alpha=0.85,
                    edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
ax.set_title('VISUALIZATION 4: Precision, Recall, F1-Score per Class',
                fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Flower Class', fontsize=10)
ax.set_ylabel('Score', fontsize=10)
# FIX: Adjusted ticks to align with the renamed x_indices variable
ax.set_xticks(x_indices + bar_width) 
ax.set_xticklabels(class_names, fontsize=9)
ax.set_ylim(0, 1.1)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('viz4_classification_report.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[VIZ 4 SAVED] viz4_classification_report.png")


# Optimal K Selection (Elbow Method)

k_range = range(1, 21)
error_rates = []

for k in k_range:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train, y_train)
    error_rates.append(1 - knn_k.score(X_test, y_test))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(k_range, error_rates, 'o-', color='#E74C3C',
        markersize=8, linewidth=2.5, markerfacecolor='white',
        markeredgewidth=2.5, label='Error Rate')

best_k  = k_range[error_rates.index(min(error_rates))]
best_err = min(error_rates)
ax.annotate(f'   Optimal K={best_k}\n   Error={best_err:.2f}',
            xy=(best_k, best_err),
            xytext=(best_k + 2, best_err + 0.05),
            arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2),
            fontsize=10, fontweight='bold', color='#E74C3C')

ax.axvline(x=best_k, color='#E74C3C', linestyle='--', alpha=0.6)
ax.set_title('VISUALIZATION 5: Tuning the Engine - Choosing Optimal K\n(The Elbow Curve)',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('K Value (Number of Neighbors)', fontsize=10)
ax.set_ylabel('Error Rate', fontsize=10)
ax.set_xticks(list(k_range))
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('viz5_optimal_k.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[VIZ 5 SAVED] viz5_optimal_k.png")


# Decision Boundary (Petal L vs Petal W)
# Re-train on 2 features only for 2D boundary plot
X_2d        = X[:, 2:4]  # Safe now: 'X' was not overwritten!
X_2d_sc     = StandardScaler().fit_transform(X_2d)
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
    X_2d_sc, y, test_size=0.2, random_state=42)

model_2d = KNeighborsClassifier(n_neighbors=5)
model_2d.fit(X_tr2, y_tr2)

h = 0.02  # step size in the mesh
x_min, x_max = X_2d_sc[:, 0].min() - 0.5, X_2d_sc[:, 0].max() + 0.5
y_min, y_max = X_2d_sc[:, 1].min() - 0.5, X_2d_sc[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(9, 7))
ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')

for cls in range(3):
    mask = y == cls
    ax.scatter(X_2d_sc[mask, 0], X_2d_sc[mask, 1],
               color=colors[cls], label=class_names[cls],
               edgecolor='white', linewidths=0.5, s=70, alpha=0.9)
    
ax.set_title('VISUALIZATION 6: KNN Decision Boundary\n(Petal Length vs Petal Width - Scaled)',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Petal Length (Scaled)', fontsize=10)
ax.set_ylabel('Petal Width (Scaled)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('viz6_decision_boundary.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[VIZ 6 SAVED] viz6_decision_boundary.png")


# Final Summary

print("\n" + "=" * 50)
print("   FINAL SUMMARY")
print("=" * 50)
print(f"\n Algorithm : KNN (K=5)")
print(f"Dataset   : Iris (150 samples, 4 features, 3 classes)")
print(f"Split     : 80% Train / 20% Test")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Optimal K : {best_k}")
print(f"\n 6 Visualizations saved to current directory.")
print(f"\n[PIPELINE COMPLETE] Data Classification with KNN - All steps executed successfully!")