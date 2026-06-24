import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def save_performance_report(y_true, y_pred, project_name):
    """
    Generates and saves a confusion matrix and detailed text report.
    """
    # 1. Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Clean', 'Buggy'], 
                yticklabels=['Clean', 'Buggy'])
    plt.title(f"Confusion Matrix: {project_name.upper()}")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f"results_1/{project_name}_confusion_matrix.png")
    plt.close()

    # 2. Textual Classification Report (Precision, Recall, F1)
    report = classification_report(y_true, y_pred, target_names=['Clean', 'Buggy'])
    with open(f"results_1/{project_name}_report.txt", "w") as f:
        f.write(f"--- Performance Report: {project_name.upper()} ---\n")
        f.write(report)
    
    print(f"✅ Metrics report and Confusion Matrix saved to results/")