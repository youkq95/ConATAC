import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix
import seaborn as sns


def load_data(prob_path):
    """加载概率文件和标签文件"""
    label_path = prob_path.replace('probs', 'labels')
    y_prob = np.load(prob_path)
    y_true = np.load(label_path)
    # 如果输出多列概率，取正类的概率
    if y_prob.ndim > 1 and y_prob.shape[1] > 1:
        y_prob = y_prob[:, 1]
    return y_true, y_prob


def plot_roc(y_true, y_prob, color, label_text, filename):
    """计算并绘制 ROC 曲线"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 5), dpi=300)
    plt.plot(fpr, tpr, color=color, lw=2, label=f'{label_text} (AUC = {auc_value:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title('ROC Curve', fontsize=20)
    plt.legend(loc="lower right", fontsize=12, frameon=False)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()


def plot_pr(y_true, y_prob, color, label_text, filename):
    """计算并绘制 PR 曲线"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    avg_precision = np.mean(precision)
    plt.figure(figsize=(6, 5), dpi=300)
    plt.plot(recall, precision, color=color, lw=2, label=f'{label_text} (AP = {avg_precision:.2f})')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall', fontsize=18)
    plt.ylabel('Precision', fontsize=18)
    plt.title('PR Curve', fontsize=20)
    plt.legend(loc="lower left", fontsize=12, frameon=False)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_prob, filename):
    """绘制阈值下的混淆矩阵"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    y_pred = (y_prob >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix\n (Threshold = {optimal_threshold:.2f})', fontsize=20)
    plt.xlabel('Predicted', fontsize=18)
    plt.ylabel('Actual', fontsize=18)
    plt.tight_layout()
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()


def main(cell_line, mode, bw_type):
    base_dir = f'{cell_line}/{mode}/traindata_{bw_type}/'

    if not os.path.exists(base_dir + 'picture'):
        os.makedirs(base_dir + 'picture', exist_ok=True)

    prob_file_old = f'{base_dir}predict_test/test_probs.npy'
    y_true_old, y_prob_old = load_data(prob_file_old)

    plot_roc(y_true_old, y_prob_old, color='purple', label_text=f'{cell_line} {mode}',
             filename=f'{base_dir}/picture/roc.pdf')
    plot_pr(y_true_old, y_prob_old, color='purple', label_text=f'{cell_line} {mode}',
            filename=f'{base_dir}/picture/pr.pdf')
    plot_confusion_matrix(y_true_old, y_prob_old, filename=f'{base_dir}/picture/confusion_matrix.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process input parameters.')
    parser.add_argument('--cell_line', type=str, required=True, help='The cell line name, e.g., GM12878')
    parser.add_argument('--mode', type=str, required=True, help='The mode, e.g., P')
    parser.add_argument('--bw_type', type=str, required=True, help='The BW type, e.g., sc')

    args = parser.parse_args()
    main(args.cell_line, args.mode, args.bw_type)