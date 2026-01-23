import os
import matplotlib.pyplot as plt


def plot_training_curves(out_dir: str, train_losses, val_losses, train_accs, val_accs):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(train_accs, label="train_acc")
    plt.plot(val_accs, label="val_acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig(os.path.join(out_dir, "acc_curve.png"))
    plt.close()


def plot_roc(out_path: str, fpr, tpr):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.savefig(out_path)
    plt.close()


def plot_histogram(out_path: str, known_scores, unknown_scores):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.hist(known_scores, bins=30, alpha=0.6, label="known")
    plt.hist(unknown_scores, bins=30, alpha=0.6, label="unknown")
    plt.legend()
    plt.xlabel("score")
    plt.ylabel("count")
    plt.savefig(out_path)
    plt.close()
