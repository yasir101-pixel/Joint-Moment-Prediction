import os
import sys
import numpy as np
import argparse

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_prep import (load_maryam_dataset, loso_split,
                       encode_dataset_as_images, MOMENT_COLS)
from model_dnn import train_dnn
from model_tl import train_transfer_model
from evaluate import evaluate_model, print_results, save_results

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
IMU_ROOT    = '/scratch/yasir071/maryam_data/Maryam_Dataset/IMU_100Hz_AllSubjects'
MOM_ROOT    = '/scratch/yasir071/maryam_data/Maryam_Dataset/JointMoments'
OUTPUT_DIR  = '/scratch/yasir071/output/liew_replication'
MOMENT_NAMES = MOMENT_COLS


def run_loso(model_type='dnn'):
    """
    Run full Leave-One-Subject-Out cross-validation.
    model_type: 'dnn' or 'tl'
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load dataset
    print(f"\nLoading Maryam dataset...")
    dataset = load_maryam_dataset(IMU_ROOT, MOM_ROOT)
    n_subjects = len(dataset)
    print(f"Loaded {n_subjects} subjects.")

    all_results = []

    for test_idx in range(n_subjects):
        test_subj = dataset[test_idx][0]
        print(f"\n{'='*60}")
        print(f"LOSO Fold {test_idx+1}/{n_subjects} — Test subject: {test_subj}")
        print(f"{'='*60}")

        # Split data
        X_train, y_train, X_test, y_test, scaler = loso_split(
            dataset, test_subject_idx=test_idx
        )
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")

        # Encode as images for CNN/VGG16
        print("Encoding windows as images...")
        X_train_img = encode_dataset_as_images(X_train)
        X_test_img  = encode_dataset_as_images(X_test)
        print(f"Image shapes: train={X_train_img.shape}, test={X_test_img.shape}")

        # Use 10% of training as validation
        n_val = max(1, int(0.1 * len(X_train_img)))
        X_val_img = X_train_img[-n_val:]
        y_val     = y_train[-n_val:]
        X_tr_img  = X_train_img[:-n_val]
        y_tr      = y_train[:-n_val]

        n_outputs = y_train.shape[1]

        # Train selected model
        if model_type == 'dnn':
            print("\nTraining custom DNN (MLDNN)...")
            model, history = train_dnn(X_tr_img, y_tr, X_val_img, y_val,
                                       n_outputs=n_outputs)
        elif model_type == 'tl':
            print("\nTraining VGG16 Transfer Learning (MLTL)...")
            model, history = train_transfer_model(X_tr_img, y_tr,
                                                  X_val_img, y_val,
                                                  n_outputs=n_outputs)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Evaluate
        print("\nEvaluating...")
        y_pred = model.predict(X_test_img)
        results = evaluate_model(y_test, y_pred, moment_names=MOMENT_NAMES)
        print_results(results, model_name=f"{model_type.upper()} - {test_subj}")

        # Save per-fold results
        fold_csv = os.path.join(OUTPUT_DIR,
                                f"results_{model_type}_fold{test_idx+1}_{test_subj}.csv")
        save_results(results, model_name=model_type, output_path=fold_csv)
        all_results.append(results)

        # Save model
        model_path = os.path.join(OUTPUT_DIR,
                                  f"model_{model_type}_fold{test_idx+1}.keras")
        model.save(model_path)
        print(f"Model saved to {model_path}")

    # Aggregate results across all folds
    print(f"\n{'='*60}")
    print(f"FINAL AVERAGE ACROSS ALL {n_subjects} FOLDS")
    print(f"{'='*60}")
    avg_rmse    = np.mean([r['AVERAGE']['RMSE']     for r in all_results])
    avg_relrmse = np.mean([r['AVERAGE']['relRMSE']  for r in all_results])
    avg_r       = np.mean([r['AVERAGE']['pearson_r'] for r in all_results])
    print(f"Avg RMSE:    {avg_rmse:.4f} Nm/kg")
    print(f"Avg relRMSE: {avg_relrmse:.2f}%")
    print(f"Avg r:       {avg_r:.4f}")

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, f"summary_{model_type}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Model: {model_type.upper()}\n")
        f.write(f"Subjects: {n_subjects}\n")
        f.write(f"Avg RMSE:    {avg_rmse:.4f} Nm/kg\n")
        f.write(f"Avg relRMSE: {avg_relrmse:.2f}%\n")
        f.write(f"Avg r:       {avg_r:.4f}\n")
    print(f"Summary saved to {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='dnn',
                        choices=['dnn', 'tl'],
                        help='Model type: dnn or tl')
    args = parser.parse_args()
    run_loso(model_type=args.model)