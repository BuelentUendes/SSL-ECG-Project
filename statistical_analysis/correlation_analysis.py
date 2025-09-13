import os
import json

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from utils.torch_utilities import (
    split_indices_by_participant_groups,
    get_participant_cv_splitter,
)
from utils.helper_paths import DATA_PATH, RESULTS_PATH, SAVED_MODELS_PATH
from utils.torch_utilities import (
    load_processed_data,
)
from models.tstcc import (
    data_generator_from_arrays,
    Trainer,
    base_Model,
    TC,
    Config as ECGConfig,
    encode_representations,
    optimize_tstcc_hyperparameters,
)

from latentmi import lmi

def handle_missing_data(data, drop_values=True, verbose=True):
    """Handle missing values and infinity values in the data."""
    if isinstance(data, np.ndarray):
        # Reshape 3D arrays to 2D by flattening the last dimension
        if data.ndim == 3:
            original_shape = data.shape
            data_reshaped = data.reshape(data.shape[0], -1)
            df = pd.DataFrame(data_reshaped)
            was_3d = True
        else:
            df = pd.DataFrame(data)
            was_3d = False
        was_numpy = True
    else:
        df = data.copy()
        was_numpy = False
        was_3d = False

    original_data_len = len(df)

    # Identify rows and columns with infinity values
    inf_mask = df.isin([np.inf, -np.inf])
    rows_with_inf = inf_mask.any(axis=1)
    cols_with_inf = inf_mask.any(axis=0)

    if verbose:
        print(f"Rows with infinity values: {rows_with_inf.sum()}")
        print(f"Columns with infinity values:")
        for col in df.columns[cols_with_inf]:
            inf_count = inf_mask[col].sum()
            print(f"  - {col}: {inf_count} infinity values ({(inf_count / len(df)) * 100:.2f}%)")

    # Identify rows and columns with NaN values
    nan_mask = df.isna()
    rows_with_nan = nan_mask.any(axis=1)
    cols_with_nan = nan_mask.any(axis=0)

    if verbose:
        print(f"Rows with NaN values: {rows_with_nan.sum()}")
        print(f"Columns with NaN values:")
        for col in df.columns[cols_with_nan]:
            nan_count = nan_mask[col].sum()
            print(f"  - {col}: {nan_count} NaN values ({(nan_count / len(df)) * 100:.2f}%)")

    if drop_values:
        clean_data = df[~df.isin([np.inf, -np.inf]).any(axis=1)]
        clean_data = clean_data.dropna(axis=0)

        dropped_percent = ((original_data_len - len(clean_data)) / original_data_len) * 100
        if verbose:
            print(f"Dropping these rows removed {np.round(dropped_percent, 4)}% of the original data")

        if was_numpy:
            return clean_data.values
        else:
            return clean_data
    else:
        if was_numpy:
            return df.values
        else:
            return df


def main(
        fs: float=1000,
        window_size: int =30,
        step_size: int=10,
        seed: int=42,
        train_ratio_encoder: float = 0.75,
        model_name: str = "TSTCC",
        tcc_epochs: int=40,
        tcc_batch_size: int=128,
        tc_timesteps: int =70,
        tc_hidden_dim: int = 128,
        cc_temperature: float=0.07,
        cc_use_cosine: bool=False,
        use_s3_layers: bool=False,
        initial_num_segments: int=2,
        num_s3_layers: int=2,
        segment_multiplier: int=1,
        jitter_scale_ratio: float=0.001,
        jitter_ratio: float=0.001,
        max_segment: int=8,
):

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{0}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        # Important note:
        # For TSTCC the MPS is not supported due to some binary operation that does not work on MPS.
    else:
        device = torch.device("cpu")

    label_map = {"baseline": 0, "mental_stress": 1}
    window_data_path_features = os.path.join(
        DATA_PATH, "interim", "ECG_features", str(fs), str(window_size), str(step_size), 'windowed_data.h5'
    )

    X_features, y_features, groups, feature_names = load_processed_data(window_data_path_features, label_map=label_map, domain_features=True)
    y_features = y_features.astype(np.float32)

    print("=== Handling missing values (Drop missing values) ===")
    X_features_clean = handle_missing_data(X_features, drop_values=True, verbose=True)

    if len(X_features_clean) != len(X_features):
        print(f"Updating labels and groups after dropping {len(X_features) - len(X_features_clean)} samples")
        X_df_features = pd.DataFrame(X_features)
        valid_rows = ~(X_df_features.isin([np.inf, -np.inf]).any(axis=1) | X_df_features.isna().any(axis=1))
        y_features = y_features[valid_rows]
        groups = groups[valid_rows]
        X_features = X_features_clean

    # Raw ECG feature data path:
    window_size_tstcc = 30
    step_size_tstcc = 10

    window_data_path = os.path.join(
        DATA_PATH, "interim", "ECG", str(fs), str(window_size_tstcc), str(step_size_tstcc), 'windowed_data.h5'
    )

    # Load the ECG representation now (for now only the TSTCC)
    X, y, groups = load_processed_data(window_data_path, label_map=label_map)
    y = y.astype(np.float32)

    pretrain_data = "all_labels"


    model_save_path = os.path.join(
        SAVED_MODELS_PATH, "ECG", str(fs), model_name, pretrain_data, f"{seed}", str(window_size_tstcc),
        str(step_size_tstcc), str(train_ratio_encoder)
    )

    model_file_name = "tstcc.pt"

    # Check if we have a locally saved model and no forced retraining
    if os.path.exists(os.path.join(model_save_path, model_file_name)):
        print("We found a pretrained model. Load the pretrained weights")
        ckpt_path = os.path.join(model_save_path, model_file_name)

        # rebuild model
        cfg = ECGConfig(fs, window_size_tstcc)
        cfg.num_epoch = tcc_epochs
        cfg.batch_size = tcc_batch_size
        cfg.TC.timesteps = tc_timesteps
        cfg.TC.hidden_dim = tc_hidden_dim
        cfg.Context_Cont.temperature = cc_temperature
        cfg.Context_Cont.use_cosine_similarity = cc_use_cosine
        cfg.use_s3_layers = use_s3_layers
        cfg.initial_num_segments = initial_num_segments
        cfg.num_s3_layers = num_s3_layers
        cfg.segment_multiplier = segment_multiplier

        # Here we can set the augmentations (potentially optimized)
        cfg.augmentation.jitter_ratio = jitter_ratio
        cfg.augmentation.jitter_scale_ratio = jitter_scale_ratio
        cfg.augmentation.max_seg = max_segment

        model = base_Model(cfg).to(device)
        tc_head = TC(cfg, device).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["encoder"])
        tc_head.load_state_dict(state["tc_head"])

    # Set the model to eval mode
    model.eval()
    tc_head.eval()

    train_repr, _ = encode_representations(X, y, model, tc_head, tcc_batch_size, device)

    print(f"TSTCC embeddings shape: {train_repr.shape}")
    print(f"ECG features shape: {X_features.shape}")
    print(f"Number of feature names: {len(feature_names)}")
    
    # Ensure we have matching samples between features and embeddings
    min_samples = min(len(X_features), len(train_repr))
    X_features_matched = X_features[:min_samples]
    train_repr_matched = train_repr[:min_samples]

    # Do
    print(f"Using {min_samples} matched samples for correlation analysis")
    
    # Compute correlation matrix between features (55) and embeddings (32)
    correlation_matrix = np.zeros((len(feature_names), train_repr_matched.shape[1]))
    
    for i, feature_name in enumerate(feature_names):
        for j in range(train_repr_matched.shape[1]):
            corr_coef, _ = pearsonr(X_features_matched[:, i], train_repr_matched[:, j])
            correlation_matrix[i, j] = corr_coef
    
    # Create DataFrame for better visualization
    embedding_names = [f"Embedding_{i}" for i in range(train_repr_matched.shape[1])]
    corr_df = pd.DataFrame(
        correlation_matrix,
        index=feature_names,
        columns=embedding_names
    )
    
    print("\n=== Correlation Matrix Statistics ===")
    print(f"Correlation matrix shape: {correlation_matrix.shape}")
    print(f"Max absolute correlation: {np.abs(correlation_matrix).max():.4f}")
    print(f"Mean absolute correlation: {np.abs(correlation_matrix).mean():.4f}")
    
    # Find strongest correlations
    max_corr_idx = np.unravel_index(np.abs(correlation_matrix).argmax(), correlation_matrix.shape)
    max_corr_feature = feature_names[max_corr_idx[0]]
    max_corr_embedding = embedding_names[max_corr_idx[1]]
    max_corr_value = correlation_matrix[max_corr_idx]
    
    print(f"Strongest correlation: {max_corr_feature} <-> {max_corr_embedding} = {max_corr_value:.4f}")
    
    # Save correlation matrix
    results_path = os.path.join(RESULTS_PATH, "correlation_analysis")
    os.makedirs(results_path, exist_ok=True)
    
    # Save as CSV
    corr_df.to_csv(os.path.join(results_path, f"feature_embedding_correlation_matrix_feature_{window_size}_{step_size}.csv"))
    
    # Create and save heatmap
    plt.figure(figsize=(12, 16))
    sns.heatmap(corr_df, 
                cmap='RdBu_r', 
                center=0, 
                annot=False, 
                fmt='.2f',
                xticklabels=True,
                yticklabels=True,
                cbar_kws={'label': 'Pearson Correlation Coefficient'})
    plt.title('Correlation between ECG Features and TSTCC Embeddings', fontsize=14)
    plt.xlabel('TSTCC Embedding Dimensions', fontsize=12)
    plt.ylabel('ECG Features', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f"feature_embedding_correlation_heatmap_feature_{window_size}_{step_size}.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Find top correlations for each embedding dimension
    print("\n=== Top 3 Features per Embedding Dimension ===")
    for j in range(min(5, train_repr_matched.shape[1])):  # Show first 5 dimensions
        abs_corrs = np.abs(correlation_matrix[:, j])
        top_indices = np.argsort(abs_corrs)[-3:][::-1]
        print(f"\nEmbedding_{j}:")
        for idx in top_indices:
            feature_name = feature_names[idx]
            corr_val = correlation_matrix[idx, j]
            print(f"  {feature_name}: {corr_val:.4f}")
    
    print(f"\nCorrelation analysis complete. Results saved to: {results_path}")
    
    # ===============================================
    # LINEAR PROBING: PREDICTING ECG FEATURES FROM EMBEDDINGS
    # ===============================================
    print("\n" + "="*60)
    print("LINEAR PROBING ANALYSIS")
    print("="*60)
    
    def linear_probe_ecg_features_participant_based(X_emb, y_feature, groups, feature_name, train_ratio=0.8, cv_folds=5, seed=42):
        """
        Perform linear probing to predict an ECG feature from embeddings using participant-based train/test split.
        
        Parameters:
        - X_emb: embeddings (n_samples, n_embedding_dims)
        - y_feature: target ECG feature values (n_samples,)
        - groups: group identifiers for samples (n_samples,)
        - feature_name: name of the ECG feature being predicted
        - train_ratio: ratio for train/test split by participants
        - cv_folds: number of cross-validation folds
        - seed: random seed for reproducibility
        
        Returns:
        - test_r2: test set R² score 
        - cv_r2: cross-validated R² score on training set
        - best_alpha: optimal regularization parameter
        """
        # Step 1: Split by participant to get train/test split
        train_idx, train_p, test_idx, test_p = split_indices_by_participant_groups(
            groups,
            train_ratio=train_ratio,
            label_fraction=1.0,  # Use all data for regression
            seed=seed
        )
        
        X_train_all = X_emb[train_idx]
        y_train_all = y_feature[train_idx]
        groups_train_all = groups[train_idx]
        
        X_test = X_emb[test_idx]
        y_test = y_feature[test_idx]
        
        print(f"  Training samples: {X_train_all.shape[0]}, Test samples: {X_test.shape[0]}")
        print(f"  Training participants: {len(np.unique(groups_train_all))}, Test participants: {len(np.unique(groups[test_idx]))}")
        
        # Step 2: Set up participant-based cross-validation on training data (for regression)
        unique_participants = np.unique(groups_train_all)
        n_participants = len(unique_participants)
        
        print(f"Total participants: {n_participants}")
        
        if n_participants < 3:
            print(f"    Not enough participants for CV (need ≥3, have {n_participants}), using simple train/test evaluation")
            cv_splitter = None
            n_splits = None
        else:
            from sklearn.model_selection import GroupKFold
            cv_splitter = GroupKFold(n_splits=min(cv_folds, n_participants))
            n_splits = cv_splitter.n_splits
            print(f"Using {n_splits}-Fold Group CV ({n_splits} splits)")
            if n_participants % n_splits != 0:
                print(f"Note: {n_participants} participants don't divide evenly by {n_splits}")
                print("Some folds will have different numbers of participants")
        
        if cv_splitter is None:
            print(f"    Not enough participants for CV, using simple train/test evaluation")
            # Just do train/test evaluation
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_all)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Ridge with default alpha
            ridge = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=3)
            ridge.fit(X_train_scaled, y_train_all)
            
            y_pred = ridge.predict(X_test_scaled)
            test_r2 = r2_score(y_test, y_pred)
            
            return test_r2, test_r2, ridge.alpha_, y_pred, y_test
        
        # Step 3: Perform cross-validation on training data for hyperparameter tuning
        alphas = np.logspace(-3, 3, 50)
        cv_scores = []
        alpha_values = []
        
        print(f"  Performing {n_splits}-fold participant-based CV for {feature_name}")
        
        for fold_idx, (cv_train_idx, cv_val_idx) in enumerate(cv_splitter.split(X_train_all, y_train_all, groups_train_all)):
            # Split CV data
            X_cv_train, X_cv_val = X_train_all[cv_train_idx], X_train_all[cv_val_idx]
            y_cv_train, y_cv_val = y_train_all[cv_train_idx], y_train_all[cv_val_idx]
            
            # Standardize
            scaler = StandardScaler()
            X_cv_train_scaled = scaler.fit_transform(X_cv_train)
            X_cv_val_scaled = scaler.transform(X_cv_val)
            
            # Train Ridge regression with built-in CV for alpha selection
            ridge_cv = RidgeCV(alphas=alphas, cv=3)
            ridge_cv.fit(X_cv_train_scaled, y_cv_train)
            
            # Predict on validation set
            y_cv_pred = ridge_cv.predict(X_cv_val_scaled)
            fold_r2 = r2_score(y_cv_val, y_cv_pred)
            
            cv_scores.append(fold_r2)
            alpha_values.append(ridge_cv.alpha_)
            
            print(f"    Fold {fold_idx + 1}: R² = {fold_r2:.4f}, α = {ridge_cv.alpha_:.4f}")
        
        # Step 4: Train final model on all training data and evaluate on test set
        cv_r2 = np.mean(cv_scores)
        best_alpha = np.mean(alpha_values)
        
        # Final training and test evaluation
        scaler_final = StandardScaler()
        X_train_scaled_final = scaler_final.fit_transform(X_train_all)
        X_test_scaled_final = scaler_final.transform(X_test)
        
        # Use the best alpha from CV
        ridge_final = RidgeCV(alphas=[best_alpha], cv=3)
        ridge_final.fit(X_train_scaled_final, y_train_all)
        
        y_test_pred = ridge_final.predict(X_test_scaled_final)
        test_r2 = r2_score(y_test, y_test_pred)
        
        print(f"  CV R² = {cv_r2:.4f}, Test R² = {test_r2:.4f}, Best α = {best_alpha:.4f}")
        
        return test_r2, cv_r2, best_alpha, y_test_pred, y_test
    
    # Ensure we have the same groups for both embeddings and features
    groups_matched = groups[:min_samples]
    
    # Perform linear probing for each ECG feature
    linear_probing_results = {}
    
    print(f"Performing linear probing for {len(feature_names)} ECG features...")
    print(f"Data shape: Embeddings {train_repr_matched.shape}, Features {X_features_matched.shape}")
    print(f"Groups shape: {groups_matched.shape}, Unique groups: {len(np.unique(groups_matched))}")
    
    for i, feature_name in enumerate(feature_names):
        print(f"\n[{i+1}/{len(feature_names)}] Processing {feature_name}")
        
        y_target = X_features_matched[:, i]
        
        # Check for any remaining invalid values
        valid_mask = np.isfinite(y_target)
        if not np.all(valid_mask):
            print(f"  Warning: {(~valid_mask).sum()} invalid values found, skipping this feature")
            continue
            
        try:
            test_r2, cv_r2, best_alpha, predictions, true_values = linear_probe_ecg_features_participant_based(
                train_repr_matched, y_target, groups_matched, feature_name, 
                train_ratio=0.8, cv_folds=5, seed=seed
            )
            
            linear_probing_results[feature_name] = {
                'test_r2': test_r2,
                'cv_r2': cv_r2,
                'best_alpha': best_alpha,
                'predictions': predictions,
                'true_values': true_values
            }
            
            print(f"  Test R² = {test_r2:.4f}, CV R² = {cv_r2:.4f}, α = {best_alpha:.4f}")
            
        except Exception as e:
            print(f"  Error processing {feature_name}: {e}")
            continue
    
    # Analyze and visualize linear probing results
    print(f"\n=== Linear Probing Results Summary ===")
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Feature': list(linear_probing_results.keys()),
        'Test_R2': [results['test_r2'] for results in linear_probing_results.values()],
        'CV_R2': [results['cv_r2'] for results in linear_probing_results.values()],
        'Best_Alpha': [results['best_alpha'] for results in linear_probing_results.values()]
    })
    
    # Sort by test R² score
    results_df = results_df.sort_values('Test_R2', ascending=False)
    
    print(f"Successfully analyzed {len(results_df)} features")
    print(f"Test R² statistics:")
    print(f"  Mean Test R²: {results_df['Test_R2'].mean():.4f}")
    print(f"  Median Test R²: {results_df['Test_R2'].median():.4f}")
    print(f"  Max Test R²: {results_df['Test_R2'].max():.4f}")
    print(f"  Features with Test R² > 0.1: {(results_df['Test_R2'] > 0.1).sum()}")
    print(f"  Features with Test R² > 0.3: {(results_df['Test_R2'] > 0.3).sum()}")
    
    print(f"CV R² statistics:")
    print(f"  Mean CV R²: {results_df['CV_R2'].mean():.4f}")
    print(f"  Median CV R²: {results_df['CV_R2'].median():.4f}")
    print(f"  Max CV R²: {results_df['CV_R2'].max():.4f}")
    print(f"  Features with CV R² > 0.1: {(results_df['CV_R2'] > 0.1).sum()}")
    print(f"  Features with CV R² > 0.3: {(results_df['CV_R2'] > 0.3).sum()}")
    
    # Show top 10 most predictable features
    print(f"\nTop 10 Most Predictable ECG Features (by Test R²):")
    for idx, row in results_df.head(10).iterrows():
        print(f"  {row['Feature']}: Test R² = {row['Test_R2']:.4f}, CV R² = {row['CV_R2']:.4f}")
    
    # Save results
    results_df.to_csv(os.path.join(results_path, f"linear_probing_results_{window_size}_{step_size}.csv"), index=False)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: R² distribution
    plt.subplot(2, 2, 1)
    plt.hist(results_df['CV_R2'], bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(results_df['CV_R2'].mean(), color='red', linestyle='--', label=f'Mean: {results_df["CV_R2"].mean():.3f}')
    plt.axvline(0.1, color='orange', linestyle='--', label='R² = 0.1')
    plt.axvline(0.3, color='green', linestyle='--', label='R² = 0.3')
    plt.xlabel('Cross-Validated R²')
    plt.ylabel('Number of Features')
    plt.title('Distribution of Linear Probing R² Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Top features
    plt.subplot(2, 2, 2)
    top_features = results_df.head(15)
    bars = plt.barh(range(len(top_features)), top_features['CV_R2'])
    plt.yticks(range(len(top_features)), top_features['Feature'], fontsize=8)
    plt.xlabel('Cross-Validated R²')
    plt.title('Top 15 Most Predictable Features')
    plt.grid(True, alpha=0.3)
    
    # Color bars based on R² value
    for i, (bar, r2) in enumerate(zip(bars, top_features['CV_R2'])):
        if r2 > 0.3:
            bar.set_color('green')
        elif r2 > 0.1:
            bar.set_color('orange')
        else:
            bar.set_color('gray')
    
    # Plot 3: Regularization parameter distribution
    plt.subplot(2, 2, 3)
    plt.hist(np.log10(results_df['Best_Alpha']), bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('log₁₀(Best Alpha)')
    plt.ylabel('Number of Features')
    plt.title('Distribution of Optimal Regularization Parameters')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Scatter plot of example predictions vs true values for best feature
    plt.subplot(2, 2, 4)
    best_feature = results_df.iloc[0]['Feature']
    best_result = linear_probing_results[best_feature]
    
    plt.scatter(best_result['true_values'], best_result['predictions'], alpha=0.6)
    
    # Add perfect prediction line
    min_val = min(min(best_result['true_values']), min(best_result['predictions']))
    max_val = max(max(best_result['true_values']), max(best_result['predictions']))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predictions vs True Values\n{best_feature} (R² = {results_df.iloc[0]["CV_R2"]:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f"linear_probing_analysis_{window_size}_{step_size}.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nLinear probing analysis complete. Results saved to: {results_path}")
    
    # ===============================================
    # CANONICAL CORRELATION ANALYSIS (CCA)
    # ===============================================
    # print("\n" + "="*60)
    # print("CANONICAL CORRELATION ANALYSIS")
    # print("="*60)
    #
    # # Standardize the data for CCA
    # scaler_features = StandardScaler()
    # scaler_embeddings = StandardScaler()
    #
    # X_features_scaled = scaler_features.fit_transform(X_features_matched)
    # train_repr_scaled = scaler_embeddings.fit_transform(train_repr_matched)
    #
    # # Determine number of canonical components (min of dimensions)
    # n_components = min(X_features_matched.shape[1], train_repr_matched.shape[1])
    # print(f"Computing {n_components} canonical components")
    #
    # # Perform CCA
    # cca = CCA(n_components=n_components)
    # cca.fit(X_features_scaled, train_repr_scaled)
    #
    # # Transform data to canonical space
    # X_c, Y_c = cca.transform(X_features_scaled, train_repr_scaled)
    #
    # # Calculate canonical correlations
    # canonical_correlations = []
    # for i in range(n_components):
    #     corr = np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
    #     canonical_correlations.append(corr)
    #
    # canonical_correlations = np.array(canonical_correlations)
    #
    # print(f"\nCanonical Correlations:")
    # for i, corr in enumerate(canonical_correlations[:10]):  # Show first 10
    #     print(f"  Component {i+1}: {corr:.4f}")
    #
    # print(f"\nCanonical Correlation Statistics:")
    # print(f"  Maximum canonical correlation: {canonical_correlations.max():.4f}")
    # print(f"  Mean canonical correlation: {canonical_correlations.mean():.4f}")
    # print(f"  Number of components > 0.5: {(canonical_correlations > 0.5).sum()}")
    # print(f"  Number of components > 0.7: {(canonical_correlations > 0.7).sum()}")
    #
    # # Get canonical weights (loadings)
    # feature_weights = cca.x_weights_  # Weights for features
    # embedding_weights = cca.y_weights_  # Weights for embeddings
    #
    # # Create DataFrames for canonical weights
    # feature_weights_df = pd.DataFrame(
    #     feature_weights,
    #     index=feature_names,
    #     columns=[f"CanonComp_{i+1}" for i in range(n_components)]
    # )
    #
    # embedding_weights_df = pd.DataFrame(
    #     embedding_weights,
    #     index=[f"Embedding_{i}" for i in range(train_repr_matched.shape[1])],
    #     columns=[f"CanonComp_{i+1}" for i in range(n_components)]
    # )
    #
    # # Save standardized data for R CCA analysis
    # print("\n=== Saving Standardized Data for R Analysis ===")
    #
    # # Save standardized ECG features
    # ecg_features_df = pd.DataFrame(
    #     X_features_scaled,
    #     columns=feature_names
    # )
    # ecg_features_file = os.path.join(results_path, f"standardized_ecg_features_feature_{window_size}_{step_size}.csv")
    # ecg_features_df.to_csv(ecg_features_file, index=False)
    # print(f"Saved standardized ECG features ({ecg_features_df.shape}) to: {ecg_features_file}")
    #
    # # Save standardized TSTCC embeddings
    # tstcc_embeddings_df = pd.DataFrame(
    #     train_repr_scaled,
    #     columns=[f"Embedding_{i}" for i in range(train_repr_scaled.shape[1])]
    # )
    # tstcc_embeddings_file = os.path.join(results_path, f"standardized_tstcc_embeddings_feature_{window_size}_{step_size}.csv")
    # tstcc_embeddings_df.to_csv(tstcc_embeddings_file, index=False)
    # print(f"Saved standardized TSTCC embeddings ({tstcc_embeddings_df.shape}) to: {tstcc_embeddings_file}")
    #
    # # Important: To analyze the script:
    # # https://stats.oarc.ucla.edu/r/dae/canonical-correlation-analysis/
    #
    # # Save CCA results
    # pd.DataFrame(canonical_correlations,
    #              columns=['Canonical_Correlation']).to_csv(
    #     os.path.join(results_path, f"canonical_correlations_feature_{window_size}_{step_size}.csv")
    # )
    #
    # feature_weights_df.to_csv(
    #     os.path.join(results_path, f"cca_feature_weights_feature_{window_size}_{step_size}.csv")
    # )
    #
    # embedding_weights_df.to_csv(
    #     os.path.join(results_path, f"cca_embedding_weights_feature_{window_size}_{step_size}.csv")
    # )
    #
    # # Visualize canonical correlations
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(range(1, len(canonical_correlations) + 1), canonical_correlations, 'bo-')
    # plt.xlabel('Canonical Component')
    # plt.ylabel('Canonical Correlation')
    # plt.title('Canonical Correlations')
    # plt.grid(True, alpha=0.3)
    # plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='0.5 threshold')
    # plt.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='0.7 threshold')
    # plt.legend()
    #
    # # Cumulative variance explained approximation
    # plt.subplot(1, 2, 2)
    # squared_corrs = canonical_correlations**2
    # cumsum_squared = np.cumsum(squared_corrs) / np.sum(squared_corrs)
    # plt.plot(range(1, len(cumsum_squared) + 1), cumsum_squared, 'go-')
    # plt.xlabel('Canonical Component')
    # plt.ylabel('Cumulative Squared Correlation Ratio')
    # plt.title('Cumulative Canonical Variance')
    # plt.grid(True, alpha=0.3)
    #
    # plt.tight_layout()
    # plt.savefig(os.path.join(results_path, f"canonical_correlation_analysis_feature_{window_size}_{step_size}.png"),
    #             dpi=300, bbox_inches='tight')
    # plt.close()
    #
    # # Analyze most important features for first few canonical components
    # print(f"\n=== Top 5 Most Important Features for First 3 Canonical Components ===")
    # for comp_idx in range(min(5, n_components)):
    #     feature_weights_comp = np.abs(feature_weights[:, comp_idx])
    #     top_feature_indices = np.argsort(feature_weights_comp)[-5:][::-1]
    #
    #     embedding_weights_comp = np.abs(embedding_weights[:, comp_idx])
    #     top_embedding_indices = np.argsort(embedding_weights_comp)[-5:][::-1]
    #
    #     print(f"\nCanonical Component {comp_idx + 1} (correlation = {canonical_correlations[comp_idx]:.4f}):")
    #     print("  Top 5 Contributing Features:")
    #     for idx in top_feature_indices:
    #         weight = feature_weights[idx, comp_idx]
    #         print(f"    {feature_names[idx]}: {weight:.4f}")
    #
    #     print("  Top 5 Contributing Embeddings:")
    #     for idx in top_embedding_indices:
    #         weight = embedding_weights[idx, comp_idx]
    #         print(f"    Embedding_{idx}: {weight:.4f}")
    #
    # # Create heatmap for first few canonical components
    # n_show = min(10, n_components)
    # plt.figure(figsize=(14, 8))
    #
    # plt.subplot(1, 2, 1)
    # sns.heatmap(feature_weights_df.iloc[:, :n_show],
    #             cmap='RdBu_r', center=0,
    #             xticklabels=True, yticklabels=True,
    #             cbar_kws={'label': 'Feature Weight'})
    # plt.title(f'Feature Weights for First {n_show} Canonical Components')
    # plt.xlabel('Canonical Components')
    # plt.ylabel('ECG Features')
    # plt.yticks(rotation=0, fontsize=6)
    #
    # plt.subplot(1, 2, 2)
    # sns.heatmap(embedding_weights_df.iloc[:, :n_show],
    #             cmap='RdBu_r', center=0,
    #             xticklabels=True, yticklabels=True,
    #             cbar_kws={'label': 'Embedding Weight'})
    # plt.title(f'Embedding Weights for First {n_show} Canonical Components')
    # plt.xlabel('Canonical Components')
    # plt.ylabel('TSTCC Embeddings')
    #
    # plt.tight_layout()
    # plt.savefig(os.path.join(results_path, f"canonical_weights_heatmap_feature_{window_size}_{step_size}.png"),
    #             dpi=300, bbox_inches='tight')
    # plt.close()
    #
    # print(f"\nCanonical Correlation Analysis complete. All results saved to: {results_path}")

if __name__ == "__main__":
    main()









