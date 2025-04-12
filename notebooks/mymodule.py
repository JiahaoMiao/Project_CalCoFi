# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gsw  # Oceanographic calculations (used for density calculations)
from matplotlib.colors import LogNorm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import PolynomialCountSketch
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


# --------------------------
# Data Preparation Functions
# --------------------------
def prepare_data(df, features, targets, test_size=0.2, random_state=42):
    """Prepare data for modeling. Returns X_train, X_test, y_train, y_test"""
    X = df[features]
    y = df[targets] 
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Add intercept term
    X_train_scaled = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]
    X_test_scaled = np.c_[np.ones(X_test_scaled.shape[0]), X_test_scaled]

    return X_train_scaled, X_test_scaled, y_train, y_test

# ----------------------
# Visualization Functions
# ----------------------
def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics for both temperature and salinity.
    Returns a dictionary with r2, rmse, and mae for each target variable.
    """
    # Ensure the metrics are calculated independently for each target
    metrics = {}
    
    # Calculate metrics for each target (T_degC and Salnty)
    for target_idx, target_name in enumerate(y_true.columns):
        metrics[f'r2_{target_name}'] = r2_score(y_true[target_name], y_pred[:, target_idx])
        metrics[f'rmse_{target_name}'] = np.sqrt(mean_squared_error(y_true[target_name], y_pred[:, target_idx]))
        metrics[f'mae_{target_name}'] = mean_absolute_error(y_true[target_name], y_pred[:, target_idx])
    
    return metrics

def create_composite_limits(*arrays):
    """Create common axis limits for multiple arrays."""
    all_values = np.concatenate(arrays)
    return [all_values.min(), all_values.max()]

# ----------------------
# Visualization Functions
# ----------------------
def plot_density_scatter(ax, y_true, y_pred, metrics, lims, title, target_name):
    """Plot 2D density scatter plot with target-specific metrics and labels."""
    units = {
        "T_degC": "°C",
        "Salnty": "PSU"
    }
    
    h = ax.hist2d(y_true, y_pred, bins=(400, 400), 
                 cmap='coolwarm', norm=LogNorm(vmin=1, vmax=1000),
                 cmin=1, zorder=0)
    
    ax.plot(lims, lims, '--', c='blue', lw=1, label='Perfect fit')
    ax.set(
        xlabel=f'True {target_name} ({units.get(target_name, "")})',
        ylabel=f'Predicted {target_name} ({units.get(target_name, "")})',
        xlim=lims, ylim=lims, 
        title=f'{title} - {target_name}'
    )
    
    plt.colorbar(h[3], ax=ax, label='Data Density (log scale)')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left')
    
    # Extract metrics for this specific target
    target_metrics = {
        'r2': metrics[f'r2_{target_name}'],
        'rmse': metrics[f'rmse_{target_name}'],
        'mae': metrics[f'mae_{target_name}']
    }
    
    textstr = '\n'.join((
        f'R² = {target_metrics["r2"]:.4f}',
        f'RMSE = {target_metrics["rmse"]:.4f}',
        f'MAE = {target_metrics["mae"]:.4f}'))
    ax.text(0.05, 0.90, textstr, transform=ax.transAxes,
           verticalalignment='top', bbox=dict(alpha=0.5))


def plot_residuals_hist(ax, residuals, title, target_name):
    """Plot residuals histogram with target-specific units."""
    units = {
        "T_degC": "°C",
        "Salnty": "PSU"
    }
    
    mean_residuals = np.mean(residuals)
    std_residuals = np.std(residuals)

    residual_range = [np.min(residuals)*0.2, np.max(residuals)*0.2]

    _, _, patches = ax.hist(residuals, bins=100,range=residual_range, 
                          color='blue', alpha=0.7, edgecolor='black')
    
    ax.axvline(0, color='red', linestyle='--', linewidth=1)
    ax.set(
        xlabel=f'Residuals ({units.get(target_name, "")})',
        ylabel='Frequency',
        title=f'{title} - {target_name}'
    )
    ax.grid(alpha=0.3)
    
    legend_text = f'Mean: {mean_residuals:.4f}\nStd Dev: {std_residuals:.4f}'
    ax.legend([patches[0]], [legend_text], loc='upper left')

def create_prediction_figure(y_train, y_pred_train, y_test, y_pred_test, targets):
    """Create separate diagnostic figures for each target variable."""
    for target in targets:
        fig = plt.figure(figsize=(12, 10), dpi=150)
        
        # Get target-specific data
        train_true = y_train[target]
        train_pred = y_pred_train[:, targets.index(target)]
        test_true = y_test[target]
        test_pred = y_pred_test[:, targets.index(target)]
        
        # Create common limits
        lims = create_composite_limits(train_true, test_true, train_pred, test_pred)
        
        # # Calculate metrics
        test_metrics = calculate_metrics(y_test, y_pred_test)
        train_metrics = calculate_metrics(y_train, y_pred_train)
        
        # Calculate residuals
        res_train = train_true - train_pred
        res_test = test_true - test_pred
        
        # Create subplots
        ax1 = fig.add_subplot(2, 2, 1)
        plot_density_scatter(ax1, train_true, train_pred, train_metrics, lims, 'Train Set', target)
        
        ax2 = fig.add_subplot(2, 2, 2)
        plot_residuals_hist(ax2, res_train, 'Train Set Residuals', target)
        
        ax3 = fig.add_subplot(2, 2, 3)
        plot_density_scatter(ax3, test_true, test_pred, test_metrics, lims, 'Test Set', target)
        
        ax4 = fig.add_subplot(2, 2, 4)
        plot_residuals_hist(ax4, res_test, 'Test Set Residuals', target)
        
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        fig.suptitle(f'Prediction Diagnostics - {target}', y=1.02, fontsize=14)
        plt.tight_layout()
        plt.show()

# ----------------------
# Analysis feature removal
# ----------------------
def analyze_feature_cases(df, original_features, cases, targets):
    """Run analysis for all feature cases"""
    results = []
    
    for case_name, excluded in cases:
        current_features = [f for f in original_features if f not in excluded]
        
        # Prepare data
        X_train, X_test, y_train, y_test = prepare_data(df, current_features, targets)
        
        # # Train model using normal equation
        beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train.values
        
        # # Predictions
        y_pred_train = X_train @ beta
        y_pred_test = X_test @ beta
    
        # Calculate metrics
        train_metrics = calculate_metrics(y_train, y_pred_train)
        test_metrics = calculate_metrics(y_test, y_pred_test)
     
        # Store results with clear naming
        results.append({
            'Case': case_name,
            'Features Used': len(current_features),
            'Temp Train R²': train_metrics['r2_T_degC'],
            'Temp Test R²': test_metrics['r2_T_degC'],
            'Temp Test RMSE': test_metrics['rmse_T_degC'],
            'Sal Train R²': train_metrics['r2_Salnty'],
            'Sal Test R²': test_metrics['r2_Salnty'],
            'Sal Test RMSE': test_metrics['rmse_Salnty'],
        })
    
    return pd.DataFrame(results)

# Gaussian (RBF) Kernel
def rbf_kernel(X, Y, gamma=0.1):
    X_norm = np.sum(X**2, axis=1)[:, np.newaxis]
    Y_norm = np.sum(Y**2, axis=1)[np.newaxis, :]
    dist_sq = X_norm + Y_norm - 2 * np.dot(X, Y.T)
    return np.exp(-gamma * dist_sq)

# Polynomial Kernel
def polynomial_kernel(X, Y, degree=2, gamma=1, coef0=1):
    return (gamma* np.dot(X, Y.T) + coef0) ** degree

def kernel_train_predict(X_train, y_train, X_test, kernel_function, n_components=1000, **kernel_params):
    """Train/predict using Nyström-approximated kernel."""
    # Approximate kernel matrix using landmarks
    nystroem = Nystroem(
        kernel=kernel_function,
        n_components=min(n_components, X_train.shape[0]),
        random_state=42,
        n_jobs=-1,
        **kernel_params
    )
    
    # Transform data into approximate kernel space
    X_train_transformed = nystroem.fit_transform(X_train)
    X_test_transformed = nystroem.transform(X_test)
    
    # Train linear regression on transformed data
    model = LinearRegression()
    model.fit(X_train_transformed, y_train)
    
    # Predict
    y_pred_test = model.predict(X_test_transformed)
    y_pred_train = model.predict(X_train_transformed)
    
    return y_pred_test, y_pred_train