import os, sys
from config import *  # Import everything from config.py
from volterra_decaying_prior import btnkm
# Parameters
memory = 100          # M (system memory)
order = 3             # number of nonlinear orders to model (1, 2, 3)
base_value = 1e-6

# Create index grid for [lag, order]
lags = np.arange(1, memory + 2)
orders = np.arange(1, order + 1)

# Volterra-inspired scaling: decay with lag, decay faster with order
alpha = 0.1  # temporal decay rate
beta = 0.5   # nonlinear order decay rate

# Build prior scaling
volterra_prior = np.exp(-alpha * lags[:, None]) * np.exp(-beta * (orders[None, :] - 1))

# Encode into g/h such that ratio matches the prior (NumPy arrays)
g_array = base_value * np.ones_like(volterra_prior)
h_array = base_value * np.ones_like(volterra_prior) / volterra_prior
check_ratio_array = g_array / h_array  # should equal volterra_prior

# --- Convert to lists where each list contains a column ---
g = [g_array[:, j].tolist() for j in range(order)]
h = [h_array[:, j].tolist() for j in range(order)]
check_ratio = [check_ratio_array[:, j].tolist() for j in range(order)]

# Plot example: g/h surface
plt.figure(figsize=(7,5))
for j in range(order):
    plt.plot(lags, check_ratio[j], label=f'Order {j+1}')
plt.title('Volterra-inspired g/h Ratio')
plt.xlabel('Lag index')
plt.ylabel('g/h value')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



# Load .mat file
data = scipy.io.loadmat('/Users/hakilic/Downloads/Tensor-Network-B-splines-master/Cascaded/dataBenchmark.mat')

# Extract and reshape
u_train, y_train = data['uEst'].squeeze()[:, None], data['yEst'].squeeze()
u_test, y_test   = data['uVal'].squeeze()[:, None], data['yVal'].squeeze()

# Normalize inputs to [0, 1]
X_min, X_max = u_train.min(axis=0), u_train.max(axis=0)
range_X = np.where(X_max - X_min == 0, 1, X_max - X_min)  # prevent div by zero
u_train = (u_train - X_min) / range_X
u_test  = (u_test  - X_min) / range_X

y_mean = y_train.mean()
y_std = y_train.std()
y_train = (y_train - y_mean) / y_std

# hyper-parameters
input_dimension = 100
max_rank = 20
Kernel_Degree = 3

a, b = 1e-3, 1e-3
c, d = 1e-5 * np.ones(max_rank), 1e-6 * np.ones(max_rank)
#g, h = 1e-6 * np.ones(input_dimension+1), 1e-6 * np.ones(input_dimension+1)

rmse_list, nll_list, total_time_list = [], [], []

for seed in range(1):
    print(f"\n===== Running Volterra BTN (seed={seed}) =====")
    model = btnkm(Kernel_Degree)
    
    R, W_D, lambda_M, lambda_R = model.train(
        features=u_train,
        target=y_train,
        input_dimension=input_dimension,
        Volterra_Degree=Kernel_Degree, 
        max_rank=max_rank,
        shape_parameter_tau=a,
        scale_parameter_tau=b,
        shape_parameter_lambda_R=c,
        scale_parameter_lambda_R=d,
        shape_parameter_lambda_M=g,
        scale_parameter_lambda_M=h,
        max_iter=50,
        seed=3,
        precision_update=True,
        lambda_R_update=True,
        lambda_M_update=True,
        plot_results=False, 
        prune_rank=True,
    )


    num_degrees = len(W_D)
    fig, axes = plt.subplots(1, num_degrees, figsize=(4*num_degrees, 4), sharey=True)

    # Handle the case where num_degrees == 1
    if num_degrees == 1:
        axes = [axes]

    for deg_idx, (W, ax) in enumerate(zip(W_D, axes)):
        R = W.shape[1]
        for r in range(R):
            ax.plot(
                W[:, r],
                linewidth=1.5,
                label=f'Comp {r+1}',
                alpha=0.8
            )
        ax.set_title(f'Degree {deg_idx+1}')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xlabel('Input Dimension Index')
        if deg_idx == 0:
            ax.set_ylabel('Weight Value')
        if deg_idx == num_degrees - 1:
            ax.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.suptitle('All R Components for Each W_D (seed=0)', y=1.05)
    plt.tight_layout()
    plt.figure(figsize=(8, 5))

    for j, col in enumerate(lambda_M):  # each column = one order
        plt.plot(np.log(col), marker='o', linestyle='-', alpha=0.8, label=f'Order {j+1}')

    plt.title('Ordered λ_M Values per Order')
    plt.xlabel('Lag Index')
    plt.ylabel('log(λ_M)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    prediction_mean, prediction_std, _ = model.predict(features=u_test, input_dimension=input_dimension)

    # Unscale predictions
    prediction_mean_unscaled = prediction_mean[input_dimension:] * y_std + y_mean
    prediction_std_unscaled  = prediction_std[input_dimension:] * y_std
    y_true_unscaled = y_test[input_dimension:]

    # Compute RMSE
    rmse = np.sqrt(np.mean((prediction_mean_unscaled - y_true_unscaled) ** 2))
    # Compute NLL
    nll = np.mean(
        0.5 * np.log(2 * np.pi * prediction_std_unscaled**2)
        + 0.5 * ((y_true_unscaled - prediction_mean_unscaled) ** 2) / (prediction_std_unscaled**2)
    )

    rmse_list.append(rmse)
    nll_list.append(nll)

    # Plot only for the first run
    if seed == 0:
        time_steps = np.arange(len(y_true_unscaled))
        plt.figure(figsize=(14, 6))
        plt.plot(time_steps, y_true_unscaled, color='black', linestyle='-', linewidth=1.8, label='Actual Output')
        plt.plot(time_steps, prediction_mean_unscaled, color='black', linestyle='--', linewidth=1.8, label='Volterra BTN Prediction')
        plt.fill_between(
            time_steps,
            prediction_mean_unscaled - 3 * prediction_std_unscaled,
            prediction_mean_unscaled + 3 * prediction_std_unscaled,
            color='gray', alpha=0.2, label='Prediction ±3 std'
        )
        plt.title('Volterra BTN (First Run)')
        plt.xlabel('Time Step')
        plt.ylabel('Output')
        plt.ylim([-2.5, 15])
        plt.legend(frameon=False)
        plt.tick_params(direction='in', length=4)
        plt.tight_layout()

        plt.savefig('/Users/hakilic/Desktop/BTN_Volterra_confidence_bounds.pdf', format='pdf', bbox_inches='tight')
        plt.show()


# Compute Mean & Std over 10 runs
rmse_mean, rmse_std = np.mean(rmse_list), np.std(rmse_list)
nll_mean, nll_std   = np.mean(nll_list), np.std(nll_list)
#time_mean, time_std = np.mean(total_time_list), np.std(total_time_list)

# Results Table
results = pd.DataFrame({
    'Method': ['Volterra BTN (10 runs)', 'BMALS'],
    'RMSE_mean': [rmse_mean, None],
    'RMSE_std': [rmse_std, None],
    'NLL_mean': [nll_mean, None],
    'NLL_std': [nll_std, None], 
    #'Total_Time_mean': [time_mean, None],
    #'Total_Time_std':[time_std, None]
})


# BMALS (BMVALS) Evaluation — Multiple Runs
# ============================================================
bmals_path = '/Users/hakilic/Desktop/IFAC/IFAC/BMVALS_all_runs.csv'
bmals_df = pd.read_csv(bmals_path)
bmals_df.columns = ['run_index', 'y_pred', 'variance']

y_true_bmals = y_test[input_dimension-1:]

rmse_bmals_list, nll_bmals_list = [], []

for run in np.unique(bmals_df['run_index']):
    run_data = bmals_df[bmals_df['run_index'] == run]
    y_pred_run = run_data['y_pred'].values
    var_run = run_data['variance'].values
    std_run = np.sqrt(var_run)

    # Compute metrics
    rmse_run = np.sqrt(np.mean((y_pred_run - y_true_bmals) ** 2))
    nll_run = np.mean(
        0.5 * np.log(2 * np.pi * std_run**2) +
        0.5 * ((y_true_bmals - y_pred_run) ** 2) / (std_run**2)
    )

    rmse_bmals_list.append(rmse_run)
    nll_bmals_list.append(nll_run)

    # Plot only first run (to match BTN)
    if run == 1:
        time_steps = np.arange(len(y_true_bmals))
        plt.figure(figsize=(14, 6))
        plt.plot(time_steps, y_true_bmals, color='black', linestyle='-', linewidth=1.8, label='Actual Output')
        plt.plot(time_steps, y_pred_run, color='black', linestyle='--', linewidth=1.8, label='BMALS Prediction')
        plt.fill_between(
            time_steps,
            y_pred_run - 3 * std_run,
            y_pred_run + 3 * std_run,
            color='gray', alpha=0.2, label='Prediction ±3 std'
        )
        plt.title('BMALS (First Run)')
        plt.xlabel('Time Step')
        plt.ylabel('Output')
        plt.ylim([-2.5, 15])
        plt.legend(frameon=False)
        plt.tick_params(direction='in', length=4)
        plt.tight_layout()

        plt.savefig('/Users/hakilic/Desktop/BMVAL_confidence_bounds.pdf', format='pdf', bbox_inches='tight')
        plt.show()

# Compute mean ± std for BMALS
rmse_bmals_mean, rmse_bmals_std = np.mean(rmse_bmals_list), np.std(rmse_bmals_list)
nll_bmals_mean, nll_bmals_std = np.mean(nll_bmals_list), np.std(nll_bmals_list)

# Update results table
results.loc[1, ['RMSE_mean', 'RMSE_std', 'NLL_mean', 'NLL_std']] = [
    rmse_bmals_mean, rmse_bmals_std, nll_bmals_mean, nll_bmals_std
]

# Round and print comparison
results = results.round(5)
print("\n================ Performance Comparison over 10 Runs ================\n")
print(results.to_string(index=False))



