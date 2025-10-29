from config import *  # Import everything from config.py
from volterra_BTN import btnkm

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
g, h = 1e-6 * np.ones(input_dimension+1), 1e-6 * np.ones(input_dimension+1)

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
        precision_update=True,
        lambda_R_update=True,
        lambda_M_update=True,
        plot_results=True,
        prune_rank=True,
    )


    # Predict (mse is returned by the predict function)
prediction_mean, prediction_std, _ = model.predict(
    features=u_test, input_dimension=input_dimension
)



plt.figure(figsize=(8, 5))
plt.plot(np.log(lambda_M), marker='o', linestyle='-', alpha=0.8)
plt.title('Ordered λ_M Values')
plt.xlabel('Sample Index (sorted)')
plt.ylabel('λ_M Value')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

prediction_mean_unscaled = prediction_mean[input_dimension:, ]  * y_std + y_mean
prediction_std_unscaled = prediction_std[input_dimension:, ] * y_std

rmse = np.sqrt(np.mean((prediction_mean_unscaled - y_test[input_dimension:, ]) ** 2))

print(rmse)
# nll
nll = 0.5 * np.log(2 * np.pi * prediction_std_unscaled**2) + 0.5 * (
    (y_test[input_dimension:, ] - prediction_mean_unscaled) ** 2
) / (prediction_std_unscaled**2)

nll = np.mean(nll)

print(nll)


# Load BMALS results obtained from the MATLAB script

bmals_df = pd.read_csv('/Users/hakilic/Desktop/IFAC/IFAC/BMVALS_results.csv', header=None)
bmals_df.columns = ['y_true', 'y_pred', 'confidence_std']

y_true_bmals = bmals_df['y_true'].values
y_pred_bmals = bmals_df['y_pred'].values
conf_std_bmals = bmals_df['confidence_std'].values

time_steps_btn = np.arange(len(y_test[input_dimension:]))
time_steps_bmals = np.arange(len(y_true_bmals))


# Figure 1: Volterra BTN
# -------------------------------
plt.figure(figsize=(14, 6))
plt.plot(time_steps_btn, y_test[input_dimension:], color='black', linestyle='-', linewidth=1.8, label='Actual Output')
plt.plot(time_steps_btn, prediction_mean_unscaled, color='black', linestyle='--', linewidth=1.8, label='Volterra BTN Prediction')
plt.fill_between(
    time_steps_btn,
    prediction_mean_unscaled - 3*prediction_std_unscaled,
    prediction_mean_unscaled + 3*prediction_std_unscaled,
    color='gray', alpha=0.2, label='Prediction ±3 std'
)
plt.title('Volterra BTN')
plt.xlabel('Time Step')
plt.ylabel('Output')
plt.ylim([-2.5, 15])  # same y-axis
plt.legend(frameon=False)
plt.tick_params(direction='in', length=4)
plt.tight_layout()
plt.show()


# Figure 2: BMALS
# -------------------------------
plt.figure(figsize=(14, 6))
plt.plot(time_steps_bmals, y_true_bmals, color='black', linestyle='-', linewidth=1.8, label='Actual Output')
plt.plot(time_steps_bmals, y_pred_bmals, color='black', linestyle='--', linewidth=1.8, label='BMALS Prediction')
plt.fill_between(
    time_steps_bmals,
    y_pred_bmals - 3*conf_std_bmals,
    y_pred_bmals + 3*conf_std_bmals,
    color='gray', alpha=0.2, label='Prediction ±3 std'
)
plt.title('BMVALS')
plt.xlabel('Time Step')
plt.ylabel('Output')
plt.ylim([-2.5, 15])  # same y-axis
plt.legend(frameon=False)
plt.tick_params(direction='in', length=4)
plt.tight_layout()
plt.show()