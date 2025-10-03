from scipy.io import loadmat
from lagfeatures import lagfeatures
from config import *  # Import everything from config.py
from BTN_KM import btnkm

# Load the .mat file
data = loadmat('/Users/hakilic/Downloads/robot_arm_data/inverse_identification_without_raw_data.mat')

X_train = data['u_train'].squeeze().T
X_test = data['u_test'].squeeze().T
y_train = data['y_train'].squeeze().T[:,0]
y_test = data['y_test'].squeeze().T[:,0]


# # Compute min and max from training set
X_min = X_train.min(axis=0)
X_max = X_train.max(axis=0)

# Avoid division by zero for constant columns
range_X = X_max - X_min
range_X[range_X == 0] = 1  # prevents division by zero

# Scale train and test, overwrite variables
X_train = (X_train - X_min) / range_X
X_test  = (X_test - X_min) / range_X


y_mean = y_train.mean()
y_std = y_train.std()
y_train = (y_train - y_mean) / y_std

# Hyperparameters
input_dimension = 20
max_rank = 20

a, b = 1e-1, 1e-3
c, d = 1e-5 * np.ones(max_rank), 1e-6 * np.ones(max_rank)
g, h = 1e-6 * np.ones(input_dimension), 1e-6 * np.ones(input_dimension)

# Train model
model = btnkm(X_train.shape[1])
R, _, _, _, _, _, _ = model.train(
    features=X_train,
    target=y_train,
    input_dimension=input_dimension,
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

# Predict
# Predict (mse is returned by the predict function)
prediction_mean, prediction_std, _ = model.predict(
    features=X_test, input_dimension=input_dimension
)

prediction_mean_unscaled = prediction_mean  * y_std + y_mean
prediction_std_unscaled = prediction_std  * y_std

rmse = np.sqrt(np.mean((prediction_mean_unscaled - y_test) ** 2))

print(rmse)

# nll using raw values
nll = 0.5 * np.log(2 * np.pi * prediction_std_unscaled**2) + 0.5 * (
    (y_test - prediction_mean_unscaled) ** 2
) / (prediction_std_unscaled**2)
print("NLL:", np.mean(nll))

# Plot
time_steps = np.arange(len(y_test[:801]))
plt.figure(figsize=(14, 6))
plt.plot(time_steps, y_test[:801], label='Actual Output (y_test)', color='blue', linewidth=2)
plt.plot(time_steps, prediction_mean[:801], label='Predicted Output', color='orange', linewidth=2)
plt.fill_between(
    time_steps,
    prediction_mean[:801] - 1*prediction_std[:801],
    prediction_mean[:801] + 1*prediction_std[:801],
    color='orange',
    alpha=0.2,
    label='Prediction ±1 std'
)
plt.ylim(-6, 6)  # Limit Y-axis
plt.xlabel('Time Step', fontsize=12)
plt.ylabel('Output', fontsize=12)
plt.title('Test Output vs Predictions', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

