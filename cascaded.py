import scipy.io
from lagfeatures import lagfeatures
from config import *  # Import everything from config.py
from BTN_KM import btnkm
# Replace 'your_file.mat' with the path to your .mat file
mat = scipy.io.loadmat('/Users/hakilic/Downloads/Tensor-Network-B-splines-master/Cascaded/dataBenchmark.mat')

# Print keys to see what’s inside
print(mat.keys())

uEst = mat['uEst'].squeeze()
uVal = mat['uVal'].squeeze()
yEst = mat['yEst'].squeeze()
yVal = mat['yVal'].squeeze()

# MATLAB lags
inlags = [1, 2, 3, 4, 8, 16]
outlags = [0, 1, 2, 3, 4, 8, 16]

print("Python inlags:", inlags)
print("Python outlags:", outlags)


#Normalize the input and output training data to [0 1]
u_train = uEst #/7
y_train = yEst#/10 - 0.1
u_test = uVal #/ 7
y_test = yVal#/10 - 0.1
X_train, y_train, X_test, y_test = lagfeatures(u_train, u_test, y_train, y_test, inlags, outlags)

X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_std[X_std == 0] = 1
X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std  # Use train stats

y_mean = y_train.mean()
y_std = y_train.std()
y_train = (y_train - y_mean) / y_std


# hyper-parameters
input_dimension = 2
max_rank = 25

a, b = 1e-2, 1e-3
c, d = 1e-5 * np.ones(max_rank), 1e-6 * np.ones(max_rank)
g, h = 1e-6 * np.ones(input_dimension), 1e-6 * np.ones(input_dimension)

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
        max_iter=10,
        precision_update=True,
        lambda_R_update=True,
        lambda_M_update=True,
        plot_results=False,
        prune_rank=True,
    )

    # Predict (mse is returned by the predict function)
prediction_mean, prediction_std, _ = model.predict(
    features=X_test, input_dimension=input_dimension
)

prediction_mean_unscaled = prediction_mean * y_std + y_mean
prediction_std_unscaled = prediction_std * y_std

# nll
nll = 0.5 * np.log(2 * np.pi * prediction_std_unscaled**2) + 0.5 * (
    (y_test - prediction_mean_unscaled) ** 2
) / (prediction_std_unscaled**2)

nll = np.mean(nll)

print(nll)
# Create a time vector for plotting
time_steps = np.arange(len(y_test))

plt.figure(figsize=(14, 6))

# Plot actual test values
plt.plot(time_steps, y_test, label='Actual Output (y_test)', color='blue', linewidth=2)

# Plot predicted values
plt.plot(time_steps, prediction_mean_unscaled, label='Predicted Output', color='orange', linewidth=2)

# Optionally, add confidence intervals (±1 std)
plt.fill_between(
    time_steps,
    prediction_mean_unscaled - 3* prediction_std_unscaled,
    prediction_mean_unscaled + 3*prediction_std_unscaled,
    color='orange',
    alpha=0.2,
    label='Prediction ±1 std'
)

# Styling
plt.xlabel('Time Step', fontsize=12)
plt.ylabel('Output', fontsize=12)
plt.title('Test Output vs Predictions', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Show plot
plt.show()


a = 2

#merhaba