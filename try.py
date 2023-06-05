import tensorflow as tf
import tensorflow_probability as tfp

print('A')
# Generate some example time series data
x = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
y = tf.constant([10.0, 20.0, 30.0, 40.0, 50.0])
print('A')

# Define the GP model
kernel = tfp.math.psd_kernels.ExpSinSquared(amplitude=1.0, length_scale=1.0, period=1.0)
model = tfp.distributions.GaussianProcess(kernel=kernel, index_points=x[:, None])
print('A')

# Fit the model to the data using maximum likelihood estimation
optimizer = tf.optimizers.Adam()
neg_log_likelihood = lambda y, rv_y: -rv_y.log_prob(y)
loss_fn = lambda: neg_log_likelihood(y, model)
print('A')
for i in range(1000):
    # if i%50:
    #
    print(i)
    optimizer.minimize(loss_fn, model.trainable_variables)

# Generate predictions for the past using the fitted model
predictions = model.mean().numpy()[-2::-1]  # reverse and omit last point
print(predictions)
