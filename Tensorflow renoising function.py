import tensorflow as tf
import matplotlib.pyplot as plt

total_points = 1000

x = tf.random.uniform(shape=[total_points], minval=0, maxval=10)
noise = tf.random.normal(shape=[total_points], stddev=0.2)

k_true = 0.7
b_true = 2.0

y = x + k_true + b_true + noise

k = tf.Variable(0.0)
b = tf.Variable(0.0)

epochs = 500
learning_rate = 0.02

batch_size = 100
num_steps = total_points // batch_size

for n in range(epochs):
    with tf.GradientTape() as tape:
        f = k * x + b
        loss = tf.reduce_mean(tf.square(y - f))

    dk, db = tape.gradient(loss, [k, b])

    k.assign_sub(learning_rate * dk)
    b.assign_sub(learning_rate * db)

print(f'Real: k={k_true}, b={b_true}')
print(f'Predict: k={k}, b={b}')

y_pred = k * x + b


plt.scatter(x, y, s=2)
plt.scatter(x, y_pred, c='r', s=2)
plt.show()
