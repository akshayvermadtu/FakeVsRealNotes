import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def read_dataset():
    df = pd.read_csv("C:\\Users\\Akshay\\Desktop\\ML\\Tensorflow\\FakeVsRealNote\\bank_note_data.csv")
    X = df[df.columns[0:4]].values
    y1 = df[df.columns[4]]
    encoder = LabelEncoder()
    encoder.fit(y1)
    y = encoder.transform(y1)
    Y = one_hot_encode(y)
    print(X.shape)
    return X, Y, y1


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


X, Y, y1 = read_dataset()
model_path = "C:\\Users\\Akshay\\Desktop\\ML\\Tensorflow\\FakeVsRealNote\\main"
cost_history = np.empty(shape=[1], dtype=float)
n_dim = X.shape[1]
n_class = 2

n_hidden_1 = 10
n_hidden_2 = 10
n_hidden_3 = 10
n_hidden_4 = 10

x = tf.placeholder(tf.float32, [None, n_dim])
W = tf.Variable(tf.zeros([n_dim, n_class]))
b = tf.Variable(tf.zeros([n_class]))
y_ = tf.placeholder(tf.float32, [None, n_class])


# define model
def multilayer_perceptron(x, weights, biases):

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    # out layer
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer


# define weights and biases
weights = {
    'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_class]))
}

biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
    'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_class]))
}

# Initialize all variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Call model defined
y = multilayer_perceptron(x, weights, biases)

sess = tf.Session()
sess.run(init)
saver.restore(sess, model_path)

prediction = tf.argmax(y, 1)
correct_prediction = tf.equal(prediction, tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# print accuracy run
print("************************************************************************")
print(" 0 stands for Fake note and 1 stands for Real note")
print("************************************************************************")
for i in range(754, 768):
    prediction_run = sess.run(prediction, feed_dict={x: X[i].reshape(1, 4)})
    accuracy_run = sess.run(accuracy, feed_dict={x: X[i].reshape(1, 4), y_: Y[i].reshape(1, 2)})
    print("Original case : ", y1[i], " Predicted Values : ", prediction_run, " Accuracy : ", accuracy_run*100, "%")


