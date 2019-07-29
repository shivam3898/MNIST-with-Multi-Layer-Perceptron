import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("/tmp/data", one_hot=True)

learning_rate = 0.001
training_epochs = 20
batch_size = 100

n_classes=10
n_samples=mnist.train.num_examples
n_input=784
n_hidden_1=256
n_hidden_2=256

def multilayer_perceptron(x, weights, biases):
    layer1=tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer1=tf.nn.relu(layer1)
    
    layer2=tf.add(tf.matmul(layer1, weights['h2']), biases['b2'])
    layer2=tf.nn.relu(layer2)
    
    out_layer=tf.matmul(layer2, weights['out'])+biases['out']
    
    return out_layer

weights={
    'h1':tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2':tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases={
    'b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32)

pred=multilayer_perceptron(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess=tf.InteractiveSession()
init=tf.global_variables_initializer()
sess.run(init)

for epoch in range(training_epochs):
    avg_cost=0.0
    total_batch=int(n_samples/batch_size)
    for i in range(total_batch):
        batch_x, batch_y=mnist.train.next_batch(batch_size)
        _,c=sess.run([optimizer, cost], feed_dict={x:batch_x, y:batch_y})
        avg_cost+=c/total_batch
    print("Epoch: ", epoch+1, " cost: ", avg_cost)
print("Model has completed ", training_epochs, " epochs of training")

correct_predictions=tf.equal(tf.arg_max(pred, 1),tf.arg_max(y, 1))
correct_predictions=tf.cast(correct_predictions, "float")
accuracy=tf.reduce_mean(correct_predictions)
print(accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))