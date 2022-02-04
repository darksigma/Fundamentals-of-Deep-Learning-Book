from fdl_examples.datatools import input_data
mnist = input_data.read_data_sets("../../data/", one_hot=True)

import tensorflow as tf
import numpy as np
from fdl_examples.chapter3.multilayer_perceptron_updated import inference, loss

import matplotlib.pyplot as plt

sess = tf.Session()

x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes

	
saver = tf.train.import_meta_graph('frozen_mlp_checkpoint/model-checkpoint-547800.meta')
saver.restore(sess, 'frozen_mlp_checkpoint/model-checkpoint-547800')

var_list_opt = [None, None, None, None, None, None]
name_2_index = {
	"mlp_model/hidden_1/W:0" : 0,
	"mlp_model/hidden_1/b:0" : 1,
	"mlp_model/hidden_2/W:0" : 2,
	"mlp_model/hidden_2/b:0" : 3,
	"mlp_model/output/W:0" : 4,
	"mlp_model/output/b:0" : 5
}

for x in tf.trainable_variables():
	if x.name in name_2_index:
		index = name_2_index[x.name]
		var_list_opt[index] = x



with tf.variable_scope("mlp_init") as scope:

	output_rand = inference(x)
	cost_rand = loss(output_rand, y)

	scope.reuse_variables()

	var_list_rand = ["hidden_1/W", "hidden_1/b", "hidden_2/W", "hidden_2/b", "output/W", "output/b"]
	var_list_rand = [tf.get_variable(v) for v in var_list_rand]

	init_op = tf.variables_initializer(var_list_rand)

	sess.run(init_op)


feed_dict = {
	x: mnist.test.images,
	y: mnist.test.labels,
}

print(sess.run([cost_opt, cost_rand], feed_dict=feed_dict))

with tf.variable_scope("mlp_inter") as scope:

	alpha = tf.placeholder("float", [1, 1])

	h1_W_inter = var_list_opt[0] * (1 - alpha) + var_list_rand[0] * (alpha)
	h1_b_inter = var_list_opt[1] * (1 - alpha) + var_list_rand[1] * (alpha)
	h2_W_inter = var_list_opt[2] * (1 - alpha) + var_list_rand[2] * (alpha)
	h2_b_inter = var_list_opt[3] * (1 - alpha) + var_list_rand[3] * (alpha)
	o_W_inter = var_list_opt[4] * (1 - alpha) + var_list_rand[4] * (alpha)
	o_b_inter = var_list_opt[5] * (1 - alpha) + var_list_rand[5] * (alpha)

	h1_inter = tf.nn.relu(tf.matmul(x, h1_W_inter) + h1_b_inter)
	h2_inter = tf.nn.relu(tf.matmul(h1_inter, h2_W_inter) + h2_b_inter)
	o_inter = tf.nn.relu(tf.matmul(h2_inter, o_W_inter) + o_b_inter)

	cost_inter = loss(o_inter, y)
	tf.summary.scalar("interpolated_cost", cost_inter)


summary_writer = tf.summary.FileWriter("linear_interp_logs/",
                                        graph_def=sess.graph_def)
summary_op = tf.summary.merge_all()
results = []
for a in np.arange(-2, 2, 0.01):
	feed_dict = {
		x: mnist.test.images,
		y: mnist.test.labels,
		alpha: [[a]],
	}

	cost, summary_str = sess.run([cost_inter, summary_op], feed_dict=feed_dict)
	summary_writer.add_summary(summary_str, (a + 2)/0.01)
	results.append(cost)

plt.plot(np.arange(-2, 2, 0.01), results, 'ro')
plt.ylabel('Incurred Error')
plt.xlabel('Alpha')
plt.show()


