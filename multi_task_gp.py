#  GRAPH CODE
# ============

# Import Tensorflow and Numpy
import tensorflow as tf
import numpy as np
import pandas as pd
import gpflow as gpf
import matplotlib.pyplot as plt

# open the session
tf_graph = tf.Graph()
tf_session = tf.Session(graph=tf_graph)

# X1 = np.random.rand(10, 1)
# Y1 = np.sin(6*X1) + np.random.standard_t(3, X1.shape)*0.03
#
# X2 = X1
# Y2 = np.sin(6*X2+ 0.7) + np.random.standard_t(3, X2.shape)*0.1

# plt.plot(X1, Y1, 'x', mew=2)
# plt.plot(X2, Y2, 'x', mew=2)
#
# plt.show()
## define the multi learning task graph - this is an example, so we have two tasks
### https://jg8610.github.io/Multi-Task/
with tf_graph.as_default():


    # ======================
    # Define the Graph
    # ======================

    # Define the Placeholders
    X_holder = tf.placeholder("float", shape=[10, 10], name="X_holder")
    Y1_holder = tf.placeholder("float", shape=[10, 20], name="Y1_holder")
    Y2_holder = tf.placeholder("float", shape=[10, 20], name="Y2_holder")

    # Define the weights for the layers

    initial_shared_layer_weights = np.random.rand(10, 20)
    initial_Y1_layer_weights = np.random.rand(20, 20)
    initial_Y2_layer_weights = np.random.rand(20, 20)

    shared_layer_weights = tf.Variable(initial_shared_layer_weights, name="share_W", dtype="float32")
    Y1_layer_weights = tf.Variable(initial_Y1_layer_weights, name="share_Y1", dtype="float32")
    Y2_layer_weights = tf.Variable(initial_Y2_layer_weights, name="share_Y2", dtype="float32")

    # Construct the Layers with RELU Activations
    shared_layer = tf.nn.relu(tf.matmul(X_holder, shared_layer_weights))

    Y1_layer = tf.nn.relu(tf.matmul(shared_layer, Y1_layer_weights))
    Y2_layer = tf.nn.relu(tf.matmul(shared_layer, Y2_layer_weights))

## attach GP at each output task
## there are two tasks from above network seeting, so there will be two multi-output GPs
## the input of GP must be the output of previous layer (y1_layer and y2_layer)
with gpf.defer_build():
    # ---------------- GP -------------------------------------------
    # a Coregionalization kernel. The base kernel is Matern 3/2, and acts on the first ([0]) data dimension.
    # the 'Coregion' kernel indexes the outputs, and acts on the second ([1]) data dimension
    k1 = gpf.kernels.Matern32(1, active_dims=[0])
    coreg = gpf.kernels.Coregion(1, output_dim=2, rank=1, active_dims=[1])
    kern = k1 * coreg

    # build a variational model. This likelihood switches between Student-T noise with different variances:
    lik = gpf.likelihoods.SwitchedLikelihood([gpf.likelihoods.StudentT(), gpf.likelihoods.StudentT()])

    # Augment the time data with ones or zeros to indicate the required output dimension
    # X1_avg = np.random.rand(10, 1)
    # stacked_X1 = np.hstack((Y1_layer, np.zeros_like(Y1_layer)))
    # stacked_X2 = np.hstack((X1_avg, np.ones_like(X1_avg)))
    # X1_augmented = np.vstack((stacked_X1, stacked_X2))
    #
    # Y1_label = np.hstack((Y1_holder, np.zeros_like(Y1_holder)))
    # Y1_avg_label = np.hstack((Y1_holder, np.ones_like(Y1_holder)))
    # Y1_augmented = np.vstack((Y1_label, Y1_avg_label))
    #
    # # Augment the Y data to indicate which likelihood we should use
    # Y2_avg = Y2_layer
    # stacked_Y1 = np.hstack((Y2_layer, np.zeros_like(Y2_layer)))
    # stacked_Y2 = np.hstack((Y2_avg, np.ones_like(Y2_avg)))
    # X2_augmented = np.vstack((stacked_Y1, stacked_Y2))
    #
    # Y2_label = np.hstack((Y2_holder, np.zeros_like(Y1_holder)))
    # Y2_avg_label = np.hstack((Y2_holder, np.ones_like(Y2_holder)))
    # Y2_augmented = np.vstack((Y2_label, Y2_avg_label))

    # now build the GP model as normal
    gp_model1 = gpf.models.VGP(Y1_layer, Y1_holder, kern=kern, likelihood=lik, num_latent=1)
    gp_model2 = gpf.models.VGP(Y2_layer, Y2_holder, kern=kern, likelihood=lik, num_latent=1)


gp_model1.compile(tf_session)
gp_model2.compile(tf_session)


### Jointly train the model
### joint loss and individual losses must be minimized
## joint loss
with tf_graph.as_default():

    ## individual loss
    gpf.train.ScipyOptimizer().minimize(gp_model1)
    gpf.train.ScipyOptimizer().minimize(gp_model2)

    # Calculate joint Loss
    Joint_Loss = gp_model1.objective + gp_model2.objective

    # optimisers
    optimiser = tf.train.AdamOptimizer()
    minimise = optimiser.minimize(Joint_Loss)  # this should pick up all Trainable variables.
# with tf_graph.as_default():
#     # Calculate Loss
#     Y1_Loss = tf.nn.l2_loss(Y1_holder-Y1_layer)
#     Y2_Loss = tf.nn.l2_loss(Y2_holder-Y2_layer)
#     Joint_Loss = Y1_Loss + Y2_Loss
#
#     # optimisers
#     optimiser = tf.train.AdamOptimizer().minimize(Joint_Loss)
#     Y1_op = tf.train.AdamOptimizer().minimize(Y1_Loss)
#     Y2_op = tf.train.AdamOptimizer().minimize(Y2_Loss)

results = []
with tf_graph.as_default():
    # session.run(tf.initialize_all_variables())
    tf_session.run(tf.global_variables_initializer())

    for i in range(1000):
        _, Joint_loss = tf_session.run([optimiser, Joint_Loss],
                                    {
                                        X_holder: np.random.rand(10, 10) * 10,
                                        Y1_holder: np.random.rand(10, 20) * 10,
                                        Y2_holder: np.random.rand(10, 20) * 10
                                    })

        if i % 50 == 0:
            # run on test data
            fd = {}
            fd.update({
                        X_holder: np.random.rand(10, 10) * 10,
                        Y1_holder: np.random.rand(10, 20) * 10,
                        Y2_holder: np.random.rand(10, 20) * 10
                    })

            _, Joint_loss = tf_session.run([optimiser, Joint_Loss], feed_dict=fd)
            print("iteratiom:", i, "joint loss", Joint_loss)


            ## create test data and predict
            # mu, var = gp_model1.predict_f(stacked_xtest1)
            # mu, var = gp_model2.predict_f(stacked_xtest1)

            results.append(dict(step_no=i, loss=Joint_loss))

    tf_session.close()

print("Done!")
# ## We save the results for looking at them later.
df = pd.DataFrame(results)
df.to_pickle("output/multi-task-gp.pdpick")



