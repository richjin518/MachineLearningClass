import tensorflow.compat.v1 as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.disable_v2_behavior()

def tensorflow_demo():
    a_t = tf.constant(2)
    b_t = tf.constant(3)
    c_t = a_t + b_t
    print(c_t)
    
    with tf.Session() as sess:
        c_t_val = sess.run(c_t)
        print(c_t_val)

def graph_demo():
    a_t = tf.constant(2)
    b_t = tf.constant(3)
    c_t = a_t + b_t
    # view default graph
    # method 1: call function
    default_g = tf.get_default_graph()
    print("default graph:", default_g)
    # method 2: check attributes
    print("tensor graph ", a_t.graph)
    print("tensor graph ", b_t.graph)
    print("tensor graph ", c_t.graph)

    with tf.Session() as sess:
        print(sess.graph)

    # self define new graph
    new_g = tf.Graph()
    with new_g.as_default():
        a_new = tf.constant(20)
        b_new = tf.constant(30)
        c_new = a_new + b_new
        print("c_new: ", c_new)
        print("c_new graph ", c_new.graph)

    with tf.Session(graph=new_g) as new_sess:
        c_t_val = new_sess.run(c_new)
        print(c_t_val)
        print(new_sess.graph)
    
        graph_store_path = "D:/DL_data/graph_test"
        tf.summary.FileWriter(graph_store_path, graph=new_sess.graph)

if __name__ == "__main__":
    graph_demo()
