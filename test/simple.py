
import tensorflow as tf

def main():

    image_data = tf.gfile.FastGFile('cropped_panda.jpg', 'rb').read()

    with tf.gfile.FastGFile('classify_image_graph_def.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        print(predictions.shape)
        print(predictions)

if __name__ == '__main__':
    main()
