import os.path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.platform import gfile

from create_X import Create_X
from prepare_imagenet_data import process_image,undo_image,process_UAP, undo_UAP
from universal_pert_fastUAP import universal_perturbation_fastUAP

device = '/gpu:0'
num_classes = 10

def jacobian(y_flat, x, inds):
    n = num_classes # Not really necessary, just a quick fix.
    loop_vars = [
         tf.constant(0, tf.int32),
         tf.TensorArray(tf.float32, size=n),
    ]
    _, jacobian = tf.while_loop(
        lambda j,_: j < n,
        lambda j,result: (j+1, result.write(j, tf.gradients(y_flat[inds[j]], x))), 
        loop_vars)
    return jacobian.stack()

if __name__ == '__main__':

    with tf.device(device):
        persisted_sess = tf.Session()
        model = os.path.join('data', 'DenseNet121_graph_from_keras.pb')

        # Load model
        with gfile.FastGFile(model, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            persisted_sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')

        persisted_sess.graph.get_operations()

        persisted_input = persisted_sess.graph.get_tensor_by_name("input_1:0")
        persisted_output = persisted_sess.graph.get_tensor_by_name("fc1000/BiasAdd:0")  

        # Computing feedforward function.
        def f(image_inp): return persisted_sess.run(persisted_output, 
                                                feed_dict={persisted_input: np.reshape(image_inp, (-1, 224, 224, 3))})

        perturbation_name = 'universal_perturbation_DenseNet121'
        file_perturbation = os.path.join('data', perturbation_name+'.npy')
        pathTo_ILSVRC2012 = 'E:\\ILSVRC2012'
        pathTo_testImage = os.path.join('data', 'test_image.JPEG')

        if os.path.isfile(file_perturbation) == 0:

            # Compiling the gradient tensorflow functions.
            y_flat = tf.reshape(persisted_output, (-1,))
            inds = tf.placeholder(tf.int32, shape=(num_classes,))
            dydx = jacobian(y_flat,persisted_input,inds)

            # Computing gradient function.
            def grad_fs(image_inp, indices): return persisted_sess.run(dydx, feed_dict={persisted_input: image_inp, inds: indices}).squeeze(axis=1)

            # Load/Create data
            datafile = os.path.join('data', 'imagenet_data_torch_mode.npy')
            if os.path.isfile(file_perturbation) == 0:
                # Caution: This can take take a lot of space.
                X = Create_X(pathTo_ILSVRC2012, mode = 'torch')
            else:
                print('>> Pre-processed imagenet data detected, loading...')
                X = np.load(datafile)
                print('>> Pre-processed imagenet loaded.')

            # Running universal perturbation
            v = universal_perturbation_fastUAP(perturbation_name,
                                        X, 
                                        f, 
                                        grad_fs, 
                                        delta=0.8,
                                        xi=0.1735207357279195, #l_inf norm value upper bound after preprocessing
                                        num_classes=num_classes)

            # Saving the universal perturbation
            np.save(os.path.join(file_perturbation), v)
        else:
            print('>> Found a pre-computed universal perturbation!')
            v = np.load(file_perturbation)

        print('>> Testing the universal perturbation on an image.')

        labels = open(os.path.join('data', 'keras&caffe_ilsvrc_synsets.txt'), 'r').read().split('\n')

        image_original = process_image(pathTo_testImage, mode = 'torch')
        label_original = np.argmax(f(image_original), axis=1).flatten()
        str_label_original = labels[np.int(label_original)].split(',')[0]

        undo_image_original = undo_image(image_original,mode='torch')
        # Clip the perturbation to make sure images fit in uint8
        clipped_v = np.clip(undo_image_original+undo_UAP(v[0,:,:,:], mode = 'torch'), 0 ,255) - np.clip(undo_image_original, 0, 255)

        image_perturbed = image_original + process_UAP(clipped_v, mode = 'torch')
        label_perturbed = np.argmax(f(image_perturbed), axis=1).flatten()
        str_label_perturbed = labels[np.int(label_perturbed)].split(',')[0]

        # Show original and perturbed image
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(undo_image_original.astype(dtype='uint8'), interpolation=None)
        plt.title(str_label_original)

        plt.subplot(1, 2, 2)
        plt.imshow(undo_image(image_perturbed, mode = 'torch').astype(dtype='uint8'), interpolation=None)
        plt.title(str_label_perturbed)

        plt.show()
        
