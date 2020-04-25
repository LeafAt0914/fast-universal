import os
import numpy as np
from deepfool_fastUAP import deepfool_fastUAP
from prepare_imagenet_data import process_image, undo_image, process_UAP, undo_UAP

def universal_perturbation_fastUAP(dataset, f, grads, delta=0.8, max_iter_uni = np.inf, xi=10, num_classes=10, overshoot=0.02, max_iter_df=10):
    """
    :param dataset: Images of size MxHxWxC (M: number of images)
    :param f: feedforward function (input: images, output: values of activation BEFORE softmax).
    :param grads: gradient functions with respect to input (as many gradients as classes).
    :param delta: the desired fooling rate (default = 80% fooling rate)
    :param max_iter_uni: optional other termination criterion (maximum number of iteration, default = np.inf)
    :param xi: controls the l_inf magnitude of the perturbation (depend on image preprocess methods, default = 10)
    :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
    :param max_iter_df: maximum number of iterations for deepfool (default = 10)
    :return: the universal perturbation.
    """
    #initialization
    v = 0

    fooling_rate = 0.0
    itr = 0
    num_images =  np.shape(dataset)[0] # The images should be stacked ALONG FIRST DIMENSION

    while fooling_rate < 1-delta and itr < max_iter_uni:
        # Shuffle the dataset
        np.random.shuffle(dataset)
        est_labels_orig = np.zeros((num_images))

        #compute origin estimated labels
        for k in range(0, num_images):
            est_labels_orig[k] = int(np.argmax(np.array(f(dataset[k:(k+1), :, :, :])).flatten()))
            if k == 0:
                print('>> computing origin labels...')
            if (k+1) % 500==0:
                print('>>No.', k + 1)

        print ('Starting pass number ', itr)

        # Go through the data set and compute the perturbation increments sequentially
        for k in range(0, num_images):
            cur_img = dataset[k:(k+1), :, :, :]

            if est_labels_orig[k] == int(np.argmax(np.array(f(cur_img+v)).flatten())):
                print('>> k = ', k, ', pass #', itr)

                # Compute adversarial perturbation
                dr,iter,_,_ = deepfool_fastUAP(cur_img + v, v, f, grads, num_classes=num_classes, overshoot=overshoot, max_iter=max_iter_df)

                # Make sure it converged...
                if iter < max_iter_df-1:
                    v = v + dr
                    # clip updated universal perturbation
                    v = np.sign(v) * np.minimum(abs(v), xi)
                else:
                    print("UAP didn`t update.")
            
        itr = itr + 1

        #fooling rate test
        est_labels_pert = np.zeros((num_images))
        batch_size = 5
        num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size)))
        # Compute the estimated labels in batches
        for ii in range(0, num_batches):
            m = (ii * batch_size)
            M = min((ii+1)*batch_size, num_images)
            est_labels_pert[m:M] = np.argmax(f(dataset[m:M, :, :, :] + v), axis=1).flatten()

        # Compute the fooling rate
        fooling_rate = float(np.sum(est_labels_pert != est_labels_orig) / float(num_images))
        print('FOOLING RATE = ', fooling_rate)
        
    return v
