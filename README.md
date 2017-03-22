# Semisupervised-learning-with-DCGANs
**It is a work in progress**

==> Using DCGANs to learn useful representations during the adversarial training process and using the learned features to classify images with relatively little training data

==>Main aim of this project : To use the power of unsupervised representation learning with DCGANs to build discriminative image classifiers which can be trained with relatively little training data as compared to a fully supervised paradigm using a semisupervised learning approach.

==>Heavily influenced by :<br>
   Paper - <a href="https://arxiv.org/pdf/1511.06434.pdf">Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks</a> <br>

   Code - <a href="https://github.com/jacobgil/keras-dcgan">Keras Implementation of DCGANs</a>

==>Steps involves in the process,<br>
    1. Train the Deep Convolutional Generative Adversarial Networks with the datset in an unsupervised manner. <br>
    2. Reuse the Discriminator for semi supervised classification on the MNIST/CIFAR10 datasets.<br>
      &nbsp;&nbsp; -> This is done by freezing the weights of the discriminator network while supervised training.<br>
      &nbsp;&nbsp;-> Currently working with one FC softmax layer on top of the leaned feauture layers

==>Instructions for running:
For working with the MNIST dataset,

Training the model - python test.py --mode train --batch_size 128 --epoch_num 200

To generate Images - python test.py --mode generate --batch_size 64 --nice

To run the semisupervised classifier - python mnist.py

For working with the CIFAR 10 Dataset,

Training the model - python cifar_gan.py --mode train --batch_size 128 --epoch_num 200

To generate Images - python cifar_gan.py --mode generate --batch_size 64 --nice

To run the semisupervised classifier - python cifar_gan.py

