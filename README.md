# RAD-PAW
RAD-PAW: Reconstructed Adversarial Defense via Pixel Attention Weight

Requirements:
Python 3.5, Tensorflow-gpu-1.3, Tflearn-0.3.2.

Folder benign are examples of orinial images in NSFW,which can be classified correctly. Folder adv are some adversarial examples that lead to misclassification.
In defense folder are reconstructed images that can be correctly identified.

Running the model
We can use the tran2_test.py script to run the  model which is a Not suitable for work classifier.
python train2_test.py
