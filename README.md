# RAD-PAW
RAD-PAW: Reconstructed Adversarial Defense via Pixel Attention Weight

Requirements
-----------------------
Python 3.5 <br>
Tensorflow-gpu-1.3 <br>
Tflearn-0.3.2. <br>

Folder benign are examples of orinial images in NSFW,which can be classified correctly. <br>
Folder adv are some adversarial examples that lead to misclassification. <br>
In defense folder are reconstructed images that can be correctly identified. <br>

Running the model
-------------------------
We can use the tran2_test.py script to run the  model which is a Not suitable for work classifier: <br>
python train2_test.py <br>
