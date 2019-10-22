import tensorflow as tf 
import numpy as np 
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import time
# %matplotlib inline

digit_origin = 8
digit_target = 3

classifier = tf.keras.models.load_model('Models/classifier_capacity1_simple.model', compile=True)
gan = tf.keras.models.load_model('Models/gan_digit8_rich.h5')

classifier.trainable = False
combined_networkInput = tf.keras.layers.Input(shape=(10,))
x = (gan(combined_networkInput) + 1.0)/2.0
new_shape = tf.convert_to_tensor([1,28,28,1],dtype=tf.int32)
x = tf.reshape(x,new_shape,name=None)
combined_networkOutput = classifier(x)
combined_network = tf.keras.models.Model(inputs=combined_networkInput, outputs=combined_networkOutput)


noise_change = 0.5

found = False

loss_object = tf.keras.losses.CategoricalCrossentropy()

input_label = np.zeros((1,10))
input_label[0][digit_target] = 1.0

start = time.time()

while(not found):
        noise = np.random.normal(0,1,size=[1,10])
        noise = tf.cast(noise,tf.float32)

        print("try a new seed")
        j = 0
        while(j < 25 and not found):
                prediction = combined_network(noise)
                loss = loss_object(input_label,prediction)
                grad = tf.gradients(loss,noise)[0]
                perturbations = tf.sign(grad)
                noise -= noise_change*perturbations
                result_target = K.eval(combined_network(noise))[0][3]
                print("confidence of 3",result_target)
                print("confidence of 8",K.eval(combined_network(noise))[0][8])
                if(result_target > 0.5):
                      generated_image = K.eval(gan(noise))[0]
                      print("Confidence in %d is %f" % (digit_target,result_target))
                      plt.imshow(generated_image.reshape(1,28,28)[0],cmap='gray')
                      plt.axis("off")
                      found = True
                j += 1

end = time.time()
print("time: %f" % (end - start))

plt.savefig('adversarial_examples_white_box.png')
plt.show()