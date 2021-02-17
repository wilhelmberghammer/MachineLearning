# Neural Style Transfer

I tried to follow the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576v2.pdf) but I did some things differently. 

For example I didn't use their, more complicated loss function, I simply used MSE.

I also didn't use noise as input, I used the content.

This might be the reason for some differences. I didn't notice any differences when changing the alpha and beta values. The biggest change happens when changing the learning rate. A higher learning rate leads to more significant changens to the original content.

<br>

## Result Example
**Content:**

![content alisha](https://github.com/wilhelmberghammer/MachineLearning/blob/main/neural_style_transfer/content_images/alisha.jpg)

**Style image:**

![style dog](https://github.com/wilhelmberghammer/MachineLearning/blob/main/neural_style_transfer/style_images/dog_abstract_style.jpg)

**Result:**

![Result](https://github.com/wilhelmberghammer/MachineLearning/blob/main/neural_style_transfer/generated_images/alisha_2.png)