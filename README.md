# GAN-Monet

Background
Monet style images generation

This Kaggle competition focuses on building a Generative Adversarial Network (GAN) capable of generating 7,000 to 10,000 Monet style images from real world photos. The goal is to train a model that can transform natural photographs into images that resemble the artistic style of Claude Monet, capturing his color palette, brushstroke patterns, contrast, and overall artistic feel.

What is GAN?

Generative Adversarial Network (GAN) is a deep learning model that uses two competing neural networks—a generator and a discriminator—to create new, realistic data that is similar to a training dataset. The generator creates fake data, while the discriminator tries to distinguish between the real data and the fake data. This adversarial competition drives both networks to improve: the generator gets better at creating convincing fakes, and the discriminator gets better at spotting them.(from https://aws.amazon.com/what-is/gan/)

Evaluation metric: MiFID

MiFID is a specialized adaptation of the Fréchet Inception Distance (FID)—a commonly used metric for evaluating generative models. It measures how close the generated images are to the real Monet paintings. Lower MiFID scores indicate better performance.

Data
The dataset contains two distinct collections of images: Monet-style paintings and photographic landscape images. These datasets are unpaired, meaning there is no direct 1 to 1 mapping between a photo and a Monet painting. This makes the problem an unpaired image to image translation task, which is why CycleGAN is an appropriate model choice.

Data Processing
Duplicate check
There are duplicate images in the Photo datasets, so I remove them to ensure that the GAN will not memorize repeated images.
Balance check
Since the number of Monet images is 300 and the number of photos is 7068, they are unbalanced. So I am cycling the Monet images to make sure there is a Monet image training with a photo.
RGB check
After analyzing the color distributions across both datasets, Monet images are generally brighter with more pixels in the higher intensity ranges. This difference highlights the characteristic color style of Monet paintings, which the GAN will learn to emulate.

Model Architecture¶
CycleGAN has two generators and two discriminators. The idea is one generator turns photos into Monet-style paintings, and the other turns Monet paintings back into photos. The discriminators help the generators improve by trying to tell real from fake images.

The PatchGAN discriminator converts the 256256 image into a 3131 patch map. It outputs a map of scores, which has one score per receptive patch in the input image.

For residual block, I applied two 3×3 convolutions, with instance norm + ReLU between them, then added the input back. So output = x + F(x). This residual connection helps training deeper networks by allowing gradients to flow through the identity path.

ResNet Generator Inital create a 7×7 large kernel helps capture larger context,then downsampling with 2 convolution layers,apply residual block to keep the connection, then upsampling 2 convolution layers back to resize the image to 256*256. Finally, let's produce 3×256×256 output in range -1 to 1.

Result and Analysis¶
Training Improvements:

Epochs: Increased from 20 to 40

Max steps per epoch: Increased from 20 to 40

Batch size: 4

Result: 91.08062 to 76.96256

These changes gave the generator more exposure to the training images and helped the model improve the Monet-style details. Ideally, more epochs and steps and less batch size would produce even better results, but we were limited by GPU memory and training time.

Reference¶
What is a GAN? https://aws.amazon.com/what-is/gan/
Monet cyclegan tutorial: https://www.kaggle.com/code/amyjang/monet-cyclegan-tutorial

