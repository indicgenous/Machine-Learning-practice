import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
import numpy as np
import os
from srgan_architecture import build_generator, build_discriminator, build_vgg
from data_loader import DataLoader

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Hyperparameters
batch_size = 16
lr = 1e-4
epochs = 10000
dataset_dir = 'path/to/DIV2K_train_HR'

# Load data
data_loader = DataLoader(dataset_dir, batch_size)

# Build models
generator = build_generator((64, 64, 3))
discriminator = build_discriminator((256, 256, 3))
vgg = build_vgg()

# Compile models
bce = BinaryCrossentropy(from_logits=False)
mse = MeanSquaredError()
d_optimizer = Adam(lr)
g_optimizer = Adam(lr)

# Training loop
for epoch in range(epochs):
    for batch in range(len(data_loader)):
        imgs_lr, imgs_hr = data_loader[batch]

        # Generate high-resolution images from low-resolution images
        fake_hr = generator.predict(imgs_lr)

        # Train Discriminator
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        d_loss_real = discriminator.train_on_batch(imgs_hr, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_hr, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        real_features = vgg.predict(imgs_hr)
        fake_features = vgg.predict(fake_hr)

        g_loss_content = mse(real_features, fake_features)
        g_loss_adversarial = bce(real_labels, discriminator.predict(fake_hr))
        g_loss = g_loss_content + 1e-3 * g_loss_adversarial

        generator.train_on_batch(imgs_lr, [imgs_hr, real_labels])

        # Print progress
        print(f"Epoch: {epoch+1}/{epochs}, Batch: {batch+1}/{len(data_loader)}, "
              f"D Loss: {d_loss}, G Loss: {g_loss.numpy()}")

    # Save model checkpoints
    if (epoch + 1) % 100 == 0:
        generator.save(f'models/generator_epoch_{epoch+1}.h5')
        discriminator.save(f'models/discriminator_epoch_{epoch+1}.h5')