# Autoencoder
Image segmentation using Autoencoder cnn model(python, keras)
"Autoencoding" is a data compression algorithm where the compression and decompression functions are 
1) data-specific, 2) lossy, and 3) learned automatically from examples rather than engineered by a human.
Additionally, in almost all contexts where the term "autoencoder" is used, 
the compression and decompression functions are implemented with neural networks.

1) Autoencoders are data-specific, which means that they will only be able to compress data similar to 
what they have been trained on. This is different from, say, the MPEG-2 Audio Layer III (MP3) compression algorithm, 
which only holds assumptions about "sound" in general, but not about specific types of sounds.
An autoencoder trained on pictures of faces would do a rather poor job of compressing pictures of trees,
because the features it would learn would be face-specific.

2) Autoencoders are lossy, which means that the decompressed outputs will be degraded compared to the original 
inputs (similar to MP3 or JPEG compression). This differs from lossless arithmetic compression.

3) Autoencoders are learned automatically from data examples, which is a useful property: it means that it is easy to
train specialized instances of the algorithm that will perform well on a specific type of input. 
It doesn't require any new engineering, just appropriate training data.

To build an autoencoder, you need three things: an encoding function, a decoding function,
and a distance function between the amount of information loss between the compressed representation of your
data and the decompressed representation (i.e. a "loss" function). The encoder and decoder will be chosen to be parametric
functions (typically neural networks), and to be differentiable with respect to the distance function, so the parameters
of the encoding/decoding functions can be optimize to minimize the reconstruction loss, using Stochastic Gradient Descent.
It's simple! And you don't even need to understand any of these words to start using autoencoders in practice.

##Are they good at data compression?

Usually, not really. In picture compression for instance, it is pretty difficult to train an autoencoder that does a better job than a basic algorithm like JPEG, and typically the only way it can be achieved is by restricting yourself to a very specific type of picture (e.g. one for which JPEG does not do a good job). The fact that autoencoders are data-specific makes them generally impractical for real-world data compression problems: you can only use them on data that is similar to what they were trained on, and making them more general thus requires lots of training data. But future advances might change this, who knows.

