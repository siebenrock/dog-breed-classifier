# Dog Breed Classifier

## Convolutional Neural Networks

In this [notebook](https://github.com/siebenrock/dog-breed-classifier/blob/master/dog_breed_classifier.ipynb), an algorithm is developed which accepts an user-supplied image file of a dog as input and provides an estimate of the dog's breed out of 133 options.

First, a convolutional neural network (CNN) is developed from scratch with random initialization. Later, a transfer learning is added to improve accuracy with bottleneck features taken from the pre-trained `ResNet-50` model. The model is created using [Keras 2.1.6](https://github.com/keras-team/keras/releases/tag/2.1.6) and 6680 dog images for training, 835 validation images, and 836 test images.

A project through Udacity's Deep Learning Nanodegree were some template code has already been provided.

## Code

All of the code is provided in the Jupyter notebook `dog_breed_classifier.ipynb`. Written in Python 3.

## Performance

- Accuracy with CNN from scratch: 6.70%
- Accuracy with CNN using transfer learning: 81.34%

##  Evaluation

Testing the algorithm on sample images. Is the dog's breed prediction accurate? Images taken from [Unsplash](https://unsplash.com).

```python
final_predict("samples/photo1.jpeg")
```

```
It is a dog with breed Golden retriever.
```

![png](https://raw.githubusercontent.com/siebenrock/dog-breed-classifier/master/evaluation/golden_retriever.png)

```python
final_predict("samples/photo2.jpeg")
```

```
It is a dog with breed Alaskan malamute.
```

![png](https://raw.githubusercontent.com/siebenrock/dog-breed-classifier/master/evaluation/alaskan_malamute.png)
