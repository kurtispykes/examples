# What is Layer?
Layer is a [Declarative MLOps Platform](https://layer.co/) that empowers Data Science teams to implement end-to-end machine learning with minimal effort. Layer is a managed solution and can be used without the hassle involved in configuring and setting up servers. 


# Getting started
This project illustrates how to train an image classification model with Layer.
Let's start by installing Layer:
```
pip install layer-sdk
```

Clone this  image classification project:
```
layer clone https://github.com/layerml/examples.git
```
Change into the cats-and-dogs folder run the project:
```
cd examples/cats-and-dogs-classification
```
Run the project
```
layer start

```
## File Structure

```yaml
.
|____.layer
| |____project.yaml
|____models
| |____model
| | |____model.yaml
| | |____requirements.txt
| | |____model.py
|____README.md
|____data
| |____dataset
| | |____dataset.yaml
| |____features
| | |____category
| | | |____requirements.txt
| | | |____category.py
| | |____dataset.yaml


```