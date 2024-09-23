# Multimodal-emotion-recognition-based-on-a-fusion-of-audiovisual-information-with-temporal-dynamics

This repository contains the code of the models, training, test and dataset generation used in the paper ["Multimodal emotion recognition based on a fusion of audiovisual information with temporal dynamics"](https://link.springer.com/article/10.1007/s11042-024-20227-6?utm_source=rct_congratemailt&utm_medium=email&utm_campaign=oa_20240918&utm_content=10.1007/s11042-024-20227-6)


## Dataset

To reproduce the experiments, it is necessary to download the 3 datasets used:

- [X] [SAVEE](http://kahlan.eps.surrey.ac.uk/savee/)

- [X] [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)

- [X] [RAVDESS](https://zenodo.org/records/1188976#.YFZuJ0j7SL8)

## Embedders

To extract the features these embedders were used:

- [X] For the audio [X-Vector](https://huggingface.co/pyannote/embedding) (Registration needed)
- [X] For the images [VGG-Face](https://github.com/rcmalli/keras-vggface)

## EmbraceNet

The EmbraceNet architecture used in the paper is the extracted from the paper ["EmbraceNet: A robust deep learning architecture for multimodal classification"](https://www.sciencedirect.com/science/article/pii/S1566253517308242) The code is in the [github repository](https://github.com/idearibosome/embracenet)
