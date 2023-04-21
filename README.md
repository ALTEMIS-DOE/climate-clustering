### Climate-Clustering - Clustering Autoencoder for AI-generated climate patterns
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

Climate-clustering is a new data-driven climate classification workflow based on an unsupervised deep learning technique that can accommodate the vast volume of spatiotemporal numerical climate projection data. We aim to identify climate patterns and distinct zones that capture multiple climate variables as well as their future changes under different climate change scenarios. 
The project was also supported by [Frontier Development Lab 2022](https://frontierdevelopmentlab.org/fdl-2022#adaptation)

<img src="https://earimediaprodweb.azurewebsites.net/Api/v1/Multimedia/9fff8718-4d44-4e62-89fa-d5f8e645d1ef/Rendition/low-res/Content/Public" width="320" height="250">

### Quick Start
To install base pip packages,
```
$ pip3 install -r ./requirements.txt   
```

To install a custom library, you need to run
```
$ python3 ./setup_py develop
```
This command lets `lib4climex` library download in your virtual env.   

Finally, to enable multi-gpu training, you may need to install [horovod](https://horovod.readthedocs.io/en/stable/install_include.html). Otherwise, comment off `from horovod import tensorflow as hvd` in training python script.
