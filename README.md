# DeepLearningTutorial


This repo serves as an educational suite for deep learning topics.
Each module has slides or a short manuscript as well as a code example to
accompany it.
All code examples will run in a reasonable amount of time on a CPU.
Feel free to suggest a new topic to me. More are on the way.

## Module Progression
    * You will want to start with the `intro` module.
        * First you will want to read the slideshow located at `intro/slides/MerrillDLSlides.pdf`.
        * Then navigate to `intro/code` and analyze and run the `tf_mnist_classifier.py` script.
            * You will have to install the libraries from `requirements.txt` with the command.
            * I recommend a virtual environment
              ```
              $ virtualenv  vpython # Create new virtual evironment
              $ source vpython/bin/activate
              $ pip install -r requirements.txt
              $ ./tf_mnist_classifier.py
          ```
    * Next try out the `vae` module.
        * Read the manuscript located at `vae/tutorial/MerrillVAE.pdf`.
        * Then navigate to `vae/code` and run the `tf_mnist_vae.py` script.
          ```
          $ ./tf_mnist_vae.py
          ```
    
## TODO
    * Uncertainty Modeling techniques
    * Other types of autoencoders
    * Segmentation
    * Pruning
    * Sparse neural networks 
