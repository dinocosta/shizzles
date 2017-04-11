# Shizzles

Brain MRI Segmentation for Skull Extraction using [Keras](https://keras.io).

## Process

If you don't really want to read through this section you could check out [these slides](https://docs.google.com/presentation/d/1BseUfe7vOGd64-h1fXnJj-xjA2Lvjn2bUrZo1EiqlA4/edit?usp=sharing) which give you a brief overview of what we are trying to accomplish and how we are structuring our project.

1. Pre-Processing

    It all starts with Pre-Processing, where we found two problems:

    - The scans can vary in dimensions
    - There's a non uniform brightness value distribution between scans, one scan max value can be 40 and the other can be 800 (for example.

    That's where `normalization.py` comes in, this module is responsible for:
    
    - Normalizing scan dimensions, by adding or removing rows and columns of all black vortexes
    - Normalizing brightness valus between 0 and 255.

2. Brain Tissue Classification

    After Pre-Processing a Neural Network is repsonsible for correctlying classifying a scan according to the possibility of spotting brain tissue in the scan or not.
    
    Given a MRI exam with multiple scans there's a possibility that, in some of those scans, the brain is not visible and, as such, it is essential that these scans are not fed to the next neural network so as to save computational cycles.
    
3. MRI Segmentation

    The last, and final, Neural Network is a CNN (Convulotional Neural Network) capable of analyzing a MRI scan and create a mask that segments the brain tissue in the scan.
    
    Using the generated mask we are capable of "filtering" only the brain tissue from the MRI scans.
