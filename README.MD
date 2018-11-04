# keras-text-classification

This project is an example project showing how to handle text in a classification machine learning algorithm. 
The dataset used is a free repository available at [repository site](https://archive.ics.uci.edu/ml/machine-learning-databases/00380/)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

This project was developed using the open source Anaconda distribution with python 3.6.6 . It is highly recommended to create a virtual environment of python in your machine and install the necessary packages listed in the [requirements.txt](https://github.com/jolasman/keras-text-classification/blob/master/requirements.txt) file.


### Installing

(Optional, you can use it without Anaconda, you can just using a python virtual environment) Firstly, install [Anaconda](https://www.anaconda.com/download/) on your machine. Now run the following commands to create a new virtual environment with python 3.6.
```
conda create -n yourEnvironmentName python=3.6 anaconda
```

Activate the new environment and run the command below to install the necessary packages to run the system.
```
pip install -r requirements.txt
```

Now, if you run the command ```python training.py``` the system should run and start training the Fully Connected Neural Network (NN)

The system uses the ```/dataset/Youtube01-Psy.csv``` file to the training phase. Then, running the ```python loading.py``` the system will give you a console interface to input the sentence to evaluate, giving us the predictions for each one.



<!-- ## Deployment

Additional notes about how to deploy this on a live system -->

## Built With

* [Anaconda](https://www.anaconda.com/download/) - The open source distribution used.
* [Python](https://www.python.org/) - Language used.
* [Keras](https://keras.io/) - Used to create the NN.
* [Scikit-learn](http://scikit-learn.org/stable/) - Used to process the data and to analyse results.
* [Tensorflow](https://www.tensorflow.org/) - Used by keras.


## Tutorial

You can see the video tutorials: [Playlist](https://www.youtube.com/watch?v=9KJKGRlbekM&list=PLqlmTSqjqZ0T86Cu7J5lB5JHqbUVvxTpq&index=2&t=).

<!-- ## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us. -->

<!-- ## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags).  -->




## Authors

* **Joel Carneiro** - *Initial work* - [GitHub](https://github.com/jolasman)

See also the list of [contributors](https://github.com/jolasman/keras-text-classification/graphs/contributors) who participated in this project.

<!-- ## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details -->

<!-- ## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc -->

