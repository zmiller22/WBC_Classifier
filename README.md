Contained in this folder is the code to initialize, train, test, and implement a convolutional neural network that is 
able to identify images of 4 types of white blood cell types taken by a microscope. Though it is currently optimized 
to identify blood cell images, it can be optimized to recognize any set of images you choose by adjusting a few key 
parameters that I will discuss below.

I also want to give a huge thank you to Adrian Rosebrock for creating an awesome tutorial on image recogniton using neural 
networks.

Before I discuss the different files and how to use them, it is important to note the libraries and packages used in 
the code. You will need to install the following libraries and packages along with a few standard python libraries 
like scipy, numpy, and pickle in order for the code to run on your computer.
- TensorFlow
- Keras 
- OpenCV
- scikit-learn
- matlibplot
- imutil

Within Blood_Cell_CNN there are 4 folders along with 4 python files. The folders are:
- classification_images
- dataset2-master
- output

**my_results** contains the output files from when I trained the network. It has the trained model, the label-binarizer, and
the plot of the history from my training. I will discuss later how you can use my model to classify images.

**dataset2-master** is the blood cell image dataset that I used this CNN for. It will be also be empty by default because of 
the size of the folders containing the blood cell images. If you want to use this CNN to identify the same pictures that 
I did, you must go to https://www.kaggle.com/paultimothymooney/blood-cells and download the folder called 'dataset2-master'
(this will take a while as it contains thousands of images). Once you have downloaded the folder, paste the contents into
your empty dataset2-master while perserving its file structure. If you decide to train on different images, you do not 
have to do this and you can delete the empty dataset2-master.

**output** contains the files that the CNN will save to after it is trained. I will discuss this more below.


Blood_Cell_CNN contains 4 main python files
- smallvggnet.py 
- train_vgg.py
- evaluate.py
- classify.py

**smallvggnet.py** contains the SmallVGGNet class. This was written by Adrian Rosebrock and is a smaller version of the 
standard VGG CNN architecture. It is called by train_vgg.py when the CNN is initialized

**train_vgg.py** is the file that reads in the pictures, formats them, and then trains the neural network on them. After
training, it will save the model and label-binarizer to files that you specify, and will plot the accuracy and loss over
the course of training to a file that you specify. There are two acceptable usages. If you run the file without adding any 
command line arguments, then it will run with all the path variables set to the defaults that work with my file structure. 
You can also run it with command line arguments, in which case you must enter the paths to all the necessary files on 
your system. Below is example usage with each path set to its default and below that is an explanation of what each argument
does (you can also type 'help' at the command line to see an explanation of what each argument is)

python train_vgg.py --train_data dataset2-master/images/TRAIN  --test_data dataset2-master/images/TEST --model output/smallvggnet.model --label-bin output/smallvggnet_lb.pickle --plot output/smallvggnet_plot.png

- --train_data is the path to the folder containing the training images. The folder name (or names) will be used as the 
    label so name your folders accordingly
- --test_data is the same as --train_data except you should enter the path to the folder containing the testing data instead
    of the training data
- --model is the path to the file you want the model to be saved to after training
- --label-bin is the path to the file that you want the label-binarizer to be saved to after training
- --plot is the path of the file that you want to plot the results to

**evaluate.py** will load in a trained model and test it on your test data. It will output a summary of a few important
for evaluating your model and can optionally list out each item in your test data, what the CNN classified it as, and 
how confident it was. By default it will load the model output by train_vgg.py and test it on the test images of the 
blood cells. However, you can also enter command line arguments to load a different model and test it on different data.
Below is example usage with each path set to the defualt path, and below that is an explaination of what each argument 
does (you can also type 'help' at the command line to see an  explaination of what each argument is)

python evaluate.py --show_data 1 --test_data dataset2-master/images/TEST --model output/smallvggnet.model --label-bin output/smallvggnet_lb.pickle

All of the parameters are the same as in train_vgg.py except for the first argument --show_data. This argument allows you
to view a list of each classification that the CNN made on an image, what the image acutally was, and how confident the 
CNN was in its prediction. This list could be very long, but is often good for use when debugging the network or trying
to understand the types of mistakes it making. Make this parameter a 1 if you want to see the list, and 0 otherwise. This 
argument is required even when running the program with all paths set to the defaults.

**classify.py** allows you to put the CNN to work on some folder of images that you want classified. It will print out
each files name, what it classified it as, and its level of confidence to the console as an output. Below is example usage 
with each path set to the defualt path, and below that is an explaination of what each argument does (you can also type 
'help' at the command line to see an  explaination of what each argument is).

python classify.py --image dataset2-master/images/TEST --model output/smallvggnet.model --label-bin output/smallvggnet_lb.pickle

All of the parameters are the same as they were in train_vgg.py except for the first one --image. Here --image should be 
the path to the image or images that you want to classify. This argument is required even when running as a default.


Finally, the output folder contains the results of my last run of train_vgg.py. Therefore, the defualts for classify.py 
and evaluate.py will use my last trained model and label-binarizer. If you run train_vgg.py with defaults, output will 
now contain the results of your run of train_vgg.py. However, my results can still be accessed in the my_results folder.
