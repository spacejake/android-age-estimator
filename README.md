# Age Estimation App for Android #
 This fun android app can estimate the ages of people using face detection and tensorflow. 
 
 ## Overview ## 
Multi face detection is accomplished using a pre-trianed LBP cascade classifier. The detected faces are extracted using the bounding box and feed into a pre-trained Inceptionv3 model. The Inception model's output is interpreted to get the best guessed age range class for each face and the results are displaied on screen.  
  
The pre-trained classifier and Inception models are included in this repository.
  
The application was created for a class assignment, as such please exclude the poor coding choices due to the hastyness under which this app was developed. With this app, I was able to learn some basics regarding android development using OpenCV and Tensorflow.

!(Screenshot)[]

## Building the app ##
Import the application into Android Studio as a gradel application. It requires a device with a Camera and Android OS version <= 23. Everything should be included, aside from build dependencies provided by Android Studio.

## How to Use ##
