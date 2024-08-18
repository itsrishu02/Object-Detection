```bash

# Required Libraries

- numpy
- os
- cv2
- pickle
- time
- scipy
- sklearn
- mediapipe
- matplotlib



##  To Train the Model

It requires two arguments:
a) `1` to represent the training part.
b) `<train_folder>`: It should contain two folders named "open" and "closed" and respective images in them.


Python main.py 1 train

##  To Test the Model

It requires two arguments:
a) `2` to represent the testing part.
b) `<test_folder>`: It should contain two folders named "open" and "closed" and respective images in them.


Python main.py 2 test

### Next commands can be excuted only after training the model

## Part 2: Application Type 1 (Webcam)

It requires only one argument:
a) `3` to represent the webcam model.


python main.py 3


## Part 3: Application Type 2 (Image)

It requires two arguments:
a) `4` to represent the image input model.
b) Image address.


python main.py 4 img.jpg

