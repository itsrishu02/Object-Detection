import numpy as np
import sys
import os
import cv2
import pickle
import time
import pyautogui
from scipy import ndimage
import mediapipe as mp
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt


####################################################################
# Calculating Hog Features

def compute_gradient(image):
    # Compute the x and y gradients of the image
    gx = ndimage.sobel(image, axis=0)
    gy = ndimage.sobel(image, axis=1)

    # Compute the magnitude and orientation
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180

    return magnitude, orientation

def compute_histogram(magnitude, orientation, bins=9):
    # Compute the histogram
    histogram = np.zeros(bins)
    bin_width = 180 // bins

    # Compute the bin each gradient orientation belongs to
    bin_indices = np.floor(orientation / bin_width).astype(int)
    bin_indices_next = (bin_indices + 1) % bins

    # Compute the weights for interpolation
    weights = (orientation - bin_indices * bin_width) / bin_width

    for i in range(bins):
        # Add the magnitude to the relevant bin, using interpolation
        curr_bin_mask = (bin_indices == i)
        histogram[i] += np.sum(magnitude[curr_bin_mask] * (1 - weights[curr_bin_mask]))

        next_bin_mask = (bin_indices_next == i)
        histogram[i] += np.sum(magnitude[next_bin_mask] * weights[next_bin_mask])

    return histogram

def normalize_block(histograms):
    # Normalize the block
    block = np.hstack(histograms)
    norm = np.linalg.norm(block)
    if norm > 0:
        return block / norm
    else:
        return block

def compute_hog_features(image, cell_size=(8, 8), block_size=(2, 2)):
    # Compute the gradient magnitude and orientation
    magnitude, orientation = compute_gradient(image)

    # Compute the histogram of oriented gradients
    cell_histograms = np.zeros((image.shape[0] // cell_size[0], image.shape[1] // cell_size[1], 9))
    for i in range(cell_histograms.shape[0]):
        for j in range(cell_histograms.shape[1]):
            cell_histograms[i, j] = compute_histogram(
                magnitude[i*cell_size[0]:(i+1)*cell_size[0], j*cell_size[1]:(j+1)*cell_size[1]],
                orientation[i*cell_size[0]:(i+1)*cell_size[0], j*cell_size[1]:(j+1)*cell_size[1]]
            )

    # Normalize the histograms in overlapping blocks
    hog_features = []
    for i in range(cell_histograms.shape[0] - block_size[0] + 1):
        for j in range(cell_histograms.shape[1] - block_size[1] + 1):
            block_histograms = cell_histograms[i:i+block_size[0], j:j+block_size[1]]
            hog_features.append(normalize_block(block_histograms))

    return np.hstack(hog_features)

#########################################################################
# Creating bounding box

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=True, 
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)

def increase_bbox(bbox, scale_factor, img_shape):
    x, y, w, h = bbox
    delta_w = int((scale_factor - 1) * w / 2)
    delta_h = int((scale_factor - 1) * h / 2)
    x -= delta_w
    y -= delta_h
    w += 2 * delta_w
    h += 2 * delta_h

    # Ensure bounding box remains within image boundaries
    x = max(0, x)
    y = max(0, y)
    w = min(w, img_shape[1] - x)
    h = min(h, img_shape[0] - y)

    return x, y, w, h

########################################################################
# Extracting roi from given image

def extract_roi_from_image(img, scale_factor=1.3):

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_points = []
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * img.shape[1])
                y = int(landmark.y * img.shape[0])
                landmark_points.append([x, y])

            landmark_points = np.array(landmark_points)  
            x, y, w, h = cv2.boundingRect(landmark_points) 
            x, y, w, h = increase_bbox((x, y, w, h), scale_factor, img.shape)
            
            # Extract region of interest (ROI) using bounding box coordinates
            roi = img[y:y+h, x:x+w]
            return roi
        

    return None


#########################################################################
# Extracting featurs from train and test dataset

def total_features(dataset_folder_closed,dataset_folder_open):
    # Initialize lists to store HOG feature vectors and labels
    hog_features = []
    labels = []

    # Process closed hand images
    for filename in os.listdir(dataset_folder_closed):
        if filename.endswith(".jpg"):
            # Load the image
            img_path = os.path.join(dataset_folder_closed, filename)
            img = cv2.imread(img_path)
            img = cv2.flip(img, 1)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(imgRGB)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_points = []
                    for landmark in hand_landmarks.landmark:
                        x = int(landmark.x * img.shape[1])
                        y = int(landmark.y * img.shape[0])
                        landmark_points.append([x, y])

                    landmark_points = np.array(landmark_points)  
                    x, y, w, h = cv2.boundingRect(landmark_points) 
                    scale_factor = 1.3
                    x, y, w, h = increase_bbox((x, y, w, h), scale_factor,img.shape)
                    
                    # Extract region of interest (ROI) using bounding box coordinates
                    roi = img[y:y+h, x:x+w]

                    if roi.size != 0:
                        z = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
                        im1 = cv2.resize(z, (64, 128))
                        hog_fe = compute_hog_features(im1)
                        hog_features.append(hog_fe)
                        labels.append(1)  # Label for closed hand
                    else:
                        print("Skipping HOG computation for ROI due to insufficient size.")


    # Process open hand images
    for filename in os.listdir(dataset_folder_open):
        if filename.endswith(".jpg"):
            # Load the image
            img_path = os.path.join(dataset_folder_open, filename)
            img = cv2.imread(img_path)
            img = cv2.flip(img, 1)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(imgRGB)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_points = []
                    for landmark in hand_landmarks.landmark:
                        x = int(landmark.x * img.shape[1])
                        y = int(landmark.y * img.shape[0])
                        landmark_points.append([x, y])

                    landmark_points = np.array(landmark_points)  
                    x, y, w, h = cv2.boundingRect(landmark_points) 
                    scale_factor = 1.3
                    x, y, w, h = increase_bbox((x, y, w, h), scale_factor,img.shape)
                    
                    # Extract region of interest (ROI) using bounding box coordinates
                    roi = img[y:y+h, x:x+w]

                    if roi.size != 0:
                        m = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
                        im2 = cv2.resize(m, (64, 128))
                        hog_f = compute_hog_features(im2)
                        hog_features.append(hog_f)
                        labels.append(0)  # Label for open hand
                    else:
                        print("Skipping HOG computation for ROI due to insufficient size.")

    # Convert lists to numpy arrays
    hog_features = np.array(hog_features)
    labels = np.array(labels)

    return hog_features, labels

##########################################################################################
# Training model using SVM

def training_model(tr_cl,tr_op):

    print("Finding hog features of train set")
    X_train, y_train = total_features(tr_cl,tr_op)

    
    print("Training model")

    # Flatten HOG feature vectors
    X_train_flattened = np.array([i.flatten() for i in X_train])

    # Initialize SVM classifier
    svm_model = svm.SVC(kernel='linear')

    # Train the SVM classifier
    svm_model.fit(X_train_flattened, y_train)

    # Save the model to a file
    with open('svm_model.pkl', 'wb') as f:
        pickle.dump(svm_model, f)



#################################################################################
# Testing model

def testing_model(te_cl,te_op):

    print("Finding hog features of test set")
    X_test, y_test = total_features(te_cl,te_op)

    X_test_flattened = np.array([j.flatten() for j in X_test])

        # Loading the saved trained model
    with open('svm_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)

    # Predict on the testing set
    y_pred = svm_model.predict(X_test_flattened)

    # Compute the confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)

    # Compute True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN)
    TP = conf_mat[1, 1]
    TN = conf_mat[0, 0]
    FP = conf_mat[0, 1]
    FN = conf_mat[1, 0]

    # Compute True Positive Rate (TPR) or Recall or Sensitivity
    TPR = TP / (TP + FN)
    print("True Positive Rate (TPR):", TPR)

    # Compute False Positive Rate (FPR)
    FPR = FP / (FP + TN)
    print("False Positive Rate (FPR):", FPR)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Calculate precision, recall, F1-score
    print(classification_report(y_test, y_pred))

    # Calculate the probability scores of each point in the training set
    y_score = svm_model.decision_function(X_test_flattened)

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    print("AUROC:", roc_auc)


    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.show()

    return None



##################################################################################

def main_train(train):

    train_closed = os.path.join(train, 'closed')
    train_open = os.path.join(train, 'open')


    training_model(train_closed, train_open)

################################################################################
def main_test(test):

    test_closed = os.path.join(test, 'closed')
    test_open = os.path.join(test, 'open')

    testing_model(test_closed, test_open)

################################################################################
    
def music_palyer():

    # Loading the saved trained model
    with open('svm_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    # Open the video capture
    cap = cv2.VideoCapture(0)

    start_time = time.time()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Capture the image after 4 seconds
        if time.time() - start_time >= 4:
            cv2.imwrite('captured_image.jpg', frame)
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, releasing the capture
    cap.release()
    cv2.destroyAllWindows()

    rois = extract_roi_from_image(frame)

    image = cv2.cvtColor(rois, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (64, 128))

    hog_ = compute_hog_features(image)

    # Flatten the HOG feature vector
    hog_vector_flattened = np.array(hog_).flatten()

    # Predict the label using the trained SVM classifier
    label = loaded_model.predict([hog_vector_flattened]) 

    # Printing the predicted label
    print("Predicted label (level):", label[0])

    # Function to pause the music player
    def pause_music():
        pyautogui.press('playpause')

    # Function to play the next song
    def play_next_song():
        pyautogui.press('nexttrack')

    # Checking the predicted label and controling the music player accordingly
    if label[0] == 0:
        # Pause the music player
        pause_music()
        print("Paused music")
    elif label[0] == 1:
        # Play the next song
        play_next_song()
        print("Played next song")

###############################################################################
        
def music_palyer_2(image_path):

    # Loading the saved trained model
    with open('svm_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)


    img = cv2.imread(image_path)

    rois = extract_roi_from_image(img)

    # Now you have your frame captured from the webcam
    image = cv2.cvtColor(rois, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (64, 128))

    hog_ = compute_hog_features(image)

    # Flatten the HOG feature vector
    hog_vector_flattened = np.array(hog_).flatten()

    # Predict the label (level) using the trained SVM classifier
    label = loaded_model.predict([hog_vector_flattened]) 

    # Printing the predicted label
    print("Predicted label (level):", label[0])

    # Function to pause the music player
    def pause_music():
        pyautogui.press('playpause')

    # Function to play the next song
    def play_next_song():
        pyautogui.press('nexttrack')

    # Checking the predicted label and controling the music player accordingly
    if label[0] == 0:
        # Pause the music player
        pause_music()
        print("Paused music")
    elif label[0] == 1:
        # Play the next song
        play_next_song()
        print("Played next song")
        
###############################################################################

if __name__ == "__main__":

    operation_type = int(sys.argv[1])

    if operation_type == 1:
        train_folder = sys.argv[2]
        main_train(train_folder)
        print("Model Trained")

    if operation_type == 2:
        test_folder = sys.argv[2]
        main_test(test_folder)


    if operation_type == 3:
        music_palyer()

    if operation_type == 4:
        image_path = sys.argv[2]
        music_palyer_2(image_path)