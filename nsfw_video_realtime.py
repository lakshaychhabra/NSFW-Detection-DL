import cv2
from keras.models import load_model
import numpy as np
from collections import deque
import warnings
warnings.filterwarnings("ignore")

model = load_model("/content/drive/My Drive/nsfw/Final_weights.h5")

labels = {0 : "Neutral", 1 : "Porn", 2 : "Sexy"}


size = 128
# input_vid = "2.mp4"
output_vid = "1.mp4"

Q = deque(maxlen=size)


from google.colab.patches import cv2_imshow
import imutils

# vs = cv2.VideoCapture(0)
vs = cv2.VideoCapture(take_photo())
writer = None
(W, H) = (None, None)
 
# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()
 
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break
 
    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # frame=imutils.resize(frame, width=min(100, frame.shape[1])) 
    fc = cv2.CascadeClassifier("drive/My Drive/nsfw/haarcascade_upperbody.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    upper=fc.detectMultiScale(gray,1.1,1)

    for (a,b,c,d) in upper:
      cv2.rectangle(frame,(a,b),(a+c,b+d),(0,0,255),2)
    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame/255.0
    frame = cv2.resize(frame, (224, 224)).astype("float32")
    
#     frame -= mean
    
    # make predictions on the frame and then update the predictions
    # queue
    preds = model.predict(np.expand_dims(frame, axis=0))[0]
    print(preds)
    Q.append(preds)

    # perform prediction averaging over the current history of
    # previous predictions

    results = np.array(Q).mean(axis=0)
    i = np.argmax(preds)
    label = labels[i]
    # draw the activity on the output frame
    text = "activity: {}:".format(label)
    cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5)

    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_vid, fourcc, 30, (W, H), True)

    # write the output frame to disk
    writer.write(output)

    # show the output image
    cv2_imshow(output)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
        
# release the file pointers
print("[INFO] cleaning up...")
# writer.release()
vs.release()
