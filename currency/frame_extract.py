import cv2
 
# Opens the Video file
cap= cv2.VideoCapture('data/vids/50.mp4')
i=0
j=0
note = 'fifty'
dir = 'data/frames/'+note+'/'
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    if i%1 == 0:
        # print(cv2.Laplacian(frame, cv2.CV_64F).var())
        if cv2.Laplacian(frame, cv2.CV_64F).var() >= 450:
            cv2.imwrite(dir+note+str(j)+'.jpg',frame)
            j+=1
    i+=1
    
 
cap.release()
cv2.destroyAllWindows()