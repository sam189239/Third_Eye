import cv2
 
# Opens the Video file
i=0
j=0
note = 'ten'
cap= cv2.VideoCapture('data/vids/'+note+'.mp4')
dir = 'data/Training frames/'+note+'/'
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    if i%10 == 0:
        # print(cv2.Laplacian(frame, cv2.CV_64F).var())
        if cv2.Laplacian(frame, cv2.CV_64F).var() >= 100:
            frame = cv2.resize(frame, (224,224))
            cv2.imwrite(dir+note+str(j)+'.jpg',frame)
            j+=1
    i+=1
    
 
cap.release()
cv2.destroyAllWindows()