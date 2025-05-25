import cv2
import numpy as np
import matplotlib.pyplot as plt

def merge(n1,n2,H):
        im1=n1[2]
        im2=n2[2]
        rows1, cols1 = im1.shape[:2]
        rows2, cols2 = im2.shape[:2]

        list_of_points_1 = np.float32([[0,0], [0, rows1],[cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2) #coordinates of a reference image
        temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2) #coordinates of second image

        list_of_points_2 = cv2.perspectiveTransform(temp_points, H)#calculate the transformation matrix

        list_of_points = np.concatenate((list_of_points_1,list_of_points_2), axis=0)

        [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
        
        translation_dist = [-x_min,-y_min]
        
        H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

        output_img = cv2.warpPerspective(im2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
        output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = im1

        return output_img

class Stitch:
    def __init__(self,im:list,nfeat=2000,MIN=5,tresh=0.6):#im= liste de liens vers des images
        self.images=[]
        self.nfeat=nfeat
        orb=cv2.ORB_create(nfeatures=nfeat)
        self.MIN_MATCH_COUNT=MIN
        self.treshold=tresh
        for i in im:
            n=cv2.imread(i)
            k,d=orb.detectAndCompute(n,None)
            self.images.append((k,d,n))
       
        

    def homography(self,n2,n1):
   
        k1,d1=n1[0],n1[1]
        k2,d2=n2[0],n2[1]

        bf=cv2.BFMatcher_create(cv2.NORM_HAMMING)
        matches=bf.knnMatch(d1,d2,k=2)
        good=[]
        count=0
        for m,n in matches:
            if m.distance <self.treshold*n.distance:
                good.append(m)
                count+=1
        
        if count > self.MIN_MATCH_COUNT:
            print("number of corresponding point: ",count)
            src_pts= np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst_pts = np.float32([ k2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M,_=cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            return M
        else:
            return None

    
        
    def stitching(self):
        image=self.images.pop()
        count=0
        while len(self.images)>0:
            count+=1
           
            next=self.images.pop()
            H=self.homography(image,next)
            print(H)
            if H is None:
                print("H null")
                self.images.insert(0,next)
            else:
                print("H good")
                im=merge(image,next,H)
                self.nfeat+=2000
                orb=cv2.ORB_create(nfeatures=self.nfeat)
                k,d=orb.detectAndCompute(im,None)
                image=(k,d,im)
                del k,d,im
            del H, next
        print("number of time we evaluated H: ",count)
        return image[2]

