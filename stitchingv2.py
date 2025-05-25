import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def merge(im1,im2,H):
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
        self.kp={}
        self.desc={}
        for j,i in enumerate(im):
            n=cv2.imread(i)
            k,d=orb.detectAndCompute(n,None)
            self.images.append(n)
            self.kp[j]=k
            self.desc[j]=np.array(d)


    def matching(self,image_indx=0):
        bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)
        ref=[]
        for j in self.desc:
            if j!=image_indx:
                #print('image index',j)
                ref+=[j]*len(self.desc[j])
        #ref=np.vstack([[j]*len(self.desc[j]) for j in self.desc if j!=image_indx])
        knn_matches= bf.knnMatch(self.desc[image_indx],np.vstack([self.desc[j] for j in self.desc if j!=image_indx]),k=2)
        #knn pour déterminer les points correspondants 
        # pour chaque point d'intérêts de l'image
        mpp=defaultdict(list)
       # print(knn_matches)
        #print(knn_matches.shape)
        for m,n in knn_matches:
          #  for m,n in match_g:
                img_j= ref[m.trainIdx]
                if m.distance<self.treshold*n.distance:
                    mpp[img_j].append(m)
        del bf, knn_matches
        return mpp
            
        

    def homography(self,indx1,indx2,good):

        k1=self.kp[indx1]
        k2=self.kp[indx2]
       # d1=self.desc[indx1*self.nfeat:(indx1+1)*self.nfeat]
        #d2=self.desc[indx2*self.nfeat:(indx2+1)*self.nfeat]

        # bf=cv2.BFMatcher_create(cv2.NORM_HAMMING)
        # matches=bf.knnMatch(d1,d2,k=2)
        # good=[]
        # count=0
        # for m,n in matches:
        #     if m.distance <self.treshold*n.distance:
        #         good.append(m)
        #         count+=1
        
       # if count > self.MIN_MATCH_COUNT:
        #print("number of corresponding point: ",count)
      
        src_pts= np.float32([k1[m.queryIdx%len(self.desc[indx1])].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([ k2[m.trainIdx%len(self.desc[indx2])].pt for m in good ]).reshape(-1,1,2)
        M,_=cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        del good, src_pts, dst_pts
        return M
        # else:
        #     return None

    def stitching(self):
        count=0
        while True:
            if len(self.images)==1:
                return self.images[0]
            print(count)
            count+=1
            match=self.matching()
            #print("match")
            maxi=None
            max=0
            for i in match:
                if len(match[i])>max:
                    max=len(match[i])
                    maxi=i
            if maxi!=None and max>self.MIN_MATCH_COUNT:
                H=self.homography(0,maxi,match[maxi])
                print(max)
                print(H)
                a=self.images.pop(0)
                b=self.images.pop(maxi-1)
                new_im=merge(b,a,H)
                #print("Merged")
                plt.imshow(a)
                plt.show()
                plt.imshow(b)
                plt.show()
                plt.imshow(new_im)
                plt.show()
                new_size=int(0.75*(len(self.desc[0])+len(self.desc[maxi])))
                orb=cv2.ORB_create(nfeatures=new_size)
                k,d=orb.detectAndCompute(new_im,None)
                #print(k,d)
                self.images.append(new_im)
                del self.kp[0], self.kp[maxi], self.desc[0], self.desc[maxi]

                new_kp={j:self.kp[i] for j,i in enumerate(self.kp)}
                new_desc={j:self.desc[i] for j,i in enumerate(self.desc)}
                self.kp=new_kp
                self.desc=new_desc
                self.kp[len(new_kp)]=k
                self.desc[len(new_desc)]=d
                del new_kp, new_desc, orb, new_im, k, d, H
            else:
                print("no good match")
                self.images.append(self.images.pop(0))
                k,d=self.kp[0],self.desc[0]
                del self.kp[0], self.desc[0]
                new_kp={j:self.kp[i] for j,i in enumerate(self.kp)}
                new_desc={j:self.desc[i] for j,i in enumerate(self.desc)}
                self.kp=new_kp
                self.desc=new_desc
                self.kp[len(new_kp)]=k
                self.desc[len(new_desc)]=d
                del new_kp, new_desc




#Essayer d'implémenter avec les meilleures paires en premier



