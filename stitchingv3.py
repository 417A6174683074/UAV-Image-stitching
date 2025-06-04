import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict



def gaussian_pyramid(img, levels):
    G = img.copy()
    gp = [G]
    for i in range(levels):
        G = cv2.pyrDown(G)
        gp.append(G)
    return gp

def laplacian_pyramid(img, levels):
    gp = gaussian_pyramid(img, levels)
    lp = [gp[-1]]
    for i in range(levels - 1, -1, -1):
        size = (gp[i].shape[1], gp[i].shape[0])
        GE = cv2.pyrUp(gp[i + 1], dstsize=size)
        L = cv2.subtract(gp[i], GE)
        lp.append(L)
    return lp[::-1]

def blend_pyramids(lpA, lpB, lpMask):
    blended = []
    for la, lb, lm in zip(lpA, lpB, lpMask):
        la = la.astype(np.float32)
        lb = lb.astype(np.float32)
        if la.shape[:2] != lm.shape[:2]:
            lm = cv2.resize(lm, (la.shape[1], la.shape[0]), interpolation=cv2.INTER_LINEAR)
        if len(lm.shape) == 2:
            lm = np.repeat(lm[:, :, np.newaxis], la.shape[2], axis=2)
        elif lm.shape[2] != la.shape[2]:
            lm = np.repeat(lm[:, :, 0:1], la.shape[2], axis=2)
        lm = lm.astype(np.float32)
        blended.append(la * lm + lb * (1.0 - lm))
    return blended
def reconstruct_from_laplacian(lp):
    img = lp[-1]
    for i in range(len(lp) - 2, -1, -1):
        size = (lp[i].shape[1], lp[i].shape[0])
        img = cv2.pyrUp(img, dstsize=size)
        img = cv2.add(img, lp[i])
    return img

def multiband_blend(img1, img2, mask, levels=4):
    # Normalize and prepare images
    img1 = img1.astype(np.float32) / 255
    img2 = img2.astype(np.float32) / 255
    mask = cv2.merge([mask, mask, mask]).astype(np.float32)

    lpA = laplacian_pyramid(img1, levels)
    lpB = laplacian_pyramid(img2, levels)
    gpM = gaussian_pyramid(mask, levels)
    blended = blend_pyramids(lpA, lpB, gpM)
    result = reconstruct_from_laplacian(blended)
    return (result * 255).astype(np.uint8)

def merge_multiband(im1, im2, H, levels=4):
    # Dimensions des images d'entrée
    rows1, cols1 = im1.shape[:2]
    rows2, cols2 = im2.shape[:2]

    # Coins des images
    pts1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)
    pts2_ = cv2.perspectiveTransform(pts2, H)

    all_pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(all_pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_pts.max(axis=0).ravel() + 0.5)
    trans = [-xmin, -ymin]

    # Translation matrix
    H_translation = np.array([[1, 0, trans[0]], [0, 1, trans[1]], [0, 0, 1]])

    # Canevas de sortie
    warped_img2 = cv2.warpPerspective(im2, H_translation @ H, (xmax - xmin, ymax - ymin))
    canvas_img1 = np.zeros_like(warped_img2)
    canvas_img1[trans[1]:trans[1] + rows1, trans[0]:trans[0] + cols1] = im1

    # Masques binaires pour la fusion
    mask1 = np.any(canvas_img1 > 0, axis=2).astype(np.float32)
    mask2 = np.any(warped_img2 > 0, axis=2).astype(np.float32)
    overlap = np.logical_and(mask1, mask2).astype(np.float32)

    # Génère un alpha progressif horizontal dans la zone d'overlap
    alpha = np.zeros_like(overlap)
    if np.sum(overlap) > 0:
        y_indices, x_indices = np.where(overlap > 0)
        x_min = np.min(x_indices)
        x_max = np.max(x_indices)
        alpha_gradient = np.linspace(1, 0, x_max - x_min + 1)
        for i, x in enumerate(range(x_min, x_max + 1)):
            alpha[:, x][overlap[:, x] > 0] = alpha_gradient[i]
    alpha = alpha.astype(np.float32)

    # Appliquer le blending multi-bande
    result = multiband_blend(canvas_img1, warped_img2, alpha, levels)
    return result


def GetNewFrameSizeAndMatrix(HomographyMatrix, Sec_ImageShape, Base_ImageShape):
    # Reading the size of the image
    (Height, Width) = Sec_ImageShape


    InitialMatrix = np.array([[0, Width - 1, Width - 1, 0],
                              [0, 0, Height - 1, Height - 1],
                              [1, 1, 1, 1]])

  
    FinalMatrix = np.dot(HomographyMatrix, InitialMatrix)

    [x, y, c] = FinalMatrix
    x = np.divide(x, c)
    y = np.divide(y, c)

    # Finding the dimentions of the stitched image frame and the "Correction" factor
    min_x, max_x = int(round(min(x))), int(round(max(x)))
    min_y, max_y = int(round(min(y))), int(round(max(y)))

    New_Width = max_x
    New_Height = max_y
    Correction = [0, 0]
    if min_x < 0:
        New_Width -= min_x
        Correction[0] = abs(min_x)
    if min_y < 0:
        New_Height -= min_y
        Correction[1] = abs(min_y)

    # Again correcting New_Width and New_Height
    # Helpful when secondary image is overlaped on the left hand side of the Base image.
    if New_Width < Base_ImageShape[1] + Correction[0]:
        New_Width = Base_ImageShape[1] + Correction[0]
    if New_Height < Base_ImageShape[0] + Correction[1]:
        New_Height = Base_ImageShape[0] + Correction[1]

    # Finding the coordinates of the corners of the image if they all were within the frame.
    x = np.add(x, Correction[0])
    y = np.add(y, Correction[1])
    OldInitialPoints = np.float32([[0, 0],
                                   [Width - 1, 0],
                                   [Width - 1, Height - 1],
                                   [0, Height - 1]])
    NewFinalPonts = np.float32(np.array([x, y]).transpose())

    # Updating the homography matrix. Done so that now the secondary image completely
    # lies inside the frame
    HomographyMatrix = cv2.getPerspectiveTransform(OldInitialPoints, NewFinalPonts)
    
    return [New_Height, New_Width], Correction, HomographyMatrix





def image_similarity_hist(img1, img2):
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])

    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)

    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)  # ∈ [-1, 1]
    return score

def crop_valid_region(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y+h, x:x+w]

# def merge2(im1,img2,H):
#     h,w=im1.shape[:2]
#     warped_img2 = cv2.warpPerspective(img2, H, canvas_size=(int(h*1.25),int(w*1.25)))
#     mask1 = (img1_canvas > 0).astype(np.uint8)
#     mask2 = (warped_img2 > 0).astype(np.uint8)
#     intersection = cv2.bitwise_and(mask1, mask2)

#     alpha = compute_alpha_mask(intersection)
#     blended = multiband_blend(img1_canvas, warped_img2, alpha)


def find_best_matching_patch(img1: np.ndarray, imgs, patch_size0=1500, patch_size1=2000, step: int=500,test=False):
    """
    Balaye img1 par patchs et cherche leur meilleur match dans img2 via matchTemplate.
    Renvoie les coordonnées du patch dans img1 et la position correspondante dans img2.
    """
    h1, w1 = img1.shape[:2]
    best_score = -1
    best_patch_coords = None
    best_match_coords = None
    best_im=None
    for k,img2 in enumerate(imgs):
        if test and image_similarity_hist(img1,img2)<0.6:
            continue
        for y in range(0, h1 - patch_size1, step):
            for x in range(0, w1 - patch_size0, step):
                patch = img1[y:y+patch_size1, x:x+patch_size0]
                res = cv2.matchTemplate(img2, patch, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > best_score:
                    best_score = max_val
                    best_patch_coords = (x, y)
                    best_match_coords = max_loc
                    best_im=k
                if best_score>0.7:
                    return best_patch_coords, best_match_coords, (patch_size0,patch_size1), best_im

    return best_patch_coords, best_match_coords, (patch_size0,patch_size1), best_im


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
        
        self.MIN_MATCH_COUNT=MIN
        self.treshold=tresh
        for j,i in enumerate(im):
            n=cv2.imread(i)
            self.images.append(n)



    # def matching(self,image_indx=0):
    #     bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)
    #     ref=[]
    #     for j in self.desc:
    #         if j!=image_indx:
    #             ref+=[j]*len(self.desc[j])
    #     knn_matches= bf.knnMatch(self.desc[image_indx],np.vstack([self.desc[j] for j in self.desc if j!=image_indx]),k=2)
    #     #knn pour déterminer les points correspondants 
    #     # pour chaque point d'intérêts de l'image
    #     mpp=defaultdict(list)
    #     for m,n in knn_matches:
    #       #  for m,n in match_g:
                
    #             img_j= ref[m.trainIdx]
    #             if m.distance<self.treshold*n.distance:
    #                 m.trainIdx-=ref.index(img_j)
    #                 mpp[img_j].append(m)
    #                 #indices[img_j]+=1
    #     del bf, knn_matches,ref
    #     return mpp
            
    # def show_matches(self, im1, im2,src, dst,color=(0,255,0)):
    #     img1 = self.images[idx1]
    #     img2 = self.images[idx2]
    #     kp1 = self.kp[idx1]
    #     kp2 = self.kp[idx2]
    #     # match=matches.copy()
    #     # for m,j in match:
    #     #     m.trainIdx=j
    #     match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
    #                                 matchesMask=None,
    #                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    #     h1, w1 = img1.shape[:2]
        
    #     # Redessine les lignes avec plus d'intensité
    #     for i, m in enumerate(matches):
            
    #         pt1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
    #         pt2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int))
    #         pt2_shifted = (int(pt2[0] + w1), int(pt2[1]))  # car concaténation horizontale

    #         cv2.line(match_img, pt1, pt2_shifted, color, thickness=2, lineType=cv2.LINE_AA)
    #         cv2.circle(match_img, pt1, 4, (255,0,0), -1)
    #         cv2.circle(match_img, pt2_shifted, 4, (255,0,0), -1)

    #     plt.figure(figsize=(16, 8))
    #     plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    #     plt.axis('off')
    #     plt.show()

    def compute_homography_from_best_roi(self,img1, img2,x1, y1, x2, y2, w, h):
        roi1 = img1[y1:y1+h, x1:x1+w]
        roi2 = img2[y2:y2+h, x2:x2+w]
        print("zone d'intérêt:")
        plt.imshow(roi1)
        plt.show()
        plt.imshow(roi2)
        plt.show()
        sift = cv2.SIFT_create(2000)
        kp1, des1 = sift.detectAndCompute(roi1, None)
        kp2, des2 = sift.detectAndCompute(roi2, None)

        if des1 is None or des2 is None or len(kp1) < self.MIN_MATCH_COUNT or len(kp2) < self.MIN_MATCH_COUNT:
            return None

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = [m for m, n in matches if m.distance < self.treshold * n.distance]
        if len(good) < self.MIN_MATCH_COUNT:
            print("not enough good values")
            return None
        print(len(good))
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        src_pts += np.array([x1, y1])
        dst_pts += np.array([x2, y2])

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H


    def stitching(self):
        count=0
        while True:
            if len(self.images)==1:
                return self.images[0]
            print("iteration ",count)
            count+=1
            #match=self.matching()
            #print("match")
            img1=self.images.pop(0)
            print("matching...")
            (x1, y1), (x2, y2), (w, h), indice = find_best_matching_patch(img1, self.images)
           # (x,y),loc,cible=find_super(sub_im,self.images)
            img2=self.images.pop(indice)
            print("computing homography...")
            H=self.compute_homography_from_best_roi(img1,img2,x1,y1,x2,y2,w,h)
            
            # [new_height, new_width], Correction, NewHomographyMatrix = GetNewFrameSizeAndMatrix(H, img2.shape[:2], img1.shape[:2])
            # WarpedImage = cv2.warpPerspective(img2, NewHomographyMatrix, (new_width, new_height))
            # canvas = np.zeros((new_height, new_width, 3), dtype=img1.dtype)
            # img1[Correction[1]:Correction[1] + WarpedImage.shape[0], Correction[0]:Correction[0] + WarpedImage.shape[1]] = WarpedImage
            # res=canvas

            #res= merge_multiband(img1, img2, H)
            # crop_valid_region(merge(img2,img1,H))
            res=merge(img2,img1,H)
            print("images originales:")
            plt.imshow(img1)
            plt.show()
            plt.imshow(img2)
            plt.show()
            print("résultat:")
            plt.imshow(res)
            plt.show()
            self.images.append(res)
              
