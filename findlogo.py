import cv2
import numpy as np
import argparse

def detect_logo(logo_path, photo_path):
    # Load images in grayscale
    logo = cv2.imread(logo_path, cv2.IMREAD_GRAYSCALE)
    photo = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
    
    if logo is None or photo is None:
        print("Error loading images.")
        return
    
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=2000)
    
    # Detect keypoints and compute descriptors
    kp1, des1 = orb.detectAndCompute(logo, None)
    kp2, des2 = orb.detectAndCompute(photo, None)
    
    # Use a Brute Force Matcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    if len(good_matches) < 3:
        print("Not enough good matches found.")
        return
    
    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Estimate affine transformation
    M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
    
    if M is None:
        print("Affine transformation could not be estimated.")
        return
    
    # Get the logo's bounding box
    h, w = logo.shape[:2]
    logo_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    transformed_corners = cv2.transform(logo_corners, M)
    
    # Draw detected region on the photo
    photo_color = cv2.imread(photo_path)
    cv2.polylines(photo_color, [np.int32(transformed_corners)], isClosed=True, color=(0, 255, 0), thickness=2)
    
    # Draw matches
    match_img = cv2.drawMatches(logo, kp1, photo_color, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Display results
    #cv2.imshow("Detected Logo", photo_color)
    #cv2.imshow("Feature Matches", match_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite("detected_logo.jpg", photo_color)
    cv2.imwrite("feature_matches.jpg", match_img)
    print("Results saved as 'detected_logo.jpg' and 'feature_matches.jpg'")

    
    # Print affine parameters
    print("Affine Transformation Matrix:")
    print(M)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logo", help="Path to the logo image")
    parser.add_argument("photo", help="Path to the photo containing the logo")
    args = parser.parse_args()
    
    detect_logo(args.logo, args.photo)
