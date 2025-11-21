def find_corners_from_mask(polygon_points, epsilon_factor=0.02):
    
    # 1. Prepare points for OpenCV
    polygon_points_cv = polygon_points.reshape(-1, 1, 2).astype(np.int32)
    
    # 2. Calculate perimeter (necessary for setting epsilon)
    perimeter = cv2.arcLength(polygon_points_cv, True)
    epsilon = perimeter * epsilon_factor
    
    # 3. Approximate the polygon
    approx_corners = cv2.approxPolyDP(polygon_points_cv, epsilon, True)
    
    # 4. Success Check: If 4 corners are found, we return them
    if len(approx_corners) == 4:
        return approx_corners.reshape(4, 2)
        
    # 5. Fallback: If simplification fails, find the four most extreme points
    reshaped_points = polygon_points.reshape(-1, 2)
    
    sum_coords = reshaped_points.sum(axis=1)
    tl = reshaped_points[np.argmin(sum_coords)] 
    br = reshaped_points[np.argmax(sum_coords)] 
    
    diff_coords = np.diff(reshaped_points, axis=1)
    tr = reshaped_points[np.argmin(diff_coords)] 
    bl = reshaped_points[np.argmax(diff_coords)] 
    
    return np.array([tl, tr, br, bl], dtype=np.int32)
