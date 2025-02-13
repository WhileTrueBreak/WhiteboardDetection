import random
import numpy as np
from sklearn.linear_model import HuberRegressor
import cv2

def gen_points(n,a,b,c,r):
    points = []
    for _ in range(n):
        random_z = random.random()*r*2-r
        random_x = random.random()
        random_y = random.random()
        z = a*random_x+b*random_y+c
        points.append((random_x,random_y,z+random_z))
    return np.array(points)

def solve_depth(mask, depth_image):
    v, u = np.indices(mask.shape)
    u = u[mask]
    v = v[mask]
    A = np.column_stack((u,v,np.ones_like(u)))
    b = depth_image[mask]
    try:
        model = HuberRegressor()
        model.fit(A,b)
        A, B, C = model.coef_
        C += model.intercept_
        return np.array([A, B, C])
    except:
        return [0,0,0]
    print(u.shape, v.shape, depths.shape)

def solve_plane(points):
    # Ax+By+C=z
    if points.shape[0] <= 4: return [0,0,0]
    A = np.column_stack((points[:,0],points[:,1],np.ones_like(points[:,0])))
    b = points[:,2]
    try:
        model = HuberRegressor()
        model.fit(A,b)
        A, B, C = model.coef_
        C += model.intercept_
        return np.array([A, B, C])
    except:
        return [0,0,0]

def solve_mask_quad(mask):
    shapes = []
    mask = (mask*255).astype('uint8')
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        epsilon = 0.02*cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        shapes.append(approx)
    return shapes

def uv2xyz(u, v, da, db, dc, realsense):
    d = u*da+v*db+dc
    z = d*realsense.depth_scale
    x = (u-realsense.color_intrinsics.ppx)*z/realsense.color_intrinsics.fx
    y = (v-realsense.color_intrinsics.ppy)*z/realsense.color_intrinsics.fy
    return np.array([x,y,z])

def plane2uv(x, y, z, a, b, c, realsense):
    normal = np.array([a,b,-1])/np.sqrt(a**2+b**2+1)

    beta = np.arcsin(normal[0])
    alpha = np.atan2(-normal[1]/np.cos(beta), normal[2]/np.cos(beta))

    rx = np.array([[1, 0            , 0             ], 
                   [0, np.cos(alpha),-np.sin(alpha)], 
                   [0, np.sin(alpha), np.cos(alpha) ]])
    ry = np.array([[ np.cos(beta), 0, np.sin(beta)],
                   [            0, 1,            0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    r = np.dot(rx,ry)

    xyz = np.array([x,y,z]).T
    xyz = np.dot(xyz, r)

    mask = xyz[:,2] != 0
    u = np.zeros_like(xyz[:,2])
    v = np.zeros_like(xyz[:,2])
    u[mask] = -(xyz[mask,0]*realsense.color_intrinsics.fx)/xyz[mask,2]+realsense.color_intrinsics.ppx
    v[mask] = (xyz[mask,1]*realsense.color_intrinsics.fy)/xyz[mask,2]+realsense.color_intrinsics.ppy

    return u, v

def uv2planeuv(u, v, da, db, dc, a, b, c, realsense):
    x, y, z = uv2xyz(u, v, da, db, dc, realsense)
    return plane2uv(x, y, z, a, b, c, realsense)

def uv2planeuv_direct(u, v, da, db, dc, a, b, c, realsense):
    """
    Direct transform from input (u,v) to output (u',v') on the plane.
    Note: da, db, dc cancel out, so they do not appear in the final homography.
    Also, c is not used in the original code.
    """
    # Get intrinsics
    fx = realsense.color_intrinsics.fx
    fy = realsense.color_intrinsics.fy
    pp_x = realsense.color_intrinsics.ppx
    pp_y = realsense.color_intrinsics.ppy

    # Compute rotation matrix from plane parameters (c is ignored)
    normal = np.array([a, b, -1.0])
    normal = normal / np.sqrt(a**2 + b**2 + 1)
    beta = np.arcsin(normal[0])
    alpha = np.arctan2(-normal[1] / np.cos(beta), normal[2] / np.cos(beta))
    rx = np.array([[1, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha),  np.cos(alpha)]])
    ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    r = np.dot(rx, ry)
    
    # Build the homography H' that maps from normalized coordinates to the rotated ones.
    H_prime = np.array([
        [-r[0,0], - (fx / fy) * r[0,1], -fx * r[0,2]],
        [ (fy / fx) * r[1,0], r[1,1], fy * r[1,2]],
        [r[2,0] / fx, r[2,1] / fy, r[2,2]]
    ])
    
    # Compose with the translation matrices T and T^{-1}
    T = np.array([[1, 0, pp_x],
                  [0, 1, pp_y],
                  [0, 0, 1]])
    T_inv = np.array([[1, 0, -pp_x],
                      [0, 1, -pp_y],
                      [0, 0, 1]])
    H = T @ H_prime @ T_inv

    # Prepare the input in homogeneous coordinates.
    pts_in = np.stack([u, v, np.ones_like(u)], axis=-1)  # shape (..., 3)
    pts_out = (H @ pts_in.T).T  # apply H; note the transpose to handle arrays
    
    # Normalize to obtain (u', v') in inhomogeneous coordinates.
    pts_out = pts_out / pts_out[..., 2:3]
    u_out = pts_out[..., 0]
    v_out = pts_out[..., 1]
    
    return u_out, v_out

def camera2planeuv(u1, v1, a, b, c, realsense):
    fx = realsense.color_intrinsics.fx
    fy = realsense.color_intrinsics.fy
    ppx = realsense.color_intrinsics.ppx
    ppy = realsense.color_intrinsics.ppy

    # create rotation matrix
    normal = np.array([a,b,-1])/np.sqrt(a**2+b**2+1)
    beta = np.arcsin(normal[0])
    alpha = np.atan2(-normal[1]/np.cos(beta), normal[2]/np.cos(beta))
    rx = np.array([[1, 0            , 0             ], 
                   [0, np.cos(alpha),-np.sin(alpha)], 
                   [0, np.sin(alpha), np.cos(alpha) ]])
    ry = np.array([[ np.cos(beta), 0, np.sin(beta)],
                   [            0, 1,            0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    r = np.dot(rx,ry)

    d1x = (u1-ppx)/fx
    d1y = (v1-ppy)/fy
    d1 = np.stack([d1x, d1y, np.ones_like(d1x)], axis=-1)  # shape (..., 3)
    d2 = np.dot(d1, r)

    d2x = d2[:,0]/d2[:,2]
    d2y = d2[:,1]/d2[:,2]

    u2 = fx*d2x+ppx
    v2 = fy*d2y+ppy
    return u2, v2

def plane2camerauv(u1, v1, a, b, c, realsense):
    fx = realsense.color_intrinsics.fx
    fy = realsense.color_intrinsics.fy
    ppx = realsense.color_intrinsics.ppx
    ppy = realsense.color_intrinsics.ppy

    # create rotation matrix
    normal = np.array([a,b,-1])/np.sqrt(a**2+b**2+1)
    beta = np.arcsin(normal[0])
    alpha = np.atan2(-normal[1]/np.cos(beta), normal[2]/np.cos(beta))
    rx = np.array([[1, 0            , 0             ], 
                   [0, np.cos(alpha),-np.sin(alpha)], 
                   [0, np.sin(alpha), np.cos(alpha) ]])
    ry = np.array([[ np.cos(beta), 0, np.sin(beta)],
                   [            0, 1,            0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    r = np.dot(rx,ry).T

    d1x = (u1-ppx)/fx
    d1y = (v1-ppy)/fy
    d1 = np.stack([d1x, d1y, np.ones_like(d1x)], axis=-1)  # shape (..., 3)
    d2 = np.dot(d1, r)

    d2x = d2[:,0]/d2[:,2]
    d2y = d2[:,1]/d2[:,2]

    u2 = fx*d2x+ppx
    v2 = fy*d2y+ppy
    return u2, v2

if __name__ == '__main__':
    points = gen_points(100,2,1,100,0)
    points = np.vstack((points,[1, 1, 100]))
    plane = solve_plane(points)
    print(plane)

