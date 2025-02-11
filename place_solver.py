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

def uv2xyz(u, v, da, db, dc, realsense):
    d = u*da+v*db+dc
    z = d*realsense.depth_scale
    x = (u-realsense.color_intrinsics.ppx)*z/realsense.color_intrinsics.fx
    y = (v-realsense.color_intrinsics.ppy)*z/realsense.color_intrinsics.fy
    return np.array([x,y,z])

def solve_mask_quad(mask):
    shapes = []
    mask = (mask*255).astype('uint8')
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        epsilon = 0.02*cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        shapes.append(approx)
    return shapes

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
    

    # zx = x*a
    # zy = y*b
    
    # u = np.sqrt(x**2+zx**2)*np.sign(x)
    # v = np.sqrt(y**2+zy**2)*np.sign(y)
    # return u, v

def uv2planeuv(u, v, da, db, dc, realsense, a, b, c):
    x, y, z = uv2xyz(u, v, da, db, dc, realsense)
    return plane2uv(x, y, z, a, b, c, realsense)

if __name__ == '__main__':
    points = gen_points(100,2,1,100,0)
    points = np.vstack((points,[1, 1, 100]))
    plane = solve_plane(points)
    print(plane)

