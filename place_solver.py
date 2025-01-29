import random
import numpy as np
from sklearn.linear_model import HuberRegressor

def gen_points(n,a,b,c,r):
    points = []
    for _ in range(n):
        random_z = random.random()*r*2-r
        random_x = random.random()
        random_y = random.random()
        z = a*random_x+b*random_y+c
        points.append((random_x,random_y,z+random_z))
    return np.array(points)

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

if __name__ == '__main__':
    points = gen_points(100,2,1,100,0)
    points = np.vstack((points,[1, 1, 100]))
    plane = solve_plane(points)
    print(plane)

