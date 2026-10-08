from math import radians, cos, sin, pow

import numpy as np
import matplotlib.pyplot as plt
import kagglehub #імпортую 3D моделі
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path

#PART 1 TASK 1
lynx74x2 = np.array([
[209.70, 368.42], [157.63, 332.16], [118.82, 284.21], [80.95, 224.56], [43.08, 244.44], [20.36, 266.67], [-4.26, 293.57], [2.37, 263.16], [-20.36, 292.40], [-39.29, 299.42], [-21.30, 259.65],
[-50.65, 267.84], [-39.29, 242.11], [-55.38, 240.94], [-100.83, 300.58], [-149.11, 345.03], [-172.78, 361.40], [-189.82, 300.58], [-192.66, 225.73], [-181.30, 145.03], [-168.05, 104.09], [-184.14, 66.67],
[-186.98, 31.58], [-183.20, 3.51], [-208.76, -4.68], [-197.40, -29.24], [-182.25, -44.44], [-203.08, -43.27], [-172.78, -92.40], [-131.12, -126.32], [-101.78, -147.37], [-74.32, -163.74], [-110.30, -224.56],
[-143.43, -287.72], [-161.42, -240.94], [-282.60, -221.05], [-388.64, -205.85], [-370.65, -301.75], [-339.41, -397.66], [18.46, -397.66], [345.09, -400.00], [359.29, -378.95], [367.81, -342.69], [346.98, -362.57], [363.08, -302.92], [357.40, -243.27], [348.88, -266.67], [336.57, -201.17], [290.18, -135.67], [240.00, -118.13], [258.93, -164.91], [257.99, -228.07], [252.31, -271.35], [256.09, -333.33],
[247.57, -359.06], [230.53, -307.60], [194.56, -238.60], [160.47, -181.29], [120.71, -149.71], [165.21, -132.16], [201.18, -100.58], [183.20, -99.42], [221.07, -73.68], [253.25, -24.56], [222.01, -23.39],
[251.36, -1.17], [262.72, 24.56], [234.32, 25.73], [214.44, 42.11], [202.13, 60.82], [220.12, 101.75], [234.32, 160.23], [240.00, 230.41], [232.43, 316.96], [209.70, 368.42]
])
lynx2x74 = lynx74x2.T
batman9x2 = np.array([[0, 0], [1, 0.2], [0.4, 1], [0.5, 0.4], [0, 0.8], [-0.5, 0.4], [-0.4, 1], [-1, 0.2], [0, 0]])
batman2x9 = batman9x2.T
Original =  lynx2x74

def show2d( transformed, label, origin = Original):
    plt.plot(origin[0, :], origin[1, :], label='Original')
    plt.plot(transformed[0, :], transformed[1, :], label=label)
    plt.axis('equal')
    plt.legend()
    plt.show()

def stretch(a, b, X = Original):
    Xs = X.copy()
    A = np.array([[a,0],[0,b]])
    print(A)
    return A @ Xs

#a - ширина b - висота
stretched = stretch(5, 2)
print(f"{stretched} Stretched")
show2d(stretched, 'Stretched')

def shear(a, b, X = Original):
    Xs = X.copy()
    A = np.array([[1,a], [b,1]])
    print(A)
    return  A @ Xs

#нахил
sheared = shear(0, 0.5)
print(f"{sheared} Sheared")
show2d(sheared, 'Sheared')

def reflection(a, b, X = Original):
    Xr = X.copy()
    Cooficient =  pow((a*a + b*b),-1)
    A= Cooficient * np.array([[a*a - b*b, 2*a*b],[2*a*b, b*b-a*a]])
    print(A)
    return A @ Xr

#y->-y (1,0); x->-x (0,1); y = x (1,1)
reflected = reflection( 1,-1)
print(f"{reflected} Reflected")
show2d(reflected,'Reflected' )

def rotation(Q, X = Original):
    Xr = X.copy()
    Qr = radians(Q)
    A = np.array([[cos((Qr)), -sin(Qr)], [sin(Qr), cos(Qr)]])
    print(A)
    return  A @ Xr

rotated = rotation(180)
print(f"{rotated} Rotated")
show2d( rotated, 'Rotated')

#PART 1 TASK 2
#composition
# Stretch, Sher, Rotation
Xs = stretch(2, 1)
Xss = shear(0.5, 0, Xs)
Xssr = rotation(45, Xss)
print(f"{Xssr} - Stretch, Sher, Rotation")
show2d(Xssr, "Stretch, Sher, Rotation")
# Rotation, Stretch, Sher
Xr = rotation(45)
Xrs = stretch(2, 1, Xr)
Xrss = shear(0.5, 0, Xrs)
print(f"{Xrss} - Rotation, Stretch, Sher")
show2d(Xrss, "Rotation, Stretch, Sher")
#Sher, Rotation, Stretch
Xs = shear(0.5, 0)
Xsr = rotation(45, Xs)
Xsrs = stretch(2, 1, Xsr)
print(f"{Xsrs} - Sher, Rotation, Stretch")
show2d(Xsrs, "Sher, Rotation, Stretch")


#PART 2 TASK 3
path = kagglehub.dataset_download("balraj98/modelnet40-princeton-3d-object-dataset")

def read_off(filename: str):
    with open(filename, 'r') as f:
        if 'OFF' != f.readline().strip():
            raise ValueError('Not a valid OFF header')

        n_verts, n_faces, _ = map(int, f.readline().strip().split())

        verts = [list(map(float, f.readline().strip().split())) for _ in range(n_verts)]

        faces = [list(map(int, f.readline().strip().split()))[1:] for _ in range(n_faces)]

        return np.array(verts), faces

file_path = Path(path) / "ModelNet40" / "airplane" / "train" / "airplane_0001.off"
vertices, faces = read_off(file_path)

def plot_off(vertices, faces):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection([vertices[face] for face in faces],
                            alpha=0.3, edgecolor='k')
    ax.add_collection3d(mesh)

    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:,2], s=2, c='r')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.auto_scale_xyz(vertices[:, 0], vertices[:, 1], vertices[:,2])

    plt.show()

print(vertices)
plot_off(vertices,faces)

def rotate_xy (X, Q):
    Qr = radians(Q)
    A = np.array([[cos(Qr), -sin(Qr), 0], [sin(Qr), cos(Qr), 0], [0, 0, 1]])
    print(A)
    return X @ A.T

print(rotate_xy(vertices, 45))
plot_off(rotate_xy(vertices, 45),faces)

def rotate_yz (X, Q):
    Qr = radians(Q)
    A = np.array([[1,0,0], [0, cos(Qr), -sin(Qr)], [0, sin(Qr), cos(Qr), ]])
    print(A)
    return X @ A.T

print(rotate_yz(vertices, 45))
plot_off(rotate_yz(vertices, 45),faces)

def rotate_xz (X, Q):
    Qr = radians(Q)
    A = np.array([[cos(Qr), 0, -sin(Qr)], [0, 1, 0], [sin(Qr), 0, cos(Qr)]])
    print(A)
    return X @ A.T

print(rotate_xz(vertices, 45))
plot_off(rotate_xz(vertices, 45),faces)

#PART 2 TASK 4
Xxy = rotate_xy(vertices, 45)
Xyz = rotate_yz(Xxy, 45)
Xzx = rotate_xz(Xyz, 45)
plot_off(Xzx, faces)
