import numpy as np
import matplotlib.pyplot as plt

fx = 800
fy = 800
cx = 0
cy = 0
s = 0

K = np.array([[fx, s, cx],
              [0, fy, cy],  
                [0,   0,   1]])

# Calculer l'inverse d'une matrice 

def invert_matrix(mat):
    try:
        return np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        raise ValueError("Matrice singulière : impossible d'appliquer l'inverse.")

class Repere:

    def __init__(self, origine = np.array([0, 0, 0]), R=np.eye(3)):
        self.rotation = R
        self.i = np.array([1, 0, 0]) @ self.rotation
        self.j = np.array([0, 1, 0]) @ self.rotation
        self.k = np.array([0, 0, 1]) @ self.rotation

def world_to_image(M_w, K, T_word_to_camera, A):
    # Transformation en coordonnées caméra
    M_c = T_word_to_camera @ M_w
    
    # Projection perspective
    I_e = K @ M_c[:3]
    
    # Conversion en coordonnées inhomogènes
    I_e = np.round(I_e[:2] / I_e[2])
    
    # Transformation en coordonnées image
    I_i = A @ np.hstack((I_e, np.array([0., 1.])))
    
    return I_i


l = 50
M_T_camera_to_word = np.array([[0, 1, 0 ],
                     [0, 0, -1 ],   
                     [-1, 0, 0]])
M_T_word_to_camera = invert_matrix(M_T_camera_to_word)

T = np.eye(4)
T[0:3, 0:3] = M_T_camera_to_word
T[0:3, 3] = np.array([0, l, 0])
T_word_to_camera = invert_matrix(T)
Rot = M_T_word_to_camera

O_w = np.array([0,0,0])
O_c = np.array([0,l,0])
Repere_C = Repere(O_c, R=Rot )
Repere_W = Repere(origine=O_w)

H = 480*2
W = 640*2
A = np.array([[-1, 0, 0, W/2],
              [0, -1, 0, H/2],
                [0, 0, 1, 0]]) 


Points = [np.array([15., 10., -5. , 1.]),np.array([-60. ,-60., -20. ,  1.]),np.array([-60. ,-60.,-10. ,  1.]) ]
images = []
for i in Points:
    img_point = world_to_image(i, K, T_word_to_camera, A)
    images.append(img_point)


print("Points dans le repère image :")
print(images)
