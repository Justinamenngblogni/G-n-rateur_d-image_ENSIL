import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers 3D projection

class camera_intrinseque:
    def __init__(self, K , A):
        self.K = K
        self.A = A  
    def project_point(self, P_camera):
        """
        Projette un point 3D dans le repère de la caméra sur le plan image.
        P_camera : coordonnées du point dans le repère de la caméra (x, y, z)
        Retourne les coordonnées 2D du point projeté sur le plan image (u, v)
        """
        x, y, z = P_camera
        if z <= 0:
            return None  # Le point est derrière la caméra
        I_ecran = K @ P_camera[:3]
        # Conversion en coordonnées inhomogènes
        I_ecran = np.round(I_ecran[:2] / I_ecran[2])
        I_image = A @np.hstack((I_ecran, np.array([0., 1.])))
        u, v = I_image[:2]  
        return np.array([u, v])
    

    # Calculer l'inverse d'une matrice 

def invert_matrix(mat):
    """
    Retourne l'inverse de `mat` ou lève une exception si non inversible.
    """
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
        self.origine = origine
    def coordonnees_point_dans_repere(self, point_global):
        """
        Calcule les coordonnées d'un point global dans ce repère.
        """
        vecteur = point_global - self.origine
        x = np.dot(vecteur, self.i)
        y = np.dot(vecteur, self.j)
        z = np.dot(vecteur, self.k)
        return np.array([x, y, z])
    

def project_point(M_world, Repere_C, camera):
    M_camera = Repere_C.coordonnees_point_dans_repere(M_world[:3])  
    I_image  = camera.project_point(M_camera)
    return I_image  


fx = 800
fy = 800
cx = 0
cy = 0
s = 0

H = 480*2 # Hauteur de l'image en pixels
W = 640*2 # Largeur de l'image en pixels

K = np.array([[fx, s, cx],
              [0, fy, cy],  
                [0,   0,   1]]) # Matrice des paramètres intrinsèques de la caméra

A = np.array([[-1, 0, 0, W/2],
              [0, -1, 0, H/2],
                [0, 0, 1, 0]])  # Matrice de transformation des coordonnées écran aux coordonnées image

l = 50  # distance entre l'origine du repère monde et l'origine du repère camera le long de y dans le repère monde
O_w = np.array([0,0,0])
O_c = np.array([0,l,0])



R_camera_to_word = np.array([[0, 1, 0 ],
                             [0, 0, -1 ],   
                             [-1, 0, 0]]) # Rotation de la camera vers le repère monde ou changement de base du monde vers la camera
R_word_to_camera = invert_matrix(R_camera_to_word) # Rotation du repère monde vers la camera ou changement de base de la camera vers le monde


C_word_to_camera = R_camera_to_word # changement de base du monde vers la camera ou rotation de la camera vers le repère monde

C_camera_to_word = R_word_to_camera # changement de base de la camera vers le monde ou rotation du repère monde vers la camera


camera = camera_intrinseque(K, A)
Repere_W = Repere(origine=O_w)
Repere_C = Repere(O_c, R=R_word_to_camera )



M_world =np.array([15.,  10., -5.,   1.])
I_image = project_point(M_world, Repere_C, camera)


d = 15
alpha_camera = np.array([d, d, l, 1])
beta_camera = np.array([-d, d, l, 1])
gamma_camera = np.array([0, d, l, 1])

alpha_image = project_point(alpha_camera,Repere_C, camera)
beta_image = project_point(beta_camera,Repere_C, camera)
gamma_image = project_point(gamma_camera,Repere_C, camera)

print("Les points images",alpha_image, beta_image, gamma_image)