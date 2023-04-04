import numpy as np

def get_perpendicular_normalized_vector(direction):
    normal_vector = np.array([-direction[1], direction[0]])
    normalized_vector = normal_vector / np.linalg.norm(normal_vector)
    return normalized_vector


def are_vectors_aligned_with_margin(a, b, margin_degrees):
    # Normalizar os vetores
    a_normalized = a / np.linalg.norm(a)
    b_normalized = b / np.linalg.norm(b)

    # Calcular o produto escalar
    dot_product = np.dot(a_normalized, b_normalized)

    # Calcular o ângulo entre os vetores (em graus)
    angle_degrees = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))

    # Verificar se o ângulo está dentro da margem especificada
    return np.abs(angle_degrees) < margin_degrees or np.abs(angle_degrees - 180) < margin_degrees

def rotate_vector_90_degrees_clockwise(vector):
    return np.array([vector[1], -vector[0]])

def rotate_vector_90_degrees_counterclockwise(vector):
    return np.array([-vector[1], vector[0]])


def norm_vector(vector):
    # Calcular a magnitude do vetor
    magnitude = np.linalg.norm(vector)

    # Dividir cada componente do vetor pela magnitude
    normalized_vector = vector / magnitude

    return normalized_vector