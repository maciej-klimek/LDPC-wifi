import numpy as np
from ldpc_matrices import get_ldpc_matrix

def encode(message, G):
    """
    Enkodowanie LDPC.
    
    Parametry:
        message - wektor bitów (numpy.array) o długości k.
        G - macierz generująca (numpy.array) wymiaru k x n.
      
    Return:
        codeword - zakodowany wektor bitów (numpy.array) o długości n.
    
    """
    message = np.array(message, dtype=int).flatten()
    codeword = np.mod(np.dot(message, G), 2)
    return codeword

if __name__ == '__main__':
    _, G = get_ldpc_matrix('648_1/2')

    # Dla przykładu: k = 6 (zgodnie z dummy macierzą w ldpc_matrices.py)
    message = np.random.randint(0, 2, size=6)
    codeword = encode(message, G)
    print("Wiadomość:", message)
    print("Kodowany wektor:", codeword)
