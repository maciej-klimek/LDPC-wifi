import numpy as np

def get_ldpc_matrix(code='648_1/2'):

    """
    Zwraca przykładową parę macierzy: 
        - H (macierz kontroli parzystości)
        - G (macierz generująca) dla wybranej konfiguracji
    
    
    kod: długość n = 648, 
    rate = 1/2, 
    czyli k = 324.
    Dla uproszczenia obliczen dajemy małą macierz aby pokazać strukturę
    Dla prawdziwego systemu macierz powina byc 1:1 zgodna ze standardem.
    """
    
    if code == '648_1/2':
        # rzeczywistości H miałaby wymiar M x N, 
        # gdzie N = 648 i M = 324 (dla rate 1/2), z blokową strukturą.
        # Tutaj przykład 6x12, tylko w celach demonstracyjnych.
        H = np.array([
            [1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0],
            [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        ], dtype=int)
        
        # Aby wyznaczyć macierz generującą G, można przyjąć jej postać: [I | P]
        # gdzie P jest obliczana na podstawie H (przy założeniu, że H = [P^T | I]).
        # W praktyce do enkodowania stosuje się algorytmy eliminacji Gaussa lub metody iterative.
        # Dla demonstracji zakładamy, że G jest podana – poniżej przykład odpowiednio dobranej macierzy.
        # W rzeczywistości ofc wymiar G to k x n; tutaj k=6, n=12.
        G = np.array([
            [1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1],
        ], dtype=int)
        
        return H, G

    else:
        raise ValueError("Wybrana konfiguracja LDPC nie jest obsługiwana.")

if __name__ == '__main__':
    H, G = get_ldpc_matrix()
    print("Macierz H:")
    print(H)
    print("Macierz G:")
    print(G)
