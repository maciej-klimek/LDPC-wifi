# decoder.py
import numpy as np

def belief_propagation_decode(llr, H, max_iter=50, damping=0.5):
    """
    Dekodowanie LDPC metodą belief propagation (sum–product) z dampingiem.
    
    Parametry:
      llr      – wektor LLR (log–likelihood ratio) długości n, otrzymany np. z demodulatora.
      H        – macierz kontroli parzystości (numpy.array) wymiaru m x n.
      max_iter – maksymalna liczba iteracji (domyślnie 50).
      damping  – współczynnik damping, wartość z przedziału (0,1).
      
    Zwraca:
      decoded_bits – zdekodowany wektor bitów (numpy.array) o długości n.
    """
    m, n = H.shape

    # Indeksy połączeń: dla każdego check node (wiersz) lista indeksów zmiennych, 
    # dla każdej zmiennej lista indeksów check nodes
    check_nodes = [np.where(H[i, :] == 1)[0] for i in range(m)]
    var_nodes   = [np.where(H[:, j] == 1)[0] for j in range(n)]

    # Inicjalizacja wiadomości: q[i][j] – wiadomość od zmiennej j do check node i
    # oraz r[i][j] – wiadomość od check node i do zmiennej j
    q = np.zeros((m, n))
    r = np.zeros((m, n))
    
    # Inicjalizacja q: dla każdej krawędzi (i,j) ustawiamy q[i,j] = llr[j]
    for j in range(n):
        for i in var_nodes[j]:
            q[i, j] = llr[j]
            
    # Dla damping, przechowujemy poprzednie wartości r
    prev_r = np.zeros((m, n))

    # Iteracyjna aktualizacja
    for iteration in range(max_iter):
        # Krok 1: aktualizacja wiadomości od check nodes do zmiennych z dampingiem
        for i in range(m):
            for j in check_nodes[i]:
                product = 1.0
                for j2 in check_nodes[i]:
                    if j2 != j:
                        product *= np.tanh(q[i, j2] / 2.0)
                # Unikamy problemów związanych z arctanh
                eps = 1e-12
                product = np.clip(product, -1 + eps, 1 - eps)
                new_r = 2.0 * np.arctanh(product)
                # Damping: mieszamy poprzednią wartość z nową
                r[i, j] = damping * prev_r[i, j] + (1 - damping) * new_r
                prev_r[i, j] = r[i, j]

        # Krok 2: aktualizacja wiadomości od zmiennych do check nodes oraz obliczenie uaktualnionych LLR
        updated_llr = np.copy(llr)
        for j in range(n):
            for i in var_nodes[j]:
                # Sumujemy wiadomości od wszystkich innych check nodes przyłączonych do zmiennej j
                sum_r = 0
                for i2 in var_nodes[j]:
                    if i2 != i:
                        sum_r += r[i2, j]
                updated_llr[j] += sum_r
                q[i, j] = llr[j] + sum_r

        # Estymacja bitów: przyjmujemy, że jeśli LLR < 0 to bit równy 1, inaczej 0
        decoded = (updated_llr < 0).astype(int)

        # Sprawdzenie warunku parzystości
        syndrome = np.mod(np.dot(H, decoded.T), 2)
        if np.all(syndrome == 0):
            print(f"Zakończono po {iteration + 1} iteracjach.")
            return decoded

    print("Osiągnięto maksymalną liczbę iteracji, warunek parzystości nie został spełniony.")
    return decoded

if __name__ == '__main__':
    # Test dekodera z przykładowymi LLR
    from ldpc_matrices import get_ldpc_matrix
    H, _ = get_ldpc_matrix('648_1/2')
    llr = np.random.randn(12)
    decoded_bits = belief_propagation_decode(llr, H, max_iter=50, damping=0.5)
    print("Zdekodowany wektor bitów:", decoded_bits)
