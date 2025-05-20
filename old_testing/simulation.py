# simulation.py
import numpy as np
from ldpc_matrices import get_ldpc_matrix
from encoder import encode
from decoder import belief_propagation_decode

def bpsk_modulation(bits):
    """
    Modulacja BPSK: 0 -> +1, 1 -> -1.
    """
    return 1 - 2 * bits  # przekształcamy 0 do +1, 1 do -1

def awgn_channel(signal, sigma):
    """
    Symulacja kanału AWGN.
    
    Parametr:
      signal – wektor sygnału (numpy.array)
      sigma  – odchylenie standardowe szumu
    """
    noise = np.random.normal(0, sigma, size=signal.shape)
    return signal + noise

def llr_from_awgn(received, sigma):
    """
    Oblicz LLR dla kanału AWGN przy BPSK.
    
    Jeśli przyjmiemy, że sygnał modulowany to ±1 i szum ma wariancję sigma^2,
    LLR = 2*y/sigma^2
    """
    return 2.0 * received / (sigma**2)

def simulate_ldpc(code='648_1/2', sigma=1.0, num_trials=100):
    """
    Przeprowadza symulację enkodowania, modulacji, transmisji przez kanał AWGN,
    dekodowania i porównania oryginalnych danych z danymi zdekodowanymi.
    
    Parametry:
      code      - konfiguracja LDPC, np. '648_1/2'
      sigma     - odchylenie szumu AWGN
      num_trials- liczba bloków do symulacji
    """
    H, G = get_ldpc_matrix(code)
    
    # Dla demonstracji w naszym dummy przykładzie:
    # Załóżmy, że k = liczba wierszy w G, a n = liczba kolumn
    k = G.shape[0]
    n = G.shape[1]
    
    total_bit_errors = 0
    total_bits = 0
    successful_decodings = 0

    for trial in range(num_trials):
        # 1. Generacja losowej wiadomości
        message = np.random.randint(0, 2, size=k)
        
        # 2. Enkodowanie
        codeword = encode(message, G)
        
        # 3. Modulacja BPSK
        modulated = bpsk_modulation(codeword)
        
        # 4. Przesył przez kanał AWGN
        received = awgn_channel(modulated, sigma)
        
        # 5. Oblicz LLR z otrzymanego sygnału
        llr = llr_from_awgn(received, sigma)
        
        # 6. Dekodowanie
        decoded_codeword = belief_propagation_decode(llr, H, max_iter=50)
        
        # 7. (Opcjonalnie) Demapping zdekodowanego codewordu do wiadomości
        #   W układzie systematycznym zakładamy, że pierwsze k bitów to informacja.
        decoded_message = decoded_codeword[:k]
        
        # Zlicz błędy
        errors = np.sum(decoded_message != message)
        total_bit_errors += errors
        total_bits += k
        if errors == 0:
            successful_decodings += 1

    ber = total_bit_errors / total_bits if total_bits > 0 else None
    print(f"Symulacja ukończona. Liczba prób: {num_trials}")
    print(f"Stosunek udanych dekodowań: {successful_decodings} / {num_trials}")
    print(f"Bit Error Rate (BER): {ber:.4f}")

if __name__ == '__main__':
    # Przykładowe wywołanie symulacji
    # sigma można dostosować, aby zasymulować różne poziomy szumu.
    simulate_ldpc(code='648_1/2', sigma=0.8, num_trials=20)
