import numpy as np
from pyldpc import make_ldpc, get_message
from decoder import belief_propagation_decode
from decoderMinSum import belief_propagation_decode_min_sum


def bits_to_qpsk_symbols(bits):
    """
    Konwertuje ciąg bitów na symbole QPSK.
    Każdy symbol QPSK reprezentuje 2 bity informacji.
    
    Mapowanie:
    00 -> (1, 1)   -> +1+1j
    01 -> (1, -1)  -> +1-1j
    10 -> (-1, 1)  -> -1+1j
    11 -> (-1, -1) -> -1-1j
    
    Parametry:
      bits - wektor bitów (numpy.array)
      
    Zwraca:
      symbole QPSK jako liczby zespolone (numpy.array)
    """
    if len(bits) % 2 != 0:
        # Jeśli liczba bitów jest nieparzysta, dodaj dodatkowy bit
        bits = np.append(bits, 0)
    
    # Przekształć wektor bitów na pary bitów
    pairs = bits.reshape(-1, 2)
    
    # Inicjalizuj tablicę symboli QPSK
    symbols = np.zeros(len(pairs), dtype=complex)
    
    # Mapowanie par bitów na symbole QPSK
    for i, (bit0, bit1) in enumerate(pairs):
        real_part = 1 - 2 * bit0  # 0 -> +1, 1 -> -1
        imag_part = 1 - 2 * bit1  # 0 -> +1, 1 -> -1
        symbols[i] = complex(real_part, imag_part)
    
    return symbols


def qpsk_symbols_to_llr(received_symbols, snr):
    """
    Konwertuje odebrane symbole QPSK na wartości LLR (Log-Likelihood Ratio)
    dla każdego bitu.
    
    Parametry:
      received_symbols - wektor odebranych symboli QPSK (numpy.array, complex)
      snr - stosunek sygnału do szumu (float)
      
    Zwraca:
      wektor LLR dla każdego bitu (numpy.array)
    """
    # Oblicz wariancję szumu na podstawie SNR
    sigma2 = 1 / (2 * snr)
    
    # Inicjalizuj tablicę LLR dla każdego bitu (2 bity na symbol)
    llr = np.zeros(2 * len(received_symbols))
    
    for i, symbol in enumerate(received_symbols):
        # LLR dla pierwszego bitu (część rzeczywista)
        llr[2*i] = 2 * np.real(symbol) / sigma2
        
        # LLR dla drugiego bitu (część urojona)
        llr[2*i+1] = 2 * np.imag(symbol) / sigma2
    
    return llr


def encode_qpsk(G, bits, snr):
    """
    Koduje wiadomość za pomocą macierzy generującej G,
    a następnie moduluje za pomocą QPSK.
    
    Parametry:
      G - macierz generująca
      bits - wektor bitów wiadomości
      snr - stosunek sygnału do szumu
      
    Zwraca:
      symbole QPSK (numpy.array, complex)
    """
    # Kodowanie LDPC
    encoded_bits = np.mod(np.dot(G, bits), 2)
    
    # Modulacja QPSK
    return bits_to_qpsk_symbols(encoded_bits)


def simulate_ldpc_qpsk(n=1296, d_v=2, d_c=4, snr=2.5, num_trials=10, maxiter=100, use_custom_decoder=False):
    """
    Symulacja systemu LDPC z modulacją QPSK.
    
    Parametry:
      n - długość słowa kodowego
      d_v - stopień kolumn macierzy H
      d_c - stopień wierszy macierzy H
      snr - stosunek sygnału do szumu [dB]
      num_trials - liczba prób
      maxiter - maksymalna liczba iteracji dekodera
      use_custom_decoder - czy używać własnego dekodera zamiast pyldpc
      
    Zwraca:
      skuteczność dekodowania i średnią liczbę błędów bitowych
    """
    d_v = max(2, d_v)
    d_c = max(3, d_c)
    
    # Konwersja SNR z dB na wartość liniową
    snr_linear = 10 ** (snr / 10)

    H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
    k = G.shape[1]
    print(
        f"Symulacja LDPC z modulacją QPSK\nn={n}, k={k}, snr={snr} dB, próby={num_trials}, dekoder={'custom' if use_custom_decoder else 'pyldpc'}")
    print(f"Macierz H: {H.shape}, Macierz G: {G.shape}\n")

    successes = 0
    total_bit_errors = 0

    for trial in range(num_trials):
        print(f"=== Próba {trial + 1} ===")
        
        # Generowanie losowej wiadomości
        x = np.random.randint(0, 2, k)
        
        # Kodowanie i modulacja QPSK
        symbols = encode_qpsk(G, x, snr_linear)
        
        # Dodawanie szumu gaussowskiego (AWGN)
        # Dla QPSK, szum jest dodawany do części rzeczywistej i urojonej niezależnie
        noise_real = np.random.normal(0, np.sqrt(1/(2*snr_linear)), len(symbols))
        noise_imag = np.random.normal(0, np.sqrt(1/(2*snr_linear)), len(symbols))
        noise = noise_real + 1j * noise_imag
        
        received_symbols = symbols + noise
        
        # Demodulacja - obliczenie LLR dla każdego bitu
        llr = qpsk_symbols_to_llr(received_symbols, snr_linear)
        
        # Dekodowanie
        if use_custom_decoder:
            # Użyj własnego dekodera Min-Sum
            # x_hat_full = belief_propagation_decode_min_sum(
            #     llr, H, max_iter=maxiter, alpha=0.75)
            
            # Alternatywnie, dekoder z dampingiem:
            x_hat_full = belief_propagation_decode(
               llr, H, max_iter=maxiter, damping=0.5)
        else:
            from pyldpc import decode
            x_hat_full = decode(H, llr, snr_linear, maxiter=maxiter)

        # Wyekstrahowanie wiadomości z zakodowanego wektora
        x_hat = get_message(G, x_hat_full)

        # Zliczanie błędów
        bit_errors = np.sum(x != x_hat)
        total_bit_errors += bit_errors

        if bit_errors == 0:
            print(f"Próba {trial + 1}: SUKCES ✅")
            successes += 1
        else:
            print(f"Próba {trial + 1}: BŁĄD ❌, błędów bitowych: {bit_errors}")

    print(f"\nSkuteczność: {successes}/{num_trials}, "
          f"Średnia liczba błędów bitowych: {total_bit_errors / num_trials:.2f}")
    
    return successes/num_trials, total_bit_errors/num_trials


if __name__ == "__main__":
    # Symulacja LDPC z modulacją QPSK
    # Zmień parametry według potrzeb
    simulate_ldpc_qpsk(n=1296, d_v=2, d_c=4, snr=2.0, num_trials=10, 
                      maxiter=100, use_custom_decoder=True)