import numpy as np
from pyldpc import make_ldpc, get_message
from decoder import belief_propagation_decode
from decoderMinSum import belief_propagation_decode_min_sum


def bits_to_16qam_symbols(bits):
    """
    Konwertuje ciąg bitów na symbole 16-QAM.
    Każdy symbol 16-QAM reprezentuje 4 bity informacji.
    
    Mapowanie Gray'a dla 16-QAM:
    Cztery bity mapowane są na jedną z 16 pozycji na płaszczyźnie zespolonej.
    Pierwsze dwa bity określają znak i poziom części rzeczywistej.
    Kolejne dwa bity określają znak i poziom części urojonej.
    
    Parametry:
      bits - wektor bitów (numpy.array)
      
    Zwraca:
      symbole 16-QAM jako liczby zespolone (numpy.array)
    """
    # Upewnij się, że liczba bitów jest wielokrotnością 4
    if len(bits) % 4 != 0:
        padding = 4 - (len(bits) % 4)
        bits = np.append(bits, np.zeros(padding, dtype=int))
    
    # Przekształć wektor bitów na grupy po 4 bity
    bit_groups = bits.reshape(-1, 4)
    
    # Inicjalizuj tablicę symboli 16-QAM
    symbols = np.zeros(len(bit_groups), dtype=complex)
    
    # Mapowanie 4 bitów na symbole 16-QAM
    for i, (b0, b1, b2, b3) in enumerate(bit_groups):
        # Bity b0 i b1 determinują część rzeczywistą
        # Bity b2 i b3 determinują część urojoną
        
        # Mapowanie Gray'a dla części rzeczywistej:
        # 00 -> +3, 01 -> +1, 11 -> -1, 10 -> -3
        if b0 == 0 and b1 == 0:
            real_part = 3
        elif b0 == 0 and b1 == 1:
            real_part = 1
        elif b0 == 1 and b1 == 1:
            real_part = -1
        else:  # b0 == 1 and b1 == 0
            real_part = -3
            
        # Mapowanie Gray'a dla części urojonej:
        # 00 -> +3, 01 -> +1, 11 -> -1, 10 -> -3
        if b2 == 0 and b3 == 0:
            imag_part = 3
        elif b2 == 0 and b3 == 1:
            imag_part = 1
        elif b2 == 1 and b3 == 1:
            imag_part = -1
        else:  # b2 == 1 and b3 == 0
            imag_part = -3
        
        # Normalizacja energii - aby średnia energia symbolu wynosiła 1
        # Średnia energia 16-QAM z punktami ±1, ±3 wynosi 10
        normalization_factor = np.sqrt(10)
        symbols[i] = complex(real_part/normalization_factor, imag_part/normalization_factor)
    
    return symbols


def qam16_demapper(received_symbol, sigma2):
    """
    Demapuje odebrany symbol 16-QAM na wartości LLR dla 4 bitów.
    
    Parametry:
      received_symbol - odebrany symbol (liczba zespolona)
      sigma2 - wariancja szumu
    
    Zwraca:
      LLR dla 4 bitów w kolejności [LLR_b0, LLR_b1, LLR_b2, LLR_b3]
    """
    llr = np.zeros(4)
    
    # Normalizacja tak samo jak przy mapowaniu
    normalization_factor = np.sqrt(10)
    r_real = np.real(received_symbol) * normalization_factor
    r_imag = np.imag(received_symbol) * normalization_factor
    
    # Obliczenie LLR dla bitów b0 i b2 (bity MSB)
    # LLR_b0 = log(P(b0=0|r) / P(b0=1|r))
    llr[0] = compute_llr_msb_16qam(r_real, sigma2 * normalization_factor**2)
    llr[2] = compute_llr_msb_16qam(r_imag, sigma2 * normalization_factor**2)
    
    # Obliczenie LLR dla bitów b1 i b3 (bity LSB)
    # LLR_b1 = log(P(b1=0|r) / P(b1=1|r))
    llr[1] = compute_llr_lsb_16qam(r_real, sigma2 * normalization_factor**2)
    llr[3] = compute_llr_lsb_16qam(r_imag, sigma2 * normalization_factor**2)
    
    return llr


def compute_llr_msb_16qam(r, sigma2):
    """
    Oblicza LLR dla bitu MSB (Most Significant Bit) w mapowaniu 16-QAM.
    
    W mapowaniu Gray'a dla 16-QAM, bit MSB określa znak (region dodatni/ujemny).
    
    Parametry:
      r - wartość rzeczywista lub urojona odebranego symbolu
      sigma2 - wariancja szumu
    
    Zwraca:
      LLR dla bitu MSB
    """
    # Dla mapowania Gray'a: MSB = 0 dla wartości dodatnich, MSB = 1 dla wartości ujemnych
    # Zatem liczymy:
    # LLR_msb = log( [P(r|MSB=0)] / [P(r|MSB=1)] )
    
    # Przybliżenie LLR dla bitu MSB
    # Dla uproszczenia można przyjąć, że r > 0 => MSB = 0, r < 0 => MSB = 1
    # Ale dokładniejsze jest uwzględnienie wszystkich możliwych punktów konstelacji
    
    # Punkty dla MSB = 0: +1, +3
    # Punkty dla MSB = 1: -1, -3
    
    # Obliczenie prawdopodobieństw
    p_msb0 = np.exp(-(r-1)**2/(2*sigma2)) + np.exp(-(r-3)**2/(2*sigma2))
    p_msb1 = np.exp(-(r+1)**2/(2*sigma2)) + np.exp(-(r+3)**2/(2*sigma2))
    
    # Unikanie dzielenia przez zero
    p_msb1 = np.maximum(p_msb1, 1e-10)
    
    return np.log(p_msb0 / p_msb1)


def compute_llr_lsb_16qam(r, sigma2):
    """
    Oblicza LLR dla bitu LSB (Least Significant Bit) w mapowaniu 16-QAM.
    
    W mapowaniu Gray'a dla 16-QAM, bit LSB określa amplitudę (region wewnętrzny/zewnętrzny).
    
    Parametry:
      r - wartość rzeczywista lub urojona odebranego symbolu
      sigma2 - wariancja szumu
    
    Zwraca:
      LLR dla bitu LSB
    """
    # Dla mapowania Gray'a: LSB = 0 dla wartości skrajnych (±3), LSB = 1 dla wartości wewnętrznych (±1)
    # Zatem liczymy:
    # LLR_lsb = log( [P(r|LSB=0)] / [P(r|LSB=1)] )
    
    # Punkty dla LSB = 0: +3, -3
    # Punkty dla LSB = 1: +1, -1
    
    # Obliczenie prawdopodobieństw
    p_lsb0 = np.exp(-(r-3)**2/(2*sigma2)) + np.exp(-(r+3)**2/(2*sigma2))
    p_lsb1 = np.exp(-(r-1)**2/(2*sigma2)) + np.exp(-(r+1)**2/(2*sigma2))
    
    # Unikanie dzielenia przez zero
    p_lsb1 = np.maximum(p_lsb1, 1e-10)
    
    return np.log(p_lsb0 / p_lsb1)


def qam16_symbols_to_llr(received_symbols, snr):
    """
    Konwertuje odebrane symbole 16-QAM na wartości LLR (Log-Likelihood Ratio)
    dla każdego bitu.
    
    Parametry:
      received_symbols - wektor odebranych symboli 16-QAM (numpy.array, complex)
      snr - stosunek sygnału do szumu (float)
      
    Zwraca:
      wektor LLR dla każdego bitu (numpy.array)
    """
    # Oblicz wariancję szumu na podstawie SNR
    sigma2 = 1 / (2 * snr)
    
    # Inicjalizuj tablicę LLR dla każdego bitu (4 bity na symbol)
    llr = np.zeros(4 * len(received_symbols))
    
    for i, symbol in enumerate(received_symbols):
        # Demapowanie symbolu na 4 wartości LLR
        symbol_llr = qam16_demapper(symbol, sigma2)
        
        # Zapisanie LLR do wektora wynikowego
        llr[4*i:4*i+4] = symbol_llr
    
    return llr


def encode_16qam(G, bits, snr):
    """
    Koduje wiadomość za pomocą macierzy generującej G,
    a następnie moduluje za pomocą 16-QAM.
    
    Parametry:
      G - macierz generująca
      bits - wektor bitów wiadomości
      snr - stosunek sygnału do szumu
      
    Zwraca:
      symbole 16-QAM (numpy.array, complex)
    """
    # Kodowanie LDPC
    encoded_bits = np.mod(np.dot(G, bits), 2)
    
    # Modulacja 16-QAM
    return bits_to_16qam_symbols(encoded_bits)


def simulate_ldpc_16qam(n=1296, d_v=2, d_c=4, snr=2.5, num_trials=10, maxiter=100, use_custom_decoder=False):
    """
    Symulacja systemu LDPC z modulacją 16-QAM.
    
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
        f"Symulacja LDPC z modulacją 16-QAM\nn={n}, k={k}, snr={snr} dB, próby={num_trials}, dekoder={'custom' if use_custom_decoder else 'pyldpc'}")
    print(f"Macierz H: {H.shape}, Macierz G: {G.shape}\n")

    successes = 0
    total_bit_errors = 0

    for trial in range(num_trials):
        print(f"=== Próba {trial + 1} ===")
        
        # Generowanie losowej wiadomości
        x = np.random.randint(0, 2, k)
        
        # Kodowanie i modulacja 16-QAM
        symbols = encode_16qam(G, x, snr_linear)
        
        # Dodawanie szumu gaussowskiego (AWGN)
        # Dla 16-QAM, szum jest dodawany do części rzeczywistej i urojonej niezależnie
        noise_real = np.random.normal(0, np.sqrt(1/(2*snr_linear)), len(symbols))
        noise_imag = np.random.normal(0, np.sqrt(1/(2*snr_linear)), len(symbols))
        noise = noise_real + 1j * noise_imag
        
        received_symbols = symbols + noise
        
        # Demodulacja - obliczenie LLR dla każdego bitu
        llr = qam16_symbols_to_llr(received_symbols, snr_linear)
        
        # Dekodowanie
        if use_custom_decoder:
            # Użyj własnego dekodera Min-Sum
            x_hat_full = belief_propagation_decode_min_sum(
                llr, H, max_iter=maxiter, alpha=0.75)
            
            # Alternatywnie, dekoder z dampingiem:
            # x_hat_full = belief_propagation_decode(
            #    llr, H, max_iter=maxiter, damping=0.5)
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
    # Symulacja LDPC z modulacją 16-QAM
    # Uwaga: Dla 16-QAM zazwyczaj potrzebny jest wyższy SNR niż dla QPSK czy BPSK
    simulate_ldpc_16qam(n=1296, d_v=2, d_c=4, snr=12.0, num_trials=10, 
                      maxiter=100, use_custom_decoder=True)