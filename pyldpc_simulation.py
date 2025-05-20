import numpy as np
from pyldpc import make_ldpc, encode, decode, get_message


def simulate_ldpc(n=1296, d_v=2, d_c=4, snr=2.5, num_trials=10, maxiter=100):
    d_v = max(2, d_v)
    d_c = max(3, d_c)

    H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
    k = G.shape[1]
    print(f"Symulacja LDPC\nn={n}, k={k}, snr={snr}, próby={num_trials}")
    print(f"Macierz H: {H.shape}, Macierz G: {G.shape}\n")

    successes = 0
    total_bit_errors = 0

    for trial in range(num_trials):
        print(f"=== Próba {trial + 1} ===")
        x = np.random.randint(0, 2, k)
        y = encode(G, x, snr)

        y_noisy = y + np.random.normal(scale=1 / snr, size=y.shape)

        x_hat_full = decode(H, y_noisy, snr, maxiter=maxiter)
        x_hat = get_message(G, x_hat_full)

        bit_errors = np.sum(x != x_hat)
        total_bit_errors += bit_errors

        if bit_errors == 0:
            print(f"Próba {trial + 1}: SUKCES ✅")
            successes += 1
        else:
            print(f"Próba {trial + 1}: BŁĄD ❌, błędów bitowych: {bit_errors}")

    print(f"\nSkuteczność: {successes}/{num_trials}, "
          f"Średnia liczba błędów bitowych: {total_bit_errors / num_trials:.2f}")


if __name__ == "__main__":
    simulate_ldpc(n=1296, d_v=2, d_c=4, snr=4.0, num_trials=10, maxiter=100)
