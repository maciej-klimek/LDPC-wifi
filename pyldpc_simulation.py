import numpy as np
from pyldpc import make_ldpc, encode, get_message
from decoder import belief_propagation_decode
from decoderMinSum import belief_propagation_decode_min_sum


def simulate_ldpc(n=1296, d_v=2, d_c=4, snr=2.5, num_trials=10, maxiter=100, use_custom_decoder=False):
    d_v = max(2, d_v)
    d_c = max(3, d_c)

    H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
    k = G.shape[1]
    print(
        f"Symulacja LDPC\nn={n}, k={k}, snr={snr}, próby={num_trials}, dekoder={'custom' if use_custom_decoder else 'pyldpc'}")
    print(f"Macierz H: {H.shape}, Macierz G: {G.shape}\n")

    successes = 0
    total_bit_errors = 0

    for trial in range(num_trials):
        print(f"=== Próba {trial + 1} ===")
        x = np.random.randint(0, 2, k)
        y = encode(G, x, snr)

        y_noisy = y + np.random.normal(scale=1 / snr, size=y.shape)

        if use_custom_decoder:
            llr = 2 * y_noisy * snr
            # Damping - bardziej "kosztowny" czasowo, ale dokładniejszy
            # x_hat_full = belief_propagation_decode(
            #    llr, H, max_iter=maxiter, damping=0.5)

            # Min-sum - szybszy, ale mniej dokładny
            x_hat_full = belief_propagation_decode_min_sum(
                llr, H, max_iter=maxiter,alpha=0.75)
        else:
            from pyldpc import decode
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
    simulate_ldpc(n=1296, d_v=2, d_c=4, snr=6.0, num_trials=10,
                  maxiter=100, use_custom_decoder=True)
