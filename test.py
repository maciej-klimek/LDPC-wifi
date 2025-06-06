import numpy as np
import matplotlib.pyplot as plt
from pyldpc_bpsk_simulation import simulate_ldpc
from pyldpc_qpsk_simulation import simulate_ldpc_qpsk
from pyldpc_16qam_simulation import simulate_ldpc_16qam
import time

def test_modulation_schemes():
    """
    Testuje różne schematy modulacji LDPC i porównuje ich wydajność.
    """
    # Parametry testowe
    test_params = {
        'n_values': [648, 1296, 1944],  # Różne długości kodów
        'snr_range_bpsk': np.arange(1.0, 8.0, 1.0),
        'snr_range_qpsk': np.arange(2.0, 10.0, 1.0), 
        'snr_range_16qam': np.arange(8.0, 16.0, 1.0),
        'num_trials': 50,  # Zwiększona liczba prób dla lepszej statystyki
        'maxiter': 100,
        'd_v': 2,
        'd_c': 4
    }
    
    results = {}
    
    print("=== ROZPOCZĘCIE TESTÓW MODULACJI LDPC ===\n")
    
    for n in test_params['n_values']:
        print(f"\n🔬 TESTOWANIE DLA n={n}")
        print("="*50)
        
        results[n] = {
            'BPSK': {'snr': [], 'ber': [], 'success_rate': []},
            'QPSK': {'snr': [], 'ber': [], 'success_rate': []},
            '16-QAM': {'snr': [], 'ber': [], 'success_rate': []}
        }
        
        # Test BPSK
        print(f"\n📡 Testowanie BPSK (n={n})")
        for snr in test_params['snr_range_bpsk']:
            print(f"  SNR = {snr:.1f} dB", end=" ... ")
            start_time = time.time()
            success_rate, avg_errors = simulate_ldpc(
                n=n, 
                d_v=test_params['d_v'], 
                d_c=test_params['d_c'], 
                snr=snr, 
                num_trials=test_params['num_trials'],
                maxiter=test_params['maxiter'], 
                use_custom_decoder=True
            )
            end_time = time.time()
            
            # Obliczenie BER (Bit Error Rate)
            k = n - (n * test_params['d_v']) // test_params['d_c']  # przybliżone k
            ber = avg_errors / k if k > 0 else avg_errors / (n * 0.75)
            
            results[n]['BPSK']['snr'].append(snr)
            results[n]['BPSK']['ber'].append(ber)
            results[n]['BPSK']['success_rate'].append(success_rate)
            
            print(f"BER: {ber:.4f}, Sukces: {success_rate:.2f}, Czas: {end_time-start_time:.1f}s")
        
        # Test QPSK
        print(f"\n📡 Testowanie QPSK (n={n})")
        for snr in test_params['snr_range_qpsk']:
            print(f"  SNR = {snr:.1f} dB", end=" ... ")
            start_time = time.time()
            success_rate, avg_errors = simulate_ldpc_qpsk(
                n=n, 
                d_v=test_params['d_v'], 
                d_c=test_params['d_c'], 
                snr=snr, 
                num_trials=test_params['num_trials'],
                maxiter=test_params['maxiter'], 
                use_custom_decoder=True
            )
            end_time = time.time()
            
            # Obliczenie BER
            k = n - (n * test_params['d_v']) // test_params['d_c']
            ber = avg_errors / k if k > 0 else avg_errors / (n * 0.75)
            
            results[n]['QPSK']['snr'].append(snr)
            results[n]['QPSK']['ber'].append(ber)
            results[n]['QPSK']['success_rate'].append(success_rate)
            
            print(f"BER: {ber:.4f}, Sukces: {success_rate:.2f}, Czas: {end_time-start_time:.1f}s")
        
        # Test 16-QAM
        print(f"\n📡 Testowanie 16-QAM (n={n})")
        for snr in test_params['snr_range_16qam']:
            print(f"  SNR = {snr:.1f} dB", end=" ... ")
            start_time = time.time()
            success_rate, avg_errors = simulate_ldpc_16qam(
                n=n, 
                d_v=test_params['d_v'], 
                d_c=test_params['d_c'], 
                snr=snr, 
                num_trials=test_params['num_trials'],
                maxiter=test_params['maxiter'], 
                use_custom_decoder=True
            )
            end_time = time.time()
            
            # Obliczenie BER
            k = n - (n * test_params['d_v']) // test_params['d_c']
            ber = avg_errors / k if k > 0 else avg_errors / (n * 0.75)
            
            results[n]['16-QAM']['snr'].append(snr)
            results[n]['16-QAM']['ber'].append(ber)
            results[n]['16-QAM']['success_rate'].append(success_rate)
            
            print(f"BER: {ber:.4f}, Sukces: {success_rate:.2f}, Czas: {end_time-start_time:.1f}s")
    
    return results, test_params

def plot_results(results, test_params):
    """
    Tworzy wykresy porównawcze wyników.
    """
    n_values = test_params['n_values']
    
    # Konfiguracja kolorów i stylów
    colors = {'BPSK': 'blue', 'QPSK': 'green', '16-QAM': 'red'}
    markers = {'BPSK': 'o', 'QPSK': 's', '16-QAM': '^'}
    
    # Tworzenie subplot dla każdej długości kodu
    fig, axes = plt.subplots(2, len(n_values), figsize=(5*len(n_values), 10))
    if len(n_values) == 1:
        axes = axes.reshape(2, 1)
    
    for i, n in enumerate(n_values):
        # Wykres BER vs SNR
        ax1 = axes[0, i]
        for modulation in ['BPSK', 'QPSK', '16-QAM']:
            if results[n][modulation]['snr']:
                ax1.semilogy(results[n][modulation]['snr'], 
                           results[n][modulation]['ber'],
                           marker=markers[modulation], 
                           color=colors[modulation],
                           label=modulation,
                           linewidth=2,
                           markersize=6)
        
        ax1.set_xlabel('SNR [dB]')
        ax1.set_ylabel('Bit Error Rate (BER)')
        ax1.set_title(f'BER vs SNR (n={n})')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim([1e-4, 1])
        
        # Wykres Success Rate vs SNR
        ax2 = axes[1, i]
        for modulation in ['BPSK', 'QPSK', '16-QAM']:
            if results[n][modulation]['snr']:
                ax2.plot(results[n][modulation]['snr'], 
                        results[n][modulation]['success_rate'],
                        marker=markers[modulation], 
                        color=colors[modulation],
                        label=modulation,
                        linewidth=2,
                        markersize=6)
        
        ax2.set_xlabel('SNR [dB]')
        ax2.set_ylabel('Success Rate')
        ax2.set_title(f'Success Rate vs SNR (n={n})')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('ldpc_modulation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_spectral_efficiency_comparison(results, test_params):
    """
    Porównanie efektywności spektralnej różnych modulacji.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Efektywność spektralna (bity/symbol)
    spectral_efficiency = {'BPSK': 1, 'QPSK': 2, '16-QAM': 4}
    colors = {'BPSK': 'blue', 'QPSK': 'green', '16-QAM': 'red'}
    markers = {'BPSK': 'o', 'QPSK': 's', '16-QAM': '^'}
    
    # Używamy wyników dla n=1296 (najbardziej reprezentatywne)
    n = 1296
    if n in results:
        for modulation in ['BPSK', 'QPSK', '16-QAM']:
            if results[n][modulation]['snr']:
                # Znajdź SNR dla BER ≈ 10^-3
                target_ber = 1e-3
                snr_for_target = None
                
                for j, ber in enumerate(results[n][modulation]['ber']):
                    if ber <= target_ber:
                        snr_for_target = results[n][modulation]['snr'][j]
                        break
                
                if snr_for_target is not None:
                    ax.scatter(snr_for_target, spectral_efficiency[modulation], 
                             s=100, marker=markers[modulation], 
                             color=colors[modulation], label=modulation)
                    ax.annotate(f'{modulation}\n({snr_for_target:.1f} dB)', 
                              (snr_for_target, spectral_efficiency[modulation]),
                              xytext=(10, 10), textcoords='offset points',
                              fontsize=9, ha='left')
    
    ax.set_xlabel('Required SNR [dB] for BER ≈ 10⁻³')
    ax.set_ylabel('Spectral Efficiency [bits/symbol]')
    ax.set_title('Spectral Efficiency vs Required SNR')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('spectral_efficiency_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_table(results, test_params):
    """
    Drukuje tabelę podsumowującą wyniki.
    """
    print("\n" + "="*80)
    print("📊 PODSUMOWANIE WYNIKÓW")
    print("="*80)
    
    for n in test_params['n_values']:
        print(f"\n🔍 Wyniki dla n={n}:")
        print("-"*60)
        print(f"{'Modulacja':<10} {'Min BER':<12} {'Max Success':<12} {'SNR dla 90% sukc.':<15}")
        print("-"*60)
        
        for modulation in ['BPSK', 'QPSK', '16-QAM']:
            if results[n][modulation]['ber']:
                min_ber = min(results[n][modulation]['ber'])
                max_success = max(results[n][modulation]['success_rate'])
                
                # Znajdź SNR dla 90% sukcesu
                snr_90 = "N/A"
                for j, success in enumerate(results[n][modulation]['success_rate']):
                    if success >= 0.9:
                        snr_90 = f"{results[n][modulation]['snr'][j]:.1f} dB"
                        break
                
                print(f"{modulation:<10} {min_ber:<12.2e} {max_success:<12.2f} {snr_90:<15}")

def main():
    """
    Główna funkcja uruchamiająca testy i generująca wykresy.
    """
    print("🚀 ROZPOCZYNANIE KOMPLEKSOWYCH TESTÓW MODULACJI LDPC")
    print("Może to potrwać kilka minut w zależności od parametrów...\n")
    
    # Uruchomienie testów
    results, test_params = test_modulation_schemes()
    
    # Generowanie wykresów
    print("\n📈 Generowanie wykresów...")
    plot_results(results, test_params)
    plot_spectral_efficiency_comparison(results, test_params)
    
    # Drukowanie podsumowania
    print_summary_table(results, test_params)
    
    print("\n✅ TESTY ZAKOŃCZONE!")
    print("📁 Wykresy zostały zapisane jako:")
    print("   - ldpc_modulation_comparison.png")
    print("   - spectral_efficiency_comparison.png")

if __name__ == "__main__":
    main()