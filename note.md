## 1. Ogólne Zasady Działania LDPC

### 1.1. Podstawowa Idea i Definicja
LDPC (Low-Density Parity-Check) to rodzina kodów korekcyjnych, której główną cechą jest bardzo rzadka macierz kontroli parzystości \( H \). Kod LDPC definiuje się właśnie przez tę macierz, gdzie:
- **Macierz \( H \)**: Każdy wiersz macierzy odpowiada równaniu parzystości, a każdy element (0 lub 1) wskazuje, czy dany bit kodu bierze udział w tym równaniu.
- **Macierz generująca \( G \)**: Od niej zależy sposób enkodowania wiadomości. W praktycznych zastosowaniach wykorzystuje się często formę systematyczną, czyli taki przypadek, w którym część bitów w codeword (słowo kodowe) stanowi bezpośrednio oryginalną wiadomość, a pozostałe bity są dodawane w celu uzyskania parzystości.

### 1.2. Struktura Macierzy i Własności
Kluczową cechą macierzy \( H \) w kodach LDPC jest jej **niskie zagęszczenie** (low density):
- **Rzadkość:** Znaczna większość elementów macierzy to zera. To powoduje, że liczba niezerowych elementów (jedynek) przypadających na wiersz lub kolumnę jest stosunkowo mała.
- **Waga wierszy i kolumn:** Projektanci starają się utrzymać ustalone wartości – na przykład każdy wiersz może mieć tylko kilka jedynek, podobnie jak kolumny. To pozwala na uproszczenie obliczeń podczas dekodowania.

### 1.3. Proces Enkodowania
Enkodowanie za pomocą LDPC odbywa się przy użyciu macierzy generującej \( G \):
- **Systematyczna forma enkodera:** Przyjmujemy, że słowo kodowe \( \mathbf{c} \) ma postać \( \mathbf{c} = \mathbf{u} \cdot G \) (operacja modulo 2), gdzie \( \mathbf{u} \) to oryginalna wiadomość.  
- **Związek między \( G \) a \( H \):** Macierz \( G \) jest skonstruowana tak, by zachodziło równanie \( G \cdot H^T = \mathbf{0} \) (modulo 2), co gwarantuje, że każde słowo kodowe spełnia warunki parzystości zadane przez \( H \).

### 1.4. Algorytm Dekodowania – Belief Propagation
Najbardziej popularnym algorytmem dekodowania LDPC jest iteracyjny algorytm *belief propagation* (również określany jako sum-product):
- **Reprezentacja Grafowa:** Kod LDPC można przedstawić za pomocą grafu dwudzielnego (znanego jako *factor graph*), w którym:
  - **Węzły zmiennych (variable nodes)** reprezentują bity słowa kodowego.
  - **Węzły kontrolne (check nodes)** reprezentują równania parzystości.
- **Przekazywanie Wiadomości:** Dekodowanie polega na iteracyjnym przekazywaniu wiadomości między węzłami zmiennymi a węzłami kontrolnymi. Wiadomości te często przedstawiane są jako log–likelihood ratios (LLR), które odzwierciedlają prawdopodobieństwo, że dany bit jest równy 0 lub 1.
- **Aktualizacja w Iteracjach:**
  - **Krok od zmiennych do kontrolnych:** Każdy węzeł zmienny wysyła do sąsiadujących węzłów kontrolnych swoje „przekonanie” o wartości bitu, uwzględniając sygnał odbierany z kanału oraz wiadomości przekazane przez inne węzły kontrolne.
  - **Krok od kontrolnych do zmiennych:** Każdy węzeł kontrolny na podstawie odebranych wiadomości oblicza nową wiadomość, która informuje, jakie wartości bitów mogą spełniać przypisane równania parzystości. Typowo operacje te są realizowane przy użyciu funkcji tangens hiperbolicznego (\(\tanh\)) oraz odwrotnej funkcji (\(\operatorname{arctanh}\)).
- **Warunek Zakończenia:** Algorytm wykonuje kolejne iteracje, aż spełniony zostanie warunek parzystości (czyli wszystkie równania kontrolne będą spełnione) lub osiągnięta zostanie maksymalna liczba iteracji.

### 1.5. Zalety LDPC
- **Wysoka wydajność korekcyjna:** Przy odpowiedniej konstrukcji macierzy \( H \), LDPC osiągają bardzo bliską granicę informacyjną, co oznacza, że kod może odzyskać sygnał nawet przy bardzo niskim stosunku sygnału do szumu.
- **Niskie złożoności obliczeniowe:** Ze względu na rzadką strukturę macierzy \( H \), iteracyjne algorytmy dekodowania mają względnie niską złożoność obliczeniową, co jest korzystne w implementacjach sprzętowych i programowych.

---

## 2. LDPC w Kontekście IEEE 802.11n (WiFi)

### 2.1. Wprowadzenie do Standardu IEEE 802.11n
IEEE 802.11n to standard sieci bezprzewodowych WiFi, który wprowadza techniki zwiększające przepustowość, takie jak MIMO (wielokrotne anteny) oraz możliwość transmisji wielu strumieni jednocześnie. W ramach standardu definiowane są różne metody kodowania – zarówno tradycyjne kody splotowe, jak i opcjonalne LDPC.

### 2.2. Zastosowanie LDPC w WiFi
- **Opcjonalność i Wydajność:** W standardzie 802.11n zastosowanie kodów LDPC jest opcjonalne, ale oferuje znaczące korzyści przy wysokich przepływnościach i trudnych warunkach transmisji (np. przy niskim stosunku sygnału do szumu). LDPC umożliwiają lepszą korekcję błędów dzięki wyższej złożoności (ale i większej skuteczności) procesu dekodowania.
- **Warianty Kodów:** Standard definiuje zestaw 12 kodów LDPC o różnych długościach bloków i współczynnikach kodowych. Przykładowo:
  - **Block Lengths:** 648, 1296, 1944 bitów.
  - **Code Rates:** 1/2, 2/3, 3/4, 5/6.
  
  Konstrukcja macierzy \( H \) dla tych kodów wykorzystuje strukturę blokową, w której macierz bazowa (często określana jako “base matrix”) jest rozwijana przez powielanie cykliczne – określane przez parametr \( Z \), który odpowiada rozmiarowi submacierzy cyklicznych. Taka struktura daje:
  - Łatwiejszą implementację sprzętową dzięki regularności i cyklicznej naturze macierzy.
  - Redukcję złożoności pamięci oraz operacji arytmetycznych niezbędnych przy dekodowaniu.

### 2.3. Proces Enkodowania i Dekodowania w Kontekście WiFi
- **Enkodowanie:**  
  Urządzenie nadające (np. router lub stacja bazowa) szyfruje dane korzystając z macierzy generującej \( G \), której projekt opiera się na rozwinięciu macierzy bazowej zgodnie z określonym parametrem \( Z \). Kod systematyczny umożliwia bezpośredni dostęp do danych oryginalnych po części słowa kodowego.
  
- **Dekodowanie:**  
  Odbiornik (np. laptop, smartfon) wykonuje dekodowanie przy użyciu algorytmów iteracyjnych (często zaimplementowanych w dedykowanym sprzęcie DSP lub FPGA), które wykorzystują algorytm belief propagation. Proces dekodowania jest zoptymalizowany pod kątem złożoności oraz szybkości, aby sprostać wymaganiom transmisji bezprzewodowej, gdzie ważna jest niska latencja.
  
- **Wpływ Warunków Kanału:**  
  W praktycznym wdrożeniu LDPC w WiFi istotne jest, że rzeczywisty kanał transmisyjny może generować zakłócenia (szum, interferencje). Dzięki iteracyjnemu procesowi dekodowania, LDPC potrafią dynamicznie reagować na zmienne warunki i poprawiać zbieżność dekodera, co przekłada się na mniejszą liczbę błędów bitowych (BER).

