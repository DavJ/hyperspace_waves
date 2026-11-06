# Převod na šířící se kompaktní mód v konečné vrstvě

**2026-11-06 · pracovní výzkum.** Výpočet rozšiřuje [kompaktní Einsteinův–Maxwellův model](compact-einstein-maxwell-port.cs.md). Předpokládá reálnou prostorovou kružnici o poloměru $R$, připravenou elektromagnetickou pumpu s kompaktním indexem jedna a rovinná pole nezávislá na $x_1,x_2$. Pracujeme v nejnižším řádu slabé gravitace na plochém pozadí a v nejnižším řádu v amplitudách elektromagnetického buzení. Vlastní zakřivení statické vrstvy se projeví ve vyšších řádech tohoto rozvoje. Koeficienty odchozích vln se vztahují k ploché rovinné úloze. Odvození a překlad s pomocí AI. Úplná lidská revize se netvrdí.

<!-- BILINGUAL-UNIT: compact-radiation.layer -->

## Prostředí pro výměnu podélné hybnosti

Použijeme kanonický reálný skalár $\chi$ s $v,\ell>0$ a elektromagnetickým faktorem $Z>0$. Parametr $\epsilon$ řídí malý elektromagnetický kontrast vrstvy.

~~~math
\mathcal L_{\rm matter}=-\tfrac14 Z(\chi)F_{AB}F^{AB}
-\tfrac12\partial_A\chi\,\partial^A\chi-V(\chi),\qquad
V(\chi)=\frac{(\chi^2-v^2)^2}{2\ell^2 v^2},\qquad
Z(\chi)=e^{\epsilon(1-\chi^2/v^2)}.
~~~

Při nulových elektromagnetických amplitudách řeší rovinný kink plochou rovnici skaláru. Při použití obou rovnic hmoty se výměna mezi elektromagnetickým a skalárním tenzorem energie a hybnosti vyruší. Tlak skaláru ve směrech $x_1$ a $x_2$ je stejný i pro poruchu skaláru závislou na $t,z,y$. Do rozdílu budícího $h_+$ proto nepřispívá a současně započítává výměnu hybnosti s prostředím. Ustálený rozvoj ve slabém buzení předpokládá frekvence mimo skalární rezonance vrstvy.

~~~math
\chi_0(z)=v\tanh(z/\ell),\qquad Z(z)=e^{\epsilon W(z)},\qquad
W(z)=\operatorname{sech}^2(z/\ell),\qquad
\partial^A T^{\rm EM}_{AB}=-\tfrac14F^2\partial_B Z,\quad
\partial^A T^\chi_{AB}=+\tfrac14F^2\partial_B Z.
~~~

<!-- BILINGUAL-UNIT: compact-radiation.drive -->

## Buzení a rozdíl hybností

Oba Maxwellovy potenciály polarizujeme podél $x_1$ a k vytvoření reálných polí přičteme komplexně sdružené členy. Pumpa na $\Omega=1/R$ postupuje podél $y$ a je přesným Maxwellovým módem daného rovinného profilu $Z(z)$, protože její podélné vlnové číslo mizí. Běžný signál vstupuje zleva s amplitudou $a_{\rm in}$ a $\omega>0$. Elektromagnetické napětí na součtové frekvenci dostaneme přímo ze smíšeného členu obou intenzit pole.

~~~math
\begin{gathered}
A_{1,\rm p}=a_{\rm p}e^{i(y/R-\Omega t)},\qquad
A_{1,\rm s}=a(z)e^{-i\omega t},\qquad
(Za')'+Z\omega^2a=0,\\
\nu=\Omega+\omega,\qquad k=\sqrt{\nu^2-\Omega^2}=\sqrt{\omega^2+2\Omega\omega},
\qquad \Delta=k-\omega,\qquad (T_{11}-T_{22})_{\nu,1}=2\Omega\omega\, Z a_{\rm p}a.
\end{gathered}
~~~

Součtová frekvence je nad prahem volného šíření kompaktního módu. Odchozí metrická vlna má vpravo vlnové číslo $k$, zatímco dopadající signál přispívá podél $z$ hodnotou $\omega$. Rozdíl $\Delta$ může dodat vrstva. Pro $0<\omega<\Omega$ zůstává rozdílová frekvence $\Omega-\omega$ pod prahem volného šíření ve stejné kompaktní harmonické. Statická vrstva zachovává frekvenci.

<!-- BILINGUAL-UNIT: compact-radiation.emission -->

## Oddělení vyzářené vlny

Označme $h(z)$ amplitudu $h_+$ pro $(\nu,n=1)$. V rovnoměrných oblastech existuje vynucená metrická odezva úměrná elektromagnetickému poli. Odečteme ji řešením pro $r$. Maxwellova rovnice pro $a$ dává v uvedeném řádu gravitace a buzení přesně lokalizovaný zdroj pro $r$.

~~~math
\begin{aligned}
(-\partial_z^2-k^2)h&=2\Omega\omega\,Za_{\rm p}a/M_5^3,\qquad
r=h+a_{\rm p}a/M_5^3,\\
(-\partial_z^2-k^2)r
 &=\frac{a_{\rm p}}{M_5^3}
 \left[2\Omega\omega(Z-1)a+\frac{Z'}{Z}a'\right].
\end{aligned}
~~~

Retardovaná prostorová inverze je $i e^{ik|z-z'|}/(2k)$, jak plyne ze skoku derivace pro $-\partial_z^2-k^2$. Koeficient $\mathcal A_+$ u $e^{ikz}$ vpravo je $i/(2k)$ krát Fourierova transformace zdroje při $k$. V prvním řádu v $\epsilon$ dosadíme do lokalizovaného zdroje $a=a_{\rm in}e^{i\omega z}$. Integrace po částech dává $[2\Omega\omega-\omega\Delta]\widehat W(\Delta)=k\Delta\widehat W(\Delta)$. Transformace plyne ze substituce $u=(1+\tanh(z/\ell))/2$ v jejím integrálu.

~~~math
\mathcal A_+
=\frac{i\epsilon a_{\rm p}a_{\rm in}}{2M_5^3}\,\Delta\,\widehat W(\Delta)
+O(\epsilon^2),\qquad
\widehat W(q)=\int_{\mathbb R}W(z)e^{-iqz}\,dz
=\frac{\pi\ell^2 q}{\sinh(\pi\ell q/2)}.
~~~

Při nenulové pumpě i signálu dává kink nenulovou odchozí amplitudu vpravo v prvním řádu v $\epsilon$. Derivační člen zdroje pochází z odezvy elektromagnetické vlny na vrstvu. Uvedený koeficient vzniká započtením obou členů.

<!-- BILINGUAL-UNIT: compact-radiation.readout -->

## Čtení přes tutéž vrstvu

Vyšleme doprava metrický mód $h=H_{\rm in}e^{ikz}$ a použijeme sdruženou pumpu. Maxwellův zdroj $\partial_A(ZhF_{\rm p}^{A1*})$ má běžnou nulovou harmonickou při frekvenci $\omega$. Označme $b$ amplitudu tohoto fotonu. Odečteme rozloženou vynucenou část a pro zbytek $u$ vezmeme odchozí Maxwellovo řešení. V prvním řádu v $\epsilon$ plyne koeficient $\mathcal B_+$ u $e^{i\omega z}$ z téhož Fourierova integrálu s opačným znaménkem $\Delta$.

~~~math
\begin{aligned}
(Zb')'+Z\omega^2 b&=-\Omega\omega\,Za_{\rm p}^*h,\qquad
u=b-\tfrac12 a_{\rm p}^*h,\\
(Zu')'+Z\omega^2u&=-\tfrac12 a_{\rm p}^*Z'h',\\
\mathcal B_+&=\frac{i\epsilon a_{\rm p}^*H_{\rm in}}{4\omega}
 \,k\Delta\,\widehat W(-\Delta)+O(\epsilon^2).
\end{aligned}
~~~

Pro uvedenou konvenci reálných polí je tok energie dělený frekvencí $2\omega|a|^2$ pro běžný foton a $M_5^3 k|H|^2$ pro metrický mód, na jednotku příčné plochy a kompaktní délky. Po této normalizaci tokem jsou oba převodní koeficienty prvního řádu zápornými komplexně sdruženými protějšky, protože $W$ je reálné.

<!-- BILINGUAL-UNIT: compact-radiation.check -->

## Reprodukce a rozsah

Příkaz `python scripts/verify_source_matching.py --compact-radiation` vypíše bezrozměrný příklad $R=\ell=1$, $\omega=0.5$. Předpovězené koeficienty odchozích vln na jednotku $\epsilon$ jsou $0.530662i$ a $0.593298i$ při jednotkových vstupních amplitudách a amplitudě pumpy a jednotkovém $M_5^3$ pro výpočet koeficientů. Numerická kontrola integruje Maxwellův rozptyl ve skutečném profilu vrstvy, vypočítá oba odchozí koeficienty a ověří limit nulového kontrastu. Výpočty dávají rovinné vyzařování a čtení v předpokládaném pětirozměrném modelu. Radarová ozvěna dále vyžaduje rozptyl na cíli a přípravu kompaktní pumpy a použití v UBT vyžaduje fyzikální identifikaci kompaktní souřadnice. LEAN-PENDING.

[Odchozí Helmholtzovy Greenovy funkce](https://math.mit.edu/icg/resources/teaching/18.367/notes367.pdf) · [Fourierův popis fázového přizpůsobení](https://doi.org/10.1109/3.161322)
