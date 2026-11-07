# Jeden radarový návrat v kompaktním módu s pasivním cílem

**2026-11-07 · pracovní výzkum.** Tímto dokončujeme výpočet prvního návratu v uvedeném srovnávacím modelu s reálnou prostorovou kružnicí a oddělenými kalibrovanými porty. Výpočet předpokládá připravenou kompaktní pumpu, jediný neutrální cíl s jednou vnitřní odezvou a oddělené kanály volných vln mezi stanovištěm a cílem. Ponecháváme první příspěvek s jedním převodem při vyslání a jedním při příjmu. Přímé elektromagnetické cesty, překrývající se pole buzená pumpou a opakované cesty mezi stanovištěm a cílem by vyžadovaly jiný výpočet celého systému. Odvození a překlad s pomocí AI. Úplná lidská revize se netvrdí.

<!-- BILINGUAL-UNIT: compact-echo.target -->

## Slapový oscilátor s konečným profilem

Pišme $\varphi=y/R$ a vezměme rovinný sektor $h_{11}=-h_{22}=h_+$ [kompaktního Einsteinova–Maxwellova modelu](compact-einstein-maxwell-port.cs.md). Linearizovaná slapová křivost $R_{0101}-R_{0202}=-\ddot h_+$ je invariantní vůči linearizovaným kalibračním transformacím kolem plochého pozadí. Použijeme reálnou kvadrupólovou souřadnici $q$ s vlastní frekvencí $\omega_0>0$ a průměrovacím profilem $f=w(z-d)p(\varphi-\alpha)$. Slapový člen $-gq\ddot U_f$ přejde po integraci po částech v čase na kinetické míchání níže. Normalizace $U$ zahrnuje délku kompaktního směru. Pro reálná pole je $\langle f,U\rangle=\int fU\,d\mu$.

~~~math
\begin{aligned}
U&=\sqrt{M_5^3(2\pi R)/2}\,h_+,\qquad U_f=\langle f,U\rangle,\qquad
d\mu=dz\,\frac{d\varphi}{2\pi},\\
L&=\tfrac12\int\!\left(\dot U^2-U_z^2-R^{-2}U_\varphi^2\right)d\mu
   +\tfrac12(\dot q^2-\omega_0^2q^2)+g\dot q\,\dot U_f,\\
2K&=\|\dot U+g\dot q f\|_2^2+(1-g^2\|f\|_2^2)\dot q^2 .
\end{aligned}
~~~

Celková kvadratická energie má stejný kinetický čtverec a nezáporné gradientové a oscilátorové členy. Model je pasivní pro $g^2\|f\|_2^2<1$. Příklad používá normovaný obdélník $w=1/(2a)$ na $[-a,a]$ a reálný sudý profil $p$ s koeficienty $p_0=1$, $p_{\pm1}=0.25$, $p_{\pm2}=0.10$. Parametr $\alpha$ posouvá tento známý úhlový profil a $d$ posouvá obdélník. Vlastní odezva je díky podélnému průměru a konečnému počtu harmonických konečná.

[Dynamické kvadrupóly vázané na slapovou křivost](https://arxiv.org/abs/1608.01907)

<!-- BILINGUAL-UNIT: compact-echo.scatter -->

## Přesná lineární odezva cíle

Použijeme $e^{-i\nu t}$ s $\nu>0$ mimo prahy kanálů. U retardované odmocniny volíme kladné reálné $k_j$ v otevřených kanálech a kladné imaginární $k_j$ v uzavřených. Rovnice pole a oscilátoru dávají následující řešení. Rozptýlené pole je $\mathcal T\langle f,U_{\rm in}\rangle G_\nu f$.

~~~math
\begin{aligned}
k_j&=\sqrt{(\nu+i0)^2-j^2/R^2},\qquad G_j(s)=\frac{i}{2k_j}e^{ik_j|s|},
\qquad \Sigma=\langle f,G_\nu f\rangle,\\
U&=U_{\rm in}+g\nu^2 q\,G_\nu f,\qquad
(\omega_0^2-\nu^2)q=g\nu^2 U_f,\qquad
\mathcal T=\frac{g^2\nu^4}{\omega_0^2-\nu^2-g^2\nu^4\Sigma}.
\end{aligned}
~~~

Pro jednotkovou vlnu dopadající doprava v módu $n$ má překryv na cíli faktor $p_{-n}\widehat w(-k_n)e^{ik_nd+in\alpha}$, kde $\widehat w(k)=\int w(s)e^{-iks}ds$. Rozvojem odchozí Greenovy funkce nalevo získáme koeficient odrazu $\mathcal R_{mn}$ u $e^{-ik_m z+im\varphi}$. Koeficient průchodu $\mathcal P_{mn}$ přičte dopadající člen $\delta_{mn}$ a v rozptýleném členu nahradí $k_n+k_m$ výrazem $k_n-k_m$ a $\widehat w(-k_m)$ výrazem $\widehat w(k_m)$.

~~~math
\begin{aligned}
\mathcal R_{mn}
 &=\frac{i\mathcal T}{2k_m}\,p_m p_{-n}\,\widehat w(-k_m)\widehat w(-k_n)
       e^{i(k_m+k_n)d+i(n-m)\alpha},\\
\operatorname{Im}\Sigma
 &=\sum_{j\ {\rm open}}\frac{|p_j\widehat w(k_j)|^2}{2k_j},
\qquad
\operatorname{Im}\mathcal T=|\mathcal T|^2\operatorname{Im}\Sigma,\qquad
\sum_{m\ {\rm open}}k_m\big(|\mathcal R_{mn}|^2+|\mathcal P_{mn}|^2\big)=k_n .
\end{aligned}
~~~

Uzavřené módy vstupují do reálné části $\Sigma$. V otevřených módech je imaginární část Greenova jádra $\cos(k_j|s|)/(2k_j)$ a její průměr dává uvedený čtverec Fourierova překryvu. Interferenční člen dopředné prošlé vlny je $-\operatorname{Im}\mathcal T\,|p_n\widehat w(k_n)|^2$. Součet rozptýlených toků je podle optické identity jeho opakem. Cíl v tomto modelu mění fáze a přerozděluje tok bez absorpce.

<!-- BILINGUAL-UNIT: compact-echo.range -->

## Úplný návrat a oddělení dvou souřadnic

Při společné metrické frekvenci $\nu>2/R$ vyšleme harmonickou $n=1$ s běžnou frekvencí $\omega_1$ a použijeme [převodní koeficienty vrstvy](compact-radiation-layer.cs.md) $\mathcal A_1$ při vyslání a $\mathcal B_m$ při příjmu. Čtecí pumpa pro $m=1,2$ má frekvenci $m/R$ a vrací běžnou frekvenci $\omega_m$. Sudá vrstva má ve svém středu stejný převodní koeficient vedoucího řádu i v opačném směru šíření. Kalibrace určuje komplexní portové faktory a známé překryvy profilu. Oba návraty musí být nenulové.

~~~math
\begin{aligned}
E_{mn}&=\mathcal B_m\,\mathcal R_{mn}\,\mathcal A_n,\qquad
\omega_j=\nu-j/R,\qquad F_j=p_j\widehat w(-k_j),\\
Z(\nu)&=\frac{E_{21}/E_{11}}{K(\nu)}
       =e^{i\delta(\nu)d-i\alpha},\qquad \delta(\nu)=k_2-k_1,\qquad
K(\nu)=\frac{\mathcal B_2 k_1 F_2}{\mathcal B_1 k_2 F_1}.
\end{aligned}
~~~

Návrat ve stejném módu sám o sobě nezávisí na $\alpha$. Poměr při každé frekvenci vykrátí celý skalární rezonanční faktor $\mathcal T(\nu)$ včetně jeho fáze. Jedna frekvence stále směšuje $d$ s $\alpha$. Vezmeme dvě různé frekvence $\nu_b>\nu_a>2/R$ a podělením kalibrovaných poměrů odstraníme $\alpha$.

~~~math
\frac{Z(\nu_b)}{Z(\nu_a)}=e^{i(\delta_b-\delta_a)d},\qquad
e^{i\alpha}=\frac{e^{i\delta_a d}}{Z(\nu_a)},\qquad
\delta'(\nu)=\nu\left(\frac1{k_2}-\frac1{k_1}\right)>0.
~~~

Vzdálenost je jednoznačná na známém intervalu o šířce menší než $2\pi/(\delta_b-\delta_a)$. Pro polouzavřený interval s touto šířkou volíme právě jednoho reprezentanta. Poté je $\alpha$ určen modulo $2\pi$. Výsledek používá jeden společný faktor cíle a známé poměry harmonických překryvů. U obecného cíle s více nezávislými souřadnicemi odezvy toto vykrácení nemusí platit.

<!-- BILINGUAL-UNIT: compact-echo.verify -->

## Co výpočet stanoví

Novým výsledkem tohoto srovnávacího modelu je pasivní koherentní návrat s oddělenou rekonstrukcí podélné vzdálenosti a kompaktního úhlu i při neznámé rezonanci cíle. Spojuje slapovou odezvu, rozptyl vlny a koherentní fázovou kalibraci. Kompaktní souřadnice, příprava kompaktní pumpy a vazba cíle mají pro použití v UBT status předpokladů.

Příkaz `python scripts/verify_compact_echo.py` spustí bezrozměrný příklad $R=1$, $a=0.4$, $g=0.12$, $\nu_a=2.3$, $\nu_b=2.8$. Perioda nejednoznačnosti vzdálenosti je $22.46$. Rekonstrukce dává $d=7.4$ a $\alpha=1.2$ pro obě vlastní frekvence $\omega_0=2.45$ a $4$. Kód kontroluje průměrované Greenovo jádro kvadraturou a bilanci toku s otevřenými i uzavřenými módy. [Lean](../HyperspaceWaves/Echo.lean) ověřuje konečněrozměrný kinetický čtverec, algebru odezvy cíle, skalární optickou identitu a obě fázová vykrácení. Model pole a oddělení portů jsou předpoklady výpočtu.

[Výpočet opakovaných návratů](compact-radar-feedback.cs.md) sečte odražené dráhy a určí podmínku, za které kalibrovaný poměr zůstává platný.
