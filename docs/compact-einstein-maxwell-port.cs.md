# Geometrická elektromagnetická vazba na kompaktní harmonickou

**2026-11-05 · pracovní výzkum.** Odvozeno z uvedených srovnávacích akcí. Odvození a překlad s pomocí AI. Úplná lidská revize se netvrdí.

<!-- BILINGUAL-UNIT: compact-port.ubt -->

## Vazba ve stávající geometrii UBT

Použijme popis s volně variovaným pomocným jetem a pevnou normalizací $N_0$. Zde je $X$ Lorentzovsky reálný nosný průmět plného bikvaternionového pole $\Theta$, $Q_\mu$ je jeho kompenzující pomocný jet a $D_\mu$ používá při tomto posunu pevnou spinovou konexi. Maxwellova akce vidí tetrádu přes metriku. Její variace ve směru posunu nosného pole je

~~~math
E_\mu=N_0^{-1/2}(D_\mu X+Q_\mu),\qquad
\delta_\epsilon X=\epsilon,\quad \delta_\epsilon Q_\mu=-D_\mu\epsilon,\quad
\delta_\epsilon E_\mu=0,\qquad
\delta_\epsilon S_{\rm EM}
=\frac12\int d^4x\sqrt{-g}\,T^{\mu\nu}\delta_\epsilon g_{\mu\nu}=0.
~~~

Konečný kompenzovaný posun zachovává stejné $E_\mu$ a tím i stejnou Maxwellovu akci. Pro čistě nosnou referenci z příkladu s vinutím je to v tomto popisu přesná nula. Čtyřrozměrná rodina akcí také ponechává neurčený hlavní kompaktní kinetický člen. Fyzikální vazba na nenulovou kompaktní harmonickou vyžaduje mód s geometrickou variací, která přežije redukci vazeb, a zdroj s příslušným kompaktním překryvem.

[Přesné jádro kalibračního posunu nosného pole](https://github.com/UBT-Institute/unified-biquaternion-theory/blob/672f365ad39c46149ecbbbabe2e6178d6c89e054/research_tracks/action_selection/mm_shift_gauge_dirac.cs.md) · [Podmínka pro kompaktní akci](https://github.com/UBT-Institute/unified-biquaternion-theory/blob/672f365ad39c46149ecbbbabe2e6178d6c89e054/research_tracks/information_first/TORUS_THETA_CANONICAL_ACTION_BRIDGE_GATE.cs.md)

<!-- BILINGUAL-UNIT: compact-port.action -->

## Explicitní nenulová vazba v prostorovém kompaktním modelu

Vezměme pětirozměrnou Einsteinovu–Maxwellovu teorii na plochém pozadí se signaturou $(-++++)$, reálnou prostorovou souřadnicí $y\sim y+2\pi R$, přirozenými jednotkami a kladnou gravitační normalizací $M_5^3$. Velké indexy zahrnují $y$ a $h_{AB}=\delta G_{AB}$. Vazbu dává první variace Maxwellova členu podle metriky při pevném kovariantním $F$. Kompaktní souřadnice, akce a dostupná kompaktní elektromagnetická pumpa jsou dodatečné fyzikální předpoklady tohoto srovnání.

~~~math
S_5=\int d^4x\,dy\sqrt{-G}
\left(\frac{M_5^3}{2}\mathcal R_5-\frac14F_{AB}F^{AB}\right),\qquad
\delta S_{\rm EM,5}=\frac12\int d^4x\,dy\,h_{AB}T^{AB},
\qquad T_{AB}=F_{AC}F_B{}^C-\frac14\eta_{AB}F^2.
~~~

Pro vlny nezávislé na příčných souřadnicích $x_1,x_2$ má tenzorová polarizace $h_{11}=h_+$, $h_{22}=-h_+$ nulovou stopu i divergenci. Její průmět $(h_{11}-h_{22})/2$ je v tomto Fourierově sektoru invariantní vůči linearizovaným difeomorfismům, protože příčná hybnost mizí. Po vypuštění okrajových členů má kvadratický Einsteinův kinetický člen na této polarizaci tvar $-M_5^3\partial_C h_{AB}\partial^C h^{AB}/8$. Přispívají obě diagonální složky, což dává první člen níže. Variací $h_+$ a rozvojem všech polí jako $\sum_n f_n e^{iny/R}$ dostaneme kompaktní hmotnost $|n|/R$ a buzenou vlnovou rovnici.

~~~math
S_+^{(2)}+S_+^{\rm int}
=\int d^4x\,dy\left[
\frac{M_5^3}{4}\big((\partial_t h_+)^2-(\partial_z h_+)^2-(\partial_y h_+)^2\big)
+\frac12 h_+(T_{11}-T_{22})\right],\qquad
(\partial_t^2-\partial_z^2+n^2/R^2)h_{+,n}=\frac{(T_{11}-T_{22})_n}{M_5^3}.
~~~

Díky kladnému volnému kinetickému členu jde o šířící se polarizaci srovnávací teorie. V ortonormální tetrádě odpovídá čtyřrozměrná část variaci $\delta E_1=h_+\bar E_1/2$ a $\delta E_2=-h_+\bar E_2/2$. V popisu s volně variovaným pomocným jetem se tato metrická variace při pevném nosném poli realizuje volbou $\delta Q_\mu=\sqrt{N_0}\,\delta E_\mu$. Dynamika pětirozměrné akce je dodatečný předpoklad srovnávacího modelu.

<!-- BILINGUAL-UNIT: compact-port.overlap -->

## Buzení a čtení z téhož Maxwellova členu

Zdroj na pravé straně vlnové rovnice používá Fourierův koeficient součinu. Signál s nulovou harmonickou a pumpa s kompaktním indexem $p$ mohou budit $n=p$. Variace téhož Maxwellova členu kolem pumpy $F_{\rm p}$ dává lineární Maxwellovu rovnici pro intenzitu čteného pole $f$. Zvedání indexů v obou rovnicích používá ploché pozadí.

~~~math
\begin{aligned}
\frac{(T_{11}-T_{22})_n}{M_5^3}
 &=\frac{1}{M_5^3}\sum_{p+q=n}
 \big(F_{1C,p}F_{1,q}{}^C-F_{2C,p}F_{2,q}{}^C\big),\\
\partial_A f^{AB}
 &=\partial_A\!\left(h^A{}_C F_{\rm p}^{CB}
 +h^B{}_C F_{\rm p}^{AC}-\frac12 h^C{}_C F_{\rm p}^{AB}\right).
\end{aligned}
~~~

Čtecí pumpa s indexem $-m$ může převést metrický signál s indexem $m$ zpět na elektromagnetický signál s nulovou harmonickou. Rovnice čtení je vyjádřena pomocí intenzit polí. Pro pumpu a signál bez zdrojů se záznějová část tenzoru energie a hybnosti zachovává. U pumpy udržované zdroji by bylo potřeba do gravitačního zdroje zahrnout tenzor energie a hybnosti aparatury.

<!-- BILINGUAL-UNIT: compact-port.example -->

## Příklad rovinných vln se zachovaným tenzorem energie a hybnosti

Vezměme běžné souřadnice $(t,x_1,x_2,z)$ a zvolme oba Maxwellovy potenciály podél $x_1$. Pumpa postupuje ve směru $y$ a signál ve směru $z$. Obě hybnosti jsou v pěti rozměrech nulové co do normy a příčné vůči polarizaci. Reálné pole zapíšeme pomocí komplexních amplitud s jejich sdruženými protějšky. Člen na rozdílové frekvenci má kovektor $B$ a koeficient $b$.

~~~math
\begin{gathered}
P_A=(-\Omega,0,0,0,\Omega),\qquad K_A=(-\omega,0,0,\omega,0),
\qquad \Omega=R^{-1},\quad 0<\omega<\Omega,\\
A_1=a_{\rm p} e^{iP_A x^A-i\beta}+a_{\rm s} e^{iK_A x^A}+\mathrm{c.c.},
\qquad B_A=P_A-K_A,\qquad b=a_{\rm p}a_{\rm s}^*e^{-i\beta}.
\end{gathered}
~~~

Vypočtěme $F_{AB}=\partial_A A_B-\partial_B A_A$ a ponechme člen úměrný $e^{iB_A x^A}$. Kompaktní harmonická je jedna. Inverzní retardovaný propagátor je $D_n(\nu,k)=k^2+n^2/R^2-(\nu+i0)^2$. Poslední řádek používá sdruženou pumpu a dává zdroj čtení při sdružené hybnosti signálu.

~~~math
\begin{aligned}
B^A(T_{AC})_{\rm beat}&=0,\qquad
(T_{11}-T_{22})_{\rm beat}=-2\Omega\omega\, b,\\
D_1(\Omega-\omega,-\omega)
&=\omega^2+\Omega^2-(\Omega-\omega+i0)^2=2\Omega\omega,\qquad
h_{+,1}^{\rm beat}=-b/M_5^3,\\
(\partial_A f^{A1})_{-K}
&=\Omega\omega\, h_{+,1}^{\rm beat}\,a_{\rm p}^*e^{i\beta}.
\end{aligned}
~~~

Jde o nenulovou vynucenou metrickou odezvu s recipročním čtením. Pro tyto dvě rovinné vlny leží zázněj mimo hmotovou slupku volného gravitonu, protože $B^2=2\Omega\omega>0$. Výpočet tedy dává vazbu a zdroj čtení. Pro výpočet vyzářené vlny či ozvěny je nutné určit konečnou aparaturu, přenos hybnosti a rozptyl na cíli.

[Výpočet pro konečnou vrstvu](compact-radiation-layer.cs.md) odvozuje odchozí mód a jeho čtení za předpokladu, že statické prostředí dodá chybějící podélnou hybnost.

<!-- BILINGUAL-UNIT: compact-port.radar -->

## Co radaru dodávají dva profily pumpy

Buzení a čtení na stejné kompaktní harmonické dávají na posuvně invariantním pozadí diagonální odezvu cíle bez závislosti na úhlu. Budicí pumpa vybírající $n$ a kalibrovaná čtecí pumpa vybírající $m$ dovolují smíšenou odezvu s fází $e^{-i(m-n)\alpha}$ pro cíl posunutý o úhel $\alpha$. Rozdíl jedna dovoluje měření fáze po celém obvodu, pokud je přechod na cíli nenulový a kalibrovaný. Příprava a čtení nezávislých kompaktních profilů pumpy jsou fyzikální požadavky nad rámec běžného elektromagnetického portu s nulovou harmonickou. Také ztotožnění $y$ s imaginární částí času UBT by vyžadovalo vlastní fyzikální odvození.

<!-- BILINGUAL-UNIT: compact-port.check -->

Příkaz `python scripts/verify_source_matching.py --check` symbolickým maticovým výpočtem ověřuje záznějovou část tenzoru energie a hybnosti, její zachování, inverzní propagátor na zázněji i reciproční Maxwellův zdroj. Srovnání nedává experimentální parametry. LEAN-PENDING.

[Linearizované Einsteinovy rovnice a příčná bezestopá polarizace](https://arxiv.org/abs/gr-qc/9712019) · [Míchání polí v Einsteinově–Maxwellově teorii](https://doi.org/10.1140/epjc/s10052-023-12304-1)
