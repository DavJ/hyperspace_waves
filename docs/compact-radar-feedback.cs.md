# Lokalizace v kompaktních módech s opakovanými odrazy

**2026-11-07 · pracovní výzkum.** Rozšiřujeme výpočet první ozvěny ve srovnávacím modelu s reálnou kružnicí. Stanici a jediný slapový oscilátor s konečným profilem oddělují kanály šířících se vln. Při každé metrické frekvenci předpokládáme lineární stanici s kalibrovanými mapami vysílání a příjmu. Zachováváme odezvu izolovaného cíle včetně příspěvku uzavřených kanálů k jeho vlastní energii. Výměnu mezi oběma rozptylovači vedou otevřené kanály. Fyzikální realizace stanice a kompaktní pumpy v UBT zůstává otevřená. Odvození a překlad s pomocí AI.

[Výpočet první ozvěny a model slapového cíle](compact-radar-echo.cs.md)

<!-- BILINGUAL-UNIT: compact-feedback.sum -->

## Přesný součet opakovaných návratů

Při pevné frekvenci $\nu$ přejdeme k amplitudám normalizovaným na tok vynásobením amplitudy pole v módu $j$ faktorem $\sqrt{k_j}$. Označme $s$ vnější vstup, $x$ kompaktní vlnu vyslanou ze stanice, $y$ vlnu, která se do ní vrací, a $e$ vnější výstup. Fáze šíření zahrneme do odrazu od cíle $R_f$. Mapy $A,B,S,D$ popisují vyslání, příjem, vnitřní odraz stanice a její přímou odezvu.

~~~math
x=As+Sy,\qquad y=R_f x,\qquad e=Ds+By.
~~~

Cíl má jeden vnitřní oscilátor, takže $R_f=\tau u\ell$ s $\tau=i\mathcal T$. Vektor $u$ je vzor odchozí vlny a $\ell$ je komplexně lineární funkcionál dopadající vlny. Zavedeme $c_j=p_j\widehat w(-k_j)/\sqrt{2k_j}$. Výraz pro $\mathcal R_{mn}$ z výpočtu první ozvěny dává následující rozklad.

~~~math
u_m=c_m e^{ik_m d-im\alpha},\qquad
\ell(x)=\sum_n \frac{p_{-n}\widehat w(-k_n)}{\sqrt{2k_n}}
              e^{ik_n d+in\alpha}x_n .
~~~

Každý odraz od cíle je násobkem $u$, proto zapíšeme $y=a u$ a vyřešíme jedinou skalární rovnici.

~~~math
a=\tau\ell(As)+\lambda a,\qquad \lambda=\tau\ell(Su),\qquad
e-e_0=\frac{\tau\ell(As)}{1-\lambda}\,Bu,\qquad e_0=Ds .
~~~

Faktor $1/(1-\lambda)$ je součet všech návratů $1+\lambda+\lambda^2+\cdots$ pro $|\lambda|<1$. Míchání módů ve stanici mění tento faktor a může tak změnit amplitudu i fázi ozvěny ve stejném módu. Výstupní vzor zůstává jednorozměrný. Jde o zvláštní případ skládání rozptylových matic pro odraz s hodností jedna. [Zobecněný hvězdový součin](https://arxiv.org/abs/math-ph/0008022)

<!-- BILINGUAL-UNIT: compact-feedback.passive -->

## Proč pasivní smyčka s měřitelnou ozvěnou konverguje

Podle optické identity pro otevřené kanály platí $\|u\|^2=\|\ell\|^2=\Gamma=\operatorname{Im}\Sigma$. Bereme $\Gamma>0$. Inverzní odezva cíle má imaginární část $-\Gamma$ a holá inverzní odezva je reálná. Platí tedy $\|R_f\|=|\tau|\Gamma\le1$. Když na pasivní stanici dopadá pouze vnitřní vlna se vzorem $u$, platí $\|Su\|^2+\|Bu\|^2\le\|u\|^2$. Cauchyho–Schwarzova nerovnost dává mez

~~~math
\eta_B=\frac{\|Bu\|^2}{\Gamma},\qquad
|\lambda|\le |\tau|\Gamma\sqrt{1-\eta_B}\le\sqrt{1-\eta_B}.
~~~

Při $Bu\ne0$ je mez ostře menší než jedna, i když stanice jiný vnitřní vzor odráží dokonale. Smyčka s jednotkovým faktorem při reálné frekvenci vyžaduje vzor, na který je přijímač slepý. Opakované odrazy mohou zvětšit i zmenšit amplitudu ozvěny vůči prvnímu návratu. Pro celkový výstup $e$ sečteme tokové nerovnosti stanice a cíle. Vnitřní členy $x$ a $y$ se vyruší a zůstane $\|e\|^2+\|t\|^2\le\|s\|^2$, kde $t$ je vlna prošlá za cíl. Pro bezeztrátové bloky se započtením celého výstupního toku platí rovnost.

<!-- BILINGUAL-UNIT: compact-feedback.range -->

## Co stále určuje polohu

Odečteme koherentní přímou odezvu $e_0$ pro tentýž vstup. Jestliže kalibrované přijímače vybírají módy $1$ a $2$ s komplexními zisky $B_1,B_2$, mají obě složky ozvěny stejný faktor zpětné vazby. Pro nenulové ozvěny a známé relativní překryvy profilu dostaneme

~~~math
Z(\nu)=\frac{(e-e_0)_2}{(e-e_0)_1}\frac{B_1c_1}{B_2c_2}
      =e^{i(k_2-k_1)d-i\alpha} .
~~~

Dvě různé frekvence nad $2/R$ dávají stejnou periodu nejednoznačnosti vzdálenosti a stejnou rekonstrukci úhlu jako u prvního návratu. Výpočet vyžaduje fázovou kalibraci při obou frekvencích, buzení s $\ell(As)\ne0$ a jediný vzor cíle. Koherentní pozadí ponechané v signálu nebo více nezávislých vzorů cíle obecně poměr změní.

<!-- BILINGUAL-UNIT: compact-feedback.check -->

## Numerická a formální kontrola

Příkaz `python scripts/verify_compact_echo_feedback.py` použije bezrozměrnou bezeztrátovou stanici s unitárním Fourierovým mícháním a diagonálním příjmem. Jeden vnitřní mód se nepřevádí na vnější výstup, proto má vnitřní odraz stanice normu jedna. Kód porovná skalární součet s přímým řešením maticové soustavy a zkontroluje bilanci toku cíle, stanice i celku. Při $\nu_a=2.3$ změní opakované odrazy amplitudu ozvěny na $1.24$ a $0.80$ násobek prvního návratu pro vlastní frekvence cíle $2.27$ a $2.30$. Druhá frekvence $\nu_b=2.8$ umožní v obou případech rekonstruovat $d=7.4$ a $\alpha=1.2$. Lean ověřuje rovnici zpětné vazby s hodností jedna, skalární mez odrazu cíle, konvergenci smyčky z normalizovaných tokových předpokladů a vyrušení vnitřních toků. [Důkazy v Leanu](../HyperspaceWaves/Echo.lean)
