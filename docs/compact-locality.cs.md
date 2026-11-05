# Lokalizace, koherence mezi harmonickými a podmíněný kauzální kanál na kružnici

<!-- BILINGUAL-UNIT: locality.scope -->

**ODVOZENO_ZA_PŘEDPOKLADŮ · 2026-11-02 · pracovní výzkum.** Uzavíráme matematickou mezeru po výpočtu obsazenosti páry: obsadit Fourierovy módy je něco jiného než lokalizovat zdroj, stav nebo přijímač. Nejprve dáváme přesnou mez lokalizace s konečným počtem módů, nezávislou na navrhované akci UBT. Potom počítáme šíření pouze ve výslovně předpokládaném srovnávacím modelu redukovaného módu na reálné kružnici. Tím neodvozujeme kompaktní kinetický člen UBT, transport dříve vytvořených skalárních párů ani nový pól složené hustoty. Reálnou souřadnici y tiše neztotožňujeme s imaginární částí ψ veličiny τ=t+i*ψ. Nevybíráme počet složek Θ ani nové fundamentální pole. S pomocí AI; netvrdíme úplnou lidskou revizi. `LEAN-PENDING`.

<!-- BILINGUAL-UNIT: locality.basis -->

## Fourierův index a poloha jsou různé báze

Nechť jediná normalizovaná komplexní amplituda používá určenou konečnou sadu harmonických na kružnici o obvodu L=2*pi*R. Níže uvedených 2*N+1 indexů počítá bázové funkce, nikoli složky Θ. Každá bázová funkce má všude konstantní intenzitu. Kladná norma L² je výslovná matematická volba pro redukovanou fyzikální amplitudu; není důkazem kladnosti indefinitního invariantu úplného Θ, Tr(Θ^sharp*Θ†). Fyzikální použití musí nejprve odvodit kladný prostor módů po redukci vazeb.

```math
\phi_n(y)=\frac{e^{2\pi iny/L}}{\sqrt L},\qquad
u(y)=\sum_{n=-N}^Na_n\phi_n(y),\qquad \sum_n|a_n|^2=1.
```

<!-- BILINGUAL-UNIT: locality.populations -->

Pro kladnou jednočásticovou matici ρ s jednotkovou stopou je místní intenzita rovnoměrná, kdykoli je ρ diagonální v harmonické bázi. Zvýšení celkové obsazenosti potom tuto intenzitu zvýší rovnoměrně. Obsazení mnoha nenulových n ji nelokalizuje. Nerovnoměrná jednočásticová hustota potřebuje mimodiagonální koherence; vícebodové párové korelace mohou stále obsahovat další informaci. Invariantní stav buzený pouze invariantním zdrojem zůstává invariantní, v souladu s předchozím výběrovým pravidlem ve všech řádech. Nezávislé řízení fází nebo lokalizovaná operace jsou další fyzikální možnosti, o nichž nebylo prokázáno, že je cívka podél ψ poskytuje.

```math
I_\rho(y)=\sum_{m,n}\rho_{nm}\phi_m(y)^*\phi_n(y),\qquad
\rho_{nm}=p_n\delta_{nm}\ \Longrightarrow\ I_\rho(y)=1/L.
```

Omezuje-li tyto koherence fyzikální superselekční pravidlo náboje, je nutné odvodit také kompenzující fyzikální referenci nebo rezervoár. Geometrická mez koncentrace potřebné operace neposkytuje.

<!-- BILINGUAL-UNIT: locality.concentration -->

## Kolik dostupných harmonických může lokalizovat amplitudu?

Pro interval o šířce w a středu y_0 přesně integrujeme součiny bázových funkcí. Hermitovská koncentrační matice C je kladná a shora omezená jednotkovou maticí. Její největší vlastní číslo je přesné maximum podílu, který může normalizovaná amplituda v tomto prostoru módů soustředit do intervalu. Stopa dává slabší nutnou mez. Optimum vyžaduje koherentní vlastní vektor, nikoli nekoherentní obsazenosti se stejnými diagonálními prvky. Zde sinc(x)=sin(pi*x)/(pi*x).

```math
C_{mn}=\int_{y_0-w/2}^{y_0+w/2}\phi_m^*(y)\phi_n(y)\,dy
=\frac wL e^{2\pi i(n-m)y_0/L}\operatorname{sinc}\!\left(\frac{(n-m)w}{L}\right),
\qquad P_w=a^\dagger Ca\le\lambda_{\max}(C)
\le\min\!\left(1,\operatorname{Tr}C\right)
=\min\!\left(1,(2N+1)\frac wL\right).
```

Například pro interval pokrývající 10 % určené kružnice dostáváme níže uvedená ověřená bezrozměrná maxima. Každá diagonální směs dává přesně 10 %, včetně směsí obsazujících všechny tyto módy. Jde o geometrickou koncentraci, nikoli o účinnost přenosu, zesílení detektoru nebo pravděpodobnost nalezení jiného vesmíru. Omezení dostupné sady módů může dostupný prostor stavů pouze zmenšit; symetrické odříznutí je výslovně zvolený příklad, nikoli odvozené spektrum UBT.

| Mez N | Počet harmonických | Maximální koherentní podíl | Diagonální směs |
|---:|---:|---:|---:|
| 0 | 1 | 0.100000 | 0.100000 |
| 2 | 5 | 0.468928 | 0.100000 |
| 4 | 9 | 0.736606 | 0.100000 |
| 6 | 13 | 0.888716 | 0.100000 |
| 8 | 17 | 0.958567 | 0.100000 |

<!-- BILINGUAL-UNIT: locality.action -->

## Šíření vyžaduje samostatný dynamický operátor

Pro ověření přínosu fyzikálně dostupné lokalizace ponechme jednu neutrální redukovanou souřadnici q z dřívějšího obecného srovnávacího modelu módů a výslovně postulujme následující kvadratickou akci. Toto q neztotožňujeme se složenou hustotou ani s úplným Θ. Reálná kružnice, kladný rychlostní parametr v, hmotnostní mezera μ a lineární zdroj jsou dodatečné předpoklady. Obyčejnou prostorovou závislost potlačujeme: jde o výpočet šíření v 1+1 rozměrech, nikoli o výpočet laboratorní vzdálenosti. Periodické okraje dávají kladnou energii a uvedenou bilanci práce zdroje. Záporný koeficient gradientu by v tomto nevázaném srovnávacím modelu naopak učinil dostatečně vysoké harmonické nestabilními; nelze jej použít jen k záchraně útlumu.

```math
S=\int dt\int_0^Ldy\left[\frac12(q_t^2-v^2q_y^2-\mu^2q^2)+Jq\right],
\qquad (\partial_t^2-v^2\partial_y^2+\mu^2)q=J,
\quad\omega_n^2=\mu^2+v^2(2\pi n/L)^2,
\qquad E=\frac12\int_0^L(q_t^2+v^2q_y^2+\mu^2q^2)dy,
\quad\dot E=\int_0^LJq_t\,dy.
```

<!-- BILINGUAL-UNIT: locality.profiles -->

Použijme profily zdroje a přijímače f_A,f_B s jednotkovým integrálem L¹: J(t,y)=J_A(t)*f_A(y), O_B=int(f_B*q)dy. Tím při změně šířky profilů zachováváme normalizaci; nejde o přizpůsobené zesílení aparatury ani energetický rozpočet zdroje. Pro rovnoměrné koeficienty se Fourierovy módy navzájem nekonvertují, přesto může Greenova funkce v polohovém prostoru spojovat různé polohy. Diagonální Fourierův operátor tedy neznamená nezávislé polohové segmenty. Pro obdélníkové profily o šířkách w_A,w_B a určených středech je přesnou odezvou následující nekonečný součet; implementace výslovně uvádí konečnou Galerkinovu mez. Obdélníkové profily jsou ideální vnější zdroje, nikoli vlnové stavy s konečnou gradientovou energií.

```math
H_{BA}(\omega)=\int dy\,dy' f_B(y)G_\omega(y-y')f_A(y')
=\frac1L\sum_{n\in\mathbb Z}
\frac{e^{2\pi in(y_B-y_A)/L}\operatorname{sinc}(nw_A/L)\operatorname{sinc}(nw_B/L)}
{\mu^2-(\omega+i0)^2+v^2(2\pi n/L)^2}.
```

<!-- BILINGUAL-UNIT: locality.zero-gradient -->

## Co se stane bez kompaktního gradientu?

Je-li v=0 a není přítomen jiný nelokální nebo směšovací člen, popisuje předpokládaná kružnice nezávislé místní oscilátory. Úplný nekonečný Fourierův součet je periodická delta funkce. Nepřekrývající se podpory zdroje a přijímače potom dávají přesně nulový přenos ve všech časech, nikoli exponenciálně slabý kanál. Pro překrývající se profily vzniká následující kontaktní člen. Jde o tvrzení o tomto srovnávacím modelu místních kopií, nikoli o důkaz nemožnosti pro všechny akce UBT. Konečná Fourierova projekce přitom nahradí delta zdroj nelokálním Dirichletovým jádrem a může předstírat nenulovou křížovou odezvu i tam, kde je přesná odezva oddělených oblastí nulová.

```math
v=0:\qquad G_\omega(y-y')=\frac{\delta_L(y-y')}{\mu^2-(\omega+i0)^2},
\qquad H_{BA}(\omega)=\frac{\int f_B(y)f_A(y)dy}{\mu^2-(\omega+i0)^2}.
```

<!-- BILINGUAL-UNIT: locality.green -->

## Přesná periodická odezva pod mezerou

Pro v>0, μ>0 a reálné |ω|<μ lze bodovou Greenovu funkci nezávisle získat řešením diferenciální rovnice se skokem derivace, součtem obrazů nebo součtem všech Fourierových módů. Označme d geodetickou vzdálenost v [0,L/2]. Obrazový tvar je numericky stabilní pro velké κ*L. Na vzdálenostech malých vůči velikosti kružnice zůstává obvyklá evanescentní exponenciála. Toto jádro patří výslovně postulovanému operátoru q; jeho dosazení místo dříve vypočtené složené bubliny by bylo neoprávněné.

```math
\Delta=\mu^2-\omega^2>0,\quad\kappa=\sqrt\Delta/v,
\qquad G_\omega(d)=\sum_{\ell\in\mathbb Z}\frac{e^{-\kappa|d+\ell L|}}{2v^2\kappa}
=\frac{\cosh[\kappa(L/2-d)]}{2v^2\kappa\sinh(\kappa L/2)},
\qquad G_\omega'(0^+)-G_\omega'(L^-)=-1/v^2,
\quad \int_0^LG_\omega(y)dy=1/\Delta.
```

Skok derivace je vztažen k periodické souřadnici y na [0,L], přičemž geodetickou vzdálenost chápeme jako funkci y.

<!-- BILINGUAL-UNIT: locality.uniform -->

Je-li kterýkoli konec podél y rovnoměrný, ponechá pouze n=0 a přenos ztratí veškerou závislost na vzájemné poloze. Takový zdroj již předpokládá řízení rozprostřené po kružnici. Nelze jej nazývat vysílačem omezeným na jeden segment. Vynásobení vypočteného zdroje nebo čtení obyčejným profilem cívky v x,y,z nedodává nezávislý kompaktní profil. Elektromagnetické překryvy a skutečné řízení v kompaktní souřadnici je nutné odvodit ze stejné fyzikální akce po redukci vazeb.

```math
f_A=1/L\ \text{or}\ f_B=1/L
\quad\Longrightarrow\quad H_{BA}=\frac1{L[\mu^2-(\omega+i0)^2]}.
```

<!-- BILINGUAL-UNIT: locality.resonance -->

## Ladění k nejnižší rezonanci zesiluje rovnoměrnou složku

Přiblížení ω k μ zdola zmenšuje inverzní délku útlumu, ale dominantní odezvu činí prostorově rovnoměrnou. Bodový poměr G(d)/G(0) jde k jedné. Po odečtení rovnoměrného nulového módu zbývá konečné jádro dané Bernoulliho polynomem. Diferenciální čtení zruší divergentní rovnoměrnou složku a zachová konečnou odezvu závislou na poloze; nezíská rezonanční zesílení nulového módu. Nejde o důkaz selhání veškerého adresování poblíž rezonance: stále záleží na době příchodu v konečném čase a na nenulových módech. Samotná netlumená rezonance potřebuje řešení počáteční úlohy, nikoli nekonečnou ustálenou amplitudu nebo vymyšlenou konstantu tlumení.

```math
G_\omega(d)=\frac1{L\Delta}
+\frac L{2v^2}\left[(d/L)^2-d/L+\frac16\right]+O(\Delta),
\qquad \lim_{\omega\to\mu^-}[G_\omega(0)-G_\omega(d)]
=\frac{d(L-d)}{2Lv^2},\qquad \lim_{\omega\to\mu^-}\frac{G_\omega(d)}{G_\omega(0)}=1.
```

<!-- BILINGUAL-UNIT: locality.modal-radar -->

## Vyšší rezonance zesiluje ozvěnu závislou na poloze

Do redukovaného modelu na reálné kružnici přidejme rovnoměrné tlumení $\gamma>0$. Nehybný cíl v bodě $x$ má kladnou bodovou tuhost $\lambda>0$. Energie $E$ obsahuje kladnou energii poruchy $\lambda q(x)^2/2$. Z bilance práce plyne, že energii dodává buzení $J$.

```math
[\partial_t^2+2\gamma\partial_t+\mu^2-v^2\partial_y^2+\lambda\delta_L(y-x)]q=J,
\qquad \dot E=\int_0^L J\,\partial_tq\,dy-2\gamma\int_0^L(\partial_tq)^2\,dy.
```

Předpokládejme čtyři fázově kalibrované profily pro buzení i čtení, normalizovaný kosinus a sinus pro každou ze sousedních harmonických $n,n+1$, kde $n\ge 1$. Profily jsou rozložené po kružnici a měříme je na stejné budicí frekvenci $\omega>0$ s konvencí $e^{-i\omega t}$. Faktor $\sqrt{2/L}$ každý profil normalizuje. Jejich hodnoty v místě cíle jsou složky vektoru $\sqrt{2/L}\,u_j$.

```math
\theta=2\pi x/L,\qquad u_j=(\cos j\theta,\sin j\theta)^\top,
\qquad \omega_j^2=\mu^2+(2\pi v j/L)^2,
\qquad \chi_j=(\omega_j^2-\omega^2-2i\gamma\omega)^{-1}.
```

Od přenosu s cílem odečteme přenos prázdné kružnice. Pro výpočet jeho bloku $B_{ab}$ velikosti $2\times2$ z budicí harmonické $b$ do snímané harmonické $a$ vyřešíme pole v bodě poruchy z rovnice $q(x)=q_0(x)-\lambda G_\omega(0)q(x)$ a rozptýlené pole promítneme na snímací profily. Greenova funkce v témže bodě zahrnuje všechny harmonické kružnice. Odmocninu volíme s kladnou reálnou částí.

```math
\kappa=\frac{\sqrt{\mu^2-\omega^2-2i\gamma\omega}}{v},\qquad
G_\omega(0)=\frac{\coth(\kappa L/2)}{2v^2\kappa},\qquad
B_{ab}=-\frac{2\lambda}{L}\frac{\chi_a\chi_b}{1+\lambda G_\omega(0)}u_a u_b^\top.
```

Blok $B_{nn}$ téže harmonické je úměrný $u_nu_n^\top$. Nese dvojnásobný úhel $2n\theta$ a dává stejnou odezvu pro polohy vzdálené o $L/(2n)$. Smíšený blok $C$ dává rozdíl úhlů $(n+1)\theta-n\theta$. Jeho stopa a antisymetrická část dávají kosinus a sinus se společným komplexním faktorem. Vydělením stopou $B_{nn}$ a korekcí poměru susceptibilit prázdné kružnice dostaneme souřadnici cíle modulo $L$. Poměr Frobeniových norem navíc udává relativní velikost smíšené ozvěny.

```math
e^{i\theta}
=\frac{\chi_n}{\chi_{n+1}}
 \frac{C_{cc}+C_{ss}+i(C_{sc}-C_{cs})}{\operatorname{tr}B_{nn}},
\qquad C=B_{n+1,n},\qquad
\frac{\lVert C\rVert_F}{\lVert B_{nn}\rVert_F}
=\left|\frac{\chi_{n+1}}{\chi_n}\right|.
```

Na vlastní frekvenci prázdné kružnice $\omega_n$ platí $|\chi_n|=1/(2\gamma\omega_n)$. Pro $|\lambda G_\omega(0)|\ll1$ a izolovanou rezonanční dvojici roste ozvěna v téže harmonické jako $\gamma^{-2}$ a smíšená ozvěna jako $\gamma^{-1}$. Při pevné tuhosti a dostatečně malém tlumení změní člen $G_\omega(0)\sim(2/L)\chi_n$ v přesném jmenovateli toto škálování. Norma ozvěny v téže harmonické roste jako $|\chi_n|$, zatímco norma smíšené ozvěny se blíží konečné hodnotě $|\chi_{n+1}|$. Nejsilnější ozvěna a ozvěna, která rozliší polohu, se tedy mohou velikostí podstatně lišit.

Pro $L=12$, $v=\mu=1$, $\gamma=0.015$, $\lambda=0.01$ a $x=9.3$ má blok $n=1$ na $\omega_1$ 18,4krát větší amplitudu než při frekvenci uprostřed mezi $\mu$ a $\omega_1$. Smíšený blok na $\omega_1$ má 4,11 % normy rezonančního bloku. Jeho fáze rozliší $x=9.3$ od $x=3.3$. Všechna čísla jsou bezrozměrná. Výpočet předpokládá ustálený režim, koherentní kalibrované profily a jediný bodový cíl. Pro UBT je třeba odvodit zákon kompaktního šíření a fyzikální překryvy s módy z bikvaternionové dynamiky po redukci vazeb.

Příklad vypíše `python scripts/verify_compact_locality.py --radar-resonance`. Testy také srovnávají uzavřený vzorec s přímým maticovým výpočtem ve stojatých vlnách.

<!-- BILINGUAL-UNIT: locality.front -->

## Nenulová ustálená odezva není okamžitá komunikace

Retardované obrazové řešení téhož operátoru má přesné čelo. V konečném čase přispívá pouze konečný počet obrazů. Bodová porucha nemůže dorazit k bodovému přijímači před nejkratší dobou letu d/v. Pro konečné profily je nejdříve přípustný čas vzdálenost jejich podpor dělená v. Rovnoměrný profil má podporu všude, a proto neposkytuje kladnou dolní mez cestovní vzdálenosti. Odříznutí Fourierova součtu může dát zdánlivý signál před čelem: takový zdroj má prostorové chvosty a již není skutečně místním impulzem. Kód zaznamenává výslovný protipříklad a nikdy odříznutou odezvu neoznačuje za rychlejší fyzikální kanál. Parametr v neztotožňujeme s laboratorní rychlostí podél imaginárního času.

```math
G_R(t,d)=\frac1{2v}\sum_{\ell\in\mathbb Z}
\theta\!\left(t-\frac{|d+\ell L|}{v}\right)
J_0\!\left(\mu\sqrt{t^2-\frac{(d+\ell L)^2}{v^2}}\right),
\qquad G_R(t,d)=0\quad(t<d/v).
\qquad \int_0^\infty e^{-st}G_R(t,d)dt
=G_{\Delta=\mu^2+s^2}(d),\quad s>0.
```

<!-- BILINGUAL-UNIT: locality.ubt -->

## Zbývající podmínka pro UBT je konkrétní

Současná upstream kontrola toroidální akce výslovně uvádí, že čtyřrozměrná kanonická rodina sama nedodává kompaktní míru, kladnou kompaktní metriku ani kinetický člen D_ψ přežívající redukci. Vícerozměrný kandidát je uveden, ale nebyl vybrán jako dokončená fundamentální akce. Výzkumná větev výběru větví také odlišuje nezávislou reálnou kružnici od analytického pokračování a od dekoherovaných makroskopických větví. Naše akce pro q žádné z těchto chybějících ztotožnění nedodává. Nemění konstrukci tetrády, pevnou definici metriky, počet složek Θ ani kanonickou oboustrannou derivaci. Pro fyzikální pokračování je nutné odvodit kompaktní operátor a mapy zdroje a čtení na nedegenerovaném pozadí splňujícím rovnice pohybu po redukci vazeb; poté ověřit, zda aparatura může vytvářet koherence mezi harmonickými nebo lokalizované vazby na pozorovatelné veličiny. Samotná geometrie cívky, prvočíselný index nebo odhadnuté R nestačí.

<!-- BILINGUAL-UNIT: locality.verification -->

## Reprodukce a omezení

Nezávislé kontroly zahrnují kvadraturu vlnové funkce na intervalu, přesné meze vlastních čísel a stopy, defázování, konvoluci zdroje a čtení, periodické obrazové a Fourierovy součty, diferenciální rovnici a skok, normalizaci nulového módu, odečtení rezonance a numerickou Laplaceovu transformaci kauzálního obrazového jádra s výslovnou mezí chvostu. Besselovu rovnici uvnitř kužele a Bernoulliho limitu kontroluje CAS. Jde o matematické kontroly srovnávacího modelu, nikoli o fyzikální validaci UBT nebo experimentální výsledek. Zdrojem překladu je angličtina; netvrdíme úplnou lidskou revizi. `LEAN-PENDING`: lean/lake nejsou dostupné; nemáme přeložený formální důkaz.

```sh
python -m pytest -q
python scripts/verify_compact_locality.py --check
```

<!-- BILINGUAL-UNIT: locality.sources -->

- [Předchozí výpočet obsazenosti](occupation-channel.cs.md); [srovnávací model sektorů](intersegment-channel.md); [složená odezva](composite-response.cs.md).
- [Kontrola toroidální akce UBT, pevný současný master](https://github.com/UBT-Institute/unified-biquaternion-theory/blob/68c73508e239bfbccb1d8bd9266fac0e76f1839e/research_tracks/information_first/TORUS_THETA_CANONICAL_ACTION_BRIDGE_GATE.cs.md).
- [Taxonomie větví UBT a omezení pokračování, pevný současný master](https://github.com/UBT-Institute/unified-biquaternion-theory/blob/68c73508e239bfbccb1d8bd9266fac0e76f1839e/research_tracks/complex_time_branch_selection/psi_branch_selection.en.md).
- [Dereziński a Sikorski, arXiv:2406.08327, §5.3](https://arxiv.org/abs/2406.08327): primární reference k retardovaným Greenovým funkcím Kleinovy–Gordonovy rovnice a jejich podpoře; periodizaci na kružnici a rychlostní normalizaci odvozujeme výše.
- [NIST DLMF 24.8.1](https://dlmf.nist.gov/24.8#E1): Bernoulliho Fourierova řada pro limitu po odečtení nulového módu.
- [Výpočetní záznam](results/compact-locality.json); [provenience výzkumu](research-origin-2026-11-02-compact-locality.json).
