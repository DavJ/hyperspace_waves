# Elektromagnetická vazba na poruchu pole Θ

**2026-11-04 · pracovní výzkum.** Odvození a překlad s pomocí AI. Úplná lidská revize se nedeklaruje.

<!-- BILINGUAL-UNIT: em-coupling.action -->

Použijeme přirozené jednotky, signaturu \((-+++)\) a Maxwellův člen zapsaný v sektoru QED teorie UBT. Při variaci metriky držíme kovariantní dvojformu \(F_{\mu\nu}\) pevnou. Kanonická geometrická derivace definuje \(E_\mu=N_0^{-1/2}D_\mu\Theta\) s pevným \(N_0\). Bilineární párování používá komplexně lineární kvaternionovou operaci sharp. Kanonická metrika a její přípustné variace jsou reálné a centrální. Odvození pracuje s celým komplexním biquaternionovým nosičem pod touto podmínkou reality. Registr akce UBT vede úplnou akci ve stavu \(\texttt{DEFINED\_FAMILY\_NOT\_FINALIZED}\).

~~~math
S_{\rm EM}=-\frac14\int d^4x\,\sqrt{-g}\,F_{\mu\nu}F^{\mu\nu},
\qquad b_\sharp(X,Y)\mathbf1=\frac{X^\sharp Y+Y^\sharp X}{2},
\qquad g_{\mu\nu}=b_\sharp(E_\mu,E_\nu).
~~~

[Kanonická geometrie](https://github.com/UBT-Institute/unified-biquaternion-theory/blob/ef3e52c3438012acf9010bc43afbe0802f4c4f2d/canonical/CANONICAL_DEFINITIONS.md#L167) · [Člen QED v zakřiveném prostoru](https://github.com/UBT-Institute/unified-biquaternion-theory/blob/ef3e52c3438012acf9010bc43afbe0802f4c4f2d/canonical/interactions/qed.tex#L259) · [Registr akce](https://github.com/UBT-Institute/unified-biquaternion-theory/blob/ef3e52c3438012acf9010bc43afbe0802f4c4f2d/canonical/ACTION.en.md)

<!-- BILINGUAL-UNIT: em-coupling.variation -->

Pišme \(h_{\mu\nu}=\delta g_{\mu\nu}\). Indexy zvedáme a stopy počítáme metrikou pozadí. Variace inverzní metriky a objemové míry dává tenzor energie a hybnosti a jeho vazbu na tetrádový jet. Variace \(D_\nu\Theta\) zahrnuje i změnu geometrické konexe. Pro redukovaný mód musí variace respektovat svázané rovnice.

~~~math
\begin{aligned}
\delta g^{\mu\nu}&=-h^{\mu\nu},&
\delta\sqrt{-g}&=\tfrac12\sqrt{-\bar g}\,h^\rho{}_\rho,\\
T_{\rm EM}^{\mu\nu}&=F^\mu{}_\rho F^{\nu\rho}-\tfrac14\bar g^{\mu\nu}F^2,&
h_{\mu\nu}&=b_\sharp(\delta E_\mu,\bar E_\nu)+b_\sharp(\bar E_\mu,\delta E_\nu),\\
\delta S_{\rm EM}&=\tfrac12\int d^4x\,\sqrt{-\bar g}\,T_{\rm EM}^{\mu\nu}h_{\mu\nu}
=\int d^4x\,\sqrt{-\bar g}\,T_{\rm EM}^{\mu\nu}b_\sharp(\bar E_\mu,\delta E_\nu),&
\delta E_\nu&=N_0^{-1/2}\delta(D_\nu\Theta).
\end{aligned}
~~~

Má-li kanonicky normovaná fyzikální souřadnice \(u\) lokální algebraickou polarizaci metriky \(h_{\mu\nu}=C_{\mu\nu}u\), je jejím geometrickým zdrojem \(J_u=C_{\mu\nu}T_{\rm EM}^{\mu\nu}/2\). Pro diferenciální polarizaci působí na \(T_{\rm EM}/2\) adjungovaný operátor linearizované mapy metriky s odpovídající mírou a okrajovými podmínkami. Magnetické pole ve směru \(z\) dává v lokální inerciální soustavě \(J_u=B_z^2(C_{00}+C_{xx}+C_{yy}-C_{zz})/4\). Konformní polarizace dávají nulu, protože \(T^\mu{}_\mu=0\) ve čtyřrozměrné klasické Maxwellově teorii.

[Kompaktní Einsteinův–Maxwellův model](compact-einstein-maxwell-port.cs.md) odvozuje harmonické buzení a reciproční čtení za předpokladu fyzikální kompaktní pumpy.

<!-- BILINGUAL-UNIT: em-coupling.flat -->

Pro kontrolní výpočet zvolíme plochou afinní tetrádu a zafixujeme její geometrickou konexi na nule. V konstantní Lorentzově tetrádové bázi zapíšeme \(\delta\Theta/\sqrt{N_0}=\xi^a b_a\) s komplexními koeficienty. Na přípustných perturbacích je symetrický gradient reálný. Jeho imaginární část vymizí díky podmínce reality metriky. Integrujeme per partes pro variace s kompaktní podporou nebo při vymizení okrajových členů. Volné Maxwellovo pole splňující rovnice pohybu má zachovaný tenzor energie a hybnosti, a proto v této kontrole geometrický zdroj vymizí. Výpočet aparatury v oblastech s proudem vyžaduje zahrnout tenzory pole i zdroje.

~~~math
h_{\mu\nu}=2\partial_{(\mu}\operatorname{Re}\xi_{\nu)},\qquad
\delta S_{\rm EM}=-\int d^4x\,(\operatorname{Re}\xi_\nu)\,\partial_\mu T_{\rm EM}^{\mu\nu}=0,
\qquad\partial_\mu T_{\rm EM}^{\mu\nu}=0.
~~~

V posledním integrálu používáme reálnou část \(\xi\), která vytváří reálnou změnu metriky. Redukovaná vazba v dynamické geometrii vyžaduje vlastní výpočet konexe a vazeb. Pokud bychom deklarované levé elektromagnetické fázové působení přiřadili témuž kovariantnímu tetrádovému jetu, z komplexní linearity sharp by plynulo \(g_{\mu\nu}\mapsto e^{2i\alpha}g_{\mu\nu}\). Slučitelnost fyzikálního působení náboje s reálnou metrikou proto tvoří samostatnou podmínku.

<!-- BILINGUAL-UNIT: em-coupling.loop -->

Skalární zdroj a odečet ve schématu aparatury vycházejí z dříve spočítaného nízkoenergetického modelu. Předpokládáme fyzikální neutrální mód \(u\), těžký komplexní skalár s kladnou kanonickou normou, elektrickým nábojem \(eq\) a \(M^2(u)=M_0^2+\kappa u+\cdots\), slabé elektromagnetické pole s malými hybnostmi a nulový nezávislý holý koeficient \(uF^2\). V eukleidovském tepelném jádru má křivost konexe tvar \(ieqF_E\) a její kvadratický člen je \(s^2\Omega^2/12\). Determinant komplexního skaláru a odečtení při \(M_0\) dávají následující Minkowského člen.

~~~math
\Delta\mathcal L=\frac{e^2q^2}{192\pi^2}\log\!\frac{M^2(u)}{M_0^2}\,F^2,
\qquad c=\frac{e^2q^2}{192\pi^2}\frac{\kappa}{M_0^2},
\qquad \mathcal L_{\rm EM,eff}=-\tfrac14(1-4cu)F^2+\cdots.
~~~

V tomto modelu na pevné metrice tedy \(J_u=cF_{\rm pump}^2\). Lineární odečet kolem určeného referenčního pole plyne z variace \((1-4cu)F^{\mu\nu}\) v Maxwellově rovnici. Obě odezvy určuje stejný koeficient \(c\). Pro UBT zbývá odvodit fyzikální reprezentaci náboje, stabilní mód \(u\), derivaci hmotnosti nabitého módu \(\kappa\) a překryvy zdroje a přijímače podél \(\psi\).

[Koeficient tepelného jádra, rovnice 4.28](https://arxiv.org/abs/hep-th/0306138) · [Podmíněná vazba a energetická bilance](source-detector-matching.md)

<!-- BILINGUAL-UNIT: em-coupling.apparatus -->

Ve skalárním modelu je buzení funkcí \(F^2=2(B^2-E^2)\). Jediný toroid s časově proměnným proudem je kandidátem na zdroj běžných polí v modelu. Dvojice s opačnými toky slouží jako volitelná srovnávací geometrie. Magnetostatický výpočet konečných jader dodává běžné \(A\), tok a kontroly symetrie. Pro výpočet buzených polí a běžného přenosu k sondě potřebujeme dynamický polní nebo obvodový model zvoleného uspořádání.

~~~mermaid
flowchart TD
    Drive["Modulovaný proud"] --> Coil{"Geometrie zdroje"}
    Coil --> Single["Jeden toroid"]
    Coil --> Pair["Toroidy s opačnými toky"]
    Single --> Fields["Fyzikální E a B"]
    Pair --> Fields
    Fields --> Direct["Běžná EM vazba"]
    Direct --> Probe["Sonda a referenční pole"]
    Fields -. "Překryv s módem otevřený" .-> Mode["Fyzikální mód Theta"]
    Mode -. "Přenos podél psi otevřený" .-> Readout["Podmíněná odezva módu"]
    Readout -. "Překryv s přijímačem otevřený" .-> Probe
~~~

[Výpočet konečných toroidálních jader](https://github.com/DavJ/magnetic-field-antihemholz/blob/71e3a754e86db4d4a268e66f16a94f1d05a67342/docs/ubt-wave-bridge.md) · [Samostatný model šíření na kružnici](compact-locality.cs.md)

<!-- BILINGUAL-UNIT: em-coupling.radar -->

## Podmíněná radarová ozvěna

Radar kromě vysílače a přijímače potřebuje také návrat od cíle. V samostatném modelu na reálné kružnici umístíme oba porty do \(y=0\). Cíl modelujeme stacionární kladnou bodovou tuhostí \(\lambda\delta_L(y-d)\), přičemž \(0<d\le L/2\) je délka kratšího oblouku. Kladná tuhost přidává nezápornou potenciální energii. Pro reálný Laplaceův parametr \(s>0\) je \(G_0\) inverzí \(s^2+\mu^2-v^2\partial_y^2\) na kružnici. Vyřešíme pole v cíli a dosadíme je zpět, čímž dostaneme rozdíl odezev portu s cílem a bez cíle.

~~~math
\Delta G(s;0,0)
=-\frac{\lambda\,G_0(s;0,d)G_0(s;d,0)}
 {1+\lambda G_0(s;d,d)}.
~~~

Dva propagátory dávají nejdřívější návrat \(2d/v\), podobně jako u běžného monostatického radaru. Jmenovatel sčítá opakované interakce na cíli. První Bornova aproximace jej nahrazuje jedničkou. Skutečná odezva portu by obsahovala oba fyzikální překryvy s prostředím. Na kružnici se pozdější ozvěny mohou vracet druhým obloukem. Je-li známo \(v\), první příchod určuje pouze kratší oblouk.

Pro \(\mu=0\) a \(0\le t<L/v\) mohou přispět jen nejkratší cesty. Do rozdílu rezolvent dosadíme transformaci na přímce \(G_0(s;0,d)=e^{-sd/v}/(2vs)\) a \(G_0(s;d,d)=1/(2vs)\). Rozklad na parciální zlomky a časový posun \(2d/v\) dávají následující přesný rozdíl impulsních odezev, kde \(r_+=\max(r,0)\). Odezva nabíhá postupně i při bodovém impulzu. Konečný vyslaný pulz se s touto odezvou konvoluuje.

~~~math
\Delta G_R(t;0,0)
=-\frac1{2v}\left(1-e^{-\lambda(t-2d/v)_+/(2v)}\right),
\qquad 0\le t<L/v.
~~~

Reprodukovatelný příklad má bezrozměrné hodnoty \(L=12\), \(v=1\), \(d=2\), \(\lambda=1/4\) a \(\mu=0\). Dává náběh v \(t=4\) a rozdíl impulsních odezev \(-0.1105996\) v \(t=6\). Jde o virtuální test odečtu ozvěny. Kalibrace aparatury může měřit běžný přímý přenos a návrat od známého cíle v obyčejném prostoru. Předpověď UBT vyžaduje fyzikální propagátor, obě vazby portů a rozptylový operátor cíle. K určení původu ozvěny, kterou kalibrace nevysvětlí, bychom potřebovali tyto předpovědi.

[Čas návratu u běžného radaru](https://www.ll.mit.edu/media/6941) · [Propagátor na kružnici a kauzální čelo](compact-locality.cs.md)

<!-- BILINGUAL-UNIT: em-coupling.radar-2d -->

## Radar na reálném toru

Uvažujme plochý pravoúhlý torus s \(N\) periodickými souřadnicemi o periodách \(L_i\) a známou společnou rychlostí vlny \(v\). Každý port vysílá a přijímá ve stejném bodě \(p\). Pro jeden nehybný cíl v bodě \(x\) určuje čas první ozvěny \(T(p)\) nejkratší dráhu tam a zpět, pokud je port na tuto dráhu navázán a první ozvěnu dokážeme rozlišit. Značme \(\delta_{L_i}\) nejkratší vzdálenost po kružnici v souřadnici \(i\).

~~~math
r_p^2=\left(\frac{vT(p)}{2}\right)^2
      =\sum_{i=1}^N \delta_{L_i}(x_i-p_i)^2,
\qquad \delta_L(z)=\min_{k\in\mathbb Z}|z+kL|.
~~~

Jeden port určuje množinu bodů ve stejné vzdálenosti. Referenční port umístíme do počátku a další dva na každé ose do bodů \(\pm a_i e_i\), kde \(a_i=L_i/4\). V rozdílech \(r_\pm^2-r_0^2\) se odečtou příspěvky všech souřadnic kromě \(i\). Pro danou osu položme \(u=x_i/a_i\) s reprezentantem v \([-2,2)\) a zaveďme \(P=(r_-^2-r_+^2)/a_i^2\) a \(Q=(r_+^2+r_-^2-2r_0^2)/a_i^2\). Vyhodnocením nejkratších vzdáleností dostaneme

~~~math
(P,Q)=
\begin{cases}
(4u,2), & |u|\le 1,\\
(\operatorname{sgn}(u)(8-4|u|),\,10-8|u|), & 1<|u|\le 2.
\end{cases}
~~~

Pro \(Q=2\) vyjde \(u=P/4\). Pro \(Q<2\) vyjde \(|u|=(10-Q)/8\) a znaménko určí \(P\). Při \(P=0,Q=-6\) obě znaménka označují stejný bod modulo periodu. V tomto modelu tedy uvedené rozmístění \(2N+1\) portů stačí. Předpokládá přesné časy, známé polohy portů a rychlost a správné přiřazení prvních ozvěn jedinému cíli. Ve dvou rozměrech s \(L=(12,8)\) a \(v=1\) se první ozvěna od cíle \((10,5)\) vrátí do referenčního portu v čase \(2\sqrt{13}\approx7.2111\). Z pěti časů společně získáme zpět \((10,5)\) modulo periody. Všechny veličiny v příkladu jsou bezrozměrné.

Pro dvourozměrnou bezhmotnou vlnovou rovnici sečteme retardovanou Greenovu funkci na rovině přes periodické obrazy. S časy průchodu \(a_\ell=|x-p_T+\ell L|/v\), \(b_k=|x-p_R+kL|/v\) a kladnou bodovou tuhostí \(\lambda\) dává rozptyl v prvním řádu následující rozdíl impulzních odezev. Symbol \(*\) označuje kauzální časovou konvoluci. Uvedený tvar \(g_a\) platí pro \(t>a\), před čelem je funkce nulová.

~~~math
g_a(t)=\frac{1}{2\pi v^2\sqrt{t^2-a^2}}\quad(t>a),
\qquad
\Delta G_R^{(1)}(t)=-\lambda\sum_{\ell,k\in\mathbb Z^2}(g_{a_\ell}*g_{b_k})(t).
~~~

Mimo čela přispívají jen dvojice s \(a_\ell+b_k<t\). Pro kladné časy obou úseků \(a,b\) položme \(D=t-a-b>0\). Substituce integrační proměnné \(s=a+D\sin^2\theta\) převede konvoluci na úplný eliptický integrál \(K\) s parametrem \(m\) (modulem \(\sqrt m\)).

~~~math
(g_a*g_b)(t)=\frac{2K(m)}{(2\pi v^2)^2\sqrt{(D+2a)(D+2b)}},
\qquad m=1-\frac{4ab}{(D+2a)(D+2b)}.
~~~

První ozvěna přijde v čase, který je součtem nejkratších časů obou úseků. Impulzní odezva doznívá za každým čelem a pozdější dráhy mohou oběhnout jednu či obě periodické osy. Opakovaný rozptyl na bodě ve dvou rozměrech vyžaduje model konečného cíle nebo regularizaci divergentní Greenovy funkce v témže bodě. Pro použití v UBT je třeba nezávisle určit zákon šíření, interakci s cílem a fyzikální vazby v polohách portů. Obě souřadnice jsou zde reálné souřadnice modelu. Pro určení polohy potřebujeme nezávislá umístění portů v těchto souřadnicích.

Příkaz `python scripts/verify_compact_locality.py --radar-example-2d` vypíše pět časů, získané souřadnice a vzorky rozdílu impulzních odezev v Bornově aproximaci. Testy také srovnávají konvoluci s přímou kvadraturou a pokrývají případ s více nejkratšími drahami.

[Retardované Greenovy funkce podle dimenze](https://arxiv.org/abs/1701.00776) · [Úplný eliptický integrál](https://dlmf.nist.gov/19.2.E8)

<!-- BILINGUAL-UNIT: em-coupling.radar-reference -->

## Reference fáze v kompaktní souřadnici pro radar

Nechť $\varphi$ je úhel s periodou $2\pi$ v podmíněném kompaktním modelu. Položíme $\varphi=\psi/R_\psi$, jen pokud toto ztotožnění má fyzikální význam. Pracujeme s prostorem poruch po redukci vazeb. Posun $(T_\alpha f)(\varphi)=f(\varphi-\alpha)$ působí na celý fyzikální prostor včetně vnitřních složek. Předpokládejme, že pozadí a operátor bez cíle $D_0$ jsou invariantní vůči posunu. Jeden cíl v bodě $\alpha$ má operátor $T_\alpha V_{\rm origin}T_\alpha^{-1}$. Rozdíl odezev $\Delta R_\alpha=D_\alpha^{-1}-D_0^{-1}$ používá retardované inverze.

~~~math
D_\alpha=D_0+T_\alpha V_{\rm origin}T_\alpha^{-1},
\qquad [D_0,T_\alpha]=0,\qquad
\Delta R_\alpha=T_\alpha\Delta R_{\rm origin}T_\alpha^{-1}.
~~~

Pro zdroj $s_k$ v harmonické $k$ a přijímač $d_\ell$ v harmonické $\ell$ použijeme $T_\alpha f_k=e^{-ik\alpha}f_k$. Veškerá závislost na poloze cíle je ve fázi dané rozdílem harmonických.

~~~math
\Delta H_{\ell k}(\alpha)
=d_\ell^\dagger\Delta R_\alpha s_k
=e^{-i(\ell-k)\alpha}\Delta H_{\ell k}(0).
\qquad \Delta H_{00}(\alpha)=\Delta H_{00}(0).
~~~

Dva rovnoměrné porty mohou přes $\Delta H_{00}$ zjistit přítomnost cíle, ale při každém jeho posunu dostanou stejnou ozvěnu. Identita zahrnuje opakovaný rozptyl i libovolné míchání vnitřních složek na cíli. Časově proměnná pumpa, která zachovává symetrii kompaktních posunů, zachová tuto kovarianci i pro dvoučasovou retardovanou odezvu. Silnější rezonance v takové sestavě může zvětšit naměřenou amplitudu při stejné nejednoznačnosti polohy.

Kalibrovaný nenulový kanál s $\ell-k=1$ dává úhel modulo $2\pi$ z komplexní fáze vůči známé odezvě v počátku. Při odděleně měřených a kalibrovaných nenulových rozdílech $q$ je společná zbývající perioda $2\pi/\gcd(|q|)$. Plyne to z rovnosti jejich fázových faktorů nebo z celočíselných kombinací fází. Tvrzení předpokládá kanály s měřením fáze. Při samotném kosinu zůstává nejednoznačnost zrcadlení.

### Kandidát z plné matice a zkouška kalibrační invariance

Ve výslovně předpokládaném členu $-ZF_{\mu\nu}F^{\mu\nu}/4$ zprůměrovaném přes kompaktní úhel položme $\Theta^\sharp=\operatorname{adj}\Theta$ v reprezentaci $\mathrm{Mat}_2(\mathbb C)$. Skalár $\mathcal Q$ je invariantní vůči Lorentzově kongruenci s determinantem jedna a vůči centrální fázi. Pro reálnou změnu měřítka plného pozadí určuje tatáž variace $Z$ buzení i čtení.

~~~math
\mathcal Q(\Theta)=\operatorname{Re}\operatorname{tr}(\Theta^\sharp\Theta^\dagger),
\qquad
Z=Z_0+c\,\mathcal Q(\Theta),\qquad
\left.\delta Z\right|_{\delta\Theta=u\bar\Theta}
=2c\,u\,\mathcal Q(\bar\Theta),\quad u\in\mathbb R.
~~~

Čisté centrální vinutí $e^{iN\varphi}\Theta_0$ dává konstantní $\mathcal Q$. Jiný případ poskytuje stávající rodina plného pole Theta pro vinutí jedna, vyhodnocená v prostoročasovém počátku a posunutá o fázi $\beta$. Přímý výpočet determinantu a stopy dává následující výrazy.

~~~math
\begin{gathered}
\Theta_\beta=\sigma_3+i\begin{pmatrix}0&r\\r&b\end{pmatrix},
\qquad r^2=1+\rho\cos(\varphi-\beta),\quad
b=\rho\sin(\varphi-\beta),\quad 0<\rho<1,\\
\det\Theta_\beta=\rho e^{i(\varphi-\beta)},
\qquad
\mathcal Q(\Theta_\beta)=-2-2r^2=-4-2\rho\cos(\varphi-\beta).
\end{gathered}
~~~

Stopa má první harmonickou s fází určenou $\beta$. Pokud by tato reference byla fyzikální a pevná a zobrazená změna měřítka byla fyzikální poruchou, její koeficient by v témže vazebním členu vážil rovnoměrnou i první harmonickou. Měření fáze vyžaduje kalibrovaný přístup k oběma kvadraturám. Polohu bychom určovali relativně k referenci.

Původní konstrukce s pomocným jetem ponechává tutéž tetrádu i spinovou konexi, zatímco se nosné pole v této rodině mění. Maxwellova vazba přes metriku proto vidí tutéž geometrii. Zobrazená veličina $\mathcal Q$ se při pevné geometrii mění, takže v čistě gravitačním modelu s volně variovanými pomocnými poli neprojde zkouškou invariance vůči posunu nosného pole. Přidání vazby na tuto veličinu by vyžadovalo novou analýzu vazeb spojené akce. Pro UBT tak zůstávají otevřené dvě fyzikální podmínky, kompaktní propagátor na redukovaných módech a pozorovatelná, stabilní fázová reference.

Příkaz `python scripts/verify_source_matching.py --radar-reference` zopakuje konečný maticový příklad se dvěma vnitřními složkami v každé harmonické. Přímé inverze dávají nenulovou rovnoměrnou ozvěnu nezávislou na poloze a obnoví úhly cíle z kalibrovaného návratu v první harmonické. Kontrola také symbolicky vyhodnotí determinant a stopu plného pole Theta.

[Vinutí plného pole Theta a kvocient pomocného jetu](https://github.com/UBT-Institute/unified-biquaternion-theory/blob/672f365ad39c46149ecbbbabe2e6178d6c89e054/research_tracks/action_selection/mm_full_theta_psi_winding_decoupling_2026_11.cs.md) · [Podmínka pro kompaktní akci](https://github.com/UBT-Institute/unified-biquaternion-theory/blob/672f365ad39c46149ecbbbabe2e6178d6c89e054/research_tracks/information_first/TORUS_THETA_CANONICAL_ACTION_BRIDGE_GATE.cs.md)

<!-- BILINGUAL-UNIT: em-coupling.line -->

Stolní analog používá přenosové vedení s přizpůsobeným koncem a odpojitelnou paralelní indukčností \(L_s\) ve vzdálenosti \(d\) od měřicího portu. V bezeztrátovém modelu jsou \(L'\) a \(C'\) indukčnost a kapacita na jednotku délky. Pro uzlový tok \(\Phi\) s \(V=\partial_t\Phi\) má vedení kinetickou hustotu energie \(C'\Phi_t^2/2\), gradientovou hustotu \(\Phi_y^2/(2L')\) a energie paralelní větve je \(\Phi(d)^2/(2L_s)\). Po vydělení rovnice pole \(C'\) dostaneme model bez hmotnosti s bodovou tuhostí. Odpovídající parametry a koeficient odrazu napětí v místě paralelní větve jsou

~~~math
v=(L'C')^{-1/2},\qquad Z_0=\sqrt{L'/C'},\qquad
\lambda=\frac1{C'L_s},\qquad
\Gamma(s)=-\frac{\lambda}{2vs+\lambda}=-\frac{Z_0}{2sL_s+Z_0}.
~~~

Na portu se při jedné dopadající vlně a přizpůsobeném zdroji projeví faktor \(e^{-2sd/v}\Gamma(s)\). Porovnáme jinak shodné průběhy s připojenou a odpojenou indukčností. Ideální srovnání kalibruje známé zpoždění a odraz cíle. Skutečné průběhy ovlivní konektory, parazitní prvky a zakončení zdroje. Zde máme výpočet. Nezávislá měření aparatury a parametry UBT zůstávají otevřené.

[Tok a napětí na přenosovém vedení, rovnice 5–7](https://arxiv.org/abs/2307.14756)

<!-- BILINGUAL-UNIT: em-coupling.verification -->

Spusť `python scripts/verify_source_matching.py --check` a `python scripts/verify_compact_locality.py --check`. Průběhy s cílem a bez cíle vypíše `python scripts/verify_compact_locality.py --radar-example`. Konečné kontroly CAS pokrývají deset nezávislých symetrických variací metriky, stopu, magnetické napětí podél osy, řetězové pravidlo pro komplexní sharp a fázovou váhu. Kontrola ozvěny integruje kauzální konvoluci na cíli a srovnává ji s uzavřeným vzorcem. Stávající skalární kontroly srovnávají koeficient s integrály přes vlastní čas a Feynmanův parametr. Vymizení souřadnicového příspěvku plyne analyticky ze zachování tenzoru a okrajové podmínky. LEAN-PENDING.
