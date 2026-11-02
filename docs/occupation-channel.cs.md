# Buzení párů s opačnými indexy a čtení jejich obsazenosti

<!-- BILINGUAL-UNIT: occupation.scope -->

**ODVOZENO_ZA_PŘEDPOKLADŮ · 2026-11-01 · pracovní výzkum.** Předchozí výběrové pravidlo zakazuje nenulovou koherentní střední hodnotu nenulové harmonické ψ při zdroji s nulovou harmonickou a invariantním stavu. Nezakazuje neutrální obsazenost páry. Zde počítáme jedno možné buzení a následné čtení v dosavadním srovnávacím modelu skalární QED. Výsledek používá známou tvorbu párů; nezavádí novou vlnu ani doložený kanál UBT. Nevybíráme počet fyzikálních složek Θ, spektrum hmotností, přiřazení nábojů, nové fundamentální pole ani sousední vesmír. Používáme přirozené jednotky a signaturu (+---).

<!-- BILINGUAL-UNIT: occupation.action -->

## Obě fáze vycházejí z jedné akce

Podmíněně předpokládejme, že redukce vazeb již dala kanonické komplexní módy φ_n s kladnou energií, hmotnostmi μ_n>0 a fyzikálními náboji g_n=e*q_n. Tuto redukci zde neodvozujeme. Obyčejný foton nezávislý na ψ se váže diagonálně v n. Antičástice daného módu mají vůči jeho částicím opačný elektromagnetický náboj i opačný zachovávaný index ψ. Index označuje Fourierovu translaci, nikoli adresu vesmíru. Pro číselné předpovědi UBT je nejprve nutné odvodit jednotlivé náboje a hmotnosti.

```math
\mathcal L=-\frac14F_{\mu\nu}F^{\mu\nu}
+\sum_n\left[(D_\mu\phi_n)^*D^\mu\phi_n-\mu_n^2|\phi_n|^2\right],
\qquad D_\mu=\partial_\mu+i g_n A_\mu.
\qquad \int_0^{2\pi R}\!d\psi\,e^{i(n-m)\psi/R}=2\pi R\,\delta_{nm}.
```

<!-- BILINGUAL-UNIT: occupation.selection -->

Buzení může v rámci jednoho složkového módu vytvořit pár částice a antičástice i pro nenulové n: jeho celkový zachovávaný index je nulový. Operátor tvorby páru komutuje s generátorem indexu a oba operátory počtu jsou invariantní. Obsazenost se tedy změní bez vzniku zakázané jednobodové harmonické. Nulová harmonická hustoty obsahuje všechny složkové indexy n. Vakuové korelace a korelace stlačeného párového stavu jsou jiné pozorovatelné veličiny než koherentní střední hodnota harmonické.

```math
Q_\psi=n(N_a-N_b),\qquad [Q_\psi,a_n^\dagger b_n^\dagger]=0,
\qquad Q_\psi|k,k\rangle=0,
\qquad \langle H_m\rangle_{A_0}=0\ (m\ne0),\quad
\langle N_a\rangle=\langle N_b\rangle=N_{\boldsymbol p}.
```

<!-- BILINGUAL-UNIT: occupation.pulse -->

## Fyzikální elektrický pulz, nikoli konstantní vektorový potenciál

Pro jeden určený mód vynechme n. V časové kalibraci zvolme předepsaný homogenní elektrický pulz ve směru z. Jde o místní idealizaci na jednotku prostorového objemu, nikoli o konečné Maxwellovo řešení s cívkami. A_z zde označuje složku fyzikálního trojvektoru; kovariantní složka je A_3=-A_z. Konstanta C je kalibrační posun. Současně s C je nutné přeznačit kanonickou hybnost p_z; kinetická hybnost p_z-g*A_z zůstává invariantní. Časově proměnné pole koná práci. Samotné konstantní A, které je čistou kalibrací, páry nevytváří.

```math
A_z(t)=C-E_0T\tanh(t/T),\qquad E_z(t)=E_0\operatorname{sech}^2(t/T),
\qquad \ddot u_{\boldsymbol p}+\Omega_{\boldsymbol p}(t)^2u_{\boldsymbol p}=0,
\quad \Omega_{\boldsymbol p}^2=\mu^2+p_\perp^2+[p_z-g A_z(t)]^2.
\qquad C\mapsto C+\Delta C,\quad p_z\mapsto p_z+g\Delta C.
```

<!-- BILINGUAL-UNIT: occupation.bogoliubov -->

Vstupní vakuum určíme módem s kladnou frekvencí a jednotkovým Wronskiánem. V pozdních časech rozdělíme tentýž vyvinutý mód na kladnou a zápornou výstupní frekvenci. Jejich směšování dává střední počet párů v daném páru hybnostních módů. N je střední hodnota, nikoli pravděpodobnost detekce páru ani početní hustota. Integrace N přes kanonické hybnosti s mírou d³p/(2*pi)³ dává hustotu párů. Výstupní kinetické hybnosti obou členů jsou opačné. Wronskián slouží jako numerická kontrola, nikdy jako dodatečná normalizace nepřesné integrace.

```math
p_0=p_z-gC,\quad
\omega_{\rm in/out}=\sqrt{\mu^2+p_\perp^2+(p_0\mp gE_0T)^2},
\qquad u_{\rm in}\sim\frac{e^{-i\omega_{\rm in}t}}{\sqrt{2\omega_{\rm in}}},
\quad u\sim\frac{\alpha e^{-i\omega_{\rm out}t}+\beta e^{+i\omega_{\rm out}t}}{\sqrt{2\omega_{\rm out}}},
\qquad |\alpha|^2-|\beta|^2=1,\quad N_{\boldsymbol p}=|\beta|^2.
```

<!-- BILINGUAL-UNIT: occupation.hypergeometric -->

## Odvození a nezávislé přesné řešení

Položme x=t/T, y=(1+tanh(x))/2 a u=y^a*(1-y)^b*v(y). Transformovaná Kleinova–Gordonova rovnice je následující hypergeometrická rovnice. Analytické pokračování jejího vstupního řešení z y=0 do y=1 dává β. Poměr gama funkcí nezávisle porovnáváme s přímou integrací módu a hyperbolickým výrazem. Skalární výsledek má v čitateli znaménko plus; spinorový výraz se liší a zde jej nepoužíváme. Pro imaginární λ je kosinové pokračování po sečtení členů reálné a nezáporné.

```math
K=gE_0T^2,\quad a=-iT\omega_{\rm in}/2,\quad b=iT\omega_{\rm out}/2,
\quad\lambda=\sqrt{K^2-\tfrac14},\quad A,B=a+b+\tfrac12\pm i\lambda,\quad c=1+2a,
\qquad y(1-y)v''+[c-(A+B+1)y]v'-ABv=0,
\quad AB=(a+b)(a+b+1)+K^2,
\qquad N_{\boldsymbol p}=\frac{\omega_{\rm out}}{\omega_{\rm in}}
\left|\frac{\Gamma(c)\Gamma(2b)}{\Gamma(A)\Gamma(B)}\right|^2
=\frac{\cosh(2\pi\lambda)+\cosh[\pi T(\omega_{\rm out}-\omega_{\rm in})]}
{2\sinh(\pi T\omega_{\rm in})\sinh(\pi T\omega_{\rm out})}.
```

<!-- BILINGUAL-UNIT: occupation.limits -->

Několik limit odlišuje buzení párů od libovolně zadaného zesílení zdroje. Neutrální módy nebo nulové E_0 dávají přesně nulovou tvorbu. Při pevném nenulovém středovém podélném impulsu začíná tvorba pro slabé buzení členem E_0². Pro p_0=0 začíná až členem E_0⁴, protože tam vymizí porucha lineární ve vektorovém potenciálu; vypuštění členu A² by dalo nesprávný výsledek. Centrální mód v limitě dlouhého pulzu přechází k obvyklému skalárnímu Schwingerovu exponentu. Pro předepsané silné časově proměnné pole neexistuje univerzální ostrý jednofotonový práh: záleží na vícefotonových a tunelovacích procesech i na spektru pulzu. Výstupním částicím dodává energii buzení, nikoli přidané neutrální pole.

```math
\omega_0=\sqrt{\mu^2+p_\perp^2+p_0^2},\qquad
N_{\boldsymbol p}\sim\frac{\pi^2K^2p_0^2}{\omega_0^2\sinh^2(\pi T\omega_0)}\quad(p_0\ne0,\ E_0\to0),
\qquad N_{p_0=0}\sim\frac{\pi^2K^4}{\sinh^2(\pi T\sqrt{\mu^2+p_\perp^2})},
\qquad N_{p_0=0}\longrightarrow e^{-\pi(\mu^2+p_\perp^2)/|gE_0|}\quad(T\to\infty).
```

<!-- BILINGUAL-UNIT: occupation.statistics -->

## Informace je v náhodném počtu párů

Pro vakuový vstup je každý bosonový pár hybnostních módů dvoumódovým stlačeným stavem tvořeným stavy |k,k>. Počet párů má geometrické rozdělení, počty v obou větvích jsou stejné a jejich kovariance je kladná. Jedna větev s nezávislou účinností detekce η a bez temných impulzů má následující pravděpodobnost alespoň jednoho záchytu. Ani nenulové N nezaručuje jeden záchyt. Jde o ideální statistiku počítání, nikoli o rozpočet šumu přijímače, kapacitu kanálu nebo demonstraci komunikace. Blízko pulzu obsahuje stav také anomální koherenci úměrnou \(\alpha\beta^*\); popis pouze obsazeností zahazuje tuto fázovou informaci až po defázování nebo průměrování.

```math
P(k)=\frac{N^k}{(1+N)^{k+1}},\quad k=0,1,\ldots,
\quad \operatorname{Var}k=\operatorname{Cov}(N_a,N_b)=N(1+N),
\qquad G(z)=\sum_kP(k)z^k=\frac1{1+N(1-z)},
\quad P_{\rm click,one\ arm}=1-G(1-\eta)=\frac{\eta N}{1+\eta N}.
```

<!-- BILINGUAL-UNIT: occupation.readout -->

## Koeficient čtení plyne z energie nabitých částic

Po pulzu uvažujme defázované rozdělení výstupních částic f(k) pro jeden druh v kinetické hybnosti k; pro tento pulz jde o N vyhodnocené při p_0=k_z-gE_0T. Každý pár obsahuje i antičástici při -k se stejnou obsazeností. Slabou prostorově homogenní časově proměnnou sondu aproximujme bezsrážkovou kinetickou odezvou. Její frekvence musí být malá vůči energiím částic a párové mezeře a velká vůči relevantním srážkovým frekvencím. Počáteční obsazenosti držíme pevné, odečítáme proud pozadí a pracujeme před významným vyčerpáním nebo zpětným působením na pole, případně s vnějším proudem udržujícím pozadí. Tato aproximace nepopisuje koherentní čtení blízko pulzu ani úplnou kvantovou polarizaci pro libovolnou frekvenci a vlnový vektor.

```math
E_{\boldsymbol k}=\sqrt{\mu^2+\boldsymbol k^2},\qquad
\frac{\partial^2E(\boldsymbol k-g\delta\boldsymbol A)}{\partial A_i\partial A_j}\bigg|_0
=g^2\left(\frac{\delta_{ij}}{E_{\boldsymbol k}}-\frac{k_i k_j}{E_{\boldsymbol k}^3}\right),
\qquad D_{ij}=2g^2\int\frac{d^3k}{(2\pi)^3}f(\boldsymbol k)
\left(\frac{\delta_{ij}}{E_{\boldsymbol k}}-\frac{k_i k_j}{E_{\boldsymbol k}^3}\right),
\quad \delta j_i=-D_{ij}\delta A_j,\qquad
\epsilon_{\rm out}=2\int\frac{d^3k}{(2\pi)^3}E_{\boldsymbol k}f(\boldsymbol k).
```

<!-- BILINGUAL-UNIT: occupation.medium -->

Oba náboje přispívají s g², takže se jejich odezva sčítá i při nulovém celkovém elektrickém náboji. Faktor dva počítá oba členy páru, nikoli dvě předpokládané složky Θ. D je pro nenulovou kladnou populaci a μ>0 kladné a pro prázdnou nebo neutrální populaci vymizí. Při malých kinetických hybnostech přechází na 2*g²*n_pair/μ krát jednotkovou matici. Při Fourierově konvenci exp(-i*ω*t) je ideální bezsrážková vodivost i*D/(ω+i0). Jde o odezvu prostředí závislou na stavu a počáteční podmínce, nikoli o vakuovou Procovu hmotnost. Současný konstantní kalibrační posun a přeznačení kanonické hybnosti nemění žádnou pozorovatelnou veličinu; fyzikální sondou je změna A vůči jeho počáteční hodnotě s E=-dA/dt. Samotné rozdělení může mít proud pozadí i při nulovém náboji.

<!-- BILINGUAL-UNIT: occupation.coil -->

## Souvislost s antihelmholtzovým vektorovým potenciálem

Srovnávací model toroidálních cívek poskytuje magnetostatické A v Coulombově kalibraci, toky a obepínající cirkulaci. Tyto veličiny neurčují zde použitý elektrický budicí pulz. Lokálně nevířivé statické A mimo jádro může ovlivnit obepínající kvantovou fázi prostřednictvím holonomie, ale samo nevytváří zde počítané páry. Časová změna toku vytváří Faradayovo pole vázané na skutečnou geometrii, vratné proudy a Maxwellovy okrajové podmínky. Opačné toky mohou části tohoto pole rušit. Před nahrazením ideálního Sauterova pulzu aparaturou je nutné spočítat průběh buzení cívky, jeho nenulové místní E a překryv s fyzikálními nabitými módy. Samotný gradient A ani div(A) nejsou chybějící vazbou.

<!-- BILINGUAL-UNIT: occupation.segment -->

## Co výpočet umožňuje a co zůstává otevřené

Nyní máme podmíněný výpočet buzení → obsazenost páry → elektromagnetické čtení z jedné srovnávací akce. Přechází od lineární vakuové odezvy k vybuzenému stavu s dodanou energií; neodstraňuje dříve vypočtený útlum ve vakuu. Buzení i čtení s nulovou harmonickou jsou v ψ rovnoměrné. Fourierovy módy jsou nelokalizované a místní střední hustota invariantního stavu zůstává translačně invariantní. Změna N_n tedy nezajišťuje adresovaný vysílač v jednom segmentu ψ a přijímač v dalším. Takový kanál stále vyžaduje fyzikální spektrum, lokalizované profily zdroje a čtení, šíření či transport mezi nimi a nenulový měřený přenos odlišený od obyčejných elektromagnetických jevů. Nevyplývá nový fyzikální jev nad rámec určeného srovnávacího modelu, rychlejší čelo signálu, výkon cívky ani multivesmírný dosah.

<!-- BILINGUAL-UNIT: occupation.verification -->

## Kontroly a reprodukce

Následný [audit lokalizace](compact-locality.cs.md) kvantifikuje potřebnou koherenci mezi harmonickými a ověřuje samostatný kauzální srovnávací model na reálné kružnici. Neodvozuje transport těchto vytvořených párů ani neztotožňuje reálnou kružnici s imaginárním časem.

Kód nezávisle integruje Kleinovu–Gordonovu rovnici a porovnává její Wronskián s hyperbolickým spektrem a spojením pomocí gama funkcí. CAS kontroluje transformovanou diferenciální rovnici, Hessovu matici energie a vytvořující funkci počtů. Testy zahrnují obě větve λ, jeho nulu, mocniny slabého buzení, chování dlouhého pulzu, přeznačení při konstantní kalibraci, kladnost a rotace čtení, konečné diference energie i přímé součty počtů. Příklady jsou bezrozměrné jednotlivé módy nebo jedna buňka fázového prostoru na jednotku váhy; neuvádíme úplnou integrovanou hustotu ani parametry aparatury. Odvození a překlad vznikly s pomocí AI; zdrojem je angličtina. Netvrdíme úplnou lidskou revizi. `LEAN-PENDING`: lean/lake nejsou dostupné; netvrdíme přeložený formální důkaz ani experiment.

```sh
python -m pytest -q
python scripts/verify_occupation_channel.py --check
```

<!-- BILINGUAL-UNIT: occupation.sources -->

- [Kim, Lee a Yoon, arXiv:0807.2696, skalární rovnice 53–68 a konstrukce stlačeného stavu](https://arxiv.org/abs/0807.2696). Naše konvence konstantního posunu a kanonické hybnosti se liší od článku; frekvence a výsledné spektrum jsou výše výslovně přiřazené.
- [Předchozí složená odezva a výběrové pravidlo](composite-response.cs.md).
- [Společný srovnávací model zdroje a čtení](source-detector-matching.md).
- [Antihelmholtzový doprovodný projekt](https://github.com/DavJ/magnetic-field-antihemholz).
- [Výpočetní záznam](results/occupation-channel.json); [provenience výzkumu](research-origin-2026-11-01-occupation-channel.json).
