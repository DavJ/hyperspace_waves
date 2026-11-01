# Vypočtená odezva neutrální složené veličiny a dosah běžné fotonové dvojice

<!-- BILINGUAL-UNIT: composite-response.scope -->

## Co se počítá

**DERIVED_WITH_ASSUMPTIONS · 2026-11-01 · pracovní výzkum.** Počítáme odezvu neutrální složené veličiny, kterou audit neutrality ponechal otevřenou, nejprve jako obecnou Gaussovu kontrakci a potom pro srovnávací model jednoho komplexního fyzikálního módu s kladnou energií. Tím nedefinujeme celou Thetu jako skalár, neurčujeme počet jejích složek, nenahrazujeme tetrádovou metriku ani nepřidáváme fundamentální pole. Poloměr r je běžná prostorová vzdálenost v lokálním vakuovém výpočtu v 3+1 rozměrech; nejde o vzdálenost mezi segmenty psi. Používáme přirozené jednotky a signaturu (+---). Omezený fyzikální propagátor UBT i jeho překryvy se zdrojem a odečtem zůstávají otevřené.

<!-- BILINGUAL-UNIT: composite-response.wick -->

## Gaussova identita nezávislá na dimenzi

Nechť z je libovolný konečný vektor reálných Gaussových souřadnic s nulovou střední hodnotou a B reálná symetrická matice; odečtěme nespojené členy. Wickova kontrakce dává následující výraz. Fourierova transformace z něj činí konvoluci hybností, nikoli druhou mocninu propagátoru na stejné frekvenci. Pro původní invariant H=Tr(Theta^sharp Theta†) je B jeho kvadratická forma ve zvolených souřadnicích. Každá fyzikální redukce musí být odvozena před dosazením propagátoru. Indefinitní B sama neznamená zápornou spektrální hustotu: pro kanonické kladné skalární módy jsou váhy druhými mocninami. Kladnost skutečného omezeného Hilbertova prostoru zůstává předpokladem, nikoli výsledkem algebry.

```math
H=z^TBz,\quad G(x)=\langle z(x)z(0)^T\rangle,\qquad
\langle H(x)H(0)\rangle_c=2\operatorname{Tr}[B G(x) B G(x)^T].
\quad \rho_H(s)=2\sum_{a,b}|B_{ab}|^2\rho_{\rm pair}(s;\mu_a,\mu_b)
.
```

Ve výpočtu s konstantním pozadím nebo zmrazenými koeficienty dává nenulové pozadí celé Thety pozorovatelné veličině také lineární projekci. Nulová střední hodnota souřadnic fluktuací nenuluje toto plné pozadí. Její spojená odezva zahrnuje původní propagátor promítnutý pomocí ell; případnou jednočásticovou váhu z tohoto členu výpočet dvojice nevylučuje. Pozadí i projekce stále musejí splnit podmínky kompatibility náboje a uzamčené metriky. Níže vypočtené vakuum skalární hustoty má ell=0.

```math
H=H_0+\ell^Tz+z^TBz,\quad\ell=2Bz_0,\qquad
\langle H(x)H(0)\rangle_c=\ell^TG(x)\ell+2\operatorname{Tr}[BG(x)BG(x)^T].
```

<!-- BILINGUAL-UNIT: composite-response.spectrum -->

## Odezva skalární dvojice s pevnou normalizací

Vezměme kanonický komplexní mód phi v jeho volném, fázově invariantním vakuu a hermitovskou hustotu O=phi†phi. Jde o příklad příspěvku po fyzikální redukci, nikoli o identifikaci H=O. Použijme vnější zdroj +J O a níže uvedenou retardovanou konvenci. Spojený korelátor má kontinuum dvojice; nemá delta-funkční jednočásticovou váhu. Práh stejných hmotností je dvojnásobkem hmotnosti složky. Polynomické odečty jsou lokální kontaktní členy a nemění jádra v oddělených bodech.

```math
\chi_O^R(x)=i\theta(t)\langle[O(x),O(0)]\rangle,\quad
\rho_{\rm pair}(s;a,b)=\frac{\sqrt{[s-(a+b)^2][s-(a-b)^2]}}{16\pi^2 s}\,
\mathbf1_{s\ge(a+b)^2},\qquad a,b>0.
\quad \rho_O(s)=\frac{1}{16\pi^2}\sqrt{1-\frac{4\mu^2}{s}}\,\mathbf1_{s\ge4\mu^2}.
```

<!-- BILINGUAL-UNIT: composite-response.subtraction -->

## Jednou odečtená susceptibilita místo přiřazeného pólu

Výsledek z Feynmanova parametru souhlasí s nezávislou spektrální disperzní reprezentací. Odečet fixuje neurčenou lokální konstantu bez zavedení rezonance. Na prahu dvojice je susceptibilita konečná, ačkoli její derivace je singulární; bod větvení není částicovým pólem. Pól vázaného stavu by vyžadoval interakce a fyzikální omezení ze zvolené akce.

```math
z=(\omega+i0)^2-\boldsymbol k^2,\qquad
\widehat\Pi(z)=\Pi(z)-\Pi(0)
=-\frac1{16\pi^2}\int_0^1dx\,\log\left(1-\frac{x(1-x)z}{\mu^2}\right)
=z\int_{4\mu^2}^{\infty}\frac{\rho_O(s)}{s(s-z)}\,ds.
\quad \widehat\Pi'(0)=\frac1{96\pi^2\mu^2},\qquad
\widehat\Pi(4\mu^2)=\frac1{8\pi^2}.
```

<!-- BILINGUAL-UNIT: composite-response.static -->

## Přesná statická odezva v oddělených bodech

Prostorová Fourierova transformace spektrální reprezentace dává superpozici Yukawových jader. Statický integrál má uzavřený Besselův tvar. Nezávislá kontrola integruje druhou mocninu eukleidovského masivního propagátoru přes eukleidovský čas. Nenulová odezva pod prahem dvojice je tedy přípustná, ale její masivní volný vakuový dosah je exponenciálně potlačený. Nehmotná volná limita je uvedena zvlášť; nelze ji dosadit do níže uvedeného masivního elektromagnetického přiřazení.

```math
\chi_O^R(\omega,r)=\frac1{4\pi r}\int_{4\mu^2}^{\infty}ds\,\rho_O(s)
 e^{-r\sqrt{s-(\omega+i0)^2}},\qquad r>0,
\quad \chi_O^R(0,r)=\frac{\mu K_1(2\mu r)}{16\pi^3r^2}
\sim\frac{\sqrt\mu}{32\pi^{5/2}r^{5/2}}e^{-2\mu r}.
\quad G_E(\tau,r)=\frac{\mu K_1(\mu\sqrt{\tau^2+r^2})}{4\pi^2\sqrt{\tau^2+r^2}},
\quad \chi_O^R(0,r)=\int_{-\infty}^{\infty}G_E(\tau,r)^2\,d\tau,
\quad \lim_{\mu\to0}\chi_O^R(0,r)=\frac1{32\pi^3r^3}.
```

<!-- BILINGUAL-UNIT: composite-response.threshold -->

## Ladění frekvence mění útlum s nestejnoměrnou limitou

Při pevné kladné kappě má rozvoj pro velké vzdálenosti exponenciálu násobenou mocninou. Zvyšování frekvence k prahu dvojice v tomto srovnávacím modelu kappu snižuje; snižování frekvence nenapodobuje lepší průnik nízkofrekvenčního rádia vodičem. Přesně na prahu se rozvoj mění na jiný mocninný zákon a odezva v oddělených bodech zůstává konečná. Nejde o izolovaný rezonanční mód. Podkladové retardované propagátory mají nosič ve světelném kuželu, takže jejich spektrální superpozice nedává rychlejší čelo signálu.

```math
\kappa=\sqrt{4\mu^2-\omega^2}>0,\qquad
\chi_O^R(\omega,r)\sim\frac{(2\kappa)^{3/2}}{256\pi^{5/2}\mu\,r^{5/2}}e^{-\kappa r}
\quad(r\to\infty;\,\kappa>0),
\qquad \chi_O^R(2\mu,r)\sim\frac1{32\pi^3\mu\,r^4}.
\quad \chi_O^R(t,\boldsymbol r)=0\quad(t<|\boldsymbol r|).
```

<!-- BILINGUAL-UNIT: composite-response.matching -->

## Elektromagnetická odezva hmotnostního zdroje ze stejného modelu

Nyní ponechme existující minimálně svázaný srovnávací model skalární QED a definujme renormalizovanou hustotu derivací podle vnějšího zdroje hmotnosti na druhou J při pevných ultrafialových vazbách. J je sonda, nikoli dynamický skalár. Předpokládejme nepřítomnost holého členu J F², slabá pole a malé hybnosti. Derivace již přiřazeného skalárního smyčkového logaritmu dává lokální odezvu hustoty. Její záporné znaménko v magnetickém poli nezávisle kontrolujeme faktorem vlastního času x/sinh(x). Tato smíšená derivace je lokální odezva, nikoli samostatný propagátor O. Nízkoenergetický koeficient nelze použít pro frekvence blízké dvojnásobku téže hmotnosti bez výpočtu úplného přiřazení závislého na hybnosti.

```math
\mathcal L=|D\phi|^2-(\mu^2-J)|\phi|^2-\tfrac14F_{\mu\nu}F^{\mu\nu},\quad O=\frac{\delta\Gamma}{\delta J},
\qquad \Delta\mathcal L=\frac{e^2q^2}{192\pi^2}\log\frac{\mu^2-J}{\mu^2}F^2,
\quad O_{\rm eff}\supset c_OF^2,\qquad c_O=-\frac{e^2q^2}{192\pi^2\mu^2}.
\quad \frac{x}{\sinh x}=1-\frac{x^2}{6}+O(x^4),\qquad |p|\ll\mu.
```

<!-- BILINGUAL-UNIT: composite-response.photons -->

## Úplný elektromagnetický srovnávací model není čistě exponenciální

Jakmile ponecháme elektromagnetické interakce, tatáž hustota může vytvořit dva běžné nehmotné fotony. Její vedoucí nízkoenergetická spektrální hustota proto začíná na nule, přestože dvojice volných masivních skalárů začíná na kladném prahu. Wickova kontrakce kanonického eukleidovského Maxwellova pole dává následující normalizaci a statický dosah. Tento příspěvek je řádu e⁴; při dostatečně velké vzdálenosti převáží nad exponenciálním volným příspěvkem hmoty. Jde o kontinuum fotonové dvojice bez nového jedno-fotonového nebo složeného pólu. Netvrdíme, že součet těchto vedoucích členů je úplně resumovanou, smyčkově opravenou vakuovou odezvou.

```math
\langle F^2(x_E)F^2(0)\rangle_c=\frac{48}{\pi^4|x_E|^8},\qquad
\rho_{F^2}(s)=\frac{s^2}{4\pi^2}\mathbf1_{s\ge0},\qquad
\rho_O^{\gamma\gamma}(s)\simeq c_O^2\frac{s^2}{4\pi^2}\quad(s\ll\mu^2),
\qquad \boxed{\chi_O^{\gamma\gamma}(0,r)\simeq\frac{15c_O^2}{\pi^3 r^7}}
\quad(\mu r\gg1).
```

<!-- BILINGUAL-UNIT: composite-response.fiber -->

## Proč mocninný dosah neidentifikuje sousední segment

Fotonový dosah je běžný elektromagnetický mechanismus ve fyzikálním prostoru. V pozadí invariantním vůči posunům po psi, které obsahuje pouze běžné fotony s nulovou harmonickou složkou, se váže na nulovou harmonickou složku složené veličiny. Dva nulové indexy nemohou dodat nenulový zachovávaný index vlákna. Nenulový profil zdroje a odečtu, jiné fotonové spektrum nebo pozadí porušující invarianci vůči posunům musejí být odvozeny samostatně. Tento dosah ani jádro masivní dvojice neidentifikují segment multiversa, napětí na cívce či dosah zařízení. Ani nedávný výsledek o nevybrání vinutí čistě gravitačním modelem MM sám nedodává chráněný kanál.

```math
n_{\gamma,1}=n_{\gamma,2}=0,\qquad
m=n_{\gamma,1}+n_{\gamma,2}=0
.
```

Při invariantní akci, míře a stavu a pouze zdrojích s nulovou harmonickou složkou vynutí symetrie vůči posunům nulovou jednobodovou střední hodnotu pro nenulový index ve všech řádech. Tato fáze je Fourierův posun, nikoli elektromagnetický náboj. Tvrzení nezakazuje vytváření dvojic s opačnými indexy ani modulaci neutrálních obsazeností a korelací. Zejména nulová harmonická složka hustoty obsahuje každý index složky n, nejen složku s nulovým indexem. Takový protokol obsazenosti či korelací vyžaduje vlastní vícebodovou odezvu, fyzikální spektrum a elektromagnetický odečet; samotné výběrové pravidlo jej neprokazuje.

```math
\psi\mapsto\psi+R\alpha:\quad H_m\mapsto e^{im\alpha}H_m,\quad
\langle H_m\rangle_{A_0}=e^{im\alpha}\langle H_m\rangle_{A_0}\ (\forall\alpha)
\Longrightarrow\langle H_m\rangle_{A_0}=0\ (m\ne0).
\quad H_0=L^{-1/2}\sum_n\operatorname{Tr}(\theta_n^\sharp\theta_n^\dagger),
\qquad \langle H_mH_{-m}\rangle_{A_0}\ 	ext{is not forced to vanish}.
```

<!-- BILINGUAL-UNIT: composite-response.verification -->

## Ověření a další fyzikální podmínka

Odvození a kód vznikly s pomocí AI. Výchozím jazykem překladu je angličtina; úplnou lidskou kontrolu netvrdíme. Nezávisle kontrolujeme maticovou kontrakci, spektrální disperzi, Besselův tvar, přímý eukleidovský časový integrál, normalizaci Maxwellova tenzoru, znaménko z magnetického vlastního času a přiřazení hmotnostního zdroje konečnou diferencí. Záznam JSON uvádí přesné předpoklady, verze nástrojů, chyby a bezrozměrné příklady. `LEAN-PENDING`: lean/lake nejsou dostupné; netvrdíme ověřený formální důkaz ani experimentální validaci. Dalším výpočtem UBT je fyzikální propagátor odvozený z akce a jeho projekce na složenou veličinu, poté elektromagnetický překryv rozlišený po vlákně. Nové póly vyžadují skutečné interakce, nikoli přiřazené volné neutrální pole.

<!-- BILINGUAL-UNIT: composite-response.sources -->

## Zdroje a reprodukce

- [D’Hoker, Quantum Field Theory, §§4.2, 6.2](https://www.pa.ucla.edu/faculty-websites/dhoker-lecture-notes/graduate-courses/qft-2016.pdf).
- [NIST DLMF 10.32.8](https://dlmf.nist.gov/10.32#E8).
- [Vassilevich, hep-th/0306138, §4](https://arxiv.org/abs/hep-th/0306138).
- [Scalar-loop source matching](source-detector-matching.md).
- [UBT PR #825](https://github.com/UBT-Institute/unified-biquaternion-theory/pull/825).
- [MM winding audit](https://github.com/UBT-Institute/unified-biquaternion-theory/blob/65231d02d1ceade5068ee61b12aaa168df151a9a/research_tracks/action_selection/mm_full_theta_psi_winding_decoupling_2026_11.en.md).
- [JSON](results/composite-response.json); [provenance](research-origin-2026-11-01-composite-response.json).

```sh
python -m pytest -q
python scripts/verify_composite_response.py --check
```
