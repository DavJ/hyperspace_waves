# hyperspace_waves
Author: David Jaros

This repository contains functions for generation and detection of hyperspace waves. It is assumed that these waves, unlike regular EM waves, are able to penetrate any matter, even metal or a Farraday cage. As they propagate well in frames of reference moving by faster then light velocities, they are supposed to be possible  means of faster then light communication, and they could be possibly exploited as means of back in the time communication.

Especially balanced versions of hyperspace waves are designed to propagate both in space and hyperspace in exactly same way. Should there be any reflection of signal in hyperspace, or if the signal is just received in a distant point, both situations should lead to reception of signal before it was in deed transmitted.

## Connection to Unified Biquaternion Theory (UBT)

This repository now includes a comprehensive mathematical framework connecting hyperspace waves to the **Unified Biquaternion Theory (UBT)**: https://github.com/DavJ/unified-biquaternion-theory

### Key Features:

1. **Biquaternion Representation**: Hyperspace waves are represented as biquaternion-valued fields `Ψ_BQ = Ψ₀ + Ψ₁i + Ψ₂j + Ψ₃k`

2. **Curved Spacetime Generalization**: Wave equations generalized to curved spacetime with complex metrics

3. **Theta Function Connection**: Hyperspace waves connected to biquaternionized Jacobi theta functions, providing:
   - Natural quantization conditions
   - Modular symmetry properties
   - Connection to topological phases

4. **Modified Dispersion Relations**: For balanced waves with s₁ = -1/√2:
   - `ω²/2 = k² + m²` (allowing superluminal group velocities)
   - Complex wave vectors enabling hyperspace propagation

### Documentation:

- **[UBT_ANALYSIS.md](UBT_ANALYSIS.md)**: Complete mathematical derivation of the biquaternion formalism, curved space generalization, and theta function connections

### Code Structure:

- `workspace-pydev/hyperspace_comm/generator/`: Original hyperspace wave generation
- `workspace-pydev/hyperspace_comm/transform/`: Modified FFT for hyperspace wave detection
- `workspace-pydev/hyperspace_comm/biquaternion/`: **NEW** - Biquaternion implementation
  - `biquaternion.py`: Biquaternion arithmetic and operations
  - `wave_bq.py`: Hyperspace waves in biquaternion form
  - `theta_functions.py`: Biquaternionized Jacobi theta functions
- `workspace-pydev/hyperspace_comm/test/`: Test suites including UBT functionality tests
- `workspace-pydev/hyperspace_comm/demo_ubt.py`: **NEW** - Comprehensive demonstration of UBT connections

### Running the UBT Demonstration:

```bash
cd workspace-pydev/hyperspace_comm
python3 demo_ubt.py
```

This generates visualizations of:
- Biquaternion hyperspace wave components
- Polarized wave structures
- Theta function expansions
- Wave superposition
- Modified dispersion relations

### Testing:

```bash
cd workspace-pydev/hyperspace_comm
python3 test/test_biquaternion.py
```

For more information about hyperspace waves and related "Theory of everything" visit site www.octonion-multiverse.com


Copyright notice: files in this repository can be used based on GNU public license 

Moreover, whenever used for a derivative work (e.g. in science, commercially or even non-commercially) citation of the author is required.
