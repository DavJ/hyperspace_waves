'''
Created on Aug 27, 2016
@author: david jaros

Historical plotting example, made Python-3 compatible. No propagation claim.
'''
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import matplotlib.pyplot as plt
    import generator.generate as G
    import transform.transform as T
    plt.plot(G.generate_hyperspace_wave())
    plt.ylabel('damped waveform sample')
    plt.show()
    print(T.mfft([1, 2, 3, 4]))


if __name__ == '__main__':
    main()
