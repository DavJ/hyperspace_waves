'''
Created on Aug 27, 2016

@author: david jaros
'''


#tutorial http://matplotlib.org/users/pyplot_tutorial.html
import matplotlib.pyplot as plt
import generator.generate as G

plt.plot(G.generate_hyperspace_wave())
plt.ylabel('hyperspace wave')
plt.show()
