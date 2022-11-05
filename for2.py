import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

FONT_SIZE=22


## population matrix for n_species, for n_timesteps

Total_time=1000
timestep = 1/10

n_species = 2


T = np.arange(0,Total_time, timestep)
n_timesteps = np.size(T)

x = np.zeros((n_timesteps,n_species))

dxdt = np.zeros((n_timesteps,n_species))
##
## initial conditions

#inherent growth rate r, and interaction alpha, independent of time

r = np.ones((1,n_species))
alpha = np.ones((n_species,n_species))
x[0,:] = np.ones((1,n_species))

r = np.array([[1,0.72,1.53, 1.27]])

alpha = np.array([[1, 1.09, 1.52, 0],
                 [0, 1, 0.44, 1.36],
                 [2.33, 0, 1, 0.47],
                 [1.21, 0.51, 0.35, 1]])

r = np.array([[1,0.5]])

alpha = np.array([[1, -0.2],
                 [0.2, 1]])

#print (alpha)

x[0,:] = np.ones((1,n_species))

# numerical solver

for l in range(n_timesteps-1):
    
    for i in range(n_species):
        
        alpha_x_prod = 0
        
        for j in range(n_species):
            
            alpha_x_prod = alpha_x_prod + alpha[i,j]*x[l,j]
#            print(alpha_x_prod)
#        print('r',r[0,i])
#        print('x',x[l,i])
            
        dxdt[l,i] = r[0,i]*x[l,i]*(1-alpha_x_prod)
    
    
    ### euler method:
    
    x[l+1,:] = dxdt [l,:] * timestep + x[l,:]
    
#print(x[:5,2],T[:5])
#
#print('done')

########### animate the result
#


def animation_frame(l):
    
    ax.clear()  # Clears the figure to update the line, point,      
    for i in range(n_species):
    
        plt.plot(T[:l], x[:l,i], label='species no.='+str(i))
#        print(x[:l,i])
#        print(T[:l])

    plt.title('Time passed='+str(T[l]), weight='bold',fontsize=FONT_SIZE)
    plt.ylabel('population size', fontsize=FONT_SIZE)
    plt.xlabel('time', fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    plt.legend(fontsize=FONT_SIZE)


fig, ax= plt.subplots(figsize=(16,16))


line_ani = animation.FuncAnimation(fig, animation_frame, interval=20,   
                                   frames=n_timesteps)
plt.show()
#    
    
