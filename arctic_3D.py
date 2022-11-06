import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
FONT_SIZE=22


## population matrix for n_species, for n_timesteps

Total_time=20
timestep = 1/20

n_species = 3


T = np.arange(0,Total_time, timestep)
n_timesteps = np.size(T)

x = np.zeros((n_timesteps,n_species))

dxdt = np.zeros((n_timesteps,n_species))
##
## initial conditions

#inherent growth rate r, and interaction alpha, independent of time

# r = np.ones((1,n_species))
# alpha = np.ones((n_species,n_species))
# x[0,:] = np.ones((1,n_species))

# r = np.array([[1,0.72,1.53, 1.27]])

# alpha = np.array([[1, 1.09, 1.52, 0],
#                  [0, 1, 0.44, 1.36],
#                  [2.33, 0, 1, 0.47],
#                  [1.21, 0.51, 0.35, 1]])

#growth rate
r = np.array([[.3,-.8,-1]]) 

#interaction factors
alpha = np.array([[0, 1,0],
                 [1, 0,-1],
                 [0, 1,0]])

#print (alpha)

#x[0,:] = np.ones((1,n_species))
x[0,:] = np.array([[2,2,1]])

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
    fig.suptitle('Time passed='+str(T[l]), weight='bold',fontsize=FONT_SIZE)
   
    ax1.clear()  # Clears the figure to update the line, point,      
    ax2.clear()  # Clears the figure to update the line, point,      

#### time evol

#    for i in range(n_species):
    
    ax1.plot(T[:l], x[:l,0], label='Arctic Cod', color = 'blue')
    ax1.plot(T[l-1], x[l-1,0], 'o', color = 'blue')

    ax1.plot(T[:l], x[:l,1], label='Seal', color = 'red')
    ax1.plot(T[l-1], x[l-1,1], 'o', color = 'red')

    ax1.plot(T[:l], x[:l,2], label='Polar Bear', color = 'green')
    ax1.plot(T[l-1], x[l-1,2], 'o', color = 'green')        
#        print(x[:l,i])
#        print(T[:l])

    ax1.set_title('Time Evolution',fontsize=FONT_SIZE)
    ax1.set_ylabel('population size', fontsize=FONT_SIZE)
    ax1.set_xlabel('time', fontsize=FONT_SIZE)
    ax1.tick_params(axis='both', labelsize=FONT_SIZE)
    ax1.tick_params(axis='both', labelsize=FONT_SIZE)
    ax1.legend(fontsize=FONT_SIZE)


###### phase space
#    for i in range(n_species):
    
    ax2.plot3D(x[:l,0], x[:l,1], x[:l,2], color = 'black')
    ax2.scatter3D(x[l-1,0], x[l-1,1], x[l-1,2], 'o', color = 'black')
        
#        print(x[:l,i])
#        print(T[:l])

    ax2.set_title('Phase Space',fontsize=FONT_SIZE, y=1.1)
    ax2.set_zlabel('Arctic Cod', fontsize=FONT_SIZE)
    ax2.set_ylabel('Seal', fontsize=FONT_SIZE)
    ax2.set_xlabel('Polar Bear', fontsize=FONT_SIZE)
    ax2.xaxis.labelpad=30
    ax2.yaxis.labelpad=30
    ax2.zaxis.labelpad=30
    ax2.tick_params(axis='both', labelsize=FONT_SIZE-9)
    ax2.tick_params(axis='both', labelsize=FONT_SIZE-9)
#    ax2.legend(fontsize=FONT_SIZE)

#fig, ((ax1),(ax2)) = plt.subplots(2,1,figsize=(16,16))
fig = plt.figure(figsize=(37,20))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122,projection="3d")

line_ani = animation.FuncAnimation(fig, animation_frame, interval=50,   
                                   frames=n_timesteps)
#plt.show()
    
savefile = 'arctic3D_total_time='+str(Total_time)+'.mp4'
line_ani.save(savefile)
#line_ani.save('%s.gif'%(savefile), writer='imagemagick',fps=1000/50)    
