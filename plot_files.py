# plot_files.py

# IMPORTS --------------------------------------------------
# We first import the necessary libraries like mentionned.


import numpy as np
import matplotlib.pyplot as plt

# FUNCTIONS -------------------------------------------

def plot_convection_file(Ra,Pr,Ta,Np):
    ''' Extracts KEs from a specific Ta, Pr , Np and Ra'''
    file='KE-Ra:'+str(Ra)+'-Pr:'+str(Pr)+'-Ta:'+str(Ta)+'-Np:'+str(Np)
    with open(file,'r') as f:
        lines=f.readlines()
    n=len(lines)
    X=[]
    Y=[]
    for i in range(n):
        lines[i]=lines[i].split(' ')
        X.append(float(lines[i][0]))
        Y.append(float(lines[i][1]))
    plt.plot(X, Y, label="KE")
    plt.title('Ra='+str(Ra))
    plt.yscale("log")
    plt.legend(loc='upper right', fontsize=10)
    plt.show()


def plot_multiple_Ras(Ras,Pr,Ta,Np):
    ''' Extracts muliples KEs from a specific Ta, Pr and Np from a list of Rayleigh numbers and plots them'''
    n_files=len(Ras)
    X=[]
    Y=[]
    for j in range(n_files):
        file='KE-Ra:'+str(Ras[j])+'-Pr:'+str(Pr)+'-Ta:'+str(Ta)+'-Np:'+str(Np)
        with open(file,'r') as f:
            lines=f.readlines()
        n=len(lines)
        Xn=[]
        Yn=[]
        for i in range(n):
            lines[i]=lines[i].split(' ')
            Xn.append(float(lines[i][0]))
            Yn.append(float(lines[i][1]))
        X.append(Xn)
        Y.append(Yn)
    for k in range(n_files):
        plt.plot(X[k],Y[k],label="Ra"+str(Ras[k]))
    plt.grid()
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("viscous time")
    plt.ylabel("KE")
    plt.legend(loc='upper right', fontsize=10)
    plt.show()

def post_processing_S(Ra,Pr,Np,Ta,plot=False,save=True,gif_fps=25):
    ''' Extracts the entropies and times from a given file and outputs a GIF of the field changes '''
    filename='S-Ra:'+str(Ra)+'-Pr:'+str(Pr)+'-Ta:'+str(Ta)+'-Np:'+str(Np)
    if save==True:
        gif_name='GIF-Ra:'+str(Ra)+'-Pr:'+str(Pr)+'-Ta:'+str(Ta)+'-Np:'+str(Np)+'.gif'
    Ts,Ss=read_state_file(filename)
    n=len(Ts)
    print("########## BEGINNING POST-PROCESSING ##########")
    Smax=1
    Smin=0
    fig,ax=plt.subplots(figsize=(10,5))
    heatmap=ax.imshow(np.flip(Ss[0].T,0),cmap='RdBu_r',extent=[0,2,0,1.],vmin=Smin,vmax=Smax)
    ax.set(xlabel='X',ylabel='Y')
    plt.colorbar(heatmap)
    if plot==True:
        fig.show()
        plt.pause(3)
    if save==True:
        Images=[]
    for i in range(1,n):
        heatmap.set_data(np.flip(Ss[i].T,0))
        ax.set(xlabel='X',ylabel='Y')
        if save==True and plot==True:
            plt.pause(0.01)
            plt.draw()
            image=np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image=image.reshape(fig.canvas.get_width_height()[::-1]+(3,))
            Images.append(image)
        if plot==True and save==False:
            plt.pause(0.01)
            plt.draw()
        if plot==False and save==True:
            fig.canvas.draw()
            image=np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image=image.reshape(fig.canvas.get_width_height()[::-1]+(3,))
            Images.append(image)
    if save==True:
        print("########## GIF CREATION ##########")
        imageio.mimsave(gif_name,Images,fps=gif_fps)
        print("########## GIF FINISHED ##########")

