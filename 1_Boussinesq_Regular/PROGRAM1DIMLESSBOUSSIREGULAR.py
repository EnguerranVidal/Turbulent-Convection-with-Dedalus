# Program_1_DimlessBoussi_Regular.py

# IMPORTS --------------------------------------------------
# We first import the necessary libraries like mentionned.

import numpy as np
import matplotlib.pyplot as plt
import imageio
import time
import pickle

from functions_pickle import*

# CLASSES --------------------------------------------------
# We create a Python Object class modelizing the fluid and managing
# the calculations and storing of the resusts as a GIF montage.

class DimLess_Boussinesq_Box_2D():
    '''
    The DimLess_Boussinesq_Box_2D class uses the dimensionless Boussinesq equation to simulate thermal convection
    in a 2D box containing a fluid heated from below and cooled from above. The class uses a Marker and Cell
    staggered grid to calculate the derivatives from the finite difference method.
    The integration at each time step is done via an explicit Euler method.
    

    Fonctions :

    - __init__ : initializes the object
    - initialize_numbers : initializes the Prandtl and Rayleigh numbers
    - initialize_grid : initializes the staggered grid coordinates and parameters
    - initialize_fields : initializes the fields and gives them their initial values
    - Speeds_compute : calculates the next step speeds
    - Temp_compute : calculates the next step temperatures
    - B_calculation : calculates the RHS of the Poisson Pressure equations
    - Poisson_compute : calculates the pressure field by using a Poisson equation solver
    - dT_Calculation : calculates the next step time step to avoid program unstability
    - UV : calculates the speeds at the center of the MAC grid cells
    - RUN_Iterations : runs the simulation and stores the results in .pickle files
    - Post_processing : creates a GIF montage of the snapshots taken by the RUN_Iterations function

    '''
    def __init__(self):
        ########## Variables Fields
        self.U=None
        self.V=None
        self.P=None
        self.T=None
        
    def initialize_numbers(self,Prandtl_number=1,Rayleih_number=1800):
        self.Pr=Prandtl_number
        self.Ra=Rayleih_number

    def initialize_grid(self,grid_height=1,grid_width=2,nx=10,ny=20):
        # Grid Variables
        self.nx=nx
        self.ny=ny
        self.dx=grid_width/nx
        self.dy=grid_height/ny
        self.grid_height=grid_height
        self.grid_width=grid_width
        self.aspect_ratio=grid_width/grid_height
        # Grid Coordinates
        self.Cell_x=np.linspace(0+self.dx/2,grid_width-self.dx/2,num=nx)
        self.Cell_y=np.linspace(0+self.dy/2,grid_height-self.dy/2,num=ny)
        self.Cell_y=np.flip(self.Cell_y)
        self.Vertice_Vertical_x=np.linspace(0,grid_width,num=nx+1)
        self.Vertice_Vertical_y=np.linspace(0+self.dy/2,grid_height-self.dy/2,num=ny)
        self.Vertice_Vertical_y=np.flip(self.Vertice_Vertical_y)
        self.Vertice_Horizontal_x=np.linspace(0+self.dx/2,grid_width-self.dx/2,num=nx)
        self.Vertice_Horizontal_y=np.linspace(0,grid_height,num=ny+1)
        self.Vertice_Horizontal_y=np.flip(self.Vertice_Horizontal_y)
        self.Cell_X,self.Cell_Y=np.meshgrid(self.Cell_x,self.Cell_y)
        # Grid Meshgrids
        self.Vertice_Vertical_X,self.Vertice_Vertical_Y=np.meshgrid(self.Vertice_Vertical_x,self.Vertice_Vertical_y)
        self.Vertice_Horizontal_X,self.Vertice_Horizontal_Y=np.meshgrid(self.Vertice_Horizontal_x,self.Vertice_Horizontal_y)
        
    def initialize_fields(self,bottom_P=0,delta=0.05,template='closed_box'):
        # Speeds Initialization
        self.template=template
        self.U=delta*np.random.uniform(low=-1.0,high=1.0,size=self.Vertice_Vertical_X.shape)
        self.V=delta*np.random.uniform(low=-1.0,high=1.0,size=self.Vertice_Horizontal_X.shape)
        if self.template=='closed_box':
            self.U[0,:]=0
            self.U[:,0]=0
            self.U[:,-1]=0
            self.U[-1,:]=0
            self.V[0,:]=0
            self.V[-1,:]=0
            self.V[:,0]=0
            self.V[:,-1]=0
        #self.T=np.sin(np.pi*self.Cell_Y)
        self.T=np.zeros_like(self.Cell_Y)
        self.T[0,:]=0
        self.T[-1,:]=1
        self.P=np.zeros_like(self.Cell_Y)

    def Speeds_compute(self,dt):
        if self.template=='closed_box':
            V=0.25*(self.V[1:-2,0:-1]+self.V[1:-2,1:]+self.V[2:-1,0:-1]+self.V[2:-1,1:])
            U=0.25*(self.U[0:-1,1:-2]+self.U[0:-1,2:-1]+self.U[1:,1:-2]+self.U[1:,2:-1])
            T_V=0.5*(self.T[0:-1,1:-1]+self.T[1:,1:-1])
            self.new_U=np.empty_like(self.U)
            self.new_V=np.empty_like(self.V)
            self.new_U[1:-1,1:-1]=self.U[1:-1,1:-1]+dt*(-self.U[1:-1,1:-1]*(self.U[1:-1,2:]-self.U[1:-1,0:-2])/(2*self.dx)-V*(self.U[0:-2,1:-1]-self.U[2:,1:-1])/(2*self.dy)-(self.P[1:-1,0:-1]-self.P[1:-1,1:])/(2*self.dx)+self.Pr*(self.U[1:-1,2:]+self.U[1:-1,0:-2]-2*self.U[1:-1,1:-1])/(self.dx**2)+self.Pr*(self.U[0:-2,1:-1]+self.U[2:,1:-1]-2*self.U[1:-1,1:-1])/(self.dy**2))
            self.new_V[1:-1,1:-1]=self.V[1:-1,1:-1]+dt*(-self.V[1:-1,1:-1]*(self.V[1:-1,2:]-self.V[1:-1,0:-2])/(2*self.dx)-U*(self.V[0:-2,1:-1]-self.V[2:,1:-1])/(2*self.dy)-(self.P[0:-1,1:-1]-self.P[1:,1:-1])/(2*self.dy)+self.Pr*self.Ra*T_V+self.Pr*(self.V[1:-1,2:]+self.V[1:-1,0:-2]-2*self.V[1:-1,1:-1])/(self.dx**2)+self.Pr*(self.V[0:-2,1:-1]+self.V[2:,1:-1]-2*self.V[1:-1,1:-1])/(self.dy**2))
            self.new_U[0,:]=0
            self.new_U[:,0]=0
            self.new_U[:,-1]=0
            self.new_U[-1,:]=0
            self.new_V[0,:]=0
            self.new_V[-1,:]=0
            self.new_V[:,0]=0
            self.new_V[:,-1]=0

    def Temp_compute(self,dt):
        if self.template=='closed_box':
            U=0.5*(self.U[1:-1,1:-2]+self.U[1:-1,2:-1])
            V=0.5*(self.V[1:-2,1:-1]+self.V[2:-1,1:-1])
            VL=0.5*(self.V[1:-2,0]+self.V[2:-1,0])
            VR=0.5*(self.V[1:-2,-1]+self.V[2:-1,-1])
            self.new_T=np.empty_like(self.T)
            self.new_T[1:-1,1:-1]=self.T[1:-1,1:-1]+dt*(-U*(self.T[1:-1,2:]-self.T[1:-1,0:-2])/(2*self.dx)-V*(self.T[0:-2,1:-1]-self.T[2:,1:-1])/(2*self.dy)+(self.T[1:-1,2:]+self.T[1:-1,0:-2]-2*self.T[1:-1,1:-1])/(self.dx**2)+(self.T[0:-2, 1:-1]+self.T[2:,1:-1]-2*self.T[1:-1,1:-1])/(self.dy**2))
            self.new_T[1:-1,0]=self.T[1:-1:,0]+dt*(-VL*(self.T[0:-2,0]-self.T[2:,0])/(2*self.dy)+(2*self.T[1:-1,1]-2*self.T[1:-1,0])/(self.dx**2)+(self.T[0:-2,0]+self.T[2:,0]-2*self.T[1:-1,0])/(self.dy**2))
            self.new_T[1:-1,-1]=self.T[1:-1,-1]+dt*(-VR*(self.T[0:-2,-1]-self.T[2:,-1])/(2*self.dy)+(2*self.T[1:-1,-2]-2*self.T[1:-1,-1])/(self.dx**2)+(self.T[0:-2,-1]+self.T[2:,-1]-2*self.T[1:-1,-1])/(self.dy**2))
            self.new_T[0,:]=0
            self.new_T[-1,:]=1

    def B_calculation(self,dt):
        if self.template=='closed_box':
            VL=0.5*(self.V[1:-2,0:-2]+self.V[2:-1,0:-2])
            VR=0.5*(self.V[1:-2,2:]+self.V[2:-1,2:])
            UT=0.5*(self.U[0:-2,1:-2]+self.U[0:-2,2:-1])
            UB=0.5*(self.U[2:,1:-2]+self.U[2:,2:-1])
            B=self.Ra*self.Pr*(self.T[0:-2,1:-1]-self.T[2:,1:-1])/(2*self.dy)
            B=B-(((self.U[1:-1,2:-1]-self.U[1:-1,1:-2])/self.dx)**2+((self.V[1:-2,1:-1]-self.V[2:-1,1:-1])/self.dy)**2+((VR-VL)/(2*self.dx))*((UT-UB)/(2*self.dy)))
        return B
    
    def Poisson_compute(self,nit,dt):
        if self.template=='closed_box':
            Pn=np.empty_like(self.P)
            Pn=self.P.copy()
            B=self.B_calculation(dt)
            for i in range(nit):
                Pn=self.P.copy()
                self.P[1:-1,1:-1]=(((Pn[1:-1,2:]+Pn[1:-1,0:-2])*self.dy**2+(Pn[2:,1:-1]+Pn[0:-2, 1:-1])*self.dx**2)/(2*(self.dx**2+self.dy**2))-self.dx**2*self.dy**2/(2*(self.dx**2+self.dy**2))*B)
                self.P[:,-1]=self.P[:,-2]
                self.P[0,:]=self.P[1,:]
                self.P[:,0]=self.P[:,1]
                self.P[-1,:]=self.P[-2,:]

    def dT_Calculation(self):
        max_u=np.amax(np.absolute(self.U))
        max_v=np.amax(np.absolute(self.V))
        if max_u>max_v:
            return self.safety_coeff*(self.dx**2/max_u)
        else:
            return self.safety_coeff*(self.dy**2/max_v)

    def UV(self):
        U=0.5*(self.U[:,0:-1]+self.U[:,1:])
        V=0.5*(self.V[0:-1,:]+self.V[1:,:])
        U=0.5*U/np.sqrt(U**2+V**2)
        V=0.5*V/np.sqrt(U**2+V**2)
        return U,V

    def RUN_Iterations(self,n_iterations=500,n_poisson=50,safety_coefficient=0.01):
        t=0
        dt=0
        self.safety_coeff=safety_coefficient
        # Clearing the pickle files
        clear_file('temperatures.pkl')
        clear_file('times.pkl')
        # Saving the initial Fourier arrays
        save_unique_array('temperatures.pkl',self.T)
        save_unique_array('times.pkl',t)
        calc_time0=time.time()
        percent_5=int(n_iterations*5/100)
        print("######### BEGINNING CALCULATIONS #########")
        for i in range(n_iterations):
            if i%percent_5==0:
                printProgressBar(i+1,n_iterations,prefix='Progress:',suffix='Complete',length=20,time0=calc_time0)
            t=t+dt
            dt=self.dT_Calculation()
            self.Poisson_compute(n_poisson,dt)
            self.Temp_compute(dt)
            self.Speeds_compute(dt)
            self.U=self.new_U
            self.V=self.new_V
            self.T=self.new_T
            save_unique_array('temperatures.pkl',self.T)
            save_unique_array('times.pkl',t)
        printProgressBar(n_iterations,n_iterations,prefix='Progress:',suffix='Complete',length=20,time0=calc_time0)
        print("######### CALCULATIONS FINISHED #########")
        print(" Final calculations duration = ",round(time.time()-calc_time0,4)," seconds")

    def Post_processing(self,plot=False,save=True,skip=1,gif_fps=25):
        Temps=load_files('temperatures.pkl',frequency=skip)
        Ts=load_files('times.pkl',frequency=skip)
        n=len(Temps)
        print("######### BEGINNING POST-PROCESSING #########")
        Tmax=np.amax(Temps[0])
        Tmin=np.amin(Temps[0])
        fig,ax=plt.subplots(figsize=(10,5))
        heatmap=ax.imshow(Temps[0],cmap='Spectral_r',extent=[0,self.grid_width,0,self.grid_height],vmin=Tmin,vmax=Tmax)
        ax.set(xlabel='X',ylabel='Y',title='Time ='+str(round(Ts[0],10)))
        plt.colorbar(heatmap)
        if plot==True:
            fig.show()
            plt.pause(5)
        if save==True:
            Images=[]
        for i in range(1,n):
            heatmap.set_data(Temps[i])
            ax.set(xlabel='X',ylabel='Y',title='Time ='+str(round(Ts[i],10)))
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
            print("######### GIF CREATION #########")
            imageio.mimsave('sim.gif',Images,fps=gif_fps)
            print("######### GIF FINISHED #########")
