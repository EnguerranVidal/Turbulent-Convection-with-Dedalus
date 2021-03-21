# Program_2_DimlessBoussi_Spectral.py

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

class Spectral_DimLess_Boussinesq_Box_2D():
    '''
    The DimLess_Boussinesq_Box_2D class uses the dimensionless Boussinesq equation to simulate thermal convection
    in a 2D box containing a fluid heated from below and cooled from above using a pseudo spectral
    Fourier expansion method.
    The integration at each time step is done via an explicit Euler method.
    

    Fonctions :

    - __init__ : initializes the object
    - initialize_numbers : initializes the Prandtl and Rayleigh numbers
    - initialize_fields : initializes the grid, the fields and gives them their initial values
    - Curl_convective_term : calculates the Curl equation convective term
    - Temp_convective_term : calculates the Temperature equation convective term
    - Thomas_solver : solves the tridiagonal matrix from the Poisson equation
    - Curl_compute : calculates the next step curls
    - Temp_compute : calculates the RHS of the Poisson Pressure equations
    - Poisson_compute : calculates the pressure field by using a Poisson equation solver
    - Physical_space_calculation : allows us to calculate the spatial fields from the Fourier amplitudes
    - Time_step : calculates the next step time step to avoid program unstability
    - Velocity_calculation : calculates the speeds
    - RUN_Iterations : runs the simulation and stores the results in .pickle files
    - Post_processing : creates a GIF montage of the snapshots taken by the RUN_Iterations function

    '''
    def __init__(self):
        #Variables Fields
        self.f_Psi=None
        self.f_Omega=None
        self.f_Temp=None

    def initialize_numbers(self,Prandtl_number=1,Rayleih_number=1800):
        self.Pr=Prandtl_number
        self.Ra=Rayleih_number

    def initialize_fields(self,grid_width=1,grid_height=1,nx=100,nz=100,Fourier_limit=30):
        self.grid_width=grid_width
        self.grid_height=grid_height
        self.nx=nx
        self.nz=nz
        self.dx=grid_width/nx
        self.dz=grid_height/nz
        self.aspect_ratio=grid_width/grid_height
        self.x=np.linspace(0,grid_width,num=nx)
        self.z=np.linspace(0,grid_height,num=nz)
        self.z=np.flip(self.z)
        self.grid_X,self.grid_Z=np.meshgrid(self.x,self.z)
        self.Nn=Fourier_limit
        # Creation of the Fourier indices matrix
        self.n=np.linspace(0,self.Nn,num=self.Nn+1)
        self.f_N,self.fourier_Z=np.meshgrid(self.n,self.z)
        # Creation of the Fourier Temperatures Amplitudes
        T=np.sin(np.pi*self.z)
        self.f_Temp=np.zeros_like(self.f_N)
        for i in range(nz):
            self.f_Temp[i][0]=T[i]
        self.f_Temp[0][0]=0
        self.f_Temp[-1][0]=1
        # Creation of the Curl and Stream Function Fourier Amplitudes
        self.f_Curl=np.zeros_like(self.f_N)
        self.f_Psi=np.zeros_like(self.f_N)
        # Creation of the Poisson Tridiagonal Matrix
        self.Poisson_Sup=[]
        self.Poisson_Dia=[]
        self.Poisson_Sub=[]
        for i in range(self.Nn+1):
            sup=np.full(self.nz-1,-1/(self.dz**2))
            sub=np.full(self.nz-1,-1/(self.dz**2))
            dia=np.full(self.nz,(i*np.pi/self.aspect_ratio)**2+2/(self.dz**2))
            # adjustments to the limit values
            sup[0]=0
            dia[0]=1
            sub[-1]=0
            dia[-1]=1            # adding them to the lists
            self.Poisson_Sup.append(sup)
            self.Poisson_Dia.append(dia)
            self.Poisson_Sub.append(sub)
        assert len(self.Poisson_Sup)==self.Nn+1
        assert len(self.Poisson_Sub)==self.Nn+1
        assert len(self.Poisson_Dia)==self.Nn+1

    def Curl_convective_term(self,n):
        C=np.zeros_like(self.f_Curl[1:-1,0])
        for n_p in range(1,self.Nn+1):
            for n_pp in range(1,self.Nn+1):
                C=C+((-n_p*self.f_Curl[1:-1,n_p]*(self.f_Psi[0:-2,n_pp]-self.f_Psi[2:,n_pp])/(2*self.dz)+n_pp*self.f_Psi[1:-1,n_pp]*(self.f_Curl[0:-2,n_p]-self.f_Curl[2:,n_p])/(2*self.dz))*Kronecker(n_pp+n_p,n)-(-n_p*self.f_Curl[1:-1,n_p]*(self.f_Psi[0:-2,n_pp]-self.f_Psi[2:,n_pp])/(2*self.dz)+n_pp*self.f_Psi[1:-1,n_pp]*(self.f_Curl[0:-2,n_p]-self.f_Curl[2:,n_p])/(2*self.dz))*(Kronecker(n_pp-n_p,n)-Kronecker(n_p-n_pp,n)))
        return C*(-np.pi/(2*self.aspect_ratio))

    def Temp_convective_term(self,n):
        C=np.zeros_like(self.f_Temp[1:-1,0])
        C=-(n*np.pi/self.aspect_ratio)*self.f_Psi[1:-1,n]*(self.f_Temp[0:-2,0]-self.f_Temp[2:,0])/(2*self.dz)
        S=np.zeros_like(self.f_Temp[1:-1,0])
        if n==0:
            for n_p in range(1,self.Nn+1):
                for n_pp in range(1,self.Nn+1):
                    S=S+((-n_p*self.f_Curl[1:-1,n_p]*(self.f_Psi[0:-2,n_pp]-self.f_Psi[2:,n_pp])/(2*self.dz)+n_pp*self.f_Psi[1:-1,n_pp]*(self.f_Curl[0:-2,n_p]-self.f_Curl[2:,n_p])/(2*self.dz))*Kronecker(n_pp+n_p,0)-(-n_p*self.f_Curl[1:-1,n_p]*(self.f_Psi[0:-2,n_pp]-self.f_Psi[2:,n_pp])/(2*self.dz)+n_pp*self.f_Psi[1:-1,n_pp]*(self.f_Curl[0:-2,n_p]-self.f_Curl[2:,n_p])/(2*self.dz))*(Kronecker(n_pp-n_p,0)))
            S=S*(-np.pi/(2*self.aspect_ratio))
            return C+S
                    
        else:
            for n_p in range(1,self.Nn+1):
                for n_pp in range(1,self.Nn+1):
                    S=S+((-n_p*self.f_Temp[1:-1,n_p]*(self.f_Psi[0:-2,n_pp]-self.f_Psi[2:,n_pp])/(2*self.dz)+n_pp*self.f_Psi[1:-1,n_pp]*(self.f_Temp[0:-2,n_p]-self.f_Temp[2:,n_p])/(2*self.dz))*Kronecker(n_pp+n_p,n)-(-n_p*self.f_Temp[1:-1,n_p]*(self.f_Psi[0:-2,n_pp]-self.f_Psi[2:,n_pp])/(2*self.dz)+n_pp*self.f_Psi[1:-1,n_pp]*(self.f_Temp[0:-2,n_p]-self.f_Temp[2:,n_p])/(2*self.dz))*(Kronecker(n_pp-n_p,n)+Kronecker(n_p-n_pp,n)))
            S=S*(-np.pi/(2*self.aspect_ratio))
            return C+S
        
    def Thomas_solver(self,a,b,c,d):
        n=len(d)
        ac,bc,cc,dc=map(np.array,(a,b,c,d))
        for i in range(1, n):
            mc=ac[i-1]/bc[i-1]
            bc[i]=bc[i]-mc*cc[i-1] 
            dc[i]=dc[i]-mc*dc[i-1]
        xc=bc
        xc[-1]=dc[-1]/bc[-1]
        for i in range(n-2,-1,-1):
            xc[i]=(dc[i]-cc[i]*xc[i+1])/bc[i]
        return xc

    def Curl_compute(self,dt):
        self.new_f_Curl=np.empty_like(self.f_Curl)
        for i in range(1,self.Nn+1):
            C=self.Curl_convective_term(i)
            self.new_f_Curl[1:-1,i]=self.f_Curl[1:-1,i]+dt*(C+self.Ra*self.Pr*(i*np.pi/self.aspect_ratio)*self.f_Temp[1:-1,i]+self.Pr*(((self.f_Curl[0:-2,i]+self.f_Curl[2:,i]-2*self.f_Curl[1:-1,i])/(self.dz**2))-(i*np.pi/self.aspect_ratio)**2*self.f_Curl[1:-1,i]))
        self.new_f_Curl[0,:]=0
        self.new_f_Curl[-1,:]=0
        self.new_f_Curl[:,0]=0

    def Temp_compute(self,dt):
        self.new_f_Temp=np.empty_like(self.f_Curl)
        for i in range(self.Nn+1):
            C=self.Temp_convective_term(i)
            self.new_f_Temp[1:-1,i]=self.f_Temp[1:-1,i]+dt*(C+(((self.f_Temp[0:-2,i]+self.f_Temp[2:,i]-2*self.f_Temp[1:-1,i])/(self.dz**2))-(i*np.pi/self.aspect_ratio)**2*self.f_Temp[1:-1,i]))
        self.new_f_Temp[0,:]=0
        self.new_f_Temp[-1,1:]=0
        self.new_f_Temp[-1][0]=1

    def Poisson_compute(self):
        self.new_f_Psi=np.empty_like(self.f_Psi)
        for i in range(1,self.Nn+1):
            Curl=self.f_Curl[1:-1,i]
            Curl=Curl.T
            Sup=self.Poisson_Sup[i]
            Dia=self.Poisson_Dia[i]
            Sub=self.Poisson_Sub[i]
            Sol=self.Thomas_solver(Sub,Dia,Sup,Curl)
            self.new_f_Psi[:,i]=Sol
        self.new_f_Psi[0,:]=0
        self.new_f_Psi[-1,:]=0
        self.new_f_Psi[:,0]=0

    def Physical_space_calculation(self,variable='temp'):
        if variable=='temp':
            self.Temp=np.zeros_like(self.grid_X)
            for i in range(self.Nn+1):
                for j in range(self.nz):
                    self.Temp[j,:]=self.Temp[j,:]+self.f_Temp[j][i]*np.cos(i*np.pi*self.grid_X[j,:]/self.aspect_ratio)
        if variable=='psi':
            self.Psi=np.zeros_like(self.grid_X)
            for i in range(self.Nn+1):
                for j in range(self.nz):
                    self.Psi[j,:]=self.Psi[j,:]+self.f_Psi[j][i]*np.sin(i*np.pi*self.grid_X[j,:]/self.aspect_ratio)
        if variable=='curl':
            self.Curl=np.zeros_like(self.grid_X)
            for i in range(self.Nn+1):
                for j in range(self.nz):
                    self.Curl[j,:]=self.Curl[j,:]+self.f_Curl[j][i]*np.sin(i*np.pi*self.grid_X[j,:]/self.aspect_ratio)
             
    def Velocity_calculation(self):
        U=np.zeros_like(self.Psi)
        V=np.zeros_like(self.Psi)
        U[1:-1,1:-1]=-(self.Psi[0:-2,1:-1]-self.Psi[2:,1:-1])/(2*self.dz)
        V[1:-1,1:-1]=(self.Psi[1:-1,2:]-self.Psi[1:-1,0:-2])/(2*self.dx)
        return U,V

    def Time_step(self,V):
        v=np.amax(np.absolute(V))
        if self.Pr<1:
            if v!=0:
                dt1=(self.dz**2)/4
                dt2=(self.dz**2)/(4*v)
                return self.safety_coeff*min(dt1,dt2)
            else:
                return self.safety_coeff*(self.dz**2)/4
        elif self.Pr>=1:
            if v!=0:
                dt1=(self.dz**2)/4*self.Pr
                dt2=(self.dz**2)/(4*v)
                return self.safety_coeff*min(dt1,dt2)
            else:
                return self.safety_coeff*(self.dz**2)/4*self.Pr

    def RUN_Iterations(self,n_iterations=500,safety_coefficient=0.01):
        t=0
        dt=0
        self.safety_coeff=safety_coefficient
        # Clearing the pickle files
        clear_file('temperatures.pkl')
        clear_file('streams.pkl')
        clear_file('times.pkl')
        # Saving the initial Fourier arrays
        save_unique_array('temperatures.pkl',self.f_Temp)
        save_unique_array('streams.pkl',self.f_Psi)
        save_unique_array('times.pkl',t)
        # Calculation of the space initial arrays
        self.Physical_space_calculation(variable='psi')
        # Calculation of the initial velocities
        U,V=self.Velocity_calculation()
        calc_time0=time.time()
        percent_5=int(n_iterations*5/100)
        print("######### BEGINNING CALCULATIONS #########")
        for i in range(n_iterations):
            if i%percent_5==0:
                printProgressBar(i+1,n_iterations,prefix='Progress:',suffix='Complete',length=20,time0=calc_time0)
            dt=self.Time_step(V)
            t=t+dt
            self.Temp_compute(dt)
            self.Curl_compute(dt)
            self.Poisson_compute()
            # Updating the spectral arrays
            self.f_Psi=self.new_f_Psi
            self.f_Curl=self.new_f_Curl
            self.f_Temp=self.new_f_Temp
            # Extracting the Velocity field
            self.Physical_space_calculation(variable='psi')
            U,V=self.Velocity_calculation()
            save_unique_array('temperatures.pkl',self.f_Temp)
            save_unique_array('streams.pkl',self.f_Psi)
            save_unique_array('times.pkl',t)
        print("######### CALCULATIONS FINISHED #########")
          
    def Post_processing(self,plot=False,save=True,skip=1,gif_fps=25):
        f_Temps=load_files('temperatures.pkl',frequency=skip)
        #f_Psis=load_files('streams.pkl',frequency=skip)
        Ts=load_files('times.pkl',frequency=skip)
        n=len(f_Temps)
        Temps=[]
        Psis=[]
        print("######### BEGINNING POST-PROCESSING #########")
        for i in range(n):
            self.f_Temp=f_Temps[i]
            self.Physical_space_calculation(variable='temp')
            Temps.append(self.Temp)
            #self.f_Psi=f_Psi[i]
            #self.Physical_space_calculation(variable='psi')
            #Psis.append(self.Psi)
        print("######### POST-PROCESSING FINISHED #########")
        Tmax=np.amax(Temps[0])
        Tmin=np.amin(Temps[0])
        fig,ax=plt.subplots(figsize=(10,5))
        heatmap=ax.imshow(Temps[0],cmap='Spectral_r',extent=[0,self.grid_width,0,self.grid_height],vmin=Tmin,vmax=Tmax)
        #arrows=ax.quiver(self.grid_X[1::15,1::15],self.grid_Z[1::15,1::15],U[1::15,1::15],V[1::15,1::15])
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
