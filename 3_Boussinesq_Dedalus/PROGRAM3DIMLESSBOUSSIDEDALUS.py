# Program_3_DimlessBoussi_Dedalus.py

# IMPORTS --------------------------------------------------
# We first import the necessary libraries like mentionned.

import numpy as np
import matplotlib.pyplot as plt
from dedalus import public as de
from dedalus.extras import flow_tools
import time
import imageio

from functions_txt import*

# CLASSES --------------------------------------------------
# We create a Python Object class modelizing the fluid and managing
# the calculations and storing of the resusts as a GIF montage.

class Dedalus_Boussinesq():
    '''
    The Dedalus_Boussinesq class uses the Dedalus open source to simulate thermal convection in
    a 2Dbox containing a fluid cooled from above and heated from below. It uses the curl and
    the stream function as well as the temperature as its variables.
    

    Fonctions :

    - __init__ : initializes the object
    - problem_setup : initializes the problem parameters, equatiosn and boundary equations
    - RUN : runs the simulation
    - post_processing : creates a GIF montage of the snapshots taken by the RUN_Iterations function

    '''
    def __init__(self):
        self.domain=None
        self.solver=None
        self.problem=None

    def problem_setup(self,L=2.,nx=192,nz=96,Prandtl_number=1.,Rayleih_number=1e4,bc_type='no_slip'):
        self.L=float(L)
        self.nx=int(nx)
        self.nz=int(nz)
        x_basis = de.Fourier('x', int(nx), interval=(0, L), dealias=3/2)
        z_basis = de.Chebyshev('z',int(nz), interval=(0, 1), dealias=3/2)
        self.saving_shape=(int(nx*3/2),int(nz*3/2))
        self.domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)
        self.problem = de.IVP(self.domain, variables=['T','Tz','psi','psiz','curl','curlz'])
        self.problem.parameters['L'] = L
        self.problem.parameters['nx'] = nx
        self.problem.parameters['nz'] = nz
        self.Pr=float(Prandtl_number)
        self.Ra=float(Rayleih_number)
        self.problem.parameters['Ra'] = self.Ra
        self.problem.parameters['Pr'] = self.Pr
        # Stream function relation to the speed
        self.problem.substitutions['u'] = "-dz(psi)"
        self.problem.substitutions['v'] = "dx(psi)"
        # Derivatives values relation to the main values
        self.problem.add_equation("psiz - dz(psi) = 0")
        self.problem.add_equation("curlz - dz(curl) = 0")
        self.problem.add_equation("Tz - dz(T) = 0")
        self.problem.add_equation("curl + dx(dx(psi)) + dz(psiz) = 0")
        self.problem.add_equation("dt(curl)+Ra*Pr*dx(T)-Pr*(dx(dx(curl))+dz(curlz))=-(u*dx(curl)+v*curlz)")
        self.problem.add_equation("dt(T)-dx(dx(T))-dz(Tz)=-(u*dx(T)+v*Tz)")
        self.problem.add_bc("left(T) = 1")
        self.problem.add_bc("right(T) = 0")
        self.problem.add_bc("left(psi) = 0")
        self.problem.add_bc("right(psi) = 0")
        if bc_type not in ['no_slip','free_slip']:
            raise ValueError("Boundary Conditions must be 'no_slip' or 'free_slip'")
        else:
            if bc_type=='no_slip':
                self.problem.add_bc("left(psiz) = 0")
                self.problem.add_bc("right(psiz) = 0")
            if bc_type=='free_slip':
                self.problem.add_bc("left(dz(psiz)) = 0")
                self.problem.add_bc("right(dz(psiz)) = 0")

    def RUN(self,scheme=de.timesteppers.RK443,adding=False,sim_time=2,wall_time=np.inf,tight=False,save=20):
        self.solver = self.problem.build_solver(scheme)
        if adding:
            t=load_last_value('times.txt')
            temp=load_last_array('temperatures.txt',shape=self.saving_shape)
            Variables=load_arrays('variables.txt',frequency=1,shape=self.saving_shape)
            T=self.solver.state['T']
            Psi=self.solver.state['psi']
            Curl=self.solver.state['curl']
            T['g']=temp
            T.differentiate('z', out=self.solver.state['Tz'])
            Psi['g']=Variables[0]
            Psi.differentiate('z', out=self.solver.state['psiz'])
            Curl['g']=Variables[1]
            Curl.differentiate('z', out=self.solver.state['curlz'])
        else:
            t=0
            print("Clearing old data ...")
            clear_file('temperatures.txt')
            clear_file('times.txt')
            clear_file('variables.txt')
            # Initial conditions ----------------------------------------------------------
            print("Initializing Values ...")
            eps = 1e-4
            k = 3.117
            x,z = self.problem.domain.grids(scales=1)
            T=self.solver.state['T']
            T['g']=1-z+eps*np.sin(k*x)*np.sin(2*np.pi*z)
            T.differentiate('z', out=self.solver.state['Tz'])
        # Stopping Parameters ---------------------------------------------------------
        self.solver.stop_sim_time = sim_time # Length of simulation.
        self.solver.stop_wall_time = wall_time # Real time allowed to compute.
        self.solver.stop_iteration = np.inf # Maximum iterations allowed.
        # Control Flow ----------------------------------------------------------------
        dt = 1e-4
        if tight:
            cfl = flow_tools.CFL(self.solver,initial_dt=dt,cadence=1,
                                 safety=1,max_change=1.5,
                                 min_change=0.01,max_dt=0.01,
                                 min_dt=1e-10)
        else:
            cfl = flow_tools.CFL(self.solver, initial_dt=dt, cadence=10,
                                 safety=1,max_change=1.5,
                                 min_change=0.5,max_dt=0.01,
                                 min_dt=1e-6)
        cfl.add_velocities(('u', 'v'))
        # Flow properties (print during run; not recorded in the records files)
        flow = flow_tools.GlobalFlowProperty(self.solver, cadence=1)
        flow.add_property("sqrt(u **2 + v **2) / Ra", name='Re' )
        # MAIN COMPUTATION LOOP -------------------------------------------------------
        try:
            print("####### BEGINNING CALCULATIONS #######")
            print("Starting main loop")
            start_time = time.time()
            while self.solver.ok:
                # Recompute time step and iterate.
                dt = self.solver.step(cfl.compute_dt())
                t=t+dt
                if self.solver.iteration % 10 == 0:
                    info = "Iteration {:>5d}, Time: {:.7f}, dt: {:.2e}".format(self.solver.iteration, self.solver.sim_time, dt)
                    Re  = flow.max("Re")
                    info += ", Max Re = {:f}".format(Re)
                    print(info)
                    if np.isnan(Re):
                        raise ValueError("Reynolds number went to infinity!!"
                                         "\nRe = {}".format(Re))
                if save:
                    if self.solver.iteration % save == 0:
                        T=self.solver.state['T']
                        append_unique_array('temperatures.txt',T['g'])
                        append_unique_value('times.txt',t)
        except BaseException as e:
            print("Exception raised, triggering end of main loop.")
            raise
        finally:
            print("####### CALCULATIONS FINISHED #######")
            total_time = time.time() - start_time
            print("Iterations: {:d}".format(self.solver.iteration))
            print("Sim end time: {:.3e}".format(self.solver.sim_time))
            print("Run time: {:.3e} sec".format(total_time))
            print("END OF SIMULATION\n")
            T=self.solver.state['T']
            Psi=self.solver.state['psi']
            Curl=self.solver.state['curl']
            append_unique_array('temperatures.txt',T['g'])
            append_unique_value('times.txt',t)
            append_unique_array('variables.txt',Psi['g'])
            append_unique_array('variables.txt',Curl['g'])


    def post_processing(self,plot=False,save=True,skip=1,gif_fps=25):
        Temps=load_arrays('temperatures.txt',frequency=skip,shape=self.saving_shape)
        Ts=load_values('times.txt',frequency=skip)
        n=len(Temps)
        print("####### BEGINNING POST-PROCESSING #######")
        Tmax=np.amax(Temps[0])
        Tmin=np.amin(Temps[0])
        fig,ax=plt.subplots(figsize=(10,5))
        heatmap=ax.imshow(np.flip(Temps[0].T,0),cmap='Spectral_r',extent=[0,self.L,0,1.],vmin=Tmin,vmax=Tmax)
        ax.set(xlabel='X',ylabel='Y',title='Time ='+str(round(Ts[0],10)))
        plt.colorbar(heatmap)
        if plot==True:
            fig.show()
            plt.pause(3)
        if save==True:
            Images=[]
        for i in range(1,n):
            heatmap.set_data(np.flip(Temps[i].T,0))
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
            print("####### GIF CREATION #######")
            imageio.mimsave('sim.gif',Images,fps=gif_fps)
            print("####### GIF FINISHED #######")


