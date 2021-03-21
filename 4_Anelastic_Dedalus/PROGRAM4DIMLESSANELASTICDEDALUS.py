# Program_4_DimlessAnelastic_Dedalus.py

# IMPORTS --------------------------------------------------
# We first import the necessary libraries like mentionned.

import numpy as np
from mpi4py import MPI
import time
import matplotlib.pyplot as plt
import sys
import imageio
import os

import h5py

from dedalus import public as de
from dedalus.extras import flow_tools
import pathlib

from functions_txt import*

import logging
logger = logging.getLogger(__name__)

# FUNCTIONS -------------------------------------------
#################### ENTROPY AND KE STUDY ##############################

def Dedalus_Inelastic_S(Ra,Pr,Np,Ta,end_time,snaps=True):
    ''' Builds a problem with its equations, parameters and boundary conditions and solves it, saving values in a h5py '''
    m=1.5
    theta = 1-np.exp(-Np/m)
    Ly, Lz = 2,1
    Ny, Nz = 192,96
    Lat=np.pi/4
    initial_timestep = 1.5e-4           # Initial timestep
    snapshot_skip=10
    analysis_freq = 1.5e-3              # Frequency analysis files are outputted
    end_sim_time = end_time                  # Stop time in simulations units
    end_wall_time = np.inf              # Stop time in wall time
    end_iterations = np.inf             # Stop time in iterations
    max_dt=0.005
    save_direc = "raw_data/"
    pathlib.Path(save_direc).mkdir(parents=True, exist_ok=True)
    # Create bases and domain
    y_basis = de.Fourier('y', Ny, interval=(0, Ly), dealias=3/2)   # Fourier basis in the x
    z_basis = de.Chebyshev('z', Nz, interval=(0, Lz), dealias=3/2) # Chebyshev basis in the z
    domain = de.Domain([y_basis, z_basis], grid_dtype=np.float64)  # Defining our domain
    z = domain.grid(1, scales=1)                                   # accessing the z values
    # 2D Anelastic hydrodynamics
    problem = de.IVP(domain, variables=['p', 's', 'u', 'v', 'w', 'sz', 'uz', 'vz', 'wz'])
    problem.meta['p','s','u','w']['z']['dirichlet'] = True
    # Defining model parameters
    problem.parameters['Ly'] = Ly
    problem.parameters['Lz'] = Lz
    problem.parameters['Ra'] = Ra
    problem.parameters['Pr'] = Pr
    problem.parameters['Ta'] = Ta
    problem.parameters['Lat'] = Lat
    problem.parameters['m'] = m
    problem.parameters['theta'] = theta
    problem.parameters['X'] = Ra/Pr
    problem.parameters['Y'] = (Pr*Pr*theta) / Ra
    problem.parameters['T'] = Ta**(1/2)
    Sfilename='S-Ra:'+str(Ra)+'-Pr:'+str(Pr)+'-Ta:'+str(Ta)+'-Np:'+str(Np)
    KEfilename='KE-Ra:'+str(Ra)+'-Pr:'+str(Pr)+'-Ta:'+str(Ta)+'-Np:'+str(Np)
    # Non-constant coeffiecents
    rho_ref = domain.new_field(name='rho_ref')
    rho_ref['g'] = (1-theta*z)**m
    rho_ref.meta['y']['constant'] = True
    problem.parameters['rho_ref'] = rho_ref         # Background state for rho
    T_ref = domain.new_field(name='T_ref')
    T_ref['g'] = 1-theta*z
    T_ref.meta['y']['constant'] = True
    problem.parameters['T_ref'] = T_ref             # Background state for T
    dz_rho_ref = domain.new_field(name='dz_rho_ref')
    dz_rho_ref['g'] = -theta*m*((1-theta*z)**(m-1))
    dz_rho_ref.meta['y']['constant'] = True
    problem.parameters['dz_rho_ref'] = dz_rho_ref   # z-derivative of rho_ref
    # Defining d/dz of s, u, and w for reducing our equations to first order
    problem.add_equation("sz - dz(s) = 0")
    problem.add_equation("uz - dz(u) = 0")
    problem.add_equation("vz - dz(v) = 0")
    problem.add_equation("wz - dz(w) = 0")
    # mass continuity with rho_ref and dz(rho_ref) expanded analytically
    problem.add_equation("  (1-theta*z)*(dy(v) + wz) - theta*m*w = 0 ")
    # x-component of the momentum equation
    problem.add_equation("  rho_ref*( dt(u) - dy(dy(u)) - dz(uz) + T*(w*cos(Lat) - v*sin(Lat)) ) - dz_rho_ref*uz = -rho_ref*( v*dy(u) + w*uz ) ")
    # y-component of the momentum equation
    problem.add_equation("  rho_ref*( dt(v) - (4/3)*dy(dy(v)) - dz(vz) - (1/3)*dy(wz) + T*u*sin(Lat) ) + dy(p) - dz_rho_ref*(vz + dy(w)) = -rho_ref*( v*dy(v) + w*vz )")
    # z-component of the momentum equation
    problem.add_equation("  rho_ref*T_ref*( dt(w) - X*s - dy(dy(w)) - (4/3)*dz(wz) - (1/3)*dy(vz) - T*u*cos(Lat) ) + T_ref*dz(p) + theta*m*p + (2/3)*theta*m*rho_ref*( 2*wz - dy(v) ) = -rho_ref*T_ref*( v*dy(w) + w*wz )")
    # entropy diffusion equation
    problem.add_equation("  T_ref*( Pr*dt(s) - dy(dy(s)) - dz(sz) ) + theta*(m+1)*sz = -Pr*T_ref*( v*dy(s) + w*sz ) + 2*Y*( dy(v)*dy(v) + wz*wz + vz*dy(w) - (1/3)*(dy(v) + wz)*(dy(v) + wz) + (1/2)*(dy(u)*dy(u) + uz*uz + vz*vz + dy(w)*dy(w)) )")
    # Flux equations for use in analysis outputs
    problem.add_bc("left(w) = 0")            # Impermeable bottom boundary
    problem.add_bc("right(w) = 0", condition="(ny != 0)")   # Impermeable top boundary
    problem.add_bc("right(p) = 0", condition="(ny == 0)")   # Required for equations to be well-posed - see https://bit.ly/2nPVWIg for a related discussion
    problem.add_bc("left(uz) = 0")           # Stress-free bottom boundary
    problem.add_bc("right(uz) = 0")          # Stress-free top boundary
    problem.add_bc("left(vz) = 0")
    problem.add_bc("right(vz) = 0")
    problem.add_bc("right(s) = 0")           # Fixed entropy at upper boundary, arbitarily set to 0
    problem.add_bc("left(sz) = -1")          # Fixed flux at bottom boundary, F = F_cond
    # Build solver
    solver = problem.build_solver(de.timesteppers.RK222)
    logger.info('Solver built')
    # Initial conditions
    x = domain.grid(0)
    z = domain.grid(1)
    s = solver.state['s']
    w = solver.state['w']
    sz = solver.state['sz']
    # Random perturbations, initialized globally for same results in parallel
    gshape = domain.dist.grid_layout.global_shape(scales=1)
    slices = domain.dist.grid_layout.slices(scales=1)
    rand = np.random.RandomState(seed=42)
    noise = rand.standard_normal(gshape)[slices]
    # Linear background + perturbations damped at walls
    zb, zt = z_basis.interval
    pert =  1e-5*noise*(zt - z)*(z - zb)
    s['g'] = pert
    s.differentiate('z', out=sz)
    dt = initial_timestep # Initial timestep
    # Integration parameters --- Note if these are all set to np.inf, simulation will perpetually run.
    solver.stop_sim_time = end_sim_time
    solver.stop_wall_time = end_wall_time
    solver.stop_iteration = end_iterations
    # CFL criterion
    CFL = flow_tools.CFL(solver,initial_dt=dt,cadence=1,safety=1,max_change=1.5,min_change=0.01,max_dt=0.01,min_dt=1e-10)
    CFL.add_velocities(('v', 'w'))
    # Flow properties
    flow = flow_tools.GlobalFlowProperty(solver,cadence=10)
    flow.add_property("sqrt(u*u + v*v + w*w)",name='Re')
    # Analysis tasks
    analysis = solver.evaluator.add_file_handler(save_direc +'analysis',sim_dt=analysis_freq,max_writes=5000)
    analysis.add_task("s", layout='g', name='entropy')
    # Flux decomposition - Internal energy equation
    analysis.add_task("integ( integ( sqrt(u*u + v*v + w*w) , 'y')/Ly, 'z')/Lz", layout='g', name='Re') # Mean Reynolds number
    analysis.add_task(" integ( (integ(0.5*(u*u + v*v + w*w)*rho_ref,'y')/Ly), 'z')/Lz", layout='g', name='KE') # Mean KE
    # Creating a parameter file
    run_parameters = solver.evaluator.add_file_handler(save_direc+'run_parameters',wall_dt=1e20,max_writes=1)
    run_parameters.add_task(Ly,name="Ly")
    run_parameters.add_task(Lz,name="Lz")
    run_parameters.add_task(Ra,name="Ra")
    run_parameters.add_task(Pr,name="Pr")
    run_parameters.add_task(Np,name="Np")
    run_parameters.add_task(m,name="m")
    run_parameters.add_task(Ny,name="Ny")
    run_parameters.add_task(Nz,name="Nz")
    run_parameters.add_task("z",layout='g',name="z_grid")
    run_parameters.add_task(analysis_freq,name="ana_freq")
    run_parameters.add_task(max_dt,name="max_dt")
    try: # Main loop
        logger.info('Starting loop')
        start_time = time.time()
        while solver.ok:
            dt = CFL.compute_dt()
            dt = solver.step(dt)
            time.sleep(0.02)
            if (solver.iteration) == 1:
                # Prints various parameters to terminal upon starting the simulation
                logger.info('Parameter values imported form run_param_file.py:')
                logger.info('Ly = {}, Lz = {}; (Resolution of {},{})'.format(Ly, Lz, Ny, Nz))
                logger.info('Ra = {}, Pr = {}, Np = {}'.format(Ra, Pr, Np))
                logger.info('Analysis files outputted every {}'.format(analysis_freq))
                if end_sim_time != np.inf:
                    logger.info('Simulation finishes at sim_time = {}'.format(end_sim_time))
                elif end_wall_time != np.inf:
                    logger.info('Simulation finishes at wall_time = {}'.format(end_wall_time))
                elif end_iterations != np.inf:
                    logger.info('Simulation finishes at iteration {}'.format(end_iterations))
                else:
                    logger.info('No clear end point defined. Simulation may run perpetually.')
            if (solver.iteration-1) % 10 == 0:
                # Prints progress information include maximum Reynolds number every 10 iterations
                logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
                logger.info('Max Re = %f' %flow.max('Re'))
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        # Prints concluding information upon reaching the end of the simulation.
        end_time = time.time()
        if snaps==True:
            with open(Sfilename,'w') as file:
                    file.write(str(gshape[0])+' '+str(gshape[1]))
        logger.info('Iterations: %i' %solver.iteration)
        logger.info('Sim end time: %f' %solver.sim_time)
        logger.info('Run time: %.2f sec' %(end_time-start_time))
        logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
        with h5py.File("raw_data/analysis/analysis_s1/analysis_s1_p0.h5", mode='r') as file:
            times=file['scales']['sim_time'][:]
            data=file['tasks']['entropy'][:,:,:]
            ke=file['tasks']['KE'][:]
            ke=np.squeeze(ke)
            times=np.squeeze(times)
            n=times.shape[0]
            if snaps==True:
                for i in range(n):
                    if i%snapshot_skip==0:
                        append_unique_value(Sfilename,times[i])
                        append_unique_array(Sfilename,data[i])
            save_fct_txt(times,ke,KEfilename)
