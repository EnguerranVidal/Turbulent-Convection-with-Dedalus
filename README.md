# Turbulent-Convection-with-Dedalus

## A few words :
During an abroad intership done at the University of Exeter under the supervision of Prof. Matthew Browning, I was tasked to create a code able to model accurately a fluid heated from below and chilled from above in regards to the Boussinesq and Anelastic approximation. Once this code done, I was to use it to determine the effects of rotation and stratification on turbulent convection. All the details on the physics and concepts behind this project can be found on the [Internship Report](https://github.com/EnguerranVidal/Turbulent-Convection-with-Dedalus/blob/main/Internship_Report_2020.pdf) in the main code.

This project was done using the **Dedalus** Python library available for Linux and Mac OS only. It is copyrighted under the MIT License ( for more details see [License.md](https://github.com/EnguerranVidal/Turbulent-Convection-with-Dedalus/blob/main/License.md) ).

## Contents :

This code contains four individual and independant programs :

- **[1_Boussinesq_Regular](https://github.com/EnguerranVidal/Turbulent-Convection-with-Dedalus/tree/main/1_Boussinesq_Regular)**: A regular Boussinesq fluid modelled using regular discretization and Numpy matrices operations for its calculations. The data calculated is collected and stored using **pickle** ( the functions used for the storage abilities are in the **[functions_pickle.py](https://github.com/EnguerranVidal/Turbulent-Convection-with-Dedalus/blob/main/1_Boussinesq_Regular/functions_pickle.py)** file.

- **[2_Boussinesq_Spectral](https://github.com/EnguerranVidal/Turbulent-Convection-with-Dedalus/tree/main/2_Boussinesq_Spectral)**: A regular Boussinesq fluid modelled using Fourier series spectral discretization. The data calculated is collected and stored using **pickle** once again ( see **[functions_pickle.py](https://github.com/EnguerranVidal/Turbulent-Convection-with-Dedalus/blob/main/2_Boussinesq_Spectral/functions_pickle.py)** ).

- **[3_Boussinesq_Dedalus](https://github.com/EnguerranVidal/Turbulent-Convection-with-Dedalus/tree/main/3_Boussinesq_Dedalus)**: A Boussinesq fluid modelled using Fourier series spectral discretization handled by Dedalus. The data calculated is collected and stored using regular .txt files ( see **[functions_txt.py](https://github.com/EnguerranVidal/Turbulent-Convection-with-Dedalus/tree/main/3_Boussinesq_Dedalus/functions_txt.py)** ).

- **[4_Anelastic_Dedalus](https://github.com/EnguerranVidal/Turbulent-Convection-with-Dedalus/tree/main/4_Anelastic_Dedalus)**: An anelastic fluid handled by Dedalus. The data calculated is collected and stored using regular .txt files once again ( see **[functions_txt.py](https://github.com/EnguerranVidal/Turbulent-Convection-with-Dedalus/tree/main/4_Anelastic_Dedalus/functions_txt.py)** ).


The repository also houses other peculiar contents :

- **[ANIMATIONS](https://github.com/EnguerranVidal/Turbulent-Convection-with-Dedalus/tree/main/ANIMATIONS)**: A folder containing a few animations rendered from plots that were compiled by the **imageio** library. **1** wa sdone using the **1_Boussinesq_Regular** code while **2** to **6** were all made using **4_Anelastic_Dedalus**.

- **[Internship_Report_2020.pdf](https://github.com/EnguerranVidal/Turbulent-Convection-with-Dedalus/blob/main/Internship_Report_2020.pdf)** : The pdf report made as a conclusion to this internship.

- **[plot_files.py)](https://github.com/EnguerranVidal/Turbulent-Convection-with-Dedalus/blob/main/plot_files.py)** : .py file containing the functions responsible for the plots containing in the [Internship Report](https://github.com/EnguerranVidal/Turbulent-Convection-with-Dedalus/blob/main/Internship_Report_2020.pdf).


## Dedalus :
Dedalus is an open-source and MPI-parallelized Python library used mostly in Fluid Dynamic as it is incredibly efficient at solving non linear differential equations into closed physical domains with fixed equations and fixed boundary conditions. For more info, go visit their website :  <https://dedalus-project.org/>
