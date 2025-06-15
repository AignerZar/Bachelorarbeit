# Bachelorarbeit
Das folgende Git-Repository wird für die Bachelorarbeit mit dem Titel "Path Integral Monte Carlo Simulations and Optimizations via Variational Autoencoders". Für das Training und die Implementation des VAE wurden die Input Daten eines harmonischen Oszillators erzeugt.

Der folgende Code enthält die benötigten Files und Daten für die Implementation von einem VAE, dabei wurden zunächst die Eingabedaten mit einer PIMC simulation erzeugt diese befinden sich allerdings in dem File input\_data.csv. Dabei handelte es sich bei der Temperatur um 0.5 


## Workflow:

Open the file config.py, in this file the parameters such as number of epochs or the file paths can be adjusted.
Open the file VAE.py and run it, make sure you have the input file and all necessary requirements, see file requirements.txt, some parameters may be adjusted additionally -> see comments in code.

After running the file VAE.py, the latent variables can be seen in file latent\_variables.csv, the generated configurations can be seen in the files generated\_configurations.csv and generated\_configurations\_denorm.csv

In the file plot\_latent\_variables.py the latent space can be generated, the data should roughly follow a normal distribution.

In the file Comparator\_positions.py the position distribution of the input and output data is plotted, may parameters may be adjusted like filepath or so on -> see comments in the code. THe position distribution should roughly follow a normal distribution and the output data should match the input data.

In the file COmparator\_energies.py the energy distributions are plottet of the input and output data, some parameters may be adjusted especially the fit parameters are dependent on the data, for a description of the fit parameters see the comments in the code. 

