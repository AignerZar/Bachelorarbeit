# Bachelorarbeit
Das folgende Git-Repository wird für die Bachelorarbeit mit dem Titel "Path Integral Monte Carlo Simulations and Optimizations via Variational Autoencoders". 

Der folgende Code enthält die benötigten Files und Daten für die Implementation von einem VAE, dabei wurden zunächst die Eingabedaten mit einer PIMC simulation erzeugt diese befinden sich allerdings in dem File $input_data.csv$.

In der Datei "VAE.py" befindet sich der Code zum Variational-Autoencoder. Dabei wurden zunächst die wichtigsten Libraries importiert. 

Die eingelesenen Daten welche sich in der Datei "input_data.csv" befinden wurden vorher mithilfe einer PIMC Simulation erzeugt, es handelt sich hierbei um die Bead Positions eines H2-Moleküls.

Anschließend wurde ein VAE erstellt, hierfür wurde je für den Encoder und Decoder eine Klasse erstellt, wobei diese in der Klasse VAE dann wieder aufgerufen wurden. Weiters werden mithilfe des Latent Space dann neue Datenpunkte erzeugt welche unter dem Namen "generated_configurations.csv" abgespeichert wurden, ebenso werden die latent_space werte in einer weiteren Datei namens "latent_variables.csv" gespeichert.


In der Datei "plot_generated_examples.py" befindet sich der Code, mit welchem zunächst die einzelnen Daten geplottet werden, dann wird noch der Mittelwert der Positions berechnet und diese geplottet die Verteilungsfunktion sollte einer Normalverteilung entsprechen.

In der Datei "plot_latent_variables.py" befindet sich der Code um den Latent space zu plotten, es wird zunächst ein scatter Plot erstellt und anschließend die Verteilungen der beiden Variablen.
