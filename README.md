# Decomposition of the vibrational heat capacity

This repository is focused on the heat capacity decomposition in solid compounds, such as [ferroic crystals](https://en.wikipedia.org/wiki/Ferroics "Ferroics").

The Python language is used as a computational tool to estimate parameters of the external (lattice) vibrations with the help of 
nonlinear regression approach, along with statistical methods in order to estimate errors. 
Real experimental data is used for two different solids, tri-rubidium deuterium disulfate
Rb<sub>3</sub>D(SO<sub>4</sub>)<sub>2</sub> and lead germanate Pb<sub>5</sub>Ge<sub>3</sub>O<sub>11</sub>.
A full decomposition of the temperature dependence of vibrational heat capacity at constant pressure in wide temperature range was made 
in order to obtain the excess heat capacity. This procedure affords studying of 2nd-order phase transitions that occur in both compounds.

The environment for the code is crucial (especially about lmfit package version), specified here:

> Windows 7 SP1 x64, Python 3.3.0

> Packages:
> * numpy 1.10.1
* matplotlib 1.5.0
* scipy 0.16.1
* lmfit 0.9.2 ('minimize' function works differently than in v.0.8.x!)
