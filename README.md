# GALsPeCtrA
This repository is for essential calculations for reconstruction of synthetic galaxy spectra using the Principal Component Analysis (PCA). Mainly, it makes use for the python-FSPS code for generating the simple stellar population (SSP) spectra


# Requirements
numpy==2.4.4   
scipy==1.17.1

# FSPS Installation Guide
This project relies on <a href="https://github.com/cconroy20/fsps">FSPS</a> via its Python interface <a href="https://dfm.io/python-fsps/current/">python-fsps</a>.  
Because FSPS is a complied Fortran library, it must be installed separately before using this project. Please refer to the <a href="https://github.com/cconroy20/fsps">original guideline</a> for the complete installation and references therein.  

1. Install FSPS  
If you have `git` installed, FSPS can be ontained with the following commands:
``` Bash
cd /path/to/desired/location/
git clone https://github.com/cconroy20/fsps
cd fsps
make
```

2. Set environment variable  
You must define the `SPS_HOME` environment variable pointing to the _FSPS_ installation directory:
``` Bash
export SPS_HOME=/path/to/fsps
```
To make this persistent, add it to your shell config:
``` Bash
# For bash
echo 'export SPS_HOME=/path/to/fsps' >> ~/.bashrc  

# For zsh
echo 'export SPS_HOME=/path/to/fsps' >> ~/.zshrc
```

3. Install Python bindings  
``` Bash
pip install fsps
```

4. Verify installation  
``` Python
import fsps
sp = fsps.StellarPopulation()
wave, spec = sp.get_spectrum(tage=1.0)
```  
If this runs without error, the installation is successful.


