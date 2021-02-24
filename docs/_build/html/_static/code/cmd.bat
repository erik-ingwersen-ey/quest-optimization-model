conda activate base

~  conda info --envs                               ok | base py | at 00:09:21 
# conda environments:
#
EY-Quest-Diagnostics        /Users/EY/.conda/envs/EY-Quest-Diagnostics
                            /Users/EY/opt/anaconda3
base                  *     /opt/anaconda3

~  cd /Users/EY/Desktop/EY-Quest-Diagnostics
~/De/EY-Quest-Diagnostics ls
build                 docs                  optimization.egg-info
dist                  optimization          setup.py

~/De/EY-Quest-Diagnostics pip install -e .

Obtaining file:///Users/EY/Desktop/EY-Quest-Diagnostics
Installing collected packages: optimization
    Attempting uninstall: optimization
        Found existing installation: optimization 0.1.3
        Uninstalling optimization-0.1.3:
        Successfully uninstalled optimization-0.1.3
    Running setup.py develop for optimization
Successfully installed optimization