:: commands for conda 
conda create -n agentic_2_base python=3.11
conda activate agentic_2_base
conda env list

:: different path
:: present working directory as path is not given only the folder name is given as bacd001
:: any location path can be given in the command
:: -n is for default location. -p for new location.
conda create -p abcd001 python=3.11
conda activate <path>

:: install from a requirement.txt file
pip install -r requirement.txt 

:: create requirement.txt file 
pip freeze > requirement.txt


:: just pip list
pip list --format=freeze > requirement.txt

:: list all the environments.
conda list

:: remove the environment
conda remove -n agentic_2_base --all 



