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

:: inkscape 
inkscape input.svg --export-type=png

:: convert ipynb file to markdown
jupyter nbconvert --to markdown pydantic.ipynb --template C:\github\samratkar.github.io\_templates\jekyll.tpl 


:: - postgress backup
cd "C:\Program Files\PostgreSQL\16\bin"
pg_dump -U postgres -d journal -F c -f backup.dump
password - admin
mv -Force backup.dump C:\github\life-journal\jrnls\postgres\
restore - pg_restore -U your_username -d your_database -F c backup.dmp

:: jekyll serve
bundle exec jekyll serve 

:: uv initialization into a folder. first create the folder and then open a terminal in that folder and do and uv init
:: this creates python file, requirement files and all in toml. readme and main.py file. 
uv init

:: creation of .venv folder 
:: all environments are created here. 
:: the latest python is taken up and the env is created using that automatically. 
uv  venv 

:: activate the environment 
uv .venv\Scripts\activate 

:: adding new libraries to the toml file requirement section 
uv add "<library_name>"

:: to run the uv app
uv run 

:: to run an mcp server
:: by running uv run mcp dev you invoke the mcp inspector!
uv run mcp dev server/weather.py 

:: to use the mcp server from claude desktop app uv run mcp install 
uv run mcp install server/weather.py 


