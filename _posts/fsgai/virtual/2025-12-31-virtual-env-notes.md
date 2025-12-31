---
category: full stack llm app 
subcategory: "virtual environments"
tag : "virtual environments"
layout : mermaid
---

## 1. Creating virtual environment (python's venv module)
### 1.1 create a virtual environment folder (any name)
1. Go to the folder of your project in the file explorer panel in vscode.
2. Right click and select "Open integrated terminal"
3. In the terminal, run the command:
   ```bash 
   python -m venv .venv
   ```
   The `.venv` folder will be created in your project directory. This folder can be any folder name you choose. dot is given to hide the folder from normal view. `venv` is the python module that creates virtual environments with -m has the command line argument to run the module as a script. 
### 1.2 activate the virtual environment
1. Now, activate the virtual environment using the command:
   - On Windows:    
     ```bash
     .venv\Scripts\activate
     ```
    - On Mac/Linux:
      ```bash
      source .venv/bin/activate
      ```
### 1.3 add requirements file (any name)
- not inside the .venv folder, but in the project folder. Create a text file named `requirements.txt` and list all the packages you want to install in the virtual environment, one per line. For example:
  ```
  numpy
  pandas
  requests
  ```
### 1.4 install packages from requirements file
- In the terminal, run the command:
  ```bash
  pip install -r requirements.txt
  ```
  This command will read the `requirements.txt` file and install all the listed packages into the virtual environment.
### 1.5 Deactivate the virtual environment
- When you are done working in the virtual environment, you can deactivate it by simply running the command:
  ```bash
  deactivate
  ```
  This will return you to the global Python environment.
### 1.6 Reactivate the virtual environment
- Whenever you want to work on your project again, navigate to the project folder in the terminal and reactivate the virtual environment using the activation command mentioned in step 1.2.
### 1.7 Additional tips
- You can create multiple virtual environments for different projects to keep dependencies isolated.
- Always remember to activate the virtual environment before installing new packages or running your project scripts to ensure they use the correct dependencies.
- You can update the `requirements.txt` file anytime you add or remove packages by using the command:
  ```bash
  pip freeze > requirements.txt
  ```
  This will overwrite the existing `requirements.txt` file with the current list of installed packages in the `virtual environment.`


