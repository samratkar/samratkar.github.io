---
category: full stack llm app 
subcategory: "virtual environments"
tag : "virtual environments"
layout : mermaid
---

## Creating virtual environment (python's venv module)

1. Go to the folder of your project in the file explorer panel in vscode.
2. Right click and select "Open integrated terminal"
3. In the terminal, run the command:
   ```bash 
   python -m venv .venv
   ```
   The `.venv` folder will be created in your project directory. This folder can be any folder name you choose. dot is given to hide the folder from normal view. `venv` is the python module that creates virtual environments with -m has the command line argument to run the module as a script. 
4. Now, activate the virtual environment using the command:
   - On Windows:    
     ```bash
     .venv\Scripts\activate
     ```
    - On Mac/Linux:
      ```bash
      source .venv/bin/activate
      ```