# Facial Recognition Setup (DeepFace)

## Prequisites
Before starting, ensure you have Python 3.8+ installed on your system.
###  Environment Setup
#### Install Required Libraries
 - Download and install Miniforge (if you already have miniforge installed and your install is failing, remove the whole `~/miniforge3/` folder, then try installing again):

 #### For MacOS

 ```
 wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
 chmod +x ./Miniforge3-MacOSX-arm64.sh
sh ./Miniforge3-MacOSX-arm64.sh
 ```
 - Activate conda enviornment with Python 3.9
 ```
source ~/miniforge3/bin/activate
conda create -n myenv python=3.9
conda activate myenv
```
 - Install these dependencies (MacOS)
```
conda install -c apple tensorflow-deps
pip install tensorflow-macos==2.16.2
pip install tensorflow-metal==1.2.0

```
#### For Linux
```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
chmod +x ./Miniforge3-Linux-x86_64.sh
sh ./Miniforge3-Linix-x86_64.sh
```
 - Activate conda enviornment with Python 3.9
```
source ~/miniforge3/bin/activate
conda create -n myenv python=3.9
conda activate myenv
```
- Install these dependencies (Linux)
```
pip install tensorflow[and-cuda]==2.16.2

```
#### Next steps

- Install required python modules

```
pip install -r requirements.txt
```

## Database Setup
1. Create a folder named my_db in the same directory as emotion.py
2. Place high-quality, frontal images of the people you want to recognize inside this folder.
3. Naming Convention: Name the image files exactly as you want the identity to appear (e.g., Amitabh_Bachchan.jpg will display as "Amitabh Bachchan").


## Project Structure
Create the following file and folder structure in your project directory:

````
/Your_Project_Folder
|-- .venv/                   <-- Virtual environment folder
|-- Deepface_Recognition.py  <-- The main recognition script
|-- my_db/                   <-- **CRITICAL: Your Face Database**
|   |-- John_Doe.jpg
|   |-- Jane_Smith.png
|   |-- ... etc.
````

## Running the Application
1. Ensure you have completed steps 1, 2, and 3 (database creation and code saving).

2. Make sure your virtual environment is active.

3. Execute the script from your terminal:

```
python Deepface_Recognition.py
```
## Troubleshooting (macOS Specific)

If the camera fails to initialize, follow these steps to force authorization:

1. Stop the script.

2. Reset Camera Permissions: Open your standard macOS Terminal (not inside VS Code) and run:
```
tccutil reset Camera
```
3.  **Run the script again.** macOS should prompt you to grant camera access to your Terminal/VS Code. **You must click 'Allow'.**

4. If the camera opens but immediately closes, it means the stream is unstable. Try changing the ` CAMERA_INDEX ` (Line 11 in the code) to ` 1 ` (for an external camera) and ensure the ` time.sleep(2) ` delay remains in the code.
