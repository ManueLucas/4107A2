# Define the virtual environment name
$venvName = "4107A2G69"

# Create the virtual environment
python -m venv $venvName

# Activate the virtual environment
$venvPath = "./$venvName/Scripts/Activate"
. $venvPath

# Install dependencies from requirements.txt
pip install -r requirements.txt
