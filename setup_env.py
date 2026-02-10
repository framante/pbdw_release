import os
import subprocess
import sys
import venv

# Step 1: Set up the venv path
venv_dir = os.path.join(os.getcwd(), "venv")

# Step 2: Create virtual environment
print(f"Creating virtual environment in: {venv_dir}")
builder = venv.EnvBuilder(with_pip=True)
builder.create(venv_dir)

# Step 3: Define pip path
pip_executable = os.path.join(
    venv_dir,
    "Scripts" if os.name == "nt" else "bin",
    "pip"
)

# Step 4: Install requirements if requirements.txt exists
req_file = os.path.join(os.getcwd(), "requirements.txt")
if os.path.exists(req_file):
    print("Installing packages from requirements.txt...")
    subprocess.check_call([pip_executable, "install", "-r", req_file])
else:
    print("No requirements.txt found. Skipping package installation.")

print("Setup complete.")
