import os
import subprocess
import sys


def venv_setup():
    try:
        subprocess.run([sys.executable, "-m", "venv", ".sailorpy"], check=True)
        print("Virtual environment created successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to create virtual environment: {e}")
        sys.exit(1)


def get_venv_python():
    if sys.platform == "win32":
        return os.path.join(".sailorpy", "Scripts", "python.exe")
    return os.path.join(".sailorpy", "bin", "python")


def install_dependencies():
    try:
        venv_python = get_venv_python()
        subprocess.run(
            [venv_python, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True,
        )
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        sys.exit(1)


if __name__ == "__main__":
    venv_setup()
    install_dependencies()
    print("\nSetup complete! Activate your env with:")
    print(" $ source .sailorpy/bin/activate   -  macOS/Linux")
    print(" $ .sailorpy\\Scripts\\activate   -  Windows")
