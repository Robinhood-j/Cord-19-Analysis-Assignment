#!/usr/bin/env python3
"""
CORD-19 Project Setup Script
Automates the setup process for the CORD-19 data analysis project
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

def print_header():
    """Print setup header"""
    print("="*60)
    print("🦠 CORD-19 Data Analysis Project Setup")
    print("="*60)
    print("Setting up your environment for COVID-19 research analysis...")
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7+ is required")
        print(f"Current version: {version.major}.{version.minor}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor} is compatible")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    
    requirements = [
        'pandas>=1.5.0',
        'numpy>=1.21.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'streamlit>=1.28.0',
        'plotly>=5.15.0',
        'wordcloud>=1.9.0',
        'scipy>=1.9.0'
    ]
    
    try:
        for package in requirements:
            print(f"Installing {package.split('>=')[0]}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                package, "--quiet"
            ])
        
        print("✅ All packages installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        print("Try running manually: pip install -r requirements.txt")
        return False

def create_directory_structure():
    """Create necessary directories"""
    print("\n📁 Creating directory structure...")
    
    directories = [
        'data',
        'results',
        'results/visualizations'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}/")
    
    print("✅ Directory structure created!")

def create_requirements_file():
    """Create requirements.txt file"""
    print("\n📝 Creating requirements.txt...")
    
    requirements_content = """pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
streamlit>=1.28.0
plotly>=5.15.0
wordcloud>=1.9.0
scipy>=1.9.0
requests>=2.28.0
beautifulsoup4>=4.11.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements_content)
    
    print("✅ requirements.txt created!")

def download_sample_data():
    """Ask user if they want to download sample data"""
    print("\n📊 Data Setup Options:")
    print("1. Use sample data (quick start)")
    print("2. Download real CORD-19 data from Kaggle (manual)")
    print("3. Skip data setup")
    
    choice = input("Choose option (1-3): ").strip()
    
    if choice == "1":
        print("✅ Sample data will be generated automatically when you run the analysis")
        return True
    elif choice == "2":
        print("\n📥 To download the real CORD-19 data:")
        print("1. Visit: https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge")
        print("2. Download 'metadata.csv'")
        print("3. Place it in the project root directory")
        print("4. Rerun the analysis script")
        return True
    else:
        print("⏭️ Skipping data setup")
        return True

def create_run_scripts():
    """Create convenience run scripts"""
    print("\n🚀 Creating run scripts...")
    
    # Analysis script runner
    analysis_script = """#!/usr/bin/env python3
# Quick Analysis Runner
import sys
import subprocess

if __name__ == "__main__":
    try:
        subprocess.run([sys.executable, "cord19_analysis.py"], check=True)
    except FileNotFoundError:
        print("Error: cord19_analysis.py not found!")
        print("Make sure you have all project files in the current directory.")
    except Exception as e:
        print(f"Error running analysis: {e}")
"""
    
    with open('run_analysis.py', 'w') as f:
        f.write(analysis_script)
    
    # Streamlit app runner
    streamlit_script = """#!/usr/bin/env python3
# Quick Streamlit Runner
import sys
import subprocess

if __name__ == "__main__":
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"], check=True)
    except FileNotFoundError:
        print("Error: streamlit_app.py not found!")
        print("Make sure you have all project files in the current directory.")
    except Exception as e:
        print(f"Error running Streamlit app: {e}")
        print("Try running manually: streamlit run streamlit_app.py")
"""
    
    with open('run_streamlit.py', 'w') as f:
        f.write(streamlit_script)
    
    print("✅ Run scripts created!")
    print("  - run_analysis.py: Quick analysis runner")
    print("  - run_streamlit.py: Quick Streamlit app launcher")

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("🎉 Setup Complete! Next Steps:")
    print("="*60)
    print()
    print("1️⃣ Run the data analysis:")
    print("   python cord19_analysis.py")
    print("   # OR")
    print("   python run_analysis.py")
    print()
    print("2️⃣ Launch the interactive web app:")
    print("   streamlit run streamlit_app.py")
    print("   # OR") 
    print("   python run_streamlit.py")
    print()
    print("3️⃣ Create your GitHub repository:")
    print("   git init")
    print("   git add .")
    print("   git commit -m 'Initial CORD-19 analysis project'")
    print("   git remote add origin <your-repo-url>")
    print("   git push -u origin main")
    print()
    print("📚 For help, check README.md or the project documentation")
    print("🐛 Report issues: Open a GitHub issue")
    print("💡 Questions? Contact your course instructor")
    print()
    print("🚀 Happy analyzing! Good luck with your journey!")

def main():
    """Main setup function"""
    print_header()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directory structure
    create_directory_structure()
    
    # Create requirements file
    create_requirements_file()
    
    # Install packages
    install_success = install_requirements()
    
    # Setup data
    download_sample_data()
    
    # Create run scripts
    create_run_scripts()
    
    # Print next steps
    print_next_steps()
    
    if not install_success:
        print("\n⚠️ Some packages failed to install. Please run:")
        print("pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()