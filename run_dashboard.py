# QUICK START SCRIPT
"""
Quick start script untuk menjalankan Sales Analytics Dashboard
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'matplotlib', 
        'seaborn', 'plotly', 'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_requirements():
    """Install required packages"""
    print("🔧 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages. Please install manually:")
        print("pip install -r requirements.txt")
        return False

def check_data_file():
    """Check if data file exists"""
    data_file = "sales_visits_finalbgt_enriched.csv"
    if os.path.exists(data_file):
        print(f"✅ Data file '{data_file}' found!")
        return True
    else:
        print(f"❌ Data file '{data_file}' not found!")
        print("Please make sure the CSV file is in the same directory as this script.")
        return False

def run_dashboard():
    """Run the Streamlit dashboard"""
    print("🚀 Starting Sales Analytics Dashboard...")
    print("📊 Dashboard will open in your default browser")
    print("🔗 URL: http://localhost:8501")
    print("\n⏹️  Press Ctrl+C to stop the dashboard")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard_main.py"])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped. Goodbye!")
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install it first:")
        print("pip install streamlit")

def main():
    """Main function"""
    print("=" * 60)
    print("📊 SALES ANALYTICS DASHBOARD - QUICK START")
    print("=" * 60)
    
    # Check data file
    if not check_data_file():
        return
    
    # Check requirements
    missing = check_requirements()
    if missing:
        print(f"⚠️  Missing packages: {', '.join(missing)}")
        install_choice = input("🤔 Do you want to install them now? (y/n): ")
        if install_choice.lower() == 'y':
            if not install_requirements():
                return
        else:
            print("❌ Cannot proceed without required packages.")
            return
    else:
        print("✅ All required packages are installed!")
    
    # Run dashboard
    print("\n" + "=" * 60)
    run_dashboard()

if __name__ == "__main__":
    main()
