#!/usr/bin/env python3
"""
Enhanced Memory Agent Setup Script
Installs dependencies and sets up Ollama models for the enhanced memory system
"""

import subprocess
import sys
import os
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_ollama_installed():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def install_ollama():
    """Install Ollama based on the platform"""
    system = platform.system().lower()
    
    if system == "darwin":  # macOS
        print("🍎 Installing Ollama on macOS...")
        return run_command("brew install ollama", "Installing Ollama via Homebrew")
    
    elif system == "linux":
        print("🐧 Installing Ollama on Linux...")
        install_script = "curl -fsSL https://ollama.ai/install.sh | sh"
        return run_command(install_script, "Installing Ollama via install script")
    
    elif system == "windows":
        print("🪟 Installing Ollama on Windows...")
        print("Please download and install Ollama from: https://ollama.ai/download")
        print("After installation, restart your terminal and run this script again.")
        return False
    
    else:
        print(f"❌ Unsupported operating system: {system}")
        return False

def start_ollama_service():
    """Start Ollama service"""
    system = platform.system().lower()
    
    if system == "darwin":  # macOS
        return run_command("brew services start ollama", "Starting Ollama service")
    else:
        # For Linux and other systems, try to start the service
        return run_command("ollama serve", "Starting Ollama service")

def pull_ollama_models():
    """Pull required Ollama models"""
    models = [
        "llama2:7b",  # Main LLM model
        "nomic-embed-text"  # Embedding model
    ]
    
    for model in models:
        if not run_command(f"ollama pull {model}", f"Pulling {model}"):
            print(f"⚠️  Warning: Failed to pull {model}. You can try again later.")
            continue

def install_python_dependencies():
    """Install Python dependencies"""
    return run_command("pip install -r requirements.txt", "Installing Python dependencies")

def create_directories():
    """Create necessary directories"""
    directories = [
        "backend/memory/chroma_data",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def test_installation():
    """Test the installation"""
    print("\n🧪 Testing installation...")
    
    try:
        # Test imports
        import chromadb
        import sentence_transformers
        import streamlit
        print("✅ All Python packages imported successfully")
        
        # Test Ollama connection
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama service is running")
        else:
            print("⚠️  Ollama service might not be running")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Ollama connection error: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Enhanced Memory Agent Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Step 1: Install Ollama if not present
    if not check_ollama_installed():
        print("📦 Ollama not found. Installing...")
        if not install_ollama():
            print("❌ Failed to install Ollama. Please install manually from https://ollama.ai")
            sys.exit(1)
    else:
        print("✅ Ollama is already installed")
    
    # Step 2: Start Ollama service
    if not start_ollama_service():
        print("⚠️  Warning: Could not start Ollama service. You may need to start it manually.")
    
    # Step 3: Pull Ollama models
    print("\n📥 Pulling Ollama models (this may take a while)...")
    pull_ollama_models()
    
    # Step 4: Install Python dependencies
    print("\n📦 Installing Python dependencies...")
    if not install_python_dependencies():
        print("❌ Failed to install Python dependencies")
        sys.exit(1)
    
    # Step 5: Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Step 6: Test installation
    print("\n🧪 Testing installation...")
    if test_installation():
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Run the enhanced test: python test_enhanced_memory.py")
        print("2. Start the web interface: cd frontend && streamlit run app.py")
        print("3. Check the README.md for usage instructions")
    else:
        print("\n❌ Setup completed with errors. Please check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
