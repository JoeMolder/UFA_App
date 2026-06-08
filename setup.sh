#!/bin/bash
set -e

echo ""
echo "=================================================="
echo "  UFA Analytics — Database Setup"
echo "=================================================="
echo ""

# ── 1. Install PostgreSQL ─────────────────────────────────────────────────────
install_postgres_mac() {
    if ! command -v brew &>/dev/null; then
        echo "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        # Add brew to PATH for Apple Silicon
        eval "$(/opt/homebrew/bin/brew shellenv)" 2>/dev/null || true
    fi

    if ! command -v psql &>/dev/null; then
        echo "Installing PostgreSQL via Homebrew..."
        brew install postgresql@14
        brew link postgresql@14 --force 2>/dev/null || true
    else
        echo "✅ PostgreSQL already installed."
    fi

    echo "Starting PostgreSQL..."
    brew services start postgresql@14 2>/dev/null || true
    sleep 2
}

install_postgres_linux() {
    if ! command -v psql &>/dev/null; then
        echo "Installing PostgreSQL..."
        sudo apt-get update -qq
        sudo apt-get install -y postgresql postgresql-contrib
    else
        echo "✅ PostgreSQL already installed."
    fi

    echo "Starting PostgreSQL..."
    sudo service postgresql start 2>/dev/null || sudo systemctl start postgresql 2>/dev/null || true
    sleep 2
}

OS="$(uname -s)"
case "$OS" in
    Darwin) install_postgres_mac ;;
    Linux)  install_postgres_linux ;;
    *)
        echo "⚠️  Unsupported OS: $OS"
        echo "   Please install PostgreSQL manually: https://www.postgresql.org/download/"
        echo "   Then re-run: python pipeline.py"
        exit 1
        ;;
esac

# ── 2. Install Python dependencies ───────────────────────────────────────────
echo ""
echo "Installing Python dependencies..."
pip install --quiet psycopg2-binary requests tqdm

# ── 3. Run the pipeline ───────────────────────────────────────────────────────
echo ""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/pipeline.py"
