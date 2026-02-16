# UFA Analytics API Backend

FastAPI backend for UFA Analytics platform.

## Setup

1. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. **Configure environment variables:**
```bash
cp .env.example .env
# Edit .env with your database credentials
```

4. Run the server:
```bash
uvicorn app.main:app --reload
```

Server runs at: http://localhost:8000

## Environment Variables

Create a `.env` file (see `.env.example` for template):

```
DB_NAME=ufa_analytics
DB_USER=your_username
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
```

**⚠️ NEVER commit `.env` to Git!** It's already in `.gitignore`.

## API Endpoints

- `GET /` - Health check
- `GET /games` - List games (limit=10)
- `GET /games/{game_id}` - Get game details
- `GET /games/{game_id}/events` - Get game events
- `GET /stats/summary` - Database statistics

## API Documentation

FastAPI auto-generates interactive docs:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
