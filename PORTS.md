# Port Allocation

This project uses the following ports:

- **Frontend**: 9000 (Vite dev server)
- **Backend**: 9001 (FastAPI)

## Reserved Ranges
- 9000-9999: Application ports
- 8000-8999: Reserved for system services

## Usage
```bash
# Start backend
uvicorn backend.main:app --port 9001

# Start frontend
npm run dev # Will use 9000 from vite.config.ts
```
