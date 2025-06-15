import os

import uvicorn

if __name__ == "__main__":
    print(os.getcwd())
    print(os.listdir())
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8002, reload=False)
