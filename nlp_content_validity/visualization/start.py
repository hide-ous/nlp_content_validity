import os

import uvicorn

if __name__ == "__main__":
    # for use with docker-compose.yml
    print(os.getcwd())
    print(os.listdir())
    print(os.listdir('..'))
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8002, reload=False)
