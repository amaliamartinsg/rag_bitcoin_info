import sys
sys.path.append('.')

import uvicorn

if __name__ == "__main__":
    if sys.argv[1:2] == ["--reload"]:
        uvicorn.run("src.app:app", host="0.0.0.0", port=8008, reload=True, env_file='.env')
    else:
        uvicorn.run("src.app:app", host="0.0.0.0", port=8008, reload=False, env_file='.env')