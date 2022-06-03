import uvicorn

if __name__ == "__main__":
    import os
    port = os.environ.get('FASTAPI_PORT')
    uvicorn.run('main:app', host='0.0.0.0', port=int(port), reload=True)
