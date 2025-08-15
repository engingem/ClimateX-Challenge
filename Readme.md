### Installing Requirements

Required Python version: 3.12

Sync UV virtual environment and install dependencies:
```bash
   uv sync
```
Adding packages to the project:
```bash
   uv add fastapi --extra standard
```

Ruff formating used for this project.

```bash
   ruff format
```


Running project in development mode with reloading:

```bash
   fastapi dev main.py
```

### TESTING

For run the tests, you can use the following command:

```bash
 uv run pytest tests/ -v
```

### Large File Handling

FastAPI supports large file uploads, `File` and `UploadFile` classes can handle file uploads efficiently. 


### Concurrent Use & Performance
- FastAPI async features allow handling multiple requests concurrently.
- Using uvicorn with custom workers can improve performance.
- Use of async database drivers like `asyncpg` for PostgreSQL can enhance performance.


### Efficient Spatial Operations
- Spatial queries can be optimized for in memory dataframe like Polars to process data efficeintly.
- For database query especially with Postgis indexing will improve performance.


### Testing Scalability
- Pytest implementation tests core functionality of script and it can be configured to use sample subset of real data to see performance.
