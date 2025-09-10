from app import app


if __name__ == '__main__':
	import uvicorn

	# Run the FastAPI app in this module (so uvicorn can also import main:app)
	uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
