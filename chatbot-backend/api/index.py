from app.main import app as fastapi_app

# Vercel Python runtime will look for an ASGI app named `app` in this module.
# We simply re-export the existing FastAPI app from `app.main`.
app = fastapi_app

