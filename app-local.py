# Run the app with no audio file restrictions, and make it available on the network
from app import createUi
createUi(-1, server_name="0.0.0.0")