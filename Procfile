web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false --server.enableCORS false --server.enableXsrfProtection false --server.enableWebsocketCompression false
api: uvicorn api:app --host 0.0.0.0 --port $API_PORT
