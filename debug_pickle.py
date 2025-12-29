import pickle
import sys

print("Opening file...", flush=True)
try:
    with open('disease_prediction_model.pkl', 'rb') as f:
        print("Loading pickle...", flush=True)
        data = pickle.load(f)
        print("Pickle loaded.", flush=True)
        if hasattr(data, 'keys'):
            print(f"Keys: {data.keys()}", flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)
    import traceback
    traceback.print_exc()
