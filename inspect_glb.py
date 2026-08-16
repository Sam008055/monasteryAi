import json

file_path = 'd:/Vr-project/scene_conf50.0_blackFalse_whiteFalse_camTrue_skyFalse_max1000k.glb'
with open(file_path, 'rb') as f:
    header = f.read(12)
    magic, version, length = header[:4], int.from_bytes(header[4:8], 'little'), int.from_bytes(header[8:12], 'little')
    print(f"GLB Length: {length/(1024*1024):.2f} MB")
    
    chunk_header = f.read(8)
    chunk_len, chunk_type = int.from_bytes(chunk_header[:4], 'little'), chunk_header[4:8]
    json_data = json.loads(f.read(chunk_len).decode('utf-8'))
    
    print("Nodes:", len(json_data.get("nodes", [])))
    print("Meshes:", len(json_data.get("meshes", [])))
    print("Materials:", len(json_data.get("materials", [])))
    print("Cameras:", len(json_data.get("cameras", [])))
    
    for i, acc in enumerate(json_data.get("accessors", [])):
        if "min" in acc and "max" in acc and acc.get("type") == "VEC3":
            print(f"Accessor {i} bounds: min={acc['min']}, max={acc['max']}")
