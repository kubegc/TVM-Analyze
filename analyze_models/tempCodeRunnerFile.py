device={}
with open(os.path.join(program_path,"../create_dataset/devices.json") ,'r') as f:
    device = json.load(f)