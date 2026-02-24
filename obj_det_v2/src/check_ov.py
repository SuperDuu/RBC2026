import openvino as ov 
try: 
    core = ov.Core() 
    print("\n--- KET QUA KIEM TRA ---") 
    print("? OpenVINO load thanh cong!") 
    devices = core.available_devices 
    print(f"?? Thiet bi kha dung: {devices}") 
    for d in devices: print(f"  - {d}: {core.get_property(d, 'FULL_DEVICE_NAME')}") 
except Exception as e: 
    print(f"\n? Van loi: {e}") 
