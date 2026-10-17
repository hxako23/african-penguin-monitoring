# OpenVINO device check.
# Lists every device OpenVINO can use on this computer (for example CPU or GPU),
# with each device's full name. Run it before benchmarking, to see whether
# OpenVINO can reach the Intel integrated GPU or only the CPU.

import openvino as ov

# The Core object is OpenVINO's entry point. It knows which devices it can run on.
core = ov.Core()

# available_devices is a list of short device names, e.g. ['CPU'] or ['CPU', 'GPU']
devices = core.available_devices
print("OpenVINO available devices:", devices)

# Print the full name of each device, e.g. "CPU: Intel(R) Core(TM) Ultra 5 235".
# The full name shows which piece of hardware a short name like "GPU" really is.
for device in devices:
    try:
        full_name = core.get_property(device, "FULL_DEVICE_NAME")
        print(f"  {device}: {full_name}")
    except Exception as e:
        # Some devices cannot report their name; show the error instead of crashing
        print(f"  {device}: (could not query name - {e})")

print()

if "GPU" in devices:
    # OpenVINO can see a GPU. In Ultralytics, use device="intel:gpu"
    # (the "intel:" prefix is needed to send the model to an OpenVINO device).
    print("A GPU device is available to OpenVINO.")
    print("In Ultralytics, select it with device='intel:gpu'.")
else:
    # No GPU is visible to OpenVINO. The usual cause is that the Intel GPU
    # compute runtime is not installed, so OpenVINO cannot talk to the GPU.
    print("No GPU device is available to OpenVINO on this system.")
    print("The most likely cause is a missing Intel GPU compute runtime.")
    print()
    print("On Ubuntu or Debian, it can be installed with:")
    print("    sudo apt update")
    print("    sudo apt install intel-opencl-icd intel-level-zero-gpu level-zero")
    print()
    print("If the GPU is still not listed afterwards, check that your user")
    print("is in the 'render' and 'video' groups:")
    print("    groups $USER")
    print("    sudo usermod -aG render,video $USER")
    print("    # then log out and back in for the change to take effect")