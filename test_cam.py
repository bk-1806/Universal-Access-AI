import cv2

print("--- 📷 CAMERA DIAGNOSTIC TOOL ---")

def test_camera(index):
    print(f"\nTesting Camera Index {index}...")
    # Try different backends (Drivers)
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print(f"❌ Index {index}: Failed to open (Is it used by Zoom/Teams?)")
        return False
    
    ret, frame = cap.read()
    if ret:
        print(f"✅ Index {index}: SUCCESS! Camera is working.")
        print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
        cv2.imshow(f"Test Cam {index}", frame)
        cv2.waitKey(2000) # Show for 2 seconds
        cv2.destroyAllWindows()
        cap.release()
        return True
    else:
        print(f"⚠️ Index {index}: Opened, but returned black/empty frame.")
        cap.release()
        return False

# Test the first 3 possible cameras
found = False
for i in range(3):
    if test_camera(i):
        found = True
        break

if not found:
    print("\n❌ CRITICAL ERROR: No working camera found.")
    print("1. Check if Zoom/Teams is open.")
    print("2. Check if a privacy shutter covers the lens.")
    print("3. Restart your computer to reset the camera driver.")
else:
    print("\n✅ Camera found! The main app should use the index that worked above.")