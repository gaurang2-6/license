import sys
print('Python:', sys.version)

pkgs = ['cv2', 'numpy', 'torch', 'torchvision', 'ultralytics', 'pytesseract', 'skimage', 'kagglehub', 'PIL']
for pkg in pkgs:
    try:
        mod = __import__(pkg)
        print(f"{pkg}: {getattr(mod, '__version__', 'installed')}")
    except Exception as e:
        print(f"{pkg}: NOT INSTALLED ({e})")
