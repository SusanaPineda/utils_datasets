import os, shutil

# Este script contiene herramientas para la gestion de archivos y carpetas

def createFolders(path, delete=False):
    if delete and os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def doNothing(*Arg):
    return True

def getInto(path, destPath, function=doNothing, arg=[]):
    # Navega dentro de las carpetas de forma recursiva, copiando carpetas y archivos.
    # Trata especialmente las imagenes
    [dirpath, dirnames, filenames] = next(os.walk(path, topdown=True))
    for folder in dirnames:
        newDestPath = destPath + "/" + folder
        createFolders(newDestPath)
        getInto(path + "/" + folder, newDestPath, function, arg)
    for file in filenames:
        oFile = path+"/"+file
        dFile = destPath+"/"+file
        shutil.copyfile(oFile, dFile)
        if oFile[-3:] != "txt":
            function(oFile, dFile, arg)