from myUtils.FoldersManager import *
from myUtils.ImageProcessing import *

# Este script permite recorrer las carpetas y archivos de un directorio y los copia en otro, realizando un aumentado de datos.
# El aumentado de datos como parametros tiene:
# - Factor de aumentado
# - Lista de efectos a realizar
# Por cada imagen realizará aleatoriamente "factor de aumentado" veces uno de los procesos de la lista de efectos.

if __name__ == '__main__':
    path = "C:/Users/TTe_J/Downloads/tiny-imagenet-200/train"
    destPath = "C:/Users/TTe_J/Downloads/tiny-imagenet-200/train_aug"

    # CARACTERISTICAS DISPONIBLES
    # filtros = {"blur", "resize", "transpose_rotate"}
    # tipos = {"rangeInt", "rangeDouble", "list": rFilterList}
    # rotacion/transposicion = [FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM, ROTATE_90, ROTATE_180, ROTATE_270, TRANSPOSE] Image.method

    # [factor de aumentado, {'filtro': (tipo, [rango/lista])}]
    augFactor = 5
    filters = [augFactor, {"blur": ("rangeInt", [2,10]), "resize": ("rangeDouble", [0.15,2.5]),
                           "transpose_rotate": ("list", [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_90,
                                                         Image.ROTATE_180, Image.ROTATE_270, Image.TRANSPOSE])}]

    createFolders(destPath, True) # Crea el root
    getInto(path, destPath, imgFilters, filters)
