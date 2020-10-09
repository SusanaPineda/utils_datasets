import random
from PIL import ImageFilter, Image

# Este script contiene útiles para el tratamiento o procesado de imágenes.

def imgFilters(oFile, dFile, filters):
    # Aplica x (factores de aumento) veces distintos filtros a una imagen y la guarda en el directorio especificado,
    # agregando -_x.y- donde 'x' es el numero del aumento e 'y' la extension del archivo

    # filters = [factor de aumentado, {'filtro': (tipo, [rango/lista])}] tipo indica si es un rango o una lista de
    # opciones

    __filters = {"blur": filterBlur, "resize": filterResize, "transpose_rotate": filterTransposeRotate}
    __type = {"rangeInt": rFilterRangeInt, "rangeDouble": rFilterRangeDouble, "list": rFilterList}

    img = Image.open(oFile)
    fKeys = [f for f in filters[1].keys()]
    dFile = dFile.split('.')
    for i in range(filters[0]):
        # Random filter de los pasados como aplicables
        rFilter = fKeys[random.randint(0, len(fKeys)-1)]
        # Factor a aplicar en el filtro
        features = filters[1][rFilter]
        img = __filters[rFilter](img, __type[features[0]](features[1]))
        img.save(dFile[0]+'_'+str(i)+'.'+dFile[1])


def rFilterRangeInt(range):
    return random.randint(range[0], range[1])

def rFilterRangeDouble(range):
    return random.uniform(range[0], range[1])

def rFilterList(list):
    return list[random.randint(0, len(list)-1)]


def filterBlur(img, factor):
    return img.filter(ImageFilter.GaussianBlur(factor))

def filterResize(img, factor):
    [w,h] = img.size
    return img.resize((int(w * factor)+1, int(h * factor)+1))

def filterTransposeRotate(img, factor):
    return img.transpose(factor)