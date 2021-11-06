# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 14:12:45 2021

@author: Administrador
"""

from pdf2image import convert_from_path
import os
import glob
import random
import string

def generateRandomString(n=8):
    
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(n))


os.chdir(r"C:\Users\Administrador\Documents\camara_comercio_pdfs")
pdfs = glob.glob("*.pdf")

for pdf in pdfs:
    randomName = generateRandomString()
    pages = convert_from_path(pdf, 500)
    
    i = 0
    for page in pages:
        page.save(r"C:\Users\Administrador\Documents\output_imagenes_camaras\{}_pag_{}.jpg".format(randomName, i), 'JPEG')
        i += 1
        
        
        
        
import fitz

zoom = 2
mat = fitz.Matrix(zoom, zoom)

pdffile = r"C:\Users\Administrador\Documents\pdf_prueba_camara.pdf"
doc = fitz.open(pdffile)
page = doc.load_page(0)  # number of page
pix = page.get_pixmap(matrix=fitz.Matrix(150/72,150/72))
output = "outfile.png"
pix.save(output)

#orden: izquierda abajo, derecha abajo, derecha arriba, izquierda arriba
vertices =  [
    {
        "x": 2252,
        "y": 850
    },
    {
        "x": 2715,
        "y": 852
    },
    {
        "x": 2715,
        "y": 889
    },
    {
        "x": 2252,
        "y": 887
    }]

def sacarCentroide(vertices):
    x = (vertices[2]["x"] - vertices[0]["x"])/2 + vertices[0]["x"]
    y = (vertices[2]["y"] - vertices[0]["y"])/2 + vertices[0]["y"]
    centroid = (x, y)
    return centroid


#ideas para detectar secciones:
    
"""
con plotear el negro en una grafica bastaría, luego, se cruza con las posibles
secciones previamente mapeadas (según el emisor del documento), 

debe hacerse una función que a cada imagen le detecte y elimine el encabezado
una sección tendrá:
    - dos coordenadas de inicio
    - pagina de inicio
    - dos coordenadas de fin
    - pagina de fin
    - exportar la imagen
    
*una seccion puede empezar en la pag2 y terminar en la 8
en caso de que no funcione, se puede pasar el ocr gratuito y encontrar
aquellas palabras compuestas centradas (por coordenadas), esas serán
otra opcion: un titulo de seccion no tiene nada en la linea de arriba 
ni nada en la de abajo, se puede buscar eso...
el titulo puede ser o no en negrilla
las secciones

"""
#%%
import os

def getListaArchivos(carpeta, extensionSinPunto):
    os.chdir(carpeta)
    return [x for x in glob.glob("*.{}".format(extensionSinPunto))]


#%%
#Copiando a una carpeta solo las que tienen anotaciones
#¿cuáles imagenes tienen anotaciones?
rutaImagenes = r"C:\Users\Administrador\Documents\output_imagenes_camaras"
rutaAnotaciones = r"C:\Users\Administrador\Documents\output_etiquetas_camaras"

#anotaciones
os.chdir(rutaAnotaciones)
listaAnotaciones = [x for x in glob.glob("*.xml")]
listaAnotacionesSinExt = [os.path.splitext(x)[0] for x in listaAnotaciones]

#imgs
os.chdir(rutaImagenes)
listaImagenes = [x for x in glob.glob("*.jpg")]


#imgs con anotaciones
imgsConAnotaciones = [x for x in listaImagenes if os.path.splitext(x)[0] in listaAnotacionesSinExt]

len(listaAnotaciones) == len(imgsConAnotaciones)

#copiando
import shutil
imagenesDestino = r"C:\Users\Administrador\Documents\final\todo"
anotacionesDestino = r"C:\Users\Administrador\Documents\final\todo"

#anotaciones
os.chdir(rutaAnotaciones)
for anotacion in listaAnotaciones:
    shutil.copy(anotacion, anotacionesDestino)
    
#imagenes
os.chdir(rutaImagenes)
for img in imgsConAnotaciones:
    shutil.copy(img, imagenesDestino)
    
    
#%%
#Leer una carpeta con imagenes y anotaciones y dividirla en train y test
cuantoDarleATest = 0.1


imgsYAnotaciones = r"C:\Users\Administrador\Documents\final\todo"

nroImagenes = getListaArchivos(imgsYAnotaciones, "jpg")
nombres = [os.path.splitext(x)[0] for x in nroImagenes]

#Escogiendo el X% de nombres al azar
nroAEscoger = int(cuantoDarleATest * len(nroImagenes))

import random
listaTest = random.sample(nombres, nroAEscoger) #lista de test
listaTrain = [x for x in nombres if x not in listaTest]

#Ahora, debo copiar la lista de train y la lista de test, anotacion e img:
os.chdir(imgsYAnotaciones)

#crear train y test. si existe los elimina
if os.path.exists("train"):
    os.removedirs("train")
else:
    #si no existe, crear
    os.mkdir("train")
if os.path.exists("test"):
    os.removedirs("test")
else:
    #si no existe, crear
    os.mkdir("test")
    
#copiando a train
for elemento in listaTrain:
    #copiar imagen
    elementoImagen = elemento + ".jpg"
    shutil.copy(elementoImagen, "train")
    
    #copiar anotacion
    elementoAnotacion = elemento + ".xml"
    shutil.copy(elementoAnotacion, "train")
    

#copiando a test    
for elemento in listaTest:
    #copiar imagen
    elementoImagen = elemento + ".jpg"
    shutil.copy(elementoImagen, "test")
    
    #copiar anotacion
    elementoAnotacion = elemento + ".xml"
    shutil.copy(elementoAnotacion, "test")    

#%%
#Para convertir xmls de pascal voc a 2 archivos .csv
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


# taken from https://github.com/datitran/raccoon_dataset

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main(): # this is the part we change to work with our setup
    for directory in ['train','test']:
        image_path = os.path.join(os.getcwd(), 'todo/{}'.format(directory))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv('data/{}_labels.csv'.format(directory), index=None)
        print('Successfully converted xml to csv.')

#pathhome tiene una carpeta que se llama todo, todo tiene a train y a test
#pathome tiene una carpeta que se llama data, donde se guardará el csv
pathHome = r"C:\Users\Administrador\Documents\final"
os.chdir(pathHome)
main()


#%%
#Usar tf generate record
#se le debe poner ahi el nombre de la clase
#tuve que mover data dentro de 'todo' y el tf generate también


#%%

import matplotlib.pyplot as plt
import numpy as np

def mostrarImagen(image, nombreVentana="Imagen"):
    cv2.imshow(nombreVentana, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Convertir a blanco y negro
import cv2
img_array = cv2.imread(r"C:\Users\Administrador\Documents\output_imagenes\pagina_0.jpg")
grayImage = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
mostrarImagen(img_array)


#Código para detectar dónde está el eje X (la línea del eje X)

#La mitad de la gráfica para abajo (de 187 a 374)
end = int(img_array.shape[0] - img_array.shape[0]*0.05)
height = img_array.shape[0]
half_height = int(3*height/4)

inicio = 2244
final = 2556
means = []
file = []
for i in range(inicio, final):
    calc = np.mean(img_array[i,:])
    means.append(calc)
    file.append(i)
    

dict = {}
for key, value in zip(means, file):
    dict[key] = value
    
print(len(means))
print(len(file))
linea = min(means)
print(linea)


plt.plot(file, means)
plt.xlabel('Número de fila')
plt.ylabel('Promedio de los valores de esa fila')


plt.show()
#