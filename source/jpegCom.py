from PIL import Image

# Change the img path
filePath = '/Users/kevin/Desktop/exp-5-21/country.tif'
desDir = '/Users/kevin/Desktop/exp-5-21/country_jpeg/'

m = Image.open(filePath)

for a in range(100):
	m.save(desDir+str(a)+'.jpeg', 'jpeg', quality = 100-a)
