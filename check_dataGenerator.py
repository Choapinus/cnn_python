from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

img = load_img('./images/female/000001.jpg')
x = img_to_array(img)
print(x.shape)
x = x.reshape((1, ) + x.shape)

i = 0

for batch in datagen.flow(x, batch_size=100, save_to_dir='preview', save_prefix='female', save_format='jpeg'):
	i += 1
	if i > 20:
		break