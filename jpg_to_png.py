from PIL import Image
import os

def convert_jpg_to_png(file_path):
    image = Image.open(file_path)
    png_file_path = file_path.replace(".jpg", ".png")
    image.save(png_file_path, "PNG")
    os.remove(file_path)
    return png_file_path

dir_path = 'data/ISIC2018/images'
image_list = sorted(os.listdir(dir_path))
for i in range(len(image_list)):
    file_path = os.path.join(dir_path,image_list[i])

    png_file_path = convert_jpg_to_png(file_path)
    print(f"Conversion successful, PNG file saved in:{png_file_path}")