from PIL import Image
from bin import LaMa_Inpainter

def main():
    lama = LaMa_Inpainter()

    image = Image.open("./test_images/image.jpg")
    mask = Image.open("./test_images/mask.png")

    lama.inpaint(image, mask)

if __name__ == '__main__':
    main()
