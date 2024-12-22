from PIL import Image
from bin import LaMa_Inpainter

def main():
    lama = LaMa_Inpainter()

    image = Image.open("./test_images2/test2.png")
    mask = Image.open("./test_images2/test2_mask001.png")

    lama.inpaint(image, mask)

if __name__ == '__main__':
    main()
