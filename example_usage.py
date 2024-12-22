from PIL import Image
from LaMa_Single import LaMa_Inpainter

def main():
    lama = LaMa_Inpainter()

    image = Image.open("./test_images/image.jpg")
    mask = Image.open("./test_images/mask.png")

    inpainted_image = lama.inpaint(image, mask)

    inpainted_image.save("inpainted.jpg")

if __name__ == '__main__':
    main()
