import pyautogui
from PIL import Image, ImageGrab
import time
from numpy import asarray


def hit(key):
    pyautogui.keyDown(key)
    return


def isCollide(data):
    # Draw the rectangle for birds
    for i in range(170, 220):
        for j in range(350, 405):
            if data[i, j] > 100:
                hit("down")
                return

    for i in range(220, 280):
        for j in range(412, 480):
            if data[i, j] > 100:
                hit("up")
                return
    return


if __name__ == "__main__":
    print("Hey.. Dino game about to start in 3 seconds")
    time.sleep(2)
    # hit('up')

    while True:
        image = ImageGrab.grab().convert('L')
        data = image.load()
        isCollide(data)

        # print(asarray(image))

# #   Draw the rectangle for cactus

#         for i in range(190, 240):
#             for j in range(412, 480):
#                 data[i, j] = 200

# # Draw the rectangle for birds
#         for i in range(170, 220):
#             for j in range(350, 405):
#                 data[i, j] = 150

#         image.show()
#         break
