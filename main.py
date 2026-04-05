from environment_generation import WaterShedFactory, gen_collision_arr
from PIL import Image
from dotenv import load_dotenv
import numpy as np
import os

load_dotenv()

ENVIRONMENT_IMG = os.environ.get("ENVIRONMENT_IMG", "")
assert ENVIRONMENT_IMG

collision_arr = None

def setup_environment():
    global collision_arr
    try:
        img = Image.open(ENVIRONMENT_IMG)
        img_sz = img.size
        pixel_arr = np.array(img)
        collision_arr = gen_collision_arr(pixel_arr)
        print(collision_arr[0,0])
    except Exception as e:
        print(e)
        exit(1)
    #watershed_fact = WaterShedFactory(*img_sz)
    #watershed = watershed_fact.watershed_from_img(bgpic=ENVIRONMENT_IMG)
    #watershed.run()

def main():
    setup_environment()


if __name__ == '__main__':
    main()
