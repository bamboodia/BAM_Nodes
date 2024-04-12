import os
import random
import numpy as np
from PIL import Image
import nodes
import comfy.utils
import hashlib

def p(image):
    return image.permute([0,3,1,2])
def pb(image):
    return image.permute([0,2,3,1])

class BAM_GetShortestSide:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"Image": ("IMAGE",)}}
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("Shortest Side",)
    FUNCTION = "get_shortest_side_rounded"

    CATEGORY = "BAM Nodes"

    def get_shortest_side_rounded(self, Image):
        old_width = Image.shape[2]
        old_height = Image.shape[1]
        old_shortside = min(old_width, old_height)
        # Round down to the closest multiple of 64
        rounded_shortside = (old_shortside // 64) * 64
        print("The shortest side of the image, rounded down to the closest multiple of 8, is:" ,rounded_shortside, "pixels")
        return (rounded_shortside,)

class BAM_CropToRatio:
    aspects = ["1:1","5:4","4:3","3:2","16:10","16:9","21:9","2:1","3:1","4:1"]
    direction = ["landscape","portrait"]
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"Image": ("IMAGE",),
                             "ratio": (s.aspects,),
                             "position": (["top-left", "top-center", "top-right", "right-center", "bottom-right", "bottom-center", "bottom-left", "left-center", "center"],),
                             "direction": (s.direction,)}}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Cropped Image",)
    FUNCTION = "crop_to_ratio"

    CATEGORY = "BAM Nodes"

    def crop_to_ratio(self, Image, ratio, direction, position):
        _, oh, ow, _ = Image.shape
        x, y = ratio.split(':')
        x = int(x)
        y = int(y)
        old_ratio = ow / oh
        new_ratio = x / y
        new_width = ow
        new_height = oh

        if direction == "portrait" and new_ratio > 1:
            new_ratio = 1 / new_ratio
        elif direction == "landscape" and new_ratio < 1:
            new_ratio = 1 / new_ratio

        if old_ratio > new_ratio:
            # Crop the width
            new_width = int(oh * new_ratio)
            crop = int((ow - new_width) / 2)
        else:
            # Crop the height
            new_height = int(ow / new_ratio)
            crop = int((oh - new_height) / 2)

        if "center" in position:
            x = round((ow - new_width) / 2)
            y = round((oh - new_height) / 2)
        if "top" in position:
            y = 0
        if "bottom" in position:
            y = oh - new_height
        if "left" in position:
            x = 0
        if "right" in position:
            x = ow - new_width

        new_height = new_height - (new_height % 8)
        new_width = new_width - (new_width % 8)    

        Image = Image[:, y:y+new_height, x:x+new_width]
        return (Image,)
    
from nodes import EmptyLatentImage

class BAM_EmptyLatentImageByRatio:
    @classmethod
    def INPUT_TYPES(cls):  
        return {
        "required": {
            "ratio": (
            [   '5:4',
                '4:3',
                '3:2',
                '16:9',
                '1:1',
            ],
            {
            "default": '1:1'
             }),
            "model": (
            [   'SD1.5',
                'SDXL',
            ],
            {
            "default": 'SD1.5'
             }),
            "orientation": (
            [   'portrait',
                'landscape',
            ],
            {
            "default": 'portrait'
             }),            
            "batch_size": ("INT", {
            "default": 1,
            "min": 1,
            "max": 4096
            }),
        },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("Latent", "Width", "Height")
    FUNCTION = "generate"
    CATEGORY = "BAM Nodes"

    def generate(self, ratio, model, orientation, batch_size):
        base_size = 512 if model == 'SD1.5' else 1024
        x, y = map(int, ratio.split(':'))

        long_side, short_side = max(x, y), min(x, y)

        # Adjust base size based on ratio
        base_size *= (short_side + long_side) / (2.0 * long_side)

        # Calculate dimensions based on orientation
        if orientation == 'portrait':
            height = round(base_size * long_side / short_side / 64) * 64
            width = round(height * short_side / long_side / 64) * 64
        else:  # landscape
            width = round(base_size * long_side / short_side / 64) * 64
            height = round(width * short_side / long_side / 64) * 64

        latent = EmptyLatentImage().generate(int(width), int(height), batch_size)[0]

        return latent, int(width), int(height)
    
class BAM_RandomImageFromFolder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "load_images"

    CATEGORY = "BAM Nodes"

    @classmethod
    def IS_CHANGED(cls):
        return float("NaN")  # Always return NaN to load a new image every time

    def load_images(self, directory: str):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' cannot be found.")
        dir_files = os.listdir(directory)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        # Filter files by extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        if not dir_files:
            raise FileNotFoundError(f"No valid image files in directory '{directory}'.")

        # Select a random image file
        file = random.choice(dir_files)
        image_path = os.path.join(directory, file)

        try:
            image = nodes.LoadImage().load_image(image_path)
        except OSError as e:
            if "image file is truncated" in str(e):
                print(f"Error: Image file '{image_path}' is truncated.")
                dir_files.remove(file)
            else:
                raise e  

        return image

class BAM_Random_Float:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "granularity":(
                [   0.5, 0.2, 0.1, 0.05, 0.02, 0.01
                ],
                {
                "default": 0.1
                 }),
                "minimum": ("FLOAT", {"default": 0, "min": -18446744073709551615, "max": 18446744073709551615, "step": 0.01}),
                "maximum": ("FLOAT", {"default": 0, "min": -18446744073709551615, "max": 18446744073709551615, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "return_randm_number"

    CATEGORY = "BAM Nodes"

    def return_randm_number(self, minimum, maximum, seed, granularity):
        # Set Generator Seed
        random.seed(seed)

        # Calculate the number of steps between the minimum and maximum
        num_steps = int((maximum - minimum) / granularity)
        # Generate a random integer between 0 and num_steps, then multiply by the step size and add the minimum
        number = random.randint(0, num_steps) * granularity + minimum

        # Return number
        return (round(number, 2),)

    @classmethod
    def IS_CHANGED(cls, seed,):
        m = hashlib.sha256()
        m.update(seed)
        return m.digest().hex()
    
class BAM_OnOff:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "on": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "return_on_off"

    CATEGORY = "BAM Nodes"

    def return_on_off(self, on):
        return (1,) if on else (0,)

    @classmethod
    def IS_CHANGED(cls, on,):
        return on    

NODE_CLASS_MAPPINGS = {
    "BAM Get Shortest Side": BAM_GetShortestSide,
    "BAM Crop To Ratio": BAM_CropToRatio,
    "BAM Empty Latent By Ratio": BAM_EmptyLatentImageByRatio,
    "BAM Random Image From Folder": BAM_RandomImageFromFolder,
    "BAM Random Float": BAM_Random_Float,
    "BAM OnOff INT": BAM_OnOff,
}