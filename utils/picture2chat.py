import base64
import cv2

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def make_pic_content(text, base64_encode):
    content = [
        {
            "type": "text",
            "text": text
        }
    ]

    if isinstance(base64_encode, str):
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_encode}"
            }
        })
    elif isinstance(base64_encode, (list, tuple)):
        for base64_str in base64_encode:
            if isinstance(base64_str, str):
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_str}"
                    }
                })
            else:
                raise ValueError(f"Invalid base64 string in list: {type(base64_str)}")
    else:
        raise ValueError(f"base64_encode must be string or list, not {type(base64_encode)}")

    return content



def process_image_to_content(img, compression_ratio=1, jpeg_quality=78):

    width = int(img.shape[1] * compression_ratio)
    height = int(img.shape[0] * compression_ratio)
    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    img_bgr = cv2.cvtColor(resized_img, cv2.COLOR_BGRA2BGR)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, buffer = cv2.imencode('.jpg', img_bgr, encode_param)

    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

def process_image_path_to_content(image_path: str):
    """
    Reads an image file, encodes it to base64, and formats it for a multi-modal LLM chat.

    :param image_path: The path to the image file.
    :return: A dictionary formatted for the chat API, or None if the file doesn't exist.
    """
    if not os.path.exists(image_path):
        return None
        
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Determine mime type from file extension
        mime_type = "image/jpeg"
        if image_path.lower().endswith('.png'):
            mime_type = "image/png"

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_image}"
            }
        }
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

