import base64
import io
from PIL import Image

 
def resize_base64_image(base64_string, size=(128, 128)):
    """
    Resize an image encoded as a Base64 string.
    Args:
    base64_string (str): Base64 string of the original image.
    size (tuple): Desired size of the image as (width, height).
    Returns:
    str: Base64 string of the resized image.
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)
    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)
    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def is_base64(s):
    """Check if a string is Base64 encoded"""
    try:
        return base64.b64encode(base64.b64decode(s)) == s.encode()
    except Exception:
        return False


def split_image_text_types(docs):
    """Split numpy array images and texts"""
    images = []
    text = []
    for doc in docs:
        doc = doc.page_content  # Extract Document contents
        if is_base64(doc):
            print('hmm', doc)
            # Resize image to avoid OAI server error
            images.append(
                resize_base64_image(doc, size=(250, 250))
            )  # base64 encoded str
        else:
            print('meh', doc)
            text.append(doc)
    print('lenimg', len(images), 'lentxt', len(text))
    return {"images": images, "texts": text}