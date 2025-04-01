import cv2
import numpy as np
import pyrealsense2 as rs
import time

from FastSAM.fastsam import FastSAM, FastSAMPrompt 
from PIL import Image
import os
import argparse
from FastSAM.utils.tools import convert_box_xywh_to_xyxy

# from ultralytics import FastSAM
# from ultralytics.models.fastsam import FastSAMPrompt

def create_mask():
    points = []
    mask_path = './mask.png'

    def select_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(image_display, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow("Image", image_display)

    def generate_mask(image, points):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        points_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points_array], 255)
        return mask

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    try:
        # Wait for 1 second to allow the camera to warm up
        time.sleep(1)
        # Wait for a coherent pair of frames: depth and color    
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            raise Exception("Could not capture color frame")

        # Convert image to numpy array
        image = np.asanyarray(color_frame.get_data())
        image_display = image.copy()

        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", select_points)

        print("Click on the image to select points. Press Enter when done.")

        while True:
            cv2.imshow("Image", image_display)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter key
                break

        mask = generate_mask(image, points)

        # Save the mask image
        cv2.imwrite(mask_path, mask)
        cv2.destroyAllWindows()

        return mask_path

    finally:
        # Stop streaming
        pipeline.stop()


def create_mask_with_fastsam(model_path="FastSAM-x.pt", device="cuda", imgsz=1024, conf=0.4, iou=0.9, retina=True, better_quality=True):
    """
    Creates a mask using FastSAM model from a RealSense camera frame.
    
    Args:
        model_path: Path to the FastSAM model weights
        device: Device to run inference on ('cuda' or 'cpu')
        imgsz: Input image size for the model
        conf: Confidence threshold
        iou: IoU threshold
        retina: Whether to use retina masks
        better_quality: Whether to generate better quality output
        
    Returns:
        Path to the saved mask image
    """
    mask_path = './mask.png'
    temp_img_path = './temp_image.jpg'

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    try:
        # Wait for camera to warm up
        time.sleep(1)
        
        # Capture a frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            raise Exception("Could not capture color frame")

        # Convert image to numpy array
        image_np = np.asanyarray(color_frame.get_data())
        
        # Save the image temporarily for FastSAM to process
        cv2.imwrite(temp_img_path, image_np)
        
        # Load image with PIL for FastSAM
        input_image = Image.open(temp_img_path)
        input_image = input_image.convert("RGB")
        
        # Initialize FastSAM model
        print(f"Loading FastSAM model from {model_path}...")
        model = FastSAM(model_path)

        # Run inference on the image
        print("Running inference...")
        everything_results = model(
            input_image,
            device=device,
            retina_masks=retina,
            imgsz=imgsz,
            conf=conf,
            iou=iou
        )
        
        # Initialize prompt processor with results
        prompt_process = FastSAMPrompt(input_image, everything_results, device=device)
        ann = prompt_process.text_prompt(text='xbox')
        print(ann.shape)
        binary_mask = np.max(ann, axis=0).astype(np.uint8)
        binary_mask_255 = binary_mask * 255
        cv2.imwrite(mask_path, binary_mask_255)

        cv2.destroyAllWindows()
        return mask_path

    finally:
        # Stop streaming
        pipeline.stop()

if __name__ == "__main__":
    mask_file_path = create_mask()
    print(f"Mask saved at: {mask_file_path}")
