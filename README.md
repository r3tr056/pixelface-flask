
# PixelFace Inference Server | ETHOS 2024 (ML Challange)

## Overview

The **PixelFace Inference Server** is a web application built using Flask that allows users to upload CCTV footage for facial detection, enhancement, and 3D reconstruction. By leveraging advanced machine learning models, including BlazeFace for face detection, a variant of RestoreFormer for image enhancement, and PRNet for 3D facial reconstruction, PixelFace enables accurate identification of individuals from low-quality images.

## Features

- Real-time facial detection and enhancement from video footage.
- 3D reconstruction of detected faces with detailed textures and meshes.
- User-friendly web interface for seamless interaction.
- Optimized for performance using model acceleration techniques.

## Requirements

To run the PixelFace Inference Server, you will need:

- Python 3.7 or higher
- Flask
- ONNX Runtime
- TensorFlow or PyTorch (depending on the model implementation)
- Other dependencies specified in `requirements.txt`

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/pixelface-inference-server.git
   cd pixelface-inference-server
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the Flask server:**

   ```bash
   python server.py
   ```

   By default, the server runs on `http://127.0.0.1:5000`.

2. **Upload CCTV footage:**

   - Navigate to `http://127.0.0.1:5000` in your web browser.
   - Use the APIs to upload your video file

3. **Process the footage:**

   - Once the video is uploaded, the server will process it to detect faces, enhance their quality, and generate 3D models.
   - The results will be displayed on the web interface (PixelFace Webapp) or sent as JSON data and 3D files, allowing you to compare original and enhanced images as well as view the 3D models.

## API Endpoints

- **POST /upload**: Uploads the video file for processing.
- **GET /results**: Retrieves the processing results, including enhanced images and 3D model files.

## Inference Acceleration

The server is optimized with an acceleration platform to ensure fast inference times. This includes techniques such as model pruning, quantization, and utilizing GPUs for processing. 
