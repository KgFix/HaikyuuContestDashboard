from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import pytesseract
import io
from typing import Dict, List, Optional
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(title="Image OCR API", version="1.0.0")

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for extracted text data
extracted_data: Dict[str, Dict] = {}
processing_history: List[Dict] = []

# Configure Tesseract path if needed (Windows)
# Uncomment and adjust path if Tesseract is not in PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "message": "Image OCR API is running",
        "endpoints": {
            "/upload": "POST - Upload image for OCR processing",
            "/data": "GET - Retrieve all extracted data",
            "/data/{key}": "GET - Retrieve specific extracted data",
            "/history": "GET - Get processing history",
            "/clear": "DELETE - Clear all stored data"
        }
    }


@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    key: Optional[str] = None,
    preprocess: bool = False
):
    """
    Upload and process an image to extract text using OCR
    
    Args:
        file: Image file (PNG, JPG, JPEG)
        key: Optional key to store the result (defaults to filename)
        preprocess: Apply image preprocessing for better OCR results
    
    Returns:
        Extracted text and metadata
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Optional preprocessing for better OCR
        if preprocess:
            image = preprocess_image(image)
        
        # Extract text using Tesseract
        extracted_text = pytesseract.image_to_string(image)
        
        # Extract additional data (bounding boxes, confidence)
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        # Create storage key
        storage_key = key or file.filename or f"image_{len(extracted_data)}"
        
        # Store extracted data globally
        result = {
            "text": extracted_text,
            "filename": file.filename,
            "timestamp": datetime.now().isoformat(),
            "image_size": image.size,
            "image_mode": image.mode,
            "word_count": len(extracted_text.split()),
            "confidence": calculate_avg_confidence(ocr_data)
        }
        
        extracted_data[storage_key] = result
        processing_history.append({
            "key": storage_key,
            "timestamp": result["timestamp"],
            "filename": file.filename
        })
        
        return {
            "success": True,
            "key": storage_key,
            "data": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/data")
async def get_all_data():
    """Retrieve all extracted data from global storage"""
    return {
        "total_items": len(extracted_data),
        "data": extracted_data
    }


@app.get("/data/{key}")
async def get_data_by_key(key: str):
    """Retrieve specific extracted data by key"""
    if key not in extracted_data:
        raise HTTPException(status_code=404, detail=f"No data found for key: {key}")
    return extracted_data[key]


@app.get("/history")
async def get_history():
    """Get processing history"""
    return {
        "total_processed": len(processing_history),
        "history": processing_history
    }


@app.delete("/clear")
async def clear_data():
    """Clear all stored data"""
    extracted_data.clear()
    processing_history.clear()
    return {"message": "All data cleared successfully"}


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess image for better OCR results
    
    Args:
        image: PIL Image object
    
    Returns:
        Preprocessed image
    """
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # You can add more preprocessing here:
    # - Increase contrast
    # - Remove noise
    # - Binarization
    # - Deskewing
    
    return image


def calculate_avg_confidence(ocr_data: Dict) -> float:
    """
    Calculate average confidence score from OCR data
    
    Args:
        ocr_data: Dictionary from pytesseract.image_to_data
    
    Returns:
        Average confidence score
    """
    confidences = [
        float(conf) for conf in ocr_data['conf'] 
        if conf != '-1' and conf.replace('.', '').isdigit()
    ]
    return sum(confidences) / len(confidences) if confidences else 0.0


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
