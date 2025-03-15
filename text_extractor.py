import os
import fitz  # PyMuPDF
from paddleocr import PaddleOCR
from PIL import Image
import io
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextExtractor:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

    def convert_pdf_to_images(self, pdf_path: str) -> list:
        """Convert PDF pages to images"""
        try:
            images = []
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                pix = page.get_pixmap()
                img_data = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img_data)
                
            pdf_document.close()
            return images
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            raise

    def extract_text(self, file_path: str) -> str:
        """Extract text from resume file"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            if file_path.lower().endswith('.pdf'):
                images = self.convert_pdf_to_images(file_path)
                text = []
                
                for img in images:
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    result = self.ocr.ocr(img_byte_arr, cls=True)
                    if result:
                        for idx in range(len(result)):
                            res = result[idx]
                            for line in res:
                                text.append(line[1][0])
                
                return "\n".join(text)
            else:
                result = self.ocr.ocr(file_path, cls=True)
                return "\n".join([line[1][0] for res in result for line in res]) if result else ""
                
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            raise

    def save_extracted_text(self, text: str, original_file_path: str) -> str:
        """Save extracted text to file"""
        try:
            output_path = os.path.join(
                os.path.dirname(original_file_path),
                f"extracted_{os.path.splitext(os.path.basename(original_file_path))[0]}.txt"
            )
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            logger.info(f"Extracted text saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving text: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        extractor = TextExtractor()
        resume_path = "C:/Users/harsh/Downloads/Aryan-Rajpurkar_resume.pdf"
        print("\nExtracting text from resume...")
        extracted_text = extractor.extract_text(resume_path)
        output_file = extractor.save_extracted_text(extracted_text, resume_path)
        
        print(f"\nText extracted and saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to extract text: {str(e)}")