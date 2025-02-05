from django.http import JsonResponse
import os
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from datetime import datetime
import cv2
import pytesseract
import re
import json
import numpy as np
import face_recognition
from PIL import Image
from django.conf import settings
@api_view(['GET'])
def getRoutes(request):
    routes = [
        'GET /api',
        'GET /api/verify',
        
    ]
    return Response(routes)


def extract_text_from_image(image_path):
    """Extracts text from an image using Tesseract OCR"""
    img = cv2.imread(image_path)
    text = pytesseract.image_to_string(img)
    return text

    # Parses PAN details from extracted text
def parse_pan_details(text):
    pan_data = {}

    pan_match = re.search(r'([A-Z]{5}\d{4}[A-Z])', text)
    pan_data["PAN Number"] = pan_match.group(1) if pan_match else None

    name_match = re.search(r'Name\s*\n*([A-Z\s]+)', text)
    pan_data["Name"] = name_match.group(1).strip() if name_match else None

    dob_match = re.search(r'Date of Birth\s*\n*(\d{2}/\d{2}/\d{4})', text)
    pan_data["Date of Birth"] = dob_match.group(1) if dob_match else None

    return pan_data

    #Parses Aadhaar details from extracted text
def parse_aadhaar_details(text):
    aadhaar_data = {}

   
    enrolment_match = re.search(r'Enrolment No\.?:?\s*([\d/]+)', text)
    aadhaar_data["Enrolment No"] = enrolment_match.group(1) if enrolment_match else None

    name_match = re.search(r'([A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+)', text)
    aadhaar_data["Name"] = name_match.group(1) if name_match else None

    father_match = re.search(r'S/O\s+([A-Za-z\s]+)', text)
    aadhaar_data["Father's Name"] = father_match.group(1).strip().split("\n")[0] if father_match else None

    dob_match = re.search(r'DOB\s*:\s*(\d{2}/\d{2}/\d{4})', text)
    aadhaar_data["Date of Birth"] = dob_match.group(1) if dob_match else None

    address_lines = text.split("\n")
    address = []
    for i, line in enumerate(address_lines):
        if re.search(r'S/O\s+[A-Z]+[a-z]*', line):  
            address = address_lines[i + 1: i + 8]  
            break
    aadhaar_data["Address"] = " ".join([line.strip() for line in address if line.strip()]) if address else None

    aadhaar_match = re.search(r'(\d{4} \d{4} \d{4})', text)
    aadhaar_data["Aadhaar No"] = aadhaar_match.group(1) if aadhaar_match else None

    issue_date_match = re.search(r'Issue Date:\s*(\d{2}/\d{2}/\d{4})', text)
    aadhaar_data["Issue Date"] = issue_date_match.group(1) if issue_date_match else None

    phone_match = re.search(r'(\d{10})', text)
    aadhaar_data["Phone Number"] = phone_match.group(1) if phone_match else None

    pincode_match = re.search(r'(\d{6})', text)
    aadhaar_data["Pincode"] = pincode_match.group(1) if pincode_match else None

    return aadhaar_data




def convert_to_json(data):
    """Converts the extracted data into JSON format"""
    return json.dumps(data, indent=4)


def extract_text_from_image(image_path):
  img = cv2.imread(image_path)

  text = pytesseract.image_to_string(img)
  return text


def extract_aadhaar_image(aadhaar_image_path):
    try:
        aadhaar_image = cv2.imread(aadhaar_image_path)
        if aadhaar_image is None:
            print(f"Error: Could not read Aadhaar image at {aadhaar_image_path}")
            return None

        gray = cv2.cvtColor(aadhaar_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)


        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        photo_contour = None
        max_area = 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(contour)

            if aspect_ratio > 0.5 and aspect_ratio < 1.5 and area > 5000: 
                if area > max_area:
                    max_area = area
                    photo_contour = contour
                    photo_x, photo_y, photo_w, photo_h = x,y,w,h

        if photo_contour is not None:
            
            #crop the photo region
            aadhaar_photo = aadhaar_image[photo_y:photo_y+photo_h, photo_x:photo_x+photo_w]
            return aadhaar_photo
        else:
            print("Could not find photo region in Aadhaar image.")
            return None

    except Exception as e:
        print(f"Error extracting Aadhaar image: {e}")
        return None



def capture_webcam_image():
    try:
        video_capture = cv2.VideoCapture(0)  #0 for the default webcam

        if not video_capture.isOpened():
            print("Error: Could not open webcam.")
            return None

        ret, frame = video_capture.read()

        if ret:
            video_capture.release()
            return frame
        else:
            print("Error: Could not capture image from webcam.")
            video_capture.release()
            return None

    except Exception as e:
        print(f"Error capturing webcam image: {e}")
        return None



def compare_faces(aadhaar_image, webcam_image):
    try:
        aadhaar_image_rgb = cv2.cvtColor(aadhaar_image, cv2.COLOR_BGR2RGB)
        webcam_image_rgb = cv2.cvtColor(webcam_image, cv2.COLOR_BGR2RGB)
        
        
        
        aadhaar_encoding = face_recognition.face_encodings(aadhaar_image_rgb)
        webcam_encoding = face_recognition.face_encodings(webcam_image_rgb)

        if not aadhaar_encoding or not webcam_encoding:  
            print("Error: No faces detected in one or both images.")
            return False

        results = face_recognition.compare_faces(aadhaar_encoding, webcam_encoding[0]) # Compare Aadhaar encoding with the first webcam encoding.
        return results[0]  

    except Exception as e:
        print(f"Error comparing faces: {e}")
        return False





@api_view(['POST'])
def AadhaarOcr(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image_file')
        webcam_image = request.FILES.get('webcam_image')
        
        if not image_file or not webcam_image:
            return JsonResponse({"error": "Both image_file and webcam_image are required."}, status=400)

        # Save the image files to the 'bgimages' directory
        media_dir = os.path.join(settings.MEDIA_ROOT, 'bgimages')
        os.makedirs(media_dir, exist_ok=True)

        aadhaar_path = os.path.join(media_dir, image_file.name)
        web_path = os.path.join(media_dir, webcam_image.name)

        default_storage.save(aadhaar_path, image_file)
        default_storage.save(web_path, webcam_image)

        print(f"Aadhaar Path: {aadhaar_path}")
        print(f"Webcam Path: {web_path}")

        # Extract and process Aadhaar image
        aadhaar_image = extract_aadhaar_image(aadhaar_path)
        if aadhaar_image is not None:
            cv2.imshow("Aadhaar Image", aadhaar_image)
            cv2.waitKey(1000)
        
        # Extract text from Aadhaar image
        aadhaar_text = extract_text_from_image(aadhaar_path)
        print(f"Extracted Text: {aadhaar_text}")

        # Parse details based on document type
        # structured_data = {}
        doc_name = request.POST.get('docName')
       
        if doc_name == 'Aadhaar':
            structured_data = parse_aadhaar_details(aadhaar_text)
        elif doc_name == 'Pan':
            structured_data = parse_pan_details(aadhaar_text)
        else:
            structured_data["error"] = "Unsupported document type."

        # Compare faces if webcam image is provided
        web_img = cv2.imread(web_path)
        if web_img is not None:
            is_match = compare_faces(aadhaar_image, web_img)
            print(f"Face Match: {is_match}")
            structured_data["face match"] = "true" if is_match else "false"

            cv2.imshow("Webcam Image", web_img)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

        # Convert structured data to JSON
        json_output = convert_to_json(structured_data)
        return JsonResponse(json_output, safe=False)

    return JsonResponse({"msg": "error in handling request"}, status=400)

@api_view(['GET']) 
def PanOcr(request):
    if request.method == 'GET':
        image_file = request.FILES.get('image_file')
        # webcam_image = request.FILES.get('webcam_image')
        default_storage.save('bgimages/' + image_file.name, image_file)
        # webcam_path = default_storage.save('bgimages/' + webcam_image.name, webcam_image)
        # aadhaar_path = 'bgimages/' + image_file.name
        # webcam_path = 'bgimages/' + webcam_image.name
        
        media_dir = os.path.join(settings.MEDIA_ROOT, 'bgimages')  # Subdirectory for uploads
        os.makedirs(media_dir, exist_ok=True)  # Create directory if it doesn't exist
        pan_path = os.path.join(media_dir, image_file.name) # Construct file path
        # web_path = os.path.join(media_dir, webcam_image.name)
        # print(aadhaar_path)
        # aadhaar_image = extract_pan_image(pan_path)

        # cv2.imshow("aaddhaar image", aadhaar_image)
        # cv2.waitKey(1000)
        
        aadhaar_text = extract_text_from_image(pan_path)
        print(aadhaar_text) 
        structured_data = parse_pan_details(aadhaar_text)
        
        # web_img = cv2.imread(web_path)
        
  
        # if web_img is not None:
        #     is_match = compare_faces(aadhaar_image,  web_img)
        #     print(is_match)
        #     if is_match:
        #         structured_data["face match"] = "true"
        #         print("Faces match!")
        #     else:
        #         structured_data["face match"] = "False"
        #         print("Faces do not match.")

        #     cv2.imshow("Webcam Image",  web_img) 
        #     cv2.waitKey(1000) 
        #     cv2.destroyAllWindows()
            
        json_output = convert_to_json(structured_data)
        return JsonResponse(json_output,safe=False)

        # print(aadhaar_text)

        # print("Extracted Text:\n", json_output)


     


@api_view(['POST'])
def VerifyImage(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image')

        if image_file:
       
            image_path = default_storage.save('bgimages/' + image_file.name, image_file)
            
        else:
            image_path = 'default.jpg'
        
        

        return Response({'msg':'Image saved sucessfully'})
    

