import cv2
import pytesseract
import numpy as np
import re
import time
import sqlite3
from datetime import datetime
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
def init_db():
    conn = sqlite3.connect('toll_database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS registered_vehicles
                 (plate_number TEXT PRIMARY KEY,
                  owner_name TEXT,
                  vehicle_type TEXT,
                  registration_date TEXT)''')
                  
    c.execute('''CREATE TABLE IF NOT EXISTS toll_payments
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  plate_number TEXT,
                  payment_date TEXT,
                  payment_amount REAL,
                  toll_location TEXT,
                  FOREIGN KEY(plate_number) REFERENCES registered_vehicles(plate_number))''')
    
    conn.commit()
    conn.close()
init_db()  
def clean_plate_text(text):
    """Clean and validate detected license plate text"""
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', text)
    patterns = [
        r'[A-Z]{2,3}\d{3,4}',       
        r'\d{3,4}[A-Z]{2,3}',      
        r'[A-Z]{2}\d{2}[A-Z]{2}', 
        r'[A-Z]{1,2}\d{1,4}[A-Z]{1,3}',  
        r'\d{1,2}[A-Z]{1,3}\d{1,4}'     
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            return match.group().upper()

    return cleaned.upper()

def preprocess_image(img):
    """Enhanced image preprocessing for better plate detection"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    plate_contour = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 2.0 < aspect_ratio < 5.0 and w > 100 and h > 30:
                plate_contour = approx
                break
    
    if plate_contour is None:
        return None
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [plate_contour], 0, 255, -1)
    
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    cropped = gray[topx:bottomx+1, topy:bottomy+1]
    cropped = cv2.adaptiveThreshold(cropped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)

    cropped = cv2.medianBlur(cropped, 3)
    kernel = np.ones((1, 1), np.uint8)
    cropped = cv2.dilate(cropped, kernel, iterations=1)
    cropped = cv2.erode(cropped, kernel, iterations=1)
    scale_percent = 200 
    width = int(cropped.shape[1] * scale_percent / 100)
    height = int(cropped.shape[0] * scale_percent / 100)
    dim = (width, height)
    cropped = cv2.resize(cropped, dim, interpolation=cv2.INTER_CUBIC)
    
    return cropped

def register_vehicle(plate_number):
    """Register a new vehicle in the database"""
    conn = sqlite3.connect('toll_database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM registered_vehicles WHERE plate_number=?", (plate_number,))
    if c.fetchone():
        print("Vehicle already registered!")
        conn.close()
        return False
    print(f"\nRegistering new vehicle: {plate_number}")
    owner_name = input("Enter owner name: ")
    vehicle_type = input("Enter vehicle type (Car/Truck/Bike/etc): ")
    registration_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO registered_vehicles VALUES (?, ?, ?, ?)",
              (plate_number, owner_name, vehicle_type, registration_date))
    
    conn.commit()
    conn.close()
    print("Vehicle registered successfully!")
    return True

def record_toll_payment(plate_number):
    """Record a toll payment for a vehicle"""
    conn = sqlite3.connect('toll_database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM registered_vehicles WHERE plate_number=?", (plate_number,))
    vehicle = c.fetchone()
    
    if not vehicle:
        print("Vehicle not registered. Please register first.")
        register_choice = input("Register now? (y/n): ").lower()
        if register_choice == 'y':
            if register_vehicle(plate_number):
                return record_toll_payment(plate_number)
        else:
            conn.close()
            return False
    print(f"\nRecording toll payment for: {plate_number}")
    print(f"Vehicle Owner: {vehicle[1]}")
    print(f"Vehicle Type: {vehicle[2]}")
    
    payment_amount = float(input("Enter payment amount: "))
    toll_location = input("Enter toll location: ")
    payment_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO toll_payments (plate_number, payment_date, payment_amount, toll_location) VALUES (?, ?, ?, ?)",
              (plate_number, payment_date, payment_amount, toll_location))
    
    conn.commit()
    conn.close()
    print("Toll payment recorded successfully!")
    return True

def check_payment_status(plate_number):
    """Check if vehicle has paid toll recently"""
    conn = sqlite3.connect('toll_database.db')
    c = conn.cursor()

    c.execute('''SELECT * FROM toll_payments 
                 WHERE plate_number=? 
                 ORDER BY payment_date DESC 
                 LIMIT 1''', (plate_number,))
    last_payment = c.fetchone()
    
    conn.close()
    
    if last_payment:
        payment_date = datetime.strptime(last_payment[2], "%Y-%m-%d %H:%M:%S")
        time_since_payment = datetime.now() - payment_date
        if time_since_payment.total_seconds() < 24 * 60 * 60:
            print(f"\nVehicle {plate_number} has paid toll recently.")
            print(f"Last payment: {last_payment[2]} at {last_payment[4]}")
            print(f"Amount paid: ${last_payment[3]:.2f}")
            return True
    
    print(f"\nVehicle {plate_number} has no recent toll payment.")
    return False

def detect_plate_from_camera():
    """Main function to detect license plates from camera feed"""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\nLicense Plate Detection Started...")
    print("Press 's' to scan the current frame")
    print("Press 'p' to process toll payment for detected plate")
    print("Press 'c' to check payment status")
    print("Press 'q' to quit\n")
    
    last_scan_time = 0
    scan_cooldown = 3 
    last_detected_plate = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        cv2.putText(frame, "Press 's' to scan, 'p' for payment, 'q' to quit", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('License Plate Scanner', frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
            
        if key == ord('s'):
            current_time = time.time()
            if current_time - last_scan_time < scan_cooldown:
                print(f"Please wait {scan_cooldown - int(current_time - last_scan_time)} seconds before next scan")
                continue
                
            last_scan_time = current_time
            print("\nScanning...")
            processed_imgs = []
            processed = preprocess_image(frame)
            if processed is not None:
                processed_imgs.append(processed)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            processed_imgs.append(thresh)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            edged = cv2.Canny(blur, 50, 200)
            processed_imgs.append(edged)
            
            if not processed_imgs:
                print("No license plate detected. Try adjusting the angle or lighting.")
                continue
            custom_configs = [
                r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                r'--oem 1 --psm 7',
                r'--oem 3 --psm 11'
            ]
            
            best_text = ""
            best_confidence = 0
            
            for img in processed_imgs:
                for config in custom_configs:
                    try:
                        data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
                        
                        for i in range(len(data['text'])):
                            text = data['text'][i]
                            conf = int(data['conf'][i])
                            
                            if conf > best_confidence and len(text) >= 4:
                                cleaned_text = clean_plate_text(text)
                                if len(cleaned_text) >= 4:  
                                    best_text = cleaned_text
                                    best_confidence = conf
                    except:
                        continue
            
            if best_text:
                last_detected_plate = best_text
                print(f"Detected License Plate: {best_text} (Confidence: {best_confidence}%)")
                cv2.imshow('Processed Plate', processed_imgs[0])
                cv2.waitKey(2000)
                cv2.destroyWindow('Processed Plate')
            else:
                print("Could not read license plate. Try getting closer or better lighting.")
                cv2.imshow('Processed Plate (Debug)', processed_imgs[0])
                cv2.waitKey(2000)
                cv2.destroyWindow('Processed Plate (Debug)')
        
        elif key == ord('p') and last_detected_plate:
            record_toll_payment(last_detected_plate)
        
        elif key == ord('c') and last_detected_plate:
            check_payment_status(last_detected_plate)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Enhanced Real-Time License Plate Scanner with Toll Payment System")
    print("--------------------------------------------------------------")
    print("Instructions:")
    print("1. Point the camera at a license plate")
    print("2. Make sure the plate is well-lit and fills most of the frame")
    print("3. Hold the camera steady and parallel to the plate")
    print("4. Press 's' to scan when the plate is clearly visible")
    print("5. After detection, press 'p' to process toll payment")
    print("6. Press 'c' to check payment status for the last detected plate")
    print("7. Press 'q' to quit the application\n")
    
    detect_plate_from_camera()



