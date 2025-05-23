import os
from dotenv import load_dotenv
import cv2
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from ultralytics import YOLO
import threading

# Load environment variables from .env file
load_dotenv()
# Above Is Environment Variables

# Retrieve the email and password from environment variables
sender_email = os.getenv("SENDER_EMAIL")
receiver_email = os.getenv("RECEIVER_EMAIL")
email_password = os.getenv("EMAIL_PASSWORD")

def draw_text_with_background(frame, text, position, font_scale=0.4, color=(255, 255, 255), thickness=1, bg_color=(0, 0, 0), alpha=0.7, padding=5):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_width, text_height = text_size

    overlay = frame.copy()
    x, y = position
    cv2.rectangle(overlay, (x - padding, y - text_height - padding), (x + text_width + padding, y + padding), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

def send_email_alert(image_path):
    # Set up the MIME
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "Alert: Hardhat Missing!"
    
    body = "A hardhat was not detected for the past 10 seconds, but a person was detected. Please find the attached frame showing the situation."
    message.attach(MIMEText(body, "plain"))
    
    # Attach the image file
    with open(image_path, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={image_path}")
        message.attach(part)
    
    # Sending the email via SMTP server
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, email_password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        print("Email alert sent with attachment.")
    except Exception as e:
        print(f"Failed to send email: {e}")

def send_email_in_background(image_path):
    email_thread = threading.Thread(target=send_email_alert, args=(image_path,))
    email_thread.start()

def main():
    model = YOLO("Model/ppe.pt")  # Replace with your custom model file if needed
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera
    
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    print("Press 'q' to exit.")

    colors = [
        (255, 0, 0),  # Hardhat (Blue)
        (0, 255, 0),  # Mask (Green)
        (0, 0, 255),  # NO-Hardhat (Red)
        (255, 255, 0),  # NO-Mask (Cyan)
        (255, 0, 255),  # NO-Safety Vest (Magenta)
        (0, 255, 255),  # Person (Yellow)
        (128, 0, 128),  # Safety Cone (Purple)
        (128, 128, 0),  # Safety Vest (Olive)
        (0, 128, 128),  # Machinery (Teal)
        (128, 128, 128)  # Vehicle (Gray)
    ]

    # Initialize last time a hardhat was detected
    last_hardhat_time = time.time()
    hardhat_missing = False
    last_email_time = time.time()  # Track time of last email sent
    email_sent_flag = False
    email_sent_time = 0  # To track when to stop showing the email sent message

    # Create a resizable window
    cv2.namedWindow("YOLOv8 Annotated Feed", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from the webcam.")
            break

        # Initialize counters
        hardhat_count = 0
        vest_count = 0
        person_count = 0
        hardhat_detected = False
        person_detected = False

        # Perform YOLO inference
        results = model(frame)

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    confidence = box.conf[0]  # Confidence score
                    cls = int(box.cls[0])  # Class ID
                    label = f"{model.names[cls]} ({confidence:.2f})"

                    color = colors[cls % len(colors)]

                    # Draw the bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    draw_text_with_background(frame, label, (x1, y1 - 10), font_scale=0.4, color=(255, 255, 255), bg_color=color, alpha=0.8, padding=4)

                    if model.names[cls] == "Hardhat":
                        hardhat_count += 1
                        hardhat_detected = True
                    elif model.names[cls] == "Safety Vest":
                        vest_count += 1
                    elif model.names[cls] == "Person":
                        person_count += 1
                        person_detected = True

        # If person is detected but no hardhat detected for 10 seconds, send an email alert
        if person_detected and not hardhat_detected and (time.time() - last_email_time) >= 100:  # Check 10 seconds interval
            image_path = "no_hardhat_frame.jpg"
            cv2.imwrite(image_path, frame)  # Save the frame as an image
            send_email_in_background(image_path)  # Send email in background thread
            email_sent_flag = True
            email_sent_time = time.time()  # Track the time of sending the email
            last_email_time = time.time()  # Update last email time

        # If hardhat detected, update the last time detected
        if hardhat_detected:
            last_hardhat_time = time.time()

        # Add the counts on the sideboard
        sideboard_text = [
            f"Hardhats: {hardhat_count}",
            f"Safety Vests: {vest_count}",
            f"People: {person_count}"
        ]

        y_position = 30
        for text in sideboard_text:
            draw_text_with_background(frame, text, (10, y_position), font_scale=0.5, color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.7, padding=5)
            y_position += 30

        # Show the "Email Sent" message for 3 seconds after an email is sent
        if email_sent_flag and (time.time() - email_sent_time) < 3:
            draw_text_with_background(frame, "Email Sent", (frame.shape[1] - 100, 30), font_scale=0.5, color=(0, 255, 0), bg_color=(0, 0, 0), alpha=0.8, padding=5)

        # Resize the frame to fit the window dynamically
        resized_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)  # Resize to a fixed size

        # Display the annotated frame
        cv2.imshow("YOLOv8 Annotated Feed", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
