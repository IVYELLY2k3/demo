import smtplib
from email.message import EmailMessage

def send_alert_email(to_email, matched_image_path):
    # --- CONFIGURATION (Use your own credentials or App Password) ---
    SENDER_EMAIL = "your_email@gmail.com"  # Replace with your email
    APP_PASSWORD = "your_app_password"     # Replace with your App Password
    
    # Create the email
    msg = EmailMessage()
    msg['Subject'] = 'URGENT: Child Match Found in Surveillance Video'
    msg['From'] = SENDER_EMAIL
    msg['To'] = to_email
    msg.set_content("The AI system has detected a match for the missing child in the processed video footage. Please see the attached frame.")

    # Attach the matched image
    try:
        with open(matched_image_path, 'rb') as f:
            file_data = f.read()
            file_name = "matched_frame.jpg"
        msg.add_attachment(file_data, maintype='image', subtype='jpeg', filename=file_name)

        # Send the email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, APP_PASSWORD)
            smtp.send_message(msg)
        print("Alert email sent successfully!")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False
