from twilio.rest import Client

def trigger_voice_call(to_number, child_name="identified child"):
    """
    Initiates a voice call using Twilio API to alert authorities.
    """
    # --- CONFIGURATION (Get these from Twilio Console) ---
    # You MUST sign up for Twilio (free trial available) to get these
    ACCOUNT_SID = 'AC_YOUR_TWILIO_ACCOUNT_SID' 
    AUTH_TOKEN = 'YOUR_TWILIO_AUTH_TOKEN'
    TWILIO_NUMBER = '+1234567890' # Your Twilio phone number

    try:
        client = Client(ACCOUNT_SID, AUTH_TOKEN)

        message = f"Urgent Alert. The Child Tracing System has detected a match for {child_name}. Please check the dashboard immediately."

        call = client.calls.create(
            twiml=f'<Response><Say voice="alice">{message}</Say></Response>',
            to=to_number,
            from_=TWILIO_NUMBER
        )
        
        print(f"Call initiated successfully! Call SID: {call.sid}")
        return True
        
    except Exception as e:
        print(f"Failed to initiate call: {e}")
        return False
