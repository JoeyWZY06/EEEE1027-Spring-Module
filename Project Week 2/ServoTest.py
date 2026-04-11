import RPi.GPIO as GPIO
import time

# --- CONFIGURATION ---
SERVO_PIN = 21  # We are using GPIO 21

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)

# Create PWM instance with 50Hz frequency (Standard for Servos)
# GPIO.PWM(pin, frequency)
servo = GPIO.PWM(SERVO_PIN, 50)

# Start PWM with 0% duty cycle (so it doesn't move immediately on startup)
servo.start(0)

def set_angle(angle):
    """
    Converts degrees (0-180) to Duty Cycle (2-12)
    Formula: Duty = 2 + (angle / 18)
    """
    duty =  1.65 + (angle / 18)
    
    # Send the signal
    GPIO.output(SERVO_PIN, True)
    servo.ChangeDutyCycle(duty)
    
    # Wait for the servo to physically move there
    time.sleep(.2) 
    
    # Stop sending signal to prevent "jittering"
    GPIO.output(SERVO_PIN, False)
    servo.ChangeDutyCycle(0)

print("Starting Servo Test...")

try:
    print("Moving to 0 degrees (Right)")
    set_angle(86)
    time.sleep(1)

except KeyboardInterrupt:
    print("\nStopping...")
    servo.stop()
    GPIO.cleanup()