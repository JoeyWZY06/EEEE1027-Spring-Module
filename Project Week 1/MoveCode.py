import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Define GPIO pins for the L298N
# Left side motors
IN1 = 26
IN2 = 19
ENA = 20

# Right side motors (connected to OUT3 & OUT4)
IN3 = 13
IN4 = 6
ENB = 5

# Left and Right encoders
L_Encoder = 3
R_Encoder = 2

# Set up pins as outputs
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)

GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)
GPIO.setup(ENB, GPIO.OUT)

# Initialize PWM on Enable pins (Frequency = 1000Hz)
pwm_a = GPIO.PWM(ENA, 1000)
pwm_b = GPIO.PWM(ENB, 1000)

# Start PWM with 0% duty cycle (stopped)
pwm_a.start(0)
pwm_b.start(0)

# Move forward function
def forward(speed, duration):
    # Left Motor Logic
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    
    # Right Motor Logic
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    
    # Set Speed
    pwm_a.ChangeDutyCycle(speed)
    pwm_b.ChangeDutyCycle(speed)

    # Wait for the specified duration
    time.sleep(duration)
    
    # Stop
    stop()

# Move backward function
def backward(speed, duration):
    # Left Motor Logic
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    
    # Right Motor Logic
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    
    # Set Speed
    pwm_a.ChangeDutyCycle(speed)
    pwm_b.ChangeDutyCycle(speed)

    # Wait for the specified duration
    time.sleep(duration)
    
    # Stop
    stop()

# Turn left function
def turn_left(speed, duration):
    # Left Motor Logic
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    
    # Right Motor Logic
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    
    # Set Speed
    pwm_a.ChangeDutyCycle(speed)
    pwm_b.ChangeDutyCycle(speed)

    # Wait for the specified duration
    time.sleep(duration)
    
    # Stop
    stop()
    
# Turn right function
def turn_right(speed, duration):
    # Left Motor Logic
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    
    # Right Motor Logic
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    
    # Set Speed
    pwm_a.ChangeDutyCycle(speed)
    pwm_b.ChangeDutyCycle(speed)

    # Wait for the specified duration
    time.sleep(duration)
    
    # Stop
    stop()

# Stop function
def stop():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwm_a.ChangeDutyCycle(0)
    pwm_b.ChangeDutyCycle(0)

# --- Main Execution ---
turn_right(70, 2.2)
stop()
GPIO.cleanup()
    
    
