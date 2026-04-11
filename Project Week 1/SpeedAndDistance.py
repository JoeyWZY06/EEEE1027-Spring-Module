import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

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
L_Encoder = 17
R_Encoder = 27

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

Wheel_Circumference = 20.106 # Wheel circumference in centimeters (cm)

GPIO.setup(L_Encoder, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(R_Encoder, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Set starting pulses as 0
L_pulse_count = 0
R_pulse_count = 0

# Left pulse counter funtion
def L_Pulse_Counter(channel):
    global L_pulse_count
    L_pulse_count += 1

# Right pulse counter funtion
def R_Pulse_Counter(channel):
    global R_pulse_count
    R_pulse_count += 1
    
# Add Event Detect (Interrupt)
# This runs in the background. When the pin goes HIGH (slot detected),
# it triggers the 'count_pulse' function immediately.
GPIO.add_event_detect(L_Encoder, GPIO.RISING, callback=L_Pulse_Counter, bouncetime=1)
GPIO.add_event_detect(R_Encoder, GPIO.RISING, callback=R_Pulse_Counter, bouncetime=1)

# Calculate distance function
def calculate_distance():
    global L_pulse_count
    global R_pulse_count
    global Wheel_Circumference
    
    L_Distance = L_pulse_count * Wheel_Circumference / 20.0
    R_Distance = R_pulse_count * Wheel_Circumference / 20.0
    Avg_distance = (L_Distance + R_Distance) / 2.0
    return Avg_distance

def forward_specific_distance(distance, speed):
    global L_pulse_count
    global R_pulse_count

    start_time = time.monotonic()
    while calculate_distance() < distance :
        
        # Left Motor Logic
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
        
        # Right Motor Logic
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)
        
        # Set Speed
        pwm_a.ChangeDutyCycle(speed)
        pwm_b.ChangeDutyCycle(speed)
        
    stop()
    average_speed = calculate_distance() / (time.monotonic - start_time)
    print("Distance: ", calculate_distance())
    print("Average speed:", average_speed)
    
# --- Main Execution ---
forward_specific_distance(70, 50)



    
    
    