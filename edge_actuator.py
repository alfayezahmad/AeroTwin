"""
AeroTwin Edge Actuator Daemon

Hardware Wiring Guide (ESP32/SBC):
----------------------------------
           ESP32 MCU
       +---------------+
       |               |
       |  [GPIO 18] -----------------> [PWM Signal] Servo 1 (Main Valve)
       |               |
       |  [GPIO 19] -----------------> [PWM Signal] Servo 2 (Mist Cannon)
       |               |
       |    [5V/VIN] ----------------> [+] Power (Both Servos)
       |      [GND] -----------------> [-] GND (Both Servos)
       +---------------+

Dependencies:
$ pip install paho-mqtt gpiozero
"""

import time
import json
import logging
import threading
import paho.mqtt.client as mqtt
from gpiozero import AngularServo
from gpiozero.pins.mock import MockFactory

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

MQTT_BROKER = "localhost" # Set to broker IP for physical deployment
MQTT_PORT = 1883
TOPIC_SUB = "aerotwin/commands/dispatch"

try:
    # Attempt hardware PWM pin attachment
    valve_servo = AngularServo(18, min_angle=0, max_angle=180)
    cannon_servo = AngularServo(19, min_angle=0, max_angle=180)
except Exception as e:
    logging.warning(f"Physical PWM pins unavailable ({e}). Using Mock Factory.")
    from gpiozero import Device
    Device.pin_factory = MockFactory()
    valve_servo = AngularServo(18, min_angle=0, max_angle=180)
    cannon_servo = AngularServo(19, min_angle=0, max_angle=180)

is_mitigating = False
mitigation_thread = None

def sweep_cannon():
    """Non-blocking background loop for mist cannon pan simulation."""
    global is_mitigating
    logging.info("Mist cannon sweep sequence initiated. Valve Opening to 90 degrees.")
    valve_servo.angle = 90
    
    sweep_step = 5
    current_angle = 0
    direction = 1
    
    while is_mitigating:
        current_angle += (sweep_step * direction)
        if current_angle >= 180:
            current_angle = 180
            direction = -1
        elif current_angle <= 0:
            current_angle = 0
            direction = 1
            
        cannon_servo.angle = current_angle
        time.sleep(0.05)
        
    logging.info("Mist cannon sweep sequence terminated.")

def trigger_grap_mitigation():
    """Activates emergency closed-loop hardware actuation."""
    global is_mitigating, mitigation_thread
    if not is_mitigating:
        logging.info("🚨 EMERGENCY TRIGGER: Stage III/IV detected. Activating hardware.")
        is_mitigating = True
        mitigation_thread = threading.Thread(target=sweep_cannon, daemon=True)
        mitigation_thread.start()

def standby_mode():
    """Safely halts all physical actuation."""
    global is_mitigating, mitigation_thread
    if is_mitigating:
        logging.info("STANDBY COMMAND: Halting operations.")
        is_mitigating = False
        if mitigation_thread is not None:
            mitigation_thread.join(timeout=2.0)
            
    valve_servo.angle = 0
    cannon_servo.angle = 0
    logging.info("System is in STANDBY.")

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logging.info(f"Connected to Broker at {MQTT_BROKER}:{MQTT_PORT}")
        client.subscribe(TOPIC_SUB)
    else:
        logging.error(f"MQTT Connect Failed: {rc}")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        logging.info(f"Received Downlink: {payload}")
        stage = payload.get("grap_stage", "")
        
        if "Stage III" in stage or "Stage IV" in stage:
            trigger_grap_mitigation()
        elif "Stage I" in stage or "Normal" in stage:
            standby_mode()
            
    except json.JSONDecodeError:
        pass

def main():
    standby_mode()
    client = mqtt.Client(client_id="AeroTwin_Edge_Actuator")
    client.on_connect = on_connect
    client.on_message = on_message

    logging.info("Starting Edge Actuator Daemon...")
    while True:
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, 60)
            client.loop_forever()
        except KeyboardInterrupt:
            standby_mode()
            client.disconnect()
            break
        except Exception:
            logging.error("MQTT Broker connection dropped. Retrying...")
            time.sleep(5)

if __name__ == "__main__":
    main()
