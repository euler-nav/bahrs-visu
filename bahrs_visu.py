import serial
import threading
import serial.tools.list_ports
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import queue

# Time window for data retention in seconds
TIME_WINDOW = 30
# Data streaming rate in Hz
DATA_RATE_HZ = 100
# Plot update interval in milliseconds
PLOT_UPDATE_INTERVAL_MS = 200  # Update plot every 200 ms (5 times per second)

# Global variable to control the reading thread and animation
stop_event = threading.Event()
ani = None
ser = None

# Initialize a queue to hold parsed data
data_queue = queue.Queue(maxsize=DATA_RATE_HZ * TIME_WINDOW)  # Buffer for data

# Function to compute CRC-32
def crc32(buf):
    uCrc = 0xFFFFFFFF
    for word in buf:
        uCrc ^= word
        for _ in range(32):
            if uCrc & 0x80000000:
                uCrc = (uCrc << 1) ^ 0x04C11DB7
            else:
                uCrc <<= 1
    return uCrc & 0xFFFFFFFF

def crc32_from_bytes(data):
    buf = [int.from_bytes(data[i:i+4], byteorder='little') for i in range(0, len(data) - len(data) % 4, 4)]
    return crc32(buf)

# Function to parse the payload of message type 0x02
def parse_payload_type_02(payload, timestamp):
    sequence_number = payload[0]
    height = int.from_bytes(payload[1:3], byteorder='little', signed=True)
    vertical_velocity = int.from_bytes(payload[3:5], byteorder='little', signed=True)
    roll = int.from_bytes(payload[5:7], byteorder='little', signed=True)
    pitch = int.from_bytes(payload[7:9], byteorder='little', signed=True)
    yaw = int.from_bytes(payload[9:11], byteorder='little', signed=True)
    bitfield = payload[11]

    # Validity of each signal
    validity = {
        "height": bool(bitfield & 0x01),
        "vertical_velocity": bool(bitfield & 0x02),
        "roll": bool(bitfield & 0x04),
        "pitch": bool(bitfield & 0x08),
        "yaw": bool(bitfield & 0x10)
    }

    # Convert raw data to physical units
    height_meters = 0.16784924 * height - 1000 if validity['height'] else None
    vertical_velocity_mps = 9.155413e-3 * vertical_velocity if validity['vertical_velocity'] else None
    roll_radians = 9.587526e-5 * roll if validity['roll'] else None
    pitch_radians = 9.587526e-5 * pitch if validity['pitch'] else None
    yaw_radians = 9.587526e-5 * yaw if validity['yaw'] else None

    return {
        "timestamp": timestamp,
        "sequence_number": sequence_number,
        "height": height_meters,
        "vertical_velocity": vertical_velocity_mps,
        "roll": roll_radians,
        "pitch": pitch_radians,
        "yaw": yaw_radians,
        "validity": validity
    }

# Function to parse the entire frame
def parse_frame(data):
    if len(data) < 24:
        return None

    if data[0] == 0x4E and data[1] == 0x45:  # Check for header markers
        protocol_version = data[2] + (data[3] << 8)
        message_type = data[4]

        if message_type == 0x02:  # We are only interested in message type 0x02
            header_length = 5
            payload_length = 14  # Fixed payload length for message type 0x02
            padding_length = 1  # Padding to make total frame size a multiple of 32 bits
            expected_frame_size = header_length + payload_length + padding_length + 4

            if len(data) < expected_frame_size:
                return None

            frame_crc = int.from_bytes(data[expected_frame_size-4:expected_frame_size], byteorder='little')
            computed_crc = crc32_from_bytes(data[:expected_frame_size-4])
            if frame_crc != computed_crc:
                return None

            timestamp = time.time()
            parsed_data = parse_payload_type_02(data[header_length:header_length + payload_length], timestamp)
            data_queue.put(parsed_data)  # Put parsed data into the queue
            return parsed_data
        else:
            return None
    else:
        return None

# Function to read from the serial port
def read_from_port(ser):
    buffer = b''
    while ser.is_open and not stop_event.is_set():
        if ser.in_waiting > 0:
            buffer += ser.read(ser.in_waiting)

            while len(buffer) >= 24:  # Minimum frame size (5 header + 14 payload + 1 padding + 4 CRC)
                if buffer[0] == 0x4E and buffer[1] == 0x45:
                    if len(buffer) >= 24:
                        frame = buffer[:24]
                        parse_frame(frame)
                        buffer = buffer[24:]  # Remove the processed frame
                    else:
                        break
                else:
                    buffer = buffer[1:]  # Remove the first byte and retry
        time.sleep(0.01)  # Add a slight delay to reduce CPU usage

# Function to get available COM ports
def get_ports():
    ports = serial.tools.list_ports.comports()
    return ports

# Function to connect to the selected COM port
def connect():
    global ser, ani, start_time
    port = combo.get()
    if port:
        try:
            ser = serial.Serial(port, 115200, timeout=1, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS)
            status_label.config(text="Connected to " + port)
            start_button['state'] = 'disabled'
            stop_button['state'] = 'normal'
            stop_event.clear()
            thread = threading.Thread(target=read_from_port, args=(ser,))
            thread.start()
            start_time = time.time()  # Reset the start time when connecting
            clear_plot_data()  # Clear previous plot data and axes
            if ani is not None:
                ani.event_source.start()
            else:
                ani = FuncAnimation(fig, update_plot, interval=PLOT_UPDATE_INTERVAL_MS, cache_frame_data=False)
        except Exception as e:
            status_label.config(text="Error: " + str(e))
    else:
        status_label.config(text="Please select a port.")

# Function to disconnect from the COM port
def disconnect():
    global ser, ani
    stop_event.set()
    if ser and ser.is_open:
        ser.close()
    status_label.config(text="Disconnected")
    start_button['state'] = 'normal'
    stop_button['state'] = 'disabled'
    if ani:
        ani.event_source.stop()
    ser = None

# Function to clear plot data and axes
def clear_plot_data():
    global time_data, height_data, vertical_velocity_data, roll_data, pitch_data, yaw_data
    time_data.clear()
    height_data.clear()
    vertical_velocity_data.clear()
    roll_data.clear()
    pitch_data.clear()
    yaw_data.clear()
    for axis in ax:
        axis.cla()

# Function to update the plot
def update_plot(frame):
    current_time = time.time()
    data_chunk = []
    while not data_queue.empty():
        data = data_queue.get()
        if data:
            data_chunk.append(data)

    if data_chunk:
        for data in data_chunk:
            time_data.append(data['timestamp'] - start_time)
            height_data.append(data['height'])
            vertical_velocity_data.append(data['vertical_velocity'])
            roll_data.append(data['roll'])
            pitch_data.append(data['pitch'])
            yaw_data.append(data['yaw'])

        # Keep only the last TIME_WINDOW seconds of data
        while time_data and (current_time - start_time - time_data[0]) > TIME_WINDOW:
            time_data.pop(0)
            height_data.pop(0)
            vertical_velocity_data.pop(0)
            roll_data.pop(0)
            pitch_data.pop(0)
            yaw_data.pop(0)

        for axis in ax:
            axis.cla()

        if time_data:
            labels = ["Height (m)", "Vertical Velocity (m/s)", "Roll (rad)", "Pitch (rad)", "Yaw (rad)"]
            data_lists = [height_data, vertical_velocity_data, roll_data, pitch_data, yaw_data]
            for axis, label, data_list in zip(ax, labels, data_lists):
                axis.plot(time_data, data_list, label=label)
                axis.legend()
                axis.relim()
                axis.autoscale_view()

# Set up the GUI
root = tk.Tk()
root.title("EULER-NAV Baro-Inertial AHRS Visu")

# Dropdown for COM ports
ports = get_ports()
port_list = [port.device for port in ports]
combo_label = ttk.Label(root, text="Select COM Port:")
combo_label.pack(pady=5)
combo = ttk.Combobox(root, values=port_list, state="readonly")
combo.pack(pady=5)

# Start and Stop buttons
start_button = ttk.Button(root, text="Connect", command=connect)
start_button.pack(pady=5)
stop_button = ttk.Button(root, text="Disconnect", command=disconnect)
stop_button.pack(pady=5)
stop_button['state'] = 'disabled'

# Status label
status_label = ttk.Label(root, text="Select a COM port to start.")
status_label.pack(pady=5)

# Matplotlib figure and axes
fig, ax = plt.subplots(5, 1, figsize=(10, 10))
time_data = []
height_data = []
vertical_velocity_data = []
roll_data = []
pitch_data = []
yaw_data = []

start_time = time.time()

# Integrate the matplotlib plot with tkinter
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
canvas.draw()

# Ensure proper cleanup on close
def on_closing():
    disconnect()
    root.quit()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Start animation
ani = FuncAnimation(fig, update_plot, interval=PLOT_UPDATE_INTERVAL_MS, cache_frame_data=False)

root.mainloop()
