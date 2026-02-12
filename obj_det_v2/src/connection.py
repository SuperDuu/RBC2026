import serial
import threading
import time

class UARTManager:
    def __init__(self, port='/dev/ttyUSB0', baud=115200):
        # Khởi tạo các thuộc tính cơ bản TRƯỚC khi vào try-except
        self.lock = threading.Lock()
        self.last_packet = "S+0000E\n"
        self.running = True
        self.ser = None
        
        try:
            self.ser = serial.Serial(port, baud, timeout=0, write_timeout=0)
            print(f"UART 50Hz initialized on {port}")
        except Exception as e:
            print(f"UART Error: Không tìm thấy cổng {port}. Chế độ chạy không UART.")

        # Luồng vẫn chạy để main.py không bị lỗi khi gọi, nhưng sẽ không ghi vào serial
        self.thread = threading.Thread(target=self._send_loop_50hz, daemon=True)
        self.thread.start()

    def _send_loop_50hz(self):
        interval = 0.02
        next_t = time.perf_counter()
        
        while self.running:
            # Nếu có serial thì mới ghi, không có thì thôi nhưng luồng vẫn chạy
            if self.ser and self.ser.is_open:
                with self.lock:
                    packet = self.last_packet
                try:
                    self.ser.write(packet.encode())
                except:
                    pass
            
            next_t += interval
            sleep_time = next_t - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_t = time.perf_counter()

    def send_error(self, error_x):
        # Hàm này giờ sẽ không bao giờ lỗi AttributeError nữa
        sign = "+" if error_x >= 0 else "-"
        packet = f"S{sign}{min(9999, abs(int(error_x))):04d}E\n"
        with self.lock:
            self.last_packet = packet

    def stop(self):
        self.running = False
        if self.ser:
            self.ser.close()