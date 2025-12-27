import speech_recognition as sr
import pyttsx3
from groq import Groq
import cv2
import pytesseract
import threading
import tkinter as tk
from tkinter import scrolledtext, messagebox
import numpy as np
import time
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

# -------------------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# -------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Windows Audio Fix
try:
    import pythoncom
except ImportError:
    pythoncom = None

# Beep function
try:
    import winsound
    def beep_sound(freq=1000, duration=300):
        winsound.Beep(freq, duration)
except ImportError:
    def beep_sound(freq=1000, duration=300):
        print(f"BEEP: {freq}Hz for {duration}ms")

# Load environment variables
load_dotenv()

# Get the key securely
API_KEY = os.getenv("GROQ_API_KEY")


# YOLO Files
YOLO_WEIGHTS = "yolov3.weights"
YOLO_CFG = "yolov3.cfg"
COCO_NAMES = "coco.names"

# Vision Settings
CONFIDENCE_THRESHOLD = 0.5 
NMS_THRESHOLD = 0.3        

# -------------------------------------------------------------------------
# 2. SYNONYM MAP
# -------------------------------------------------------------------------
OBJECT_SYNONYMS = {
    'phone': 'cell phone', 'mobile': 'cell phone', 'smartphone': 'cell phone',
    'laptop': 'laptop', 'computer': 'laptop', 'pc': 'laptop',
    'tv': 'tv', 'monitor': 'tv', 'screen': 'tv', 'television': 'tv',
    'cup': 'cup', 'mug': 'cup', 'glass': 'cup',
    'bottle': 'bottle', 'water bottle': 'bottle',
    'keys': 'key', 'key': 'key', 'car keys': 'key',
    'bag': 'handbag', 'backpack': 'backpack', 'purse': 'handbag',
    'remote': 'remote', 'controller': 'remote',
    'person': 'person', 'people': 'person', 'guy': 'person', 'man': 'person', 
    'woman': 'person', 'human': 'person',
    'mouse': 'mouse', 'keyboard': 'keyboard',
    'book': 'book', 'notebook': 'book',
}

def find_tesseract():
    """Find Tesseract installation path"""
    paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe', 
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe', 
        '/usr/bin/tesseract',
        '/usr/local/bin/tesseract'
    ]
    for p in paths:
        if os.path.exists(p): 
            return p
    return None

tesseract_path = find_tesseract()
if tesseract_path: 
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

# -------------------------------------------------------------------------
# 3. MAIN APPLICATION
# -------------------------------------------------------------------------
class UniversalAssistant:
    def __init__(self, root):
        self.root = root
        self.root.title("Universal Access AI - Next Generation")
        self.root.geometry("1200x800")
        self.root.configure(bg="#0a0a0a")

        self.net = None
        self.classes = []
        self.is_listening = False
        self.recognizer = sr.Recognizer()
        self.client = None
        self.model_name = "llama-3.3-70b-versatile"
        self.camera_lock = threading.Lock()
        self.yolo_available = False
        self.ocr_available = tesseract_path is not None
        self.ai_available = False
        self.tts_engine = None
        
        # Conversation history
        self.conversation_history = []

        self._setup_ui()
        self._setup_bindings()
        threading.Thread(target=self._startup_sequence, daemon=True).start()

    def _startup_sequence(self):
        """Initialize all systems"""
        try:
            self.update_status("⚙️ INITIALIZING...", "#ff9900")
            self._fix_files()
            time.sleep(0.3)
            
            # Initialize TTS (Check only)
            self.update_status("🔊 LOADING AUDIO", "#ff9900")
            try:
                test_engine = pyttsx3.init()
                self.log_message("✅ Audio System Ready", "success")
                del test_engine
            except Exception as e:
                logging.error(f"TTS Error: {e}")
                self.log_message("⚠️ TTS Error", "warning")
            
            # Load YOLO
            self.update_status("👁️ LOADING VISION", "#9966ff")
            if self._load_yolo():
                self.yolo_available = True
                self.log_message("✅ Vision System Active", "success")
            else:
                self.log_message("⚠️ YOLO files missing", "warning")
            time.sleep(0.2)
            
            # Connect AI
            self.update_status("🤖 CONNECTING AI", "#00ccff")
            if self._connect_ai():
                self.ai_available = True
                self.log_message("✅ AI Connected", "success")
            else:
                self.log_message("⚠️ AI connection failed", "warning")
            time.sleep(0.2)
            
            # Check OCR
            if self.ocr_available:
                self.log_message("✅ OCR Available", "success")
            else:
                self.log_message("⚠️ Tesseract not found", "warning")
            
            self.update_status("🟢 SYSTEM READY", "#00ff88")
            self.force_speak("System ready. Press space for voice input. Press S to scan text. Press D to describe scene. Press F to find objects. Press H for help.")
            
        except Exception as e:
            logging.error(f"Startup Error: {e}")
            self.update_status("❌ ERROR", "#ff4444")
            self.log_message(f"❌ Error: {e}", "error")

    def force_speak(self, text):
        """Speak text - with proper threading"""
        threading.Thread(target=self._speak_logic, args=(text,), daemon=True).start()

    def _speak_logic(self, text):
        """
        FIXED VERSION: Creates a fresh TTS engine instance per thread.
        This ensures speech works correctly across all features.
        """
        try:
            # Initialize COM for Windows audio
            if pythoncom: 
                pythoncom.CoInitialize()
            
            # Create fresh engine instance for this thread
            engine = pyttsx3.init()
            
            # Configure voice settings
            voices = engine.getProperty('voices')
            if len(voices) > 1: 
                engine.setProperty('voice', voices[1].id)
            engine.setProperty('rate', 160)
            engine.setProperty('volume', 1.0)
            
            # Log to UI on main thread
            self.root.after(0, lambda: self.log_to_chat(f"🔊 {text}", "bot"))
            
            # Speak and wait for completion
            engine.say(text)
            engine.runAndWait()
            
            # Cleanup
            engine.stop()
            
            # Uninitialize COM
            if pythoncom:
                pythoncom.CoUninitialize()
            
        except Exception as e:
            logging.error(f"TTS Error: {e}")
            # Fallback: at least log the message
            self.root.after(0, lambda: self.log_message(f"⚠️ Speech error: {text}", "warning"))

    def _fix_files(self):
        """Fix file naming"""
        files = {"yolov3.cfg.txt": "yolov3.cfg", "coco.names.txt": "coco.names"}
        for bad, good in files.items():
            if os.path.exists(bad):
                try: 
                    os.rename(bad, good)
                except: 
                    pass

    def _load_yolo(self):
        """Load YOLO"""
        if not all(os.path.exists(f) for f in [YOLO_WEIGHTS, YOLO_CFG, COCO_NAMES]):
            return False
        try:
            self.net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            with open(COCO_NAMES, "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
            return True
        except Exception as e:
            logging.error(f"YOLO Load Error: {e}")
            return False

    def get_camera(self):
        """Get camera with fallback"""
        with self.camera_lock:
            try:
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if cap.isOpened(): 
                    return cap
            except:
                pass
            return cv2.VideoCapture(0)

    # ---------------------------------------------------------------------
    # VISION FEATURES
    # ---------------------------------------------------------------------

    def describe_scene(self):
        if not self.yolo_available:
            self.force_speak("Vision system not available. Please download YOLO model files.")
            return
        threading.Thread(target=self._describe_logic, daemon=True).start()

    def _describe_logic(self):
        self.force_speak("Scene analyzer activated. Analyzing what camera sees. Please hold camera steady.")
        self.update_status("👁️ ANALYZING", "#9966ff")
        
        cap = self.get_camera()
        if not cap or not cap.isOpened():
            self.force_speak("Camera error. Please check if camera is connected.")
            self.update_status("🟢 READY", "#00ff88")
            return

        # Warm up camera
        for _ in range(5): 
            cap.read()
        
        ret, frame = cap.read()
        cap.release()

        if not ret: 
            self.force_speak("Capture failed.")
            self.update_status("🟢 READY", "#00ff88")
            return

        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), (0,0,0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        class_ids, confidences, boxes = [], [], []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > CONFIDENCE_THRESHOLD:
                    cx, cy = int(detection[0]*width), int(detection[1]*height)
                    w, h = int(detection[2]*width), int(detection[3]*height)
                    x, y = int(cx - w/2), int(cy - h/2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        else:
            indices = []

        detected_objects = []
        if len(indices) > 0:
            for i in indices.flatten():
                label = str(self.classes[class_ids[i]])
                detected_objects.append(label)

        if not detected_objects:
            self.force_speak("I don't see any recognizable objects in front of the camera. Please move the camera to show different items.")
        else:
            counts = {}
            for obj in detected_objects:
                counts[obj] = counts.get(obj, 0) + 1
            
            desc_parts = []
            for obj, count in counts.items():
                if count == 1:
                    desc_parts.append(f"one {obj}")
                else:
                    desc_parts.append(f"{count} {obj}s")
            
            desc = ", ".join(desc_parts)
            self.force_speak(f"I see: {desc}")
        
        self.update_status("🟢 READY", "#00ff88")

    def find_object(self, command):
        threading.Thread(target=self._find_logic, args=(command,), daemon=True).start()

    def _find_logic(self, command):
        if not self.yolo_available:
            self.force_speak("Vision system not available. Please download YOLO files.")
            return
        
        raw_target = command.lower()
        for phrase in ["find", "where is", "locate", "search for", "look for"]:
            raw_target = raw_target.replace(phrase, "")
        for word in ["my", "the", "a", "an"]:
            raw_target = raw_target.replace(word, "")
        raw_target = raw_target.strip()

        target = raw_target
        for syn, real in OBJECT_SYNONYMS.items():
            if syn in raw_target:
                target = real
                break

        self.force_speak(f"Object finder activated. Searching for {target}. Please slowly point camera around the area.")
        self.update_status(f"🔍 FINDING {target.upper()}", "#ff4444")
        
        cap = self.get_camera()
        if not cap or not cap.isOpened(): 
            self.force_speak("Camera error. Please check your camera connection.")
            self.update_status("🟢 READY", "#00ff88")
            return

        start = time.time()
        found = False
        
        while time.time() - start < 20 and not found:
            ret, frame = cap.read()
            if not ret: 
                break
            
            h, w, _ = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), (0,0,0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.net.getUnconnectedOutLayersNames())

            boxes, confidences, class_ids = [], [], []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > CONFIDENCE_THRESHOLD:
                        cx, cy = int(detection[0]*w), int(detection[1]*h)
                        bw, bh = int(detection[2]*w), int(detection[3]*h)
                        x, y = int(cx - bw/2), int(cy - bh/2)
                        boxes.append([x, y, bw, bh])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
                if len(indices) > 0:
                    for i in indices.flatten():
                        label = str(self.classes[class_ids[i]])
                        x, y, bw, bh = boxes[i]
                        
                        if target.lower() in label.lower() or label.lower() in target.lower():
                            color = (0, 255, 0)
                            cv2.rectangle(frame, (x, y), (x+bw, y+bh), color, 3)
                            cv2.putText(frame, f"FOUND: {label}", (x, y-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
                            self.force_speak(f"Found your {target}! Look at the screen. A green box is highlighting the object.")
                            beep_sound(1000, 300)
                            found = True
                            
                            cv2.imshow("Object Found!", frame)
                            cv2.waitKey(3000)
                            break
                        else:
                            color = (255, 0, 0)
                            cv2.rectangle(frame, (x, y), (x+bw, y+bh), color, 2)
                            cv2.putText(frame, label, (x, y-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if not found:
                cv2.imshow("Searching...", frame)
                if cv2.waitKey(1) == ord('q'): 
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if not found: 
            self.force_speak(f"Could not find {target} in the camera view. Please try pointing the camera at different areas and try again.")
        
        self.update_status("🟢 READY", "#00ff88")

    def capture_and_read(self):
        if not self.ocr_available:
            self.force_speak("OCR not available. Please install Tesseract OCR software.")
            return
        self.update_status("📷 SCANNING", "#ff4444")
        self.force_speak("Text scanner activated. Hold text steady in front of camera. Scanning will begin in 3 seconds.")
        threading.Thread(target=self._ocr_logic, daemon=True).start()

    def _ocr_logic(self):
        cap = self.get_camera()
        if not cap or not cap.isOpened(): 
            self.force_speak("Camera error. Please check camera connection.")
            self.update_status("🟢 READY", "#00ff88")
            return

        self.force_speak("Prepare text now. Countdown starting. Three. Two. One.")
        start = time.time()
        while time.time() - start < 3:
            ret, frame = cap.read()
            if ret: 
                cv2.putText(frame, "ALIGN TEXT", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("Text Scanner", frame)
            if cv2.waitKey(1) == ord('q'): 
                break
        
        beep_sound(1000, 150)
        ret, frame = cap.read()
        cap.release()
        cv2.destroyAllWindows()

        if ret:
            self.force_speak("Image captured. Processing text now. Please wait.")
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.medianBlur(gray, 3)
                thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                
                text = pytesseract.image_to_string(thresh, config='--psm 6').strip()
                
                if len(text) < 2: 
                    self.force_speak("No text detected in the image. Please try again with better lighting and make sure text is clearly visible.")
                else:
                    self.log_message(f"📄 Detected Text:\n{text}", "info")
                    self.force_speak("Text successfully detected. Reading the text now.")
                    time.sleep(0.5)
                    self.force_speak(text)
            except Exception as e:
                logging.error(f"OCR Error: {e}")
                self.force_speak("Error reading text from image. Please ensure text is clear and well-lit, then try again.")
        
        self.update_status("🟢 READY", "#00ff88")

    def ask_ai(self, prompt):
        if not self.ai_available:
            self.force_speak("AI service not connected. Please check your internet connection and API key.")
            return
        threading.Thread(target=self._ai_logic, args=(prompt,), daemon=True).start()

    def _ai_logic(self, prompt):
        try:
            self.update_status("🤖 THINKING", "#00ccff")
            
            # Build conversation context
            messages = [{"role": "system", "content": "You are a helpful, friendly AI assistant for a universal accessibility interface. Keep responses conversational, clear, and concise. You help users with vision, hearing, or mobility challenges navigate their environment and access information."}]
            
            # Add recent conversation history for context
            for entry in self.conversation_history[-6:]:
                messages.append(entry)
            
            # Add current prompt
            messages.append({"role": "user", "content": prompt})
            
            # Get AI response
            chat = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                max_tokens=300,
                temperature=0.7
            )
            
            response = chat.choices[0].message.content
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Speak the response (this now works correctly!)
            self.force_speak(response)
            self.update_status("🟢 READY", "#00ff88")
            
        except Exception as e:
            logging.error(f"AI Error: {e}")
            self.force_speak("AI connection error. Please check your internet connection and try again.")
            self.update_status("🟢 READY", "#00ff88")

    def process_command(self, command):
        cmd = command.lower().strip()
        
        # Log what user said
        self.log_message(f"📝 You asked: {command}", "user")
        
        # Command routing
        if any(w in cmd for w in ["scan", "read text", "ocr"]):
            self.capture_and_read()
        elif any(w in cmd for w in ["find", "where", "locate"]):
            self.find_object(cmd)
        elif any(w in cmd for w in ["describe", "see", "look", "what do you see"]):
            self.describe_scene()
        elif any(w in cmd for w in ["exit", "quit", "goodbye", "shut down"]):
            self.force_speak("Shutting down system. Goodbye!")
            time.sleep(2)
            self.root.quit()
        elif any(w in cmd for w in ["help", "commands"]):
            help_text = "Available commands: Say scan text to read documents. Say describe scene to see what's in front of camera. Say find my phone or find my keys to locate objects. Press H key for detailed help. You can also ask me any question."
            self.force_speak(help_text)
        else:
            # For any other question, send to AI and speak response
            self.ask_ai(command)

    # ---------------------------------------------------------------------
    # UI SETUP
    # ---------------------------------------------------------------------

    def _setup_ui(self):
        # Main container
        main = tk.Frame(self.root, bg="#0a0a0a")
        main.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        tk.Label(main, text="◉ UNIVERSAL ACCESS AI", bg="#0a0a0a", 
                 fg="#00ffcc", font=("Arial", 28, "bold")).pack(pady=10)
        
        tk.Label(main, text="Next-Generation Accessibility Platform", bg="#0a0a0a",
                 fg="#666666", font=("Arial", 12)).pack()
        
        # Status
        self.status_label = tk.Label(main, text="⚙️ STARTING", bg="#ff9900",
                                     fg="white", font=("Arial", 12, "bold"),
                                     padx=20, pady=10)
        self.status_label.pack(pady=15)
        
        # Content frame
        content = tk.Frame(main, bg="#0a0a0a")
        content.pack(fill="both", expand=True)
        
        # Chat (left)
        left = tk.Frame(content, bg="#0a0a0a")
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        tk.Label(left, text="💬 CONVERSATION", bg="#0a0a0a",
                 fg="#00ffcc", font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 5))
        
        self.chat_display = scrolledtext.ScrolledText(
            left, font=("Consolas", 11), bg="#1a1a1a", fg="#ffffff",
            wrap=tk.WORD, padx=15, pady=15
        )
        self.chat_display.pack(fill="both", expand=True)
        
        self.chat_display.tag_config("bot", foreground="#00ffcc")
        self.chat_display.tag_config("user", foreground="#4d9eff")
        self.chat_display.tag_config("system", foreground="#ffaa00")
        self.chat_display.tag_config("success", foreground="#00ff88")
        self.chat_display.tag_config("warning", foreground="#ffaa00")
        self.chat_display.tag_config("error", foreground="#ff4444")
        self.chat_display.tag_config("info", foreground="#888888")
        
        # Controls (right)
        right = tk.Frame(content, bg="#0a0a0a", width=250)
        right.pack(side="right", fill="y", padx=(10, 0))
        
        tk.Label(right, text="⚡ CONTROLS", bg="#0a0a0a",
                 fg="#00ffcc", font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 10))
        
        # Buttons
        btn_config = {"font": ("Arial", 11, "bold"), "width": 20, "height": 3, "relief": "flat", "cursor": "hand2"}
        
        tk.Button(right, text="📷 SCAN TEXT\n(Press S)", bg="#ff4466", fg="white",
                  command=self.capture_and_read, **btn_config).pack(pady=5)
        
        tk.Button(right, text="🎤 VOICE INPUT\n(Press SPACE)", bg="#00cc66", fg="white",
                  command=self.start_listening_thread, **btn_config).pack(pady=5)
        
        tk.Button(right, text="👁️ DESCRIBE SCENE\n(Press D)", bg="#9966ff", fg="white",
                  command=self.describe_scene, **btn_config).pack(pady=5)
        
        tk.Button(right, text="🔍 FIND OBJECT\n(Press F)", bg="#ff8800", fg="white",
                  command=self.prompt_find_object, **btn_config).pack(pady=5)
        
        tk.Button(right, text="❓ HELP\n(Press H)", bg="#4d9eff", fg="white",
                  command=self.show_help, **btn_config).pack(pady=5)
        
        # Shortcuts info
        shortcuts = tk.Frame(right, bg="#1a1a1a")
        shortcuts.pack(fill="x", pady=(15, 0))
        
        tk.Label(shortcuts, text="⌨️ SHORTCUTS", bg="#1a1a1a",
                 fg="#00ffcc", font=("Arial", 10, "bold")).pack(pady=5)
        
        for key, action in [("SPACE", "Voice"), ("S", "Scan"), ("D", "Describe"), 
                            ("F", "Find"), ("H", "Help"), ("ESC", "Exit")]:
            tk.Label(shortcuts, text=f"[{key}] {action}", bg="#1a1a1a",
                     fg="#888888", font=("Consolas", 9)).pack()

    def prompt_find_object(self):
        self.force_speak("Object finder activated. What object should I find? For example, say phone, laptop, keys, or remote.")
        self.start_listening_thread()

    def _setup_bindings(self):
        self.root.bind('<space>', lambda e: self.start_listening_thread())
        self.root.bind('<s>', lambda e: self.capture_and_read())
        self.root.bind('<S>', lambda e: self.capture_and_read())
        self.root.bind('<d>', lambda e: self.describe_scene())
        self.root.bind('<D>', lambda e: self.describe_scene())
        self.root.bind('<f>', lambda e: self.prompt_find_object())
        self.root.bind('<F>', lambda e: self.prompt_find_object())
        self.root.bind('<h>', lambda e: self.show_help())
        self.root.bind('<H>', lambda e: self.show_help())
        self.root.bind('<Escape>', lambda e: self.root.quit())

    def show_help(self):
        help_text = """UNIVERSAL ACCESS AI - HELP GUIDE

🎤 VOICE COMMANDS:
• Press SPACE or click "VOICE INPUT"
• Say: "Scan text" or "Read this"
• Say: "Describe scene" or "What do you see"
• Say: "Find my phone" or "Where is my laptop"
• Ask any question - the AI will answer

⌨️ KEYBOARD SHORTCUTS:
• SPACE - Voice input
• S - Scan text with camera
• D - Describe scene
• F - Find object
• H - Show this help
• ESC - Exit application

🔧 FEATURES:
✅ Text-to-Speech - All responses spoken aloud
✅ Voice Control - Hands-free operation
✅ OCR - Read text from camera
✅ Object Detection - Find items with YOLO
✅ Scene Description - Understand environment
✅ AI Chat - Conversational assistance

📋 REQUIREMENTS:
• Webcam for vision features
• Microphone for voice input
• Internet connection for AI chat
• YOLO files (yolov3.weights, yolov3.cfg, coco.names)
• Tesseract OCR installed

💡 TIPS:
• Speak clearly after the beep
• Keep camera steady for best results
• Use well-lit areas for text scanning
• Ask follow-up questions for context"""
        
        messagebox.showinfo("Help Guide", help_text)
        self.force_speak("Help guide displayed on screen.")

    def _connect_ai(self):
        """Connect to Groq AI"""
        if not API_KEY: 
            return False
        try:
            self.client = Groq(api_key=API_KEY)
            # Test connection with minimal request
            self.client.chat.completions.create(
                messages=[{"role": "user", "content": "test"}],
                model=self.model_name,
                max_tokens=5
            )
            return True
        except Exception as e:
            logging.error(f"AI Connection Error: {e}")
            return False

    def log_to_chat(self, text, tag):
        """Add message to chat display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_display.insert(tk.END, f"[{timestamp}] {text}\n", tag)
        self.chat_display.see(tk.END)
    
    def log_message(self, text, tag):
        """Thread-safe logging to chat"""
        self.root.after(0, lambda: self.log_to_chat(text, tag))

    def start_listening_thread(self):
        """Start voice listening in background"""
        if not self.is_listening: 
            threading.Thread(target=self.listen_and_process, daemon=True).start()

    def listen_and_process(self):
        """Listen for voice input and process command"""
        self.is_listening = True
        self.update_status("👂 LISTENING", "#ffcc00")
        
        try:
            with sr.Microphone() as source:
                beep_sound(600, 100)
                self.log_message("🎤 Listening... Speak now!", "system")
                self.force_speak("I'm listening. Please speak now.")
                
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen for audio
                audio = self.recognizer.listen(source, timeout=8, phrase_time_limit=10)
                
                beep_sound(800, 100)
                self.log_message("🔄 Processing your speech...", "system")
                
                # Recognize speech
                cmd = self.recognizer.recognize_google(audio)
                self.log_message(f"✅ Recognized: {cmd}", "user")
                
                # Process the command
                self.process_command(cmd)
        
        except sr.WaitTimeoutError:
            self.force_speak("No speech detected. Please try again and speak clearly after the beep sound.")
            self.log_message("⚠️ No speech detected - Timeout", "warning")
        except sr.UnknownValueError:
            self.force_speak("Could not understand what you said. Please try again and speak more clearly.")
            self.log_message("⚠️ Speech not understood", "warning")
        except sr.RequestError as e:
            self.force_speak("Speech recognition service error. Please check your internet connection.")
            self.log_message(f"❌ Service error: {e}", "error")
        except Exception as e:
            logging.error(f"Listening Error: {e}")
            self.force_speak("Microphone error. Please check that your microphone is connected and working properly.")
            self.log_message(f"❌ Microphone error: {e}", "error")
        
        finally:
            self.is_listening = False
            self.update_status("🟢 READY", "#00ff88")

    def update_status(self, text, color):
        """Update status label (thread-safe)"""
        self.root.after(0, lambda: self.status_label.config(text=text, bg=color))

# -------------------------------------------------------------------------
# MAIN ENTRY POINT
# -------------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = UniversalAssistant(root)
    root.mainloop()