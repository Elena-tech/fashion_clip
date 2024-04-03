import tkinter as tk
from tkinter import Button, filedialog
from tkinter.colorchooser import askcolor
from tkinter.simpledialog import askstring

from PIL import Image, ImageDraw, ImageTk

from utils import create_folder, get_filelocation


class DrawingApp:
    def __init__(self, master, save_callback=None):
        self.master = master
        self.master.title("Sketch App")

        self.canvas = tk.Canvas(self.master, bg="white", width=400, height=400)
        self.canvas.pack()
        self.color = "black"  # Default color for drawing lines
        self.width = 5  # Default width for drawing lines

        self.image = Image.new("RGB", (400, 400), "white")
        self.draw = ImageDraw.Draw(self.image)

        self.last_x, self.last_y = None, None
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        self.canvas.bind("<Double-1>", self.add_text_on_double_click)

        self.save_button = Button(self.master, text="Save", command=self.save)
        self.save_button.pack()

        self.clear_button = Button(self.master, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        self.open_button = Button(self.master, text="Open", command=self.open_image)
        self.open_button.pack()

        self.save_callback = save_callback
        self.filename = None
        self.filedir = None

        self.setup_ui()

    def setup_ui(self):
        self.color_button = tk.Button(
            self.master, text="Choose Color", command=self.choose_color
        )
        self.color_button.pack(side="top")

        self.thicker_button = tk.Button(
            self.master, text="Thicker", command=lambda: self.adjust_width(1)
        )
        self.thicker_button.pack(side="top")

        self.thinner_button = tk.Button(
            self.master, text="Thinner", command=lambda: self.adjust_width(-1)
        )
        self.thinner_button.pack(side="top")

    def paint(self, event):
        x1, y1, x2, y2 = (event.x - 1), (event.y - 1), (event.x + 1), (event.y + 1)
        self.canvas.create_line(x1, y1, x2, y2, fill=self.color, width=self.width)
        self.draw.line([x1, y1, x2, y2], fill=self.color, width=self.width)

    def paint_old(self, event):
        paint_color = "black"
        if self.last_x and self.last_y:
            self.canvas.create_line(
                (self.last_x, self.last_y, event.x, event.y), fill=paint_color, width=2
            )
            self.draw.line(
                (self.last_x, self.last_y, event.x, event.y), fill=paint_color, width=2
            )

        self.last_x = event.x
        self.last_y = event.y

    def reset(self, event):
        self.last_x, self.last_y = None, None

    def save(self):
        # print("I am here")
        create_folder()
        self.filedir, self.filename = get_filelocation("sketch.png")
        self.image.save(self.filedir)
        print(f"Image saved as '{self.filename}'.")
        if self.save_callback:
            self.save_callback(self.filedir, self.filename)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (400, 400), "white")
        self.draw = ImageDraw.Draw(self.image)

    def open_image(self):
        # Open a file dialog and get the image file path
        filepath = filedialog.askopenfilename()

        if not filepath:  # If the user cancels the file dialog, do nothing
            return

        try:
            # Pillow versions 8.0.0 and later
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            # Older Pillow versions
            resample = Image.ANTIALIAS
        # Load the image
        self.image = Image.open(filepath)
        self.image.thumbnail((400, 400), resample)

        self.image.paste(
            self.image, (0, 0)
        )  # Paste the resized image onto the canvas' image
        self.draw = ImageDraw.Draw(self.image)  # Update the draw object for drawing

        # Display the image on the canvas
        self.photo_image = ImageTk.PhotoImage(
            self.image
        )  # Convert PIL image to PhotoImage
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)

    def add_text_on_double_click(self, event):
        # Prompt for text on double-click and add it to the canvas
        text = askstring("Text Input", "Enter the text:")
        if text:  # Only add text if the user entered something
            self.canvas.create_text(event.x, event.y, text=text, fill=self.color)
            self.draw.text((event.x, event.y), text, fill=self.color)

    def choose_color(self):
        self.color = askcolor(color=self.color)[1]  # Update the drawing color

    def adjust_width(self, delta):
        self.width = max(1, self.width + delta)
