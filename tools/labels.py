import os
import glob
import sys
import argparse
from pathlib import Path


from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QGridLayout, QProgressBar, QLineEdit, QAction

# from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtCore


class App(QWidget):

    def __init__(self, image_folder):
        super().__init__()
        self.title = "Text Recognition LABELER"
        self.left = 50
        self.top = 50
        self.width = 800
        self.height = 200

        self.list_file = list(map(str, Path(image_folder).rglob("*.jpg")))
        self.list_file.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
        self.total_file = len(self.list_file)
        self.current_index = 0
        self.current_file = self.list_file[self.current_index]
        self.layout = QGridLayout()
        self.initUI()

    def press_next(self):
        if self.current_index < self.total_file - 1:
            # Update
            self.do_save()
            self.current_index += 1
            self.current_file = self.list_file[self.current_index]
            self.progress.setValue(self.current_index + 1)
            self.update()

    def press_previous(self):
        if self.current_index > 0:
            self.do_save()
            self.current_index -= 1
            self.current_file = self.list_file[self.current_index]
            # Update
            self.progress.setValue(self.current_index + 1)
            self.update()

    def do_save(self):
        label_file = self.current_file.split(".")[0] + ".txt"
        label_text = self.textbox.text()
        with open(label_file, "w+", encoding="utf-8") as f:
            f.write(label_text)

    def get_size(self, width, height):
        max_width = 700
        if width > max_width:
            return (max_width, int(height / width * max_width))
        else:
            return (width, height)

    def update(self):
        # Update image
        pixmap = QPixmap(self.list_file[self.current_index])
        width, height = self.get_size(pixmap.width(), pixmap.height())
        self.label.setPixmap(pixmap.scaled(width, height))

        self.label.resize(width, height)
        self.update_text()

    def update_text(self):
        # Update label text
        self.textbox.clear()
        label_file = self.current_file.split(".")[0] + ".txt"
        if os.path.exists(label_file) and os.path.getsize(label_file) > 0:
            label = open(label_file, "r", encoding="utf-8").read().strip()
            self.textbox.setText(label)

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        # Init progressbar
        self.progress = QProgressBar()
        self.progress.setMaximum(self.total_file)

        # Create widget
        # Put image
        self.label = QLabel()
        pixmap = QPixmap(self.list_file[self.current_index])
        width, height = self.get_size(pixmap.width(), pixmap.height())
        self.label.setPixmap(pixmap.scaled(width, height))
        self.label.resize(width, height)

        # Text box
        self.textbox = QLineEdit()
        self.textbox.resize(280, 40)
        font = self.textbox.font()
        font.setPointSize(18)
        self.textbox.setFont(font)
        self.update_text()

        # Button
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.press_next)
        self.previous_button = QPushButton("Previous")
        self.previous_button.clicked.connect(self.press_previous)
        self.previous_button.setShortcut("Ctrl+Return")

        # Add all widgets
        self.layout.addWidget(self.label, 0, 0, 1, -1)
        self.layout.addWidget(self.textbox, 1, 0, 1, 5)
        self.layout.addWidget(self.next_button, 1, 5, 1, 1)
        self.layout.addWidget(self.progress, 2, 0, 1, 5)
        self.layout.addWidget(self.previous_button, 2, 5, 1, 1)

        # Add Keypress Event
        self.setEnterAction = QAction("Set Enter", self, shortcut=Qt.Key_Return, triggered=self.press_next)
        self.addAction(self.setEnterAction)
        self.layout.addWidget(QPushButton("Enter", self, clicked=self.setEnterAction.triggered))

        self.setLayout(self.layout)
        self.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--input_dir", type=str, required=True, help="Path to image folder")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    ex = App(args.input_dir)
    sys.exit(app.exec_())
