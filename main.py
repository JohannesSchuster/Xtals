import sys
from PyQt5.QtWidgets import QApplication
from .src.gui.interface_window import InterfaceWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InterfaceWindow()
    sys.exit(app.exec_())
