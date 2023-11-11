import sys
import socket
import json
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
from PyQt5.QtCore import pyqtSlot, QThread, pyqtSignal

class ServerThread(QThread):
    update_signal = pyqtSignal(str)

    def __init__(self):
        QThread.__init__(self)
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.host = 'localhost'
        self.port = 65432
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)

    def run(self):
        conn, addr = self.server_socket.accept()
        print(f"Connection from {addr} has been established.")
        while True:
            data = conn.recv(4096).decode('utf-8')
            if data:
                self.update_signal.emit(data)
            else:
                break
        conn.close()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 800, 600)
        self.table = QTableWidget(self)
        self.table.setSortingEnabled(True)  # Enable sorting
        layout = QVBoxLayout()
        layout.addWidget(self.table)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.server_thread = ServerThread()
        self.server_thread.update_signal.connect(self.update_table)
        self.server_thread.start()

    @pyqtSlot(str)
    def update_table(self, data):
        try:
            tasks_data = json.loads(data)
            tasks_data.sort(key=lambda x: int(x['id']))

            # Find out the maximum number of requirements for column count
            max_reqs = max(len(task['currentReqs']) for task in tasks_data)
            self.table.setRowCount(len(tasks_data))
            self.table.setColumnCount(max_reqs + 2)  # for ID, requirements, allocated agents

            headers = ["ID"] + [f"Req {i+1}" for i in range(max_reqs)] + ["Allocated Agents"]
            self.table.setHorizontalHeaderLabels(headers)

            for row, task in enumerate(tasks_data):
                self.table.setItem(row, 0, QTableWidgetItem(task['id']))
                for col, req in enumerate(task['currentReqs']):
                    self.table.setItem(row, col+1, QTableWidgetItem(str(req)))
                agents = ', '.join(map(str, task['allocatedAgents']))  # Convert agent IDs to a comma-separated string
                self.table.setItem(row, max_reqs + 1, QTableWidgetItem(agents))
            
            # Resize columns to fit contents
            self.table.resizeColumnsToContents()

            # Allow the user to sort by clicking on the header
            header = self.table.horizontalHeader()
            header.sectionClicked.connect(self.onHeaderClicked)

        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")

    def onHeaderClicked(self, logicalIndex):
        self.table.sortItems(logicalIndex)


app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
