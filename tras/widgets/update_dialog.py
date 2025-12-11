"""Update check dialog for checking if a newer version is available."""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from tras.utils.version_check import VersionInfo, check_version


class VersionCheckWorker(QtCore.QThread):
    """Worker thread for checking version in the background."""

    finished = QtCore.pyqtSignal(VersionInfo)

    def run(self):
        """Run the version check."""
        result = check_version()
        self.finished.emit(result)


class UpdateCheckDialog(QtWidgets.QDialog):
    """Dialog for checking if updates are available."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Check for Updates"))
        self.setModal(True)
        self.resize(400, 200)

        layout = QtWidgets.QVBoxLayout()

        # Status label
        self.status_label = QtWidgets.QLabel(self.tr("Checking for updates..."))
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Version info label
        self.version_label = QtWidgets.QLabel()
        self.version_label.setWordWrap(True)
        layout.addWidget(self.version_label)

        # Error label (hidden by default)
        self.error_label = QtWidgets.QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: red;")
        self.error_label.hide()
        layout.addWidget(self.error_label)

        layout.addStretch()

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.check_button = QtWidgets.QPushButton(self.tr("Check Again"))
        self.check_button.clicked.connect(self._on_check_again)
        self.check_button.setEnabled(False)
        button_layout.addWidget(self.check_button)

        self.close_button = QtWidgets.QPushButton(self.tr("Close"))
        self.close_button.clicked.connect(self.accept)
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

        # Worker thread
        self.worker = None
        self._check_version()

    def _check_version(self):
        """Start the version check in a background thread."""
        self.status_label.setText(self.tr("Checking for updates..."))
        self.version_label.clear()
        self.error_label.hide()
        self.check_button.setEnabled(False)

        self.worker = VersionCheckWorker()
        self.worker.finished.connect(self._on_version_check_complete)
        self.worker.start()

    def _on_version_check_complete(self, result: VersionInfo):
        """Handle the completion of version check."""
        self.check_button.setEnabled(True)

        if result.error:
            self.status_label.setText(self.tr("Update check failed"))
            self.error_label.setText(result.error)
            self.error_label.show()
            self.version_label.setText(
                self.tr("Local version: {}").format(result.local_version)
            )
            return

        if result.latest_version is None:
            self.status_label.setText(self.tr("Could not determine latest version"))
            self.version_label.setText(
                self.tr("Local version: {}").format(result.local_version)
            )
            return

        self.version_label.setText(
            self.tr("Local version: {}\nLatest version: {}").format(
                result.local_version, result.latest_version
            )
        )

        if result.is_up_to_date:
            self.status_label.setText(
                self.tr("You are using the latest version!")
            )
            self.status_label.setStyleSheet("color: green;")
        else:
            self.status_label.setText(
                self.tr("A newer version is available!")
            )
            self.status_label.setStyleSheet("color: orange;")

    def _on_check_again(self):
        """Re-run the version check."""
        self._check_version()

