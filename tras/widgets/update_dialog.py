"""Update check dialog for checking if a newer version is available."""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication  # type: ignore

from tras.utils.upgrade_helper import restart_application, upgrade_tras
from tras.utils.version_check import VersionInfo, check_version


class VersionCheckWorker(QtCore.QThread):
    """Worker thread for checking version in the background."""

    finished = QtCore.pyqtSignal(VersionInfo)

    def run(self):
        """Run the version check."""
        result = check_version()
        self.finished.emit(result)


class UpgradeWorker(QtCore.QThread):
    """Worker thread for upgrading TRAS in the background."""

    finished = QtCore.pyqtSignal(bool, str)  # success, message
    progress = QtCore.pyqtSignal(str)  # progress message

    def __init__(self, new_version: str):
        super().__init__()
        self.new_version = new_version

    def run(self):
        """Run the upgrade."""
        def progress_callback(msg: str):
            self.progress.emit(msg)
        
        self.progress.emit(self.tr("Starting upgrade..."))
        success, message = upgrade_tras(self.new_version, progress_callback=progress_callback)
        self.finished.emit(success, message)


class UpdateCheckDialog(QtWidgets.QDialog):
    """Dialog for checking if updates are available."""

    def __init__(self, parent=None, initial_result: VersionInfo | None = None):
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

        # Progress label (for upgrade)
        self.progress_label = QtWidgets.QLabel()
        self.progress_label.setWordWrap(True)
        self.progress_label.hide()
        layout.addWidget(self.progress_label)

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.check_button = QtWidgets.QPushButton(self.tr("Check Again"))
        self.check_button.clicked.connect(self._on_check_again)
        self.check_button.setEnabled(False)
        button_layout.addWidget(self.check_button)

        self.upgrade_button = QtWidgets.QPushButton(self.tr("Upgrade Now"))
        self.upgrade_button.clicked.connect(self._on_upgrade)
        self.upgrade_button.setEnabled(False)
        self.upgrade_button.hide()
        button_layout.addWidget(self.upgrade_button)

        self.close_button = QtWidgets.QPushButton(self.tr("Close"))
        self.close_button.clicked.connect(self.accept)
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

        # Worker threads
        self.worker = None
        self.upgrade_worker = None
        self.version_result: VersionInfo | None = None
        
        # If initial result provided, use it directly; otherwise check
        if initial_result is not None:
            self._on_version_check_complete(initial_result)
        else:
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
        self.version_result = result

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
            self.upgrade_button.hide()
        else:
            self.status_label.setText(
                self.tr("A newer version is available!")
            )
            self.status_label.setStyleSheet("color: orange;")
            self.upgrade_button.show()
            self.upgrade_button.setEnabled(True)

    def _on_check_again(self):
        """Re-run the version check."""
        self._check_version()

    def _on_upgrade(self):
        """Start the upgrade process."""
        if self.version_result is None or self.version_result.latest_version is None:
            return

        new_version = self.version_result.latest_version
        
        # Confirm upgrade
        reply = QtWidgets.QMessageBox.question(
            self,
            self.tr("Confirm Upgrade"),
            self.tr(
                "This will upgrade TRAS from version {} to version {}.\n\n"
                "The application will restart after the upgrade completes.\n\n"
                "Do you want to continue?"
            ).format(self.version_result.local_version, new_version),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )

        if reply != QtWidgets.QMessageBox.Yes:
            return

        # Disable buttons during upgrade
        self.upgrade_button.setEnabled(False)
        self.check_button.setEnabled(False)
        self.close_button.setEnabled(False)

        # Show progress
        self.progress_label.show()
        self.progress_label.setText(self.tr("Preparing upgrade..."))

        # Start upgrade worker
        self.upgrade_worker = UpgradeWorker(new_version)
        self.upgrade_worker.progress.connect(self._on_upgrade_progress)
        self.upgrade_worker.finished.connect(self._on_upgrade_complete)
        self.upgrade_worker.start()

    def _on_upgrade_progress(self, message: str):
        """Handle upgrade progress updates."""
        self.progress_label.setText(message)
        logger.info(f"Upgrade progress: {message}")

    def _on_upgrade_complete(self, success: bool, message: str):
        """Handle upgrade completion."""
        if success:
            self.progress_label.setText(self.tr("Upgrade successful! Restarting application..."))
            self.progress_label.setStyleSheet("color: green;")
            
            # Restart application after a short delay
            QtCore.QTimer.singleShot(2000, self._restart_app)
        else:
            self.progress_label.setText(self.tr("Upgrade failed: {}").format(message))
            self.progress_label.setStyleSheet("color: red;")
            self.upgrade_button.setEnabled(True)
            self.check_button.setEnabled(True)
            self.close_button.setEnabled(True)

    def _restart_app(self):
        """Restart the application."""
        if restart_application():
            # Close dialog and quit application
            self.accept()
            app = QApplication.instance()
            if app:
                app.quit()
        else:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Upgrade Complete"),
                self.tr(
                    "Upgrade completed successfully!\n\n"
                    "Please restart TRAS manually to use the new version."
                ),
            )
            self.accept()

