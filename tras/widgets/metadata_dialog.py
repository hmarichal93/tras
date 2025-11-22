"""
Metadata Dialog - Input sample information for tree ring analysis
"""
from PyQt5 import QtCore, QtWidgets


class MetadataDialog(QtWidgets.QDialog):
    """Dialog to input sample metadata (harvest year, observation, sample code)"""
    
    def __init__(self, existing_metadata=None, parent=None, default_sample_code: str = ""):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Sample Metadata"))
        self.setModal(True)
        self.resize(450, 250)
        
        layout = QtWidgets.QVBoxLayout()
        
        # Info label
        info_label = QtWidgets.QLabel(
            self.tr("Enter sample metadata. The outermost ring (closest to bark, which circumscribes "
                   "all others) will be labeled with the harvested year. Inner rings will be labeled "
                   "with decreasing years toward the pith.")
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("padding: 10px; background: #f0f0f0; border-radius: 5px;")
        layout.addWidget(info_label)
        
        # Form layout
        form_layout = QtWidgets.QFormLayout()
        form_layout.setContentsMargins(10, 20, 10, 10)
        form_layout.setSpacing(15)
        
        # Harvested Year
        self.year_input = QtWidgets.QSpinBox()
        self.year_input.setRange(1000, 3000)
        self.year_input.setValue(existing_metadata.get('harvested_year', 2024) if existing_metadata else 2024)
        self.year_input.setToolTip(self.tr("Year when the sample was harvested"))
        form_layout.addRow(self.tr("Harvested Year:"), self.year_input)
        
        # Sample Code
        self.sample_input = QtWidgets.QLineEdit()
        if existing_metadata:
            default_code = existing_metadata.get('sample_code', default_sample_code)
        else:
            default_code = default_sample_code
        self.sample_input.setText(default_code)
        self.sample_input.setPlaceholderText(self.tr("e.g., F02c, TREE-001"))
        self.sample_input.setToolTip(self.tr("Unique identifier for this sample"))
        form_layout.addRow(self.tr("Sample Code:"), self.sample_input)
        
        # Observation
        self.observation_input = QtWidgets.QTextEdit()
        self.observation_input.setPlainText(existing_metadata.get('observation', '') if existing_metadata else '')
        self.observation_input.setPlaceholderText(self.tr("Any notes or observations about this sample..."))
        self.observation_input.setMaximumHeight(80)
        self.observation_input.setToolTip(self.tr("Optional notes about the sample"))
        form_layout.addRow(self.tr("Observation:"), self.observation_input)
        
        layout.addLayout(form_layout)
        
        # Example label
        example_label = QtWidgets.QLabel(
            self.tr("Example: Harvested in 2020 with 3 rings → ring_2020 (outermost/bark), "
                   "ring_2019 (middle), ring_2018 (innermost/pith)")
        )
        example_label.setStyleSheet("color: gray; font-style: italic; padding: 5px;")
        example_label.setWordWrap(True)
        layout.addWidget(example_label)
        
        layout.addStretch()
        
        # Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def get_metadata(self):
        """Return the metadata as a dictionary"""
        return {
            'harvested_year': self.year_input.value(),
            'sample_code': self.sample_input.text().strip(),
            'observation': self.observation_input.toPlainText().strip()
        }
